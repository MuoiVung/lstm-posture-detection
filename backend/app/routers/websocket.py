"""
WebSocket router for real-time pose streaming.

Receives JPEG frames from the frontend, processes them through
MediaPipe + LSTM, and returns posture predictions.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import time
import logging

from app.services.pose_service import get_pose_service
from app.services.posture_service import get_posture_service
from app.services.health_service import HealthService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/pose")
async def pose_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time posture detection.

    Protocol:
        Client sends: Binary JPEG frame data
        Server responds: JSON with posture prediction and health risks
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    pose_service = get_pose_service()
    posture_service = get_posture_service()
    health_service = HealthService()

    frame_count = 0
    last_prediction = None

    try:
        while True:
            # Receive message (either frame bytes or text command)
            msg = await websocket.receive()
            if "text" in msg:
                try:
                    cmd = json.loads(msg["text"])
                    if cmd.get("action") == "calibrate":
                        success = posture_service.calibrate()
                        if success:
                            health_service.reset()
                        await websocket.send_json({
                            "type": "calibration_result", 
                            "success": success,
                            "message": "Calibrated successfully!" if success else "Need more frames to calibrate."
                        })
                except Exception as e:
                    logger.error(f"Failed parsing text message: {e}")
                continue

            if "bytes" not in msg:
                continue
                
            data = msg["bytes"]
            frame_count += 1

            # Extract pose landmarks
            pose_result = pose_service.extract_landmarks(data)

            if pose_result is None:
                # No pose detected
                await websocket.send_json({
                    "type": "no_pose",
                    "message": "No person detected in frame",
                    "frame": frame_count,
                })
                continue

            # Feed landmarks to LSTM service
            prediction = posture_service.add_frame(
                pose_result["key_landmarks"]
            )

            response = {
                "type": "pose_update",
                "frame": frame_count,
                "landmarks": pose_result["all_landmarks"],
                "timestamp": time.time(),
            }

            if prediction is not None:
                last_prediction = prediction

                # Update health tracking
                duration = health_service.update_tracking(
                    prediction["posture_class"],
                    frame_interval=0.067,  # ~15fps
                )

                # Get health risks
                risks = health_service.get_health_risks(
                    prediction["posture_class"],
                    duration_seconds=duration,
                )

                response["type"] = "prediction"
                response["posture"] = {
                    "class": prediction["posture_class"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction.get("all_probs", {}),
                    "is_good": prediction["posture_class"] in ["good_posture", "needs_calibration"],
                }
                response["health_risks"] = [
                    {
                        "name": r.name,
                        "description": r.description,
                        "severity": r.severity,
                        "body_part": r.body_part,
                        "recommendation": r.recommendation,
                    }
                    for r in risks
                ]
                response["tracking"] = health_service.get_summary()
            elif last_prediction is not None:
                # Send last known prediction with landmark update
                response["posture"] = {
                    "class": last_prediction["posture_class"],
                    "confidence": last_prediction["confidence"],
                    "is_good": last_prediction["posture_class"] == "good_posture",
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        posture_service.reset_buffer()
        health_service.reset()
