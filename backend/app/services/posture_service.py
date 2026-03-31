"""
Posture inference service using calibration-based relative comparison.

Maintains a frame buffer and compares current pose against
the user's calibrated "good posture" baseline.
"""

import torch
import numpy as np
import os
import logging
from typing import Optional
from collections import deque

from app.config import settings
from app.ml.model import PostureLSTM, get_device

logger = logging.getLogger(__name__)


# Upper-body landmark names (reliably visible from webcam)
UPPER_BODY_LANDMARKS = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]


def extract_relative_features(landmarks: dict) -> np.ndarray:
    """Extract webcam-friendly features normalized to shoulder center.

    Returns a feature vector of shape (21,):
      - 9 landmarks × 2 (x, y) relative to shoulder center = 18
      - shoulder_width (proxy for distance to camera)
      - shoulder_tilt_angle
      - head_tilt_ratio
    """
    # Shoulder center as origin
    left_sh = np.array(landmarks["left_shoulder"][:2], dtype=np.float32)
    right_sh = np.array(landmarks["right_shoulder"][:2], dtype=np.float32)
    sh_center = (left_sh + right_sh) / 2

    # Shoulder width as scale factor (proxy for distance to camera)
    sh_width = float(np.linalg.norm(left_sh - right_sh))
    if sh_width < 0.01:
        sh_width = 0.01  # prevent division by zero

    features = []

    # Relative positions of each landmark (x, y only) normalized by shoulder width
    for name in UPPER_BODY_LANDMARKS:
        lm = np.array(landmarks[name][:2], dtype=np.float32)
        rel = (lm - sh_center) / sh_width
        features.extend([float(rel[0]), float(rel[1])])

    # Shoulder width (absolute - proxy for forward/backward lean)
    features.append(sh_width)

    # Shoulder tilt: y-difference between left and right shoulder / width
    shoulder_tilt = (left_sh[1] - right_sh[1]) / sh_width
    features.append(float(shoulder_tilt))

    # Head tilt: nose offset from shoulder center x / width
    nose = np.array(landmarks["nose"][:2], dtype=np.float32)
    head_tilt = (nose[0] - sh_center[0]) / sh_width
    features.append(float(head_tilt))

    return np.array(features, dtype=np.float32)


class PostureService:
    """Real-time posture inference using calibration-based comparison.

    Flow:
      1. User starts camera → frames are sent to add_frame()
      2. User clicks "Calibrate" → calibrate() averages recent frames as baseline
      3. Monitoring begins → _calibrated_prediction() compares current vs baseline
    """

    CALIBRATION_FRAMES = 25  # ~2.5 seconds (at 10fps processing 5fps)

    def __init__(self):
        self.device = get_device()
        self.model: Optional[PostureLSTM] = None
        self.frame_buffer: deque = deque(maxlen=60)
        self.frame_count = 0

        # Calibration state
        self.is_calibrating = False
        self.calibration_buffer: list = []
        self.baseline: Optional[np.ndarray] = None
        self.baseline_shoulder_width: Optional[float] = None

    def add_frame(self, landmarks: dict) -> Optional[dict]:
        """Process a frame and return prediction if calibrated.

        Args:
            landmarks: Dict of landmark_name -> [x, y, z] coordinates

        Returns:
            Prediction dict or status dict
        """
        self.frame_count += 1

        # Skip frames for performance (process every 2nd frame)
        if self.frame_count % 2 != 0:
            return None

        # Extract features
        try:
            features = extract_relative_features(landmarks)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

        # If calibrating, collect frames
        if self.is_calibrating:
            self.calibration_buffer.append(features)
            progress = len(self.calibration_buffer) / self.CALIBRATION_FRAMES
            if len(self.calibration_buffer) >= self.CALIBRATION_FRAMES:
                self._finish_calibration()
                return {
                    "posture_class": "calibration_complete",
                    "confidence": 1.0,
                    "all_probs": {},
                    "calibration_progress": 1.0,
                }
            return {
                "posture_class": "calibrating",
                "confidence": progress,
                "all_probs": {},
                "calibration_progress": progress,
            }

        # Add to rolling buffer
        self.frame_buffer.append(features)

        # Need calibration first
        if self.baseline is None:
            return {
                "posture_class": "needs_calibration",
                "confidence": 1.0,
                "all_probs": {},
            }

        # Run prediction
        return self._calibrated_prediction()

    def start_calibration(self) -> bool:
        """Begin the calibration process."""
        self.is_calibrating = True
        self.calibration_buffer = []
        logger.info("Calibration started - collecting frames...")
        return True

    def _finish_calibration(self):
        """Finalize calibration from collected frames."""
        if len(self.calibration_buffer) == 0:
            return

        all_features = np.array(self.calibration_buffer)
        self.baseline = np.mean(all_features, axis=0)

        # Store baseline shoulder width separately (index 18)
        self.baseline_shoulder_width = float(self.baseline[18])

        self.is_calibrating = False
        self.calibration_buffer = []
        logger.info(f"Calibration complete! Baseline captured ({len(all_features)} frames)")

    def _calibrated_prediction(self) -> dict:
        """Prediction based on deviation from calibrated baseline."""
        if len(self.frame_buffer) < 3:
            return {
                "posture_class": "good_posture",
                "confidence": 0.5,
                "all_probs": {},
            }

        # Average last few frames for stability
        recent = np.mean(list(self.frame_buffer)[-5:], axis=0)
        diff = recent - self.baseline

        # Feature layout:
        # [0:18]  = 9 landmarks × 2 (x, y) relative positions
        # [18]    = shoulder_width
        # [19]    = shoulder_tilt
        # [20]    = head_tilt (nose x offset)

        # Key differences to detect posture:
        nose_x_diff = diff[0]   # nose relative x shift
        nose_y_diff = diff[1]   # nose relative y shift (down = positive)
        shoulder_width_diff = diff[18]  # wider = leaning forward
        shoulder_tilt_diff = diff[19]   # positive = left shoulder lower
        head_tilt_diff = diff[20]       # nose x offset change

        # Left ear / right ear y to detect lateral lean
        left_ear_y_diff = diff[3]   # left_ear y
        right_ear_y_diff = diff[5]  # right_ear y

        posture = "good_posture"
        confidence = 0.90
        scores = {
            "good_posture": 0.0,
            "forward_lean": 0.0,
            "backward_lean": 0.0,
            "left_lean": 0.0,
            "right_lean": 0.0,
        }

        # Forward lean: nose moves down AND/OR shoulders get wider (closer to camera)
        forward_score = 0.0
        if nose_y_diff > 0.08:
            forward_score += min((nose_y_diff - 0.05) / 0.15, 1.0) * 0.6
        if shoulder_width_diff > 0.02:
            forward_score += min((shoulder_width_diff - 0.01) / 0.08, 1.0) * 0.4
        scores["forward_lean"] = forward_score

        # Backward lean: nose moves up AND/OR shoulders get narrower
        backward_score = 0.0
        if nose_y_diff < -0.06:
            backward_score += min((abs(nose_y_diff) - 0.04) / 0.12, 1.0) * 0.6
        if shoulder_width_diff < -0.02:
            backward_score += min((abs(shoulder_width_diff) - 0.01) / 0.06, 1.0) * 0.4
        scores["backward_lean"] = backward_score

        # Left lean: head shifts left + shoulder tilt
        left_score = 0.0
        if nose_x_diff < -0.08:
            left_score += min((abs(nose_x_diff) - 0.05) / 0.15, 1.0) * 0.5
        if shoulder_tilt_diff > 0.06:
            left_score += min((shoulder_tilt_diff - 0.03) / 0.1, 1.0) * 0.5
        scores["left_lean"] = left_score

        # Right lean: head shifts right + shoulder tilt
        right_score = 0.0
        if nose_x_diff > 0.08:
            right_score += min((nose_x_diff - 0.05) / 0.15, 1.0) * 0.5
        if shoulder_tilt_diff < -0.06:
            right_score += min((abs(shoulder_tilt_diff) - 0.03) / 0.1, 1.0) * 0.5
        scores["right_lean"] = right_score

        # Determine posture based on highest deviation score
        max_score = max(scores.values())
        if max_score > 0.25:  # threshold to trigger bad posture
            posture = max(scores, key=scores.get)
            confidence = min(0.5 + max_score * 0.5, 0.99)
        else:
            posture = "good_posture"
            confidence = max(0.7, 1.0 - max_score * 2)

        # Build probability dict
        total = sum(scores.values()) + 0.01
        good_prob = max(0.0, 1.0 - total)
        all_probs = {"good_posture": good_prob}
        for k, v in scores.items():
            if k != "good_posture":
                all_probs[k] = v

        # Normalize probabilities
        prob_total = sum(all_probs.values())
        if prob_total > 0:
            all_probs = {k: v / prob_total for k, v in all_probs.items()}

        return {
            "posture_class": posture,
            "confidence": confidence,
            "all_probs": all_probs,
        }

    def reset_buffer(self):
        """Clear the frame buffer and calibration."""
        self.frame_buffer.clear()
        self.frame_count = 0
        self.is_calibrating = False
        self.calibration_buffer = []
        self.baseline = None


# Singleton
_posture_service: Optional[PostureService] = None


def get_posture_service() -> PostureService:
    """Get or create the PostureService singleton."""
    global _posture_service
    if _posture_service is None:
        _posture_service = PostureService()
    return _posture_service
