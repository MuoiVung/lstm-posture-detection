import { useRef, useEffect, useCallback } from "react";
import { CAPTURE_INTERVAL_MS, JPEG_QUALITY, POSE_CONNECTIONS } from "../../utils/constants";

/**
 * CameraFeed component - captures webcam frames and sends them via WebSocket.
 * Also renders pose skeleton overlay and calibration overlay.
 */
export default function CameraFeed({
  isActive,
  sendFrame,
  landmarks,
  postureClass,
  children,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const isEncoding = useRef(false);

  // Start/stop camera
  useEffect(() => {
    if (isActive) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isActive]);

  // Frame capture loop
  useEffect(() => {
    if (isActive && sendFrame) {
      intervalRef.current = setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isActive, sendFrame]);

  // Draw skeleton overlay when landmarks update
  useEffect(() => {
    if (landmarks && canvasRef.current && videoRef.current) {
      drawSkeleton(landmarks, postureClass);
    }
  }, [landmarks, postureClass]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const captureAndSend = useCallback(() => {
    if (!videoRef.current || !captureCanvasRef.current || !sendFrame) return;
    if (isEncoding.current) return; // Prevent overlapped encoding

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;

    if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    isEncoding.current = true;
    canvas.toBlob(
      (blob) => {
        isEncoding.current = false;
        if (blob) sendFrame(blob);
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }, [sendFrame]);

  const drawSkeleton = (lms, postureClass) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!lms || lms.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;

    // Determine color based on posture
    const isGood = postureClass === "good_posture";
    const pointColor = isGood ? "#00d4aa" : "#ef4444";
    const lineColor = isGood
      ? "rgba(0, 212, 170, 0.5)"
      : "rgba(239, 68, 68, 0.5)";

    // Draw connections
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;

    for (const [i, j] of POSE_CONNECTIONS) {
      if (i < lms.length && j < lms.length) {
        const a = lms[i];
        const b = lms[j];
        if (a.visibility > 0.3 && b.visibility > 0.3) {
          ctx.beginPath();
          ctx.moveTo(a.x * w, a.y * h);
          ctx.lineTo(b.x * w, b.y * h);
          ctx.stroke();
        }
      }
    }

    // Draw landmarks
    for (const lm of lms) {
      if (lm.visibility > 0.3) {
        ctx.fillStyle = pointColor;
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
        ctx.fill();

        // Glow effect
        ctx.fillStyle = isGood
          ? "rgba(0, 212, 170, 0.2)"
          : "rgba(239, 68, 68, 0.2)";
        ctx.beginPath();
        ctx.arc(lm.x * w, lm.y * h, 8, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  return (
    <div className="camera-container" id="camera-container">
      {isActive ? (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ transform: "scaleX(-1)" }}
          />
          <canvas
            ref={canvasRef}
            style={{ transform: "scaleX(-1)" }}
          />
        </>
      ) : (
        <div className="camera-placeholder">
          <span className="camera-placeholder__icon">📷</span>
          <span className="camera-placeholder__text">
            Click "Start & Calibrate" to begin
          </span>
        </div>
      )}
      {/* Calibration overlay renders as children */}
      {children}
      {/* Hidden canvas for frame capture */}
      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}
