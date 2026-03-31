"""
MediaPipe Pose extraction service.

Wraps MediaPipe Pose for extracting body landmarks from video frames.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Key landmark indices (same as training/src/preprocess.py)
LANDMARK_INDICES = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
}


class PoseService:
    """MediaPipe Pose extraction service."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("MediaPipe Pose initialized")

    def extract_landmarks(
        self, frame_bytes: bytes
    ) -> Optional[Dict]:
        """Extract pose landmarks from a JPEG frame.

        Args:
            frame_bytes: JPEG-encoded image bytes

        Returns:
            Dict with 'landmarks' (key joint coordinates) and
            'all_landmarks' (all 33 MediaPipe landmarks), or None
        """
        # Decode JPEG bytes to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None

        h, w, _ = img.shape

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        # Extract key landmarks
        key_landmarks = {}
        for name, idx in LANDMARK_INDICES.items():
            lm = results.pose_landmarks.landmark[idx]
            key_landmarks[name] = np.array(
                [lm.x, lm.y, lm.z], dtype=np.float32
            )

        # Extract all 33 landmarks for overlay rendering
        all_landmarks = []
        for lm in results.pose_landmarks.landmark:
            all_landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            })

        return {
            "key_landmarks": key_landmarks,
            "all_landmarks": all_landmarks,
            "image_size": (w, h),
        }

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


# Singleton instance
_pose_service: Optional[PoseService] = None


def get_pose_service() -> PoseService:
    """Get or create the PoseService singleton."""
    global _pose_service
    if _pose_service is None:
        _pose_service = PoseService()
    return _pose_service
