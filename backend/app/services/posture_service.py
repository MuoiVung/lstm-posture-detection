"""
Posture inference service using LSTM model.

Maintains a frame buffer and runs the trained LSTM model
when enough frames are accumulated.
"""

import torch
import numpy as np
import os
import logging
from typing import Optional, Tuple, List
from collections import deque

from app.config import settings
from app.services.feature_service import landmarks_to_features
from app.ml.model import PostureLSTM, get_device

logger = logging.getLogger(__name__)


class PostureService:
    """Real-time posture inference using trained LSTM model.

    Maintains a sliding window buffer of frames and runs inference
    when the buffer is full.
    """

    def __init__(self):
        self.device = get_device()
        self.model: Optional[PostureLSTM] = None
        self.frame_buffer: deque = deque(maxlen=settings.WINDOW_SIZE)
        self.frame_count = 0
        self.baseline_features: Optional[np.ndarray] = None
        self._load_model()

    def _load_model(self):
        """Load the trained LSTM model."""
        model_path = settings.MODEL_PATH

        if not os.path.exists(model_path):
            logger.warning(
                f"Model not found at {model_path}. "
                "Posture inference will use rule-based fallback."
            )
            return

        try:
            self.model = PostureLSTM(
                input_size=settings.INPUT_SIZE,
                hidden_size=settings.HIDDEN_SIZE,
                num_layers=settings.NUM_LAYERS,
                num_classes=settings.NUM_CLASSES,
                bidirectional=settings.BIDIRECTIONAL,
            ).to(self.device)

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            logger.info(f"Model loaded from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def add_frame(self, landmarks: dict) -> Optional[dict]:
        """Process a frame and return prediction if buffer is full.

        Args:
            landmarks: Dict of landmark_name -> [x, y, z] coordinates

        Returns:
            Prediction dict with 'posture_class', 'confidence', 'all_probs'
            or None if buffer not yet full
        """
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % settings.FRAME_SKIP != 0:
            return None

        # Extract features
        try:
            features = landmarks_to_features(landmarks)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

        # Add to buffer
        self.frame_buffer.append(features)

        # Run inference when buffer is full
        if len(self.frame_buffer) == settings.WINDOW_SIZE:
            return self._predict()

        return None

    def _predict(self) -> dict:
        """Run inference on the current buffer."""
        if self.baseline_features is None:
            return {
                "posture_class": "needs_calibration",
                "confidence": 1.0,
                "all_probs": {},
            }
        
        return self._calibrated_prediction()



        try:
            # Create input tensor: (1, window_size, features)
            sequence = np.array(list(self.frame_buffer))
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

            pred_idx = predicted.item()
            conf = confidence.item()

            # Get all class probabilities
            all_probs = {
                settings.CLASS_NAMES[i]: float(probs[0][i])
                for i in range(len(settings.CLASS_NAMES))
            }

            return {
                "posture_class": settings.CLASS_NAMES[pred_idx],
                "confidence": conf,
                "all_probs": all_probs,
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._rule_based_prediction()

    def calibrate(self):
        """Set the current frame buffer average as the Good Posture Baseline."""
        if len(self.frame_buffer) > 0:
            # Average the features in the current window to get a stable baseline
            self.baseline_features = np.mean(list(self.frame_buffer), axis=0)
            logger.info("Posture baseline calibrated successfully.")
            return True
        return False

    def _calibrated_prediction(self) -> dict:
        """Prediction based on offset from the user's calibrated baseline."""
        latest = self.frame_buffer[-1]
        diff = latest - self.baseline_features

        # Nose Y (0=x, 1=y, 2=z, etc.)
        # 0: nose_x, 1: nose_y, 2: nose_z
        nose_y_diff = diff[1] 
        nose_x_diff = diff[0]
        
        # Shoulder symmetry feature is index 46
        # Shoulder tilt angle is index 43
        shoulder_tilt_diff = diff[43]
        
        # Nose Z (depth)
        nose_z_diff = diff[2]

        posture = "good_posture"
        confidence = 0.85

        # Heuristics for Webcam (Webcam Y goes down, so +Y means moving down in frame)
        # Z comes closer (negative) and Y goes down -> Forward lean
        if nose_z_diff < -0.15 or nose_y_diff > 0.08:
            posture = "forward_lean"
        elif nose_z_diff > 0.15 or nose_y_diff < -0.05:
            posture = "backward_lean"
        elif shoulder_tilt_diff > 5 or nose_x_diff > 0.1:
            posture = "left_lean"
        elif shoulder_tilt_diff < -5 or nose_x_diff < -0.1:
            posture = "right_lean"

        probs = {
            "good_posture": 0.1,
            "forward_lean": 0.1,
            "backward_lean": 0.1,
            "left_lean": 0.1,
            "right_lean": 0.1,
        }
        probs[posture] = confidence

        return {
            "posture_class": posture,
            "confidence": confidence,
            "all_probs": probs,
        }

    def _rule_based_prediction(self) -> dict:
        """Fallback rule-based prediction when model is unavailable.

        Uses simple heuristics based on engineered features.
        """
        if len(self.frame_buffer) == 0:
            return {
                "posture_class": "good_posture",
                "confidence": 0.0,
                "all_probs": {},
            }

        # Use the latest frame's features
        latest = self.frame_buffer[-1]

        # Feature indices (matching ALL_FEATURE_COLUMNS order)
        # Raw coords: 0-38 (13 joints × 3), Engineered: 39-48
        torso_incl_idx = 48  # torso_inclination
        head_fwd_idx = 45    # head_forward_offset
        shoulder_tilt_idx = 43  # shoulder_tilt_angle
        shoulder_sym_idx = 46   # shoulder_symmetry

        torso = abs(latest[torso_incl_idx])
        head_fwd = latest[head_fwd_idx]
        shoulder_tilt = latest[shoulder_tilt_idx]
        shoulder_sym = latest[shoulder_sym_idx]

        posture = "good_posture"
        confidence = 0.7

        if torso > 20:
            posture = "forward_lean"
        elif torso < -10:
            posture = "backward_lean"
        elif abs(shoulder_tilt) > 10 and shoulder_tilt > 0:
            posture = "left_lean"
        elif abs(shoulder_tilt) > 10 and shoulder_tilt < 0:
            posture = "right_lean"

        return {
            "posture_class": posture,
            "confidence": confidence,
            "all_probs": {posture: confidence},
        }

    def reset_buffer(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
        self.frame_count = 0


# Singleton
_posture_service: Optional[PostureService] = None


def get_posture_service() -> PostureService:
    """Get or create the PostureService singleton."""
    global _posture_service
    if _posture_service is None:
        _posture_service = PostureService()
    return _posture_service
