"""
Feature engineering service.

Computes engineered features from raw landmarks, matching the
training pipeline for consistent inference.
"""

import numpy as np
import math
from typing import Dict, List


# Feature column order must match training/src/preprocess.py exactly
LANDMARK_NAMES = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]

ENGINEERED_FEATURES = [
    "left_shoulder_hip_knee_angle",
    "right_shoulder_hip_knee_angle",
    "left_ear_shoulder_hip_angle",
    "right_ear_shoulder_hip_angle",
    "shoulder_tilt_angle",
    "head_tilt_angle",
    "head_forward_offset",
    "shoulder_symmetry",
    "hip_alignment",
    "torso_inclination",
]


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b in triangle abc."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def normalize_landmarks(
    landmarks: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Normalize landmarks relative to hip center."""
    hip_center = (landmarks["left_hip"] + landmarks["right_hip"]) / 2
    shoulder_width = np.linalg.norm(
        landmarks["left_shoulder"] - landmarks["right_shoulder"]
    )
    scale = shoulder_width if shoulder_width > 0 else 1.0

    return {
        name: (coords - hip_center) / scale
        for name, coords in landmarks.items()
    }


def engineer_features(landmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate engineered features from landmarks."""
    features = {}

    features["left_shoulder_hip_knee_angle"] = calculate_angle(
        landmarks["left_shoulder"], landmarks["left_hip"], landmarks["left_knee"]
    )
    features["right_shoulder_hip_knee_angle"] = calculate_angle(
        landmarks["right_shoulder"], landmarks["right_hip"], landmarks["right_knee"]
    )
    features["left_ear_shoulder_hip_angle"] = calculate_angle(
        landmarks["left_ear"], landmarks["left_shoulder"], landmarks["left_hip"]
    )
    features["right_ear_shoulder_hip_angle"] = calculate_angle(
        landmarks["right_ear"], landmarks["right_shoulder"], landmarks["right_hip"]
    )

    left_sh = landmarks["left_shoulder"]
    right_sh = landmarks["right_shoulder"]
    shoulder_diff = left_sh - right_sh
    features["shoulder_tilt_angle"] = math.degrees(
        math.atan2(shoulder_diff[1], shoulder_diff[0] + 1e-8)
    )

    left_ear = landmarks["left_ear"]
    right_ear = landmarks["right_ear"]
    ear_diff = left_ear - right_ear
    features["head_tilt_angle"] = math.degrees(
        math.atan2(ear_diff[1], ear_diff[0] + 1e-8)
    )

    shoulder_mid = (landmarks["left_shoulder"] + landmarks["right_shoulder"]) / 2
    features["head_forward_offset"] = float(
        landmarks["nose"][2] - shoulder_mid[2]
    )
    features["shoulder_symmetry"] = float(abs(
        landmarks["left_shoulder"][1] - landmarks["right_shoulder"][1]
    ))
    features["hip_alignment"] = float(abs(
        landmarks["left_hip"][1] - landmarks["right_hip"][1]
    ))

    hip_mid = (landmarks["left_hip"] + landmarks["right_hip"]) / 2
    spine_vec = shoulder_mid - hip_mid
    vertical = np.array([0, -1, 0])
    cos_angle = np.dot(spine_vec[:2], vertical[:2]) / (
        np.linalg.norm(spine_vec[:2]) * np.linalg.norm(vertical[:2]) + 1e-8
    )
    features["torso_inclination"] = math.degrees(
        math.acos(np.clip(cos_angle, -1, 1))
    )

    return features


def landmarks_to_features(
    landmarks: Dict[str, np.ndarray]
) -> np.ndarray:
    """Convert raw landmarks to the 46-feature vector for LSTM input.

    Args:
        landmarks: Dict of landmark_name -> [x, y, z]

    Returns:
        numpy array of shape (46,)
    """
    # Normalize
    norm_landmarks = normalize_landmarks(landmarks)

    # Raw coordinates (36 features)
    raw_features = []
    for name in LANDMARK_NAMES:
        raw_features.extend(norm_landmarks[name].tolist())

    # Engineered features (10 features)
    eng = engineer_features(landmarks)
    eng_values = [eng[f] for f in ENGINEERED_FEATURES]

    return np.array(raw_features + eng_values, dtype=np.float32)
