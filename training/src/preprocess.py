"""
Preprocessing pipeline for posture detection.

Extracts MediaPipe pose landmarks from video/images, engineers features
(joint angles, relative positions), normalizes data, and saves to CSV.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import math


# Key landmark indices from MediaPipe's 33 landmarks
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
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

# Total: 12 joints × 3 coords = 36 raw features + 10 engineered = 46 features
FEATURE_COLUMNS = []

# Generate raw coordinate column names
for name in LANDMARK_INDICES.keys():
    FEATURE_COLUMNS.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

# Engineered feature names
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

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_FEATURES


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the angle at point b formed by points a, b, c.

    Args:
        a, b, c: 3D coordinate arrays [x, y, z]

    Returns:
        Angle in degrees
    """
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return angle


def extract_landmarks(
    results, image_width: int, image_height: int
) -> Optional[Dict[str, np.ndarray]]:
    """Extract key landmarks from MediaPipe pose results.

    Args:
        results: MediaPipe pose detection results
        image_width: Frame width for denormalization
        image_height: Frame height for denormalization

    Returns:
        Dictionary of landmark_name -> [x, y, z] coordinates, or None
    """
    if not results.pose_landmarks:
        return None

    landmarks = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        # Keep normalized coordinates (0-1 range) for consistency
        landmarks[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    return landmarks


def engineer_features(landmarks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate engineered features from raw landmarks.

    Features include joint angles, relative positions, and body alignment metrics.

    Args:
        landmarks: Dictionary of landmark_name -> [x, y, z]

    Returns:
        Dictionary of feature_name -> value
    """
    features = {}

    # 1. Shoulder-Hip-Knee angles (slouching indicator)
    features["left_shoulder_hip_knee_angle"] = calculate_angle(
        landmarks["left_shoulder"],
        landmarks["left_hip"],
        landmarks["left_knee"],
    )

    features["right_shoulder_hip_knee_angle"] = calculate_angle(
        landmarks["right_shoulder"],
        landmarks["right_hip"],
        landmarks["right_knee"],
    )

    # 2. Ear-Shoulder-Hip angles (forward head indicator)
    features["left_ear_shoulder_hip_angle"] = calculate_angle(
        landmarks["left_ear"],
        landmarks["left_shoulder"],
        landmarks["left_hip"],
    )

    features["right_ear_shoulder_hip_angle"] = calculate_angle(
        landmarks["right_ear"],
        landmarks["right_shoulder"],
        landmarks["right_hip"],
    )

    # 3. Shoulder tilt angle (lateral lean indicator)
    left_sh = landmarks["left_shoulder"]
    right_sh = landmarks["right_shoulder"]
    shoulder_diff = left_sh - right_sh
    features["shoulder_tilt_angle"] = math.degrees(
        math.atan2(shoulder_diff[1], shoulder_diff[0] + 1e-8)
    )

    # 4. Head tilt angle
    left_ear = landmarks["left_ear"]
    right_ear = landmarks["right_ear"]
    ear_diff = left_ear - right_ear
    features["head_tilt_angle"] = math.degrees(
        math.atan2(ear_diff[1], ear_diff[0] + 1e-8)
    )

    # 5. Head forward offset (nose relative to shoulder midpoint)
    shoulder_mid = (landmarks["left_shoulder"] + landmarks["right_shoulder"]) / 2
    nose = landmarks["nose"]
    features["head_forward_offset"] = nose[2] - shoulder_mid[2]

    # 6. Shoulder symmetry (difference in shoulder Y coords)
    features["shoulder_symmetry"] = abs(
        landmarks["left_shoulder"][1] - landmarks["right_shoulder"][1]
    )

    # 7. Hip alignment symmetry
    features["hip_alignment"] = abs(
        landmarks["left_hip"][1] - landmarks["right_hip"][1]
    )

    # 8. Torso inclination (angle of spine from vertical)
    hip_mid = (landmarks["left_hip"] + landmarks["right_hip"]) / 2
    spine_vec = shoulder_mid - hip_mid
    vertical = np.array([0, -1, 0])  # Y-axis points down in image coords
    cos_angle = np.dot(spine_vec[:2], vertical[:2]) / (
        np.linalg.norm(spine_vec[:2]) * np.linalg.norm(vertical[:2]) + 1e-8
    )
    features["torso_inclination"] = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

    return features


def normalize_landmarks(landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize landmarks relative to hip center for position invariance.

    This makes the features invariant to the person's position in the frame.

    Args:
        landmarks: Raw landmark coordinates

    Returns:
        Normalized landmark coordinates
    """
    # Use hip midpoint as origin
    hip_center = (landmarks["left_hip"] + landmarks["right_hip"]) / 2

    # Calculate scale factor (shoulder width for scale invariance)
    shoulder_width = np.linalg.norm(
        landmarks["left_shoulder"] - landmarks["right_shoulder"]
    )
    scale = shoulder_width if shoulder_width > 0 else 1.0

    normalized = {}
    for name, coords in landmarks.items():
        normalized[name] = (coords - hip_center) / scale

    return normalized


def process_frame(
    results, image_width: int, image_height: int
) -> Optional[np.ndarray]:
    """Process a single frame: extract landmarks, normalize, engineer features.

    Args:
        results: MediaPipe pose results
        image_width, image_height: Frame dimensions

    Returns:
        Feature vector of length 46, or None if no pose detected
    """
    # Extract raw landmarks
    landmarks = extract_landmarks(results, image_width, image_height)
    if landmarks is None:
        return None

    # Normalize for position/scale invariance
    norm_landmarks = normalize_landmarks(landmarks)

    # Flatten raw coordinates (36 features)
    raw_features = []
    for name in LANDMARK_INDICES.keys():
        raw_features.extend(norm_landmarks[name].tolist())

    # Engineer additional features (10 features)
    eng_features = engineer_features(landmarks)
    eng_values = [eng_features[f] for f in ENGINEERED_FEATURES]

    # Combine: 36 + 10 = 46 features
    all_features = raw_features + eng_values

    return np.array(all_features, dtype=np.float32)


def process_video(
    video_path: str,
    label: str,
    output_dir: str,
    max_frames: int = None,
) -> str:
    """Process a video file and extract pose features to CSV.

    Args:
        video_path: Path to input video
        label: Posture class label for all frames
        output_dir: Directory to save output CSV
        max_frames: Maximum frames to process (None for all)

    Returns:
        Path to output CSV file
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    rows = []
    frame_id = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")

    while cap.isOpened() and (max_frames is None or frame_id < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        features = process_frame(results, w, h)
        if features is not None:
            row = {"frame_id": frame_id, "label": label}
            for col, val in zip(ALL_FEATURE_COLUMNS, features):
                row[col] = val
            rows.append(row)

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    pose.close()

    # Save to CSV
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_{label}.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} frames to {output_path}")
    return output_path


def process_csv_dataset(
    input_csv: str,
    output_dir: str,
) -> str:
    """Process an existing CSV dataset with raw landmarks.

    Handles datasets that already have MediaPipe landmarks in CSV format,
    including the Zenodo MultiPosture dataset.
    Re-engineers features and normalizes coordinates.

    Args:
        input_csv: Path to input CSV with raw landmark data
        output_dir: Directory to save processed CSV

    Returns:
        Path to processed CSV
    """
    # Zenodo MultiPosture upper body label -> our posture class mapping
    ZENODO_LABEL_MAP = {
        "TUP": "good_posture",      # Trunk Upright
        "TLF": "forward_lean",      # Trunk Leaning Forward
        "TLB": "backward_lean",     # Trunk Leaning Backward
        "TLL": "left_lean",         # Trunk Leaning Left
        "TLR": "right_lean",        # Trunk Leaning Right
    }

    df = pd.read_csv(input_csv)
    print(f"Processing {input_csv}: {len(df)} rows, {len(df.columns)} columns")

    # Detect if this is a Zenodo MultiPosture dataset
    is_zenodo = "upperbody_label" in df.columns
    if is_zenodo:
        print(f"  Detected Zenodo MultiPosture format")
        print(f"  Upper body labels: {df['upperbody_label'].value_counts().to_dict()}")
        # Map labels
        df["label"] = df["upperbody_label"].map(ZENODO_LABEL_MAP)
        unmapped = df["label"].isnull().sum()
        if unmapped > 0:
            print(f"  Warning: {unmapped} rows with unmapped labels, dropping.")
            df = df.dropna(subset=["label"])
        print(f"  Mapped labels: {df['label'].value_counts().to_dict()}")

    processed_rows = []
    skipped = 0

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Reconstruct landmarks dict from CSV columns
        landmarks = {}
        try:
            for name, idx in LANDMARK_INDICES.items():
                # Try common column naming patterns
                x_col = f"{name}_x"
                y_col = f"{name}_y"
                z_col = f"{name}_z"

                if x_col in df.columns:
                    landmarks[name] = np.array(
                        [row[x_col], row[y_col], row.get(z_col, 0.0)],
                        dtype=np.float32,
                    )
                else:
                    # Try numeric index pattern (x0, y0, z0, x1, y1, z1, ...)
                    x_idx = f"x{idx}"
                    y_idx = f"y{idx}"
                    z_idx = f"z{idx}"
                    if x_idx in df.columns:
                        landmarks[name] = np.array(
                            [row[x_idx], row[y_idx], row.get(z_idx, 0.0)],
                            dtype=np.float32,
                        )
        except (KeyError, TypeError):
            skipped += 1
            continue

        if len(landmarks) < len(LANDMARK_INDICES):
            skipped += 1
            continue

        # Normalize and engineer features
        norm_landmarks = normalize_landmarks(landmarks)

        raw_features = []
        for name in LANDMARK_INDICES.keys():
            raw_features.extend(norm_landmarks[name].tolist())

        eng_features = engineer_features(landmarks)
        eng_values = [eng_features[f] for f in ENGINEERED_FEATURES]

        processed_row = {"frame_id": row.get("frame_id", row_idx)}

        # Get label from any available column
        if "label" in df.columns:
            processed_row["label"] = row["label"]
        elif "upperbody_label" in df.columns:
            mapped = ZENODO_LABEL_MAP.get(row["upperbody_label"])
            if mapped:
                processed_row["label"] = mapped
            else:
                skipped += 1
                continue

        for col, val in zip(ALL_FEATURE_COLUMNS, raw_features + eng_values):
            processed_row[col] = val

        processed_rows.append(processed_row)

    if skipped > 0:
        print(f"  Skipped {skipped} rows (missing landmarks or unmapped labels)")

    result_df = pd.DataFrame(processed_rows)
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_csv))[0]
    output_path = os.path.join(output_dir, f"{basename}_processed.csv")
    result_df.to_csv(output_path, index=False)

    print(f"\n=== Preprocessing Complete ===")
    print(f"  Input:  {len(df)} rows")
    print(f"  Output: {len(result_df)} rows -> {output_path}")
    print(f"  Features per row: {len(ALL_FEATURE_COLUMNS)}")
    if "label" in result_df.columns:
        print(f"  Label distribution:")
        for label, count in result_df["label"].value_counts().items():
            print(f"    {label}: {count}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess posture data")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input file/directory (video or CSV)"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Posture label for video processing"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process per video"
    )

    args = parser.parse_args()

    if os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            if args.label is None:
                print("Error: --label required for video processing")
                return
            process_video(args.input, args.label, args.output, args.max_frames)
        elif ext == ".csv":
            process_csv_dataset(args.input, args.output)
        else:
            print(f"Unsupported file format: {ext}")
    elif os.path.isdir(args.input):
        # Process all videos/CSVs in directory
        for f in sorted(os.listdir(args.input)):
            fpath = os.path.join(args.input, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                # Infer label from filename (e.g., "good_posture_001.mp4")
                label = args.label or "_".join(f.split("_")[:-1]) or "unknown"
                process_video(fpath, label, args.output, args.max_frames)
            elif ext == ".csv":
                process_csv_dataset(fpath, args.output)
    else:
        print(f"Input not found: {args.input}")


if __name__ == "__main__":
    main()
