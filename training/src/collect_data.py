"""
Self-data collection tool for posture training data.

Opens webcam, detects pose via MediaPipe, and records landmark data
with user-specified labels via keyboard shortcuts.

Controls:
    1-6: Select posture label
    SPACE: Start/stop recording
    Q: Quit and save
    R: Reset current recording
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import argparse
import time
from datetime import datetime

from preprocess import (
    extract_landmarks,
    normalize_landmarks,
    engineer_features,
    ALL_FEATURE_COLUMNS,
    LANDMARK_INDICES,
    ENGINEERED_FEATURES,
)


# Label mapping for keyboard shortcuts
LABEL_MAP = {
    ord("1"): "good_posture",
    ord("2"): "forward_lean",
    ord("3"): "backward_lean",
    ord("4"): "left_lean",
    ord("5"): "right_lean",
    ord("6"): "head_forward",
}

# Colors for visualization
STATUS_COLORS = {
    "good_posture": (0, 255, 0),      # Green
    "forward_lean": (0, 165, 255),     # Orange
    "backward_lean": (0, 255, 255),    # Yellow
    "left_lean": (255, 165, 0),        # Blue-ish
    "right_lean": (255, 0, 165),       # Purple
    "head_forward": (0, 0, 255),       # Red
}


def draw_info_panel(
    frame: np.ndarray,
    current_label: str,
    is_recording: bool,
    frame_count: int,
    total_count: int,
):
    """Draw info panel on the frame."""
    h, w, _ = frame.shape

    # Semi-transparent overlay for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(
        frame, "PostureGuard - Data Collector",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    # Current label
    color = STATUS_COLORS.get(current_label, (255, 255, 255))
    label_text = f"Label: {current_label}"
    cv2.putText(
        frame, label_text,
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )

    # Recording status
    if is_recording:
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(
            frame, f"REC {frame_count}",
            (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    else:
        cv2.putText(
            frame, "PAUSED",
            (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2,
        )

    # Total frames collected
    cv2.putText(
        frame, f"Total: {total_count}",
        (w - 150, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    # Controls help at bottom
    help_y = h - 10
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(
        frame,
        "1-6: Labels | SPACE: Record | Q: Save & Quit | R: Reset",
        (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
    )


def collect_data(output_dir: str, camera_id: int = 0):
    """Run the data collection tool.

    Args:
        output_dir: Directory to save collected data CSVs
        camera_id: Camera device ID (default 0)
    """
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    current_label = "good_posture"
    is_recording = False
    collected_data = []
    frame_count = 0
    total_count = 0

    print("\n" + "=" * 50)
    print("PostureGuard Data Collection Tool")
    print("=" * 50)
    print("\nControls:")
    for key_code, label in LABEL_MAP.items():
        print(f"  {chr(key_code)}: {label}")
    print("  SPACE: Start/Stop recording")
    print("  Q: Save and quit")
    print("  R: Reset current recording")
    print("=" * 50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Draw skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2),
            )

            # Record data if recording
            if is_recording:
                landmarks = extract_landmarks(results, w, h)
                if landmarks:
                    norm_landmarks = normalize_landmarks(landmarks)

                    raw_features = []
                    for name in LANDMARK_INDICES.keys():
                        raw_features.extend(norm_landmarks[name].tolist())

                    eng_features = engineer_features(landmarks)
                    eng_values = [eng_features[f] for f in ENGINEERED_FEATURES]

                    row = {"frame_id": total_count, "label": current_label}
                    for col, val in zip(
                        ALL_FEATURE_COLUMNS, raw_features + eng_values
                    ):
                        row[col] = val

                    collected_data.append(row)
                    frame_count += 1
                    total_count += 1

        # Draw info panel
        draw_info_panel(frame, current_label, is_recording, frame_count, total_count)

        # Show frame
        cv2.imshow("PostureGuard - Data Collector", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            is_recording = not is_recording
            status = "RECORDING" if is_recording else "PAUSED"
            print(f"[{status}] Label: {current_label}")
        elif key == ord("r"):
            collected_data = []
            frame_count = 0
            total_count = 0
            print("[RESET] All data cleared")
        elif key in LABEL_MAP:
            current_label = LABEL_MAP[key]
            frame_count = 0
            print(f"[LABEL] Switched to: {current_label}")

    # Save collected data
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    if collected_data:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"collected_{timestamp}.csv")

        df = pd.DataFrame(collected_data)
        df.to_csv(output_path, index=False)

        print(f"\n✅ Saved {len(df)} frames to {output_path}")
        print("\nLabel distribution:")
        print(df["label"].value_counts().to_string())
    else:
        print("\nNo data collected.")


def main():
    parser = argparse.ArgumentParser(
        description="Collect posture training data via webcam"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/self_collected",
        help="Output directory for collected CSVs",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device ID",
    )
    args = parser.parse_args()

    collect_data(args.output, args.camera)


if __name__ == "__main__":
    main()
