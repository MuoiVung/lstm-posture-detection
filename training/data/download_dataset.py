"""
Dataset download helper script.

Downloads publicly available posture datasets for training.
User should manually download Kaggle datasets (requires API key).
"""

import os
import urllib.request
import zipfile
import argparse


# Dataset registry
DATASETS = {
    "multiposture": {
        "name": "MultiPosture Dataset (Zenodo)",
        "description": (
            "4,800 frames of 3D skeletal coordinates (x, y, z) for 11 key body "
            "joints extracted via MediaPipe. Labeled for upper and lower body "
            "sitting postures."
        ),
        "url": None,  # Zenodo requires manual download
        "instructions": """
        ═══════════════════════════════════════════════════════
        MultiPosture Dataset - Manual Download Required
        ═══════════════════════════════════════════════════════

        1. Go to Zenodo and search for "MultiPosture dataset"
           URL: https://zenodo.org/search?q=multiposture+sitting+posture

        2. Download the dataset files (CSV format)

        3. Place the downloaded CSV files in:
           training/data/raw/multiposture/

        ═══════════════════════════════════════════════════════
        """,
    },
    "kaggle_images": {
        "name": "General Kaggle Posture Images",
        "description": "Kaggle has raw image datasets, but they require running through MediaPipe first.",
        "url": None,
        "instructions": """
        ═══════════════════════════════════════════════════════
        Kaggle Posture Datasets (Image Based)
        ═══════════════════════════════════════════════════════

        Please note: Most datasets on Kaggle for posture contain RAW IMAGES, 
        not pre-extracted MediaPipe landmarks like the Zenodo dataset above.

        If you download a Kaggle dataset (search: 'sitting posture'), you must:
        1. Extract the images.
        2. Write a script to loop through the images with `mediapipe.solutions.pose`
        3. Save the resulting landmarks to a CSV format matching this project.

        Because of this, it's highly recommended to use the Zenodo 'MultiPosture'
        dataset or use the built-in `collect_data.py` webcam tool.

        ═══════════════════════════════════════════════════════
        """,
    },
}


def show_dataset_info():
    """Display information about available datasets."""
    print("\n" + "=" * 60)
    print("Available Datasets for Posture Detection Training")
    print("=" * 60)

    for key, info in DATASETS.items():
        print(f"\n📦 {info['name']}")
        print(f"   Key: {key}")
        print(f"   {info['description']}")

    print("\n" + "=" * 60)


def download_dataset(dataset_key: str, output_dir: str):
    """Download or show instructions for a specific dataset.

    Args:
        dataset_key: Key from DATASETS registry
        output_dir: Base output directory
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return

    info = DATASETS[dataset_key]
    target_dir = os.path.join(output_dir, dataset_key)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n📦 {info['name']}")
    print(f"   Target directory: {target_dir}")

    if info["url"]:
        # Auto-download
        print(f"   Downloading from {info['url']}...")
        filename = os.path.basename(info["url"])
        filepath = os.path.join(target_dir, filename)

        urllib.request.urlretrieve(info["url"], filepath)
        print(f"   ✅ Downloaded to {filepath}")

        # Extract if zip
        if filepath.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(target_dir)
            print(f"   ✅ Extracted to {target_dir}")
    else:
        # Show manual download instructions
        print(info["instructions"])


def create_sample_data(output_dir: str):
    """Create a small sample dataset for testing the pipeline.

    Generates synthetic posture data with known patterns.
    """
    import numpy as np
    import pandas as pd

    print("\nGenerating sample dataset for pipeline testing...")

    np.random.seed(42)
    num_frames_per_class = 200
    feature_cols = []

    # Generate column names matching ALL_FEATURE_COLUMNS from preprocess.py
    landmark_names = [
        "nose", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
    ]
    for name in landmark_names:
        feature_cols.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    engineered = [
        "left_shoulder_hip_knee_angle", "right_shoulder_hip_knee_angle",
        "left_ear_shoulder_hip_angle", "right_ear_shoulder_hip_angle",
        "shoulder_tilt_angle", "head_tilt_angle",
        "head_forward_offset", "shoulder_symmetry",
        "hip_alignment", "torso_inclination",
    ]
    all_cols = feature_cols + engineered

    classes = [
        "good_posture", "forward_lean", "backward_lean",
        "left_lean", "right_lean", "head_forward",
    ]

    all_rows = []

    for cls_idx, cls_name in enumerate(classes):
        for i in range(num_frames_per_class):
            row = {"frame_id": len(all_rows), "label": cls_name}

            # Generate base features with class-specific patterns
            base = np.random.randn(len(all_cols)) * 0.1

            if cls_name == "good_posture":
                # Symmetric, upright
                base[all_cols.index("torso_inclination")] = np.random.normal(5, 2)
                base[all_cols.index("shoulder_symmetry")] = np.random.normal(0.02, 0.01)
            elif cls_name == "forward_lean":
                base[all_cols.index("torso_inclination")] = np.random.normal(25, 5)
                base[all_cols.index("head_forward_offset")] = np.random.normal(-0.15, 0.03)
            elif cls_name == "backward_lean":
                base[all_cols.index("torso_inclination")] = np.random.normal(-15, 5)
            elif cls_name == "left_lean":
                base[all_cols.index("shoulder_tilt_angle")] = np.random.normal(15, 3)
                base[all_cols.index("shoulder_symmetry")] = np.random.normal(0.08, 0.02)
            elif cls_name == "right_lean":
                base[all_cols.index("shoulder_tilt_angle")] = np.random.normal(-15, 3)
                base[all_cols.index("shoulder_symmetry")] = np.random.normal(0.08, 0.02)
            elif cls_name == "head_forward":
                base[all_cols.index("head_forward_offset")] = np.random.normal(-0.2, 0.04)
                base[all_cols.index("left_ear_shoulder_hip_angle")] = np.random.normal(140, 5)
                base[all_cols.index("right_ear_shoulder_hip_angle")] = np.random.normal(140, 5)

            for col, val in zip(all_cols, base):
                row[col] = val

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_posture_data.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Sample dataset saved to {output_path}")
    print(f"   Total frames: {len(df)}")
    print(f"   Classes: {', '.join(classes)}")
    print(f"   Features per frame: {len(all_cols)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download posture datasets")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "multiposture", "kaggle_posture", "sample"],
        help="Dataset to download (default: show all info)",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory for downloaded data",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        show_dataset_info()
    elif args.dataset == "sample":
        create_sample_data(args.output)
    else:
        download_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
