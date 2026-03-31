"""
PostureDataset - PyTorch Dataset for posture landmark sequences.

Handles loading preprocessed CSV data, creating sliding window sequences,
and applying data augmentation for training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import os
import glob


class PostureDataset(Dataset):
    """PyTorch Dataset for posture landmark sequences.

    Loads preprocessed landmark data from CSV files and creates
    sliding window sequences for LSTM input.

    Args:
        data_dir: Directory containing processed CSV files
        window_size: Number of frames per sequence (default: 30)
        stride: Step size for sliding window (default: 5)
        augment: Whether to apply data augmentation (default: False)
        noise_std: Gaussian noise standard deviation for augmentation
        scale_range: Random scaling range for augmentation
    """

    # Class labels mapping
    CLASS_LABELS = {
        "good_posture": 0,
        "forward_lean": 1,
        "backward_lean": 2,
        "left_lean": 3,
        "right_lean": 4,
    }

    LABEL_NAMES = {v: k for k, v in CLASS_LABELS.items()}

    def __init__(
        self,
        data_dir: str,
        window_size: int = 30,
        stride: int = 5,
        augment: bool = False,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range

        # Load and create sequences
        self.sequences, self.labels = self._load_and_create_sequences()

        print(
            f"Loaded {len(self.sequences)} sequences "
            f"(window={window_size}, stride={stride})"
        )
        self._print_class_distribution()

    def _load_and_create_sequences(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load CSV files and create sliding window sequences."""
        sequences = []
        labels = []

        # Find all CSV files in the data directory
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                "Run preprocessing first."
            )

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            if "label" not in df.columns:
                print(f"Warning: Skipping {csv_file} - no 'label' column found")
                continue

            # Separate features and labels
            feature_cols = [c for c in df.columns if c != "label" and c != "frame_id"]
            features = df[feature_cols].values.astype(np.float32)
            frame_labels = df["label"].values

            # Create sliding window sequences
            for start in range(0, len(features) - self.window_size + 1, self.stride):
                end = start + self.window_size
                window_features = features[start:end]
                window_labels = frame_labels[start:end]

                # Use majority label for the window
                unique, counts = np.unique(window_labels, return_counts=True)
                majority_label = unique[np.argmax(counts)]

                # Convert string label to int
                if isinstance(majority_label, str):
                    label_idx = self.CLASS_LABELS.get(majority_label, -1)
                else:
                    label_idx = int(majority_label)

                if label_idx >= 0:
                    sequences.append(window_features)
                    labels.append(label_idx)

        return sequences, labels

    def _print_class_distribution(self):
        """Print the distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution:")
        for cls_idx, count in zip(unique, counts):
            name = self.LABEL_NAMES.get(cls_idx, f"unknown_{cls_idx}")
            percentage = count / len(self.labels) * 100
            print(f"  {name}: {count} ({percentage:.1f}%)")

    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply data augmentation to a sequence.

        Augmentations:
        - Gaussian noise injection
        - Random scaling
        """
        augmented = sequence.copy()

        # Add Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, self.noise_std, augmented.shape)
            augmented = augmented + noise.astype(np.float32)

        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(*self.scale_range)
            augmented = augmented * scale

        return augmented

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Apply augmentation during training
        if self.augment:
            sequence = self._augment_sequence(sequence)

        # Convert to tensors
        x = torch.FloatTensor(sequence)
        y = torch.LongTensor([label]).squeeze()

        return x, y

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = torch.zeros(len(self.CLASS_LABELS))
        for cls_idx, count in zip(unique, counts):
            weights[cls_idx] = total / (len(unique) * count)
        return weights


def create_dataloaders(
    data_dir: str,
    window_size: int = 30,
    stride: int = 5,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create train/val/test DataLoaders from processed data.

    Args:
        data_dir: Directory containing processed CSV files
        window_size: Frames per sequence
        stride: Sliding window stride
        batch_size: Batch size for DataLoaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        num_workers: DataLoader workers (0 for macOS compatibility)
        augment_train: Whether to augment training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    # Load full dataset without augmentation first for splitting
    full_dataset = PostureDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        augment=False,
    )

    # Calculate split sizes
    total = len(full_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    # Random split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Enable augmentation for training set
    if augment_train:
        train_dataset.dataset.augment = True

    # Get class weights from full dataset
    class_weights = full_dataset.get_class_weights()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"\nDataLoader splits: train={train_size}, val={val_size}, test={test_size}")

    return train_loader, val_loader, test_loader, class_weights
