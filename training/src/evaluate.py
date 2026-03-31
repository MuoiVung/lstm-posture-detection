"""
Evaluation script for PostureLSTM model.

Generates:
- Classification report (precision, recall, F1 per class)
- Confusion matrix visualization
- Per-class accuracy breakdown
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import argparse
import os
import yaml

from model import PostureLSTM, get_device, load_model
from dataset import PostureDataset, create_dataloaders


def evaluate_model(
    model_path: str,
    data_dir: str,
    output_dir: str,
    config_path: str = None,
):
    """Evaluate a trained model on the test set.

    Args:
        model_path: Path to trained model checkpoint (.pth)
        data_dir: Path to processed data directory
        output_dir: Directory to save evaluation results
        config_path: Path to config YAML (optional, will use checkpoint config)
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config from checkpoint or file
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("No config found. Provide --config argument.")

    # Create model
    model_cfg = config["model"]
    model = PostureLSTM(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        fc_dropout=model_cfg["fc_dropout"],
        bidirectional=model_cfg["bidirectional"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test dataloader
    data_cfg = config["data"]
    _, _, test_loader, _ = create_dataloaders(
        data_dir=data_dir,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        batch_size=config["training"]["batch_size"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        num_workers=0,
        augment_train=False,
    )

    # Collect predictions
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Class names
    class_names = config.get("classes", list(PostureDataset.CLASS_LABELS.keys()))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Classification Report
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        digits=4,
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    # Save report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # 3. Overall metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100
    f1_macro = f1_score(all_targets, all_preds, average="macro") * 100
    f1_weighted = f1_score(all_targets, all_preds, average="weighted") * 100

    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (macro): {f1_macro:.2f}%")
    print(f"F1 Score (weighted): {f1_weighted:.2f}%")

    # 4. Per-class accuracy bar chart
    per_class_acc = cm_normalized.diagonal() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
    bars = ax.bar(class_names, per_class_acc, color=colors)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 110)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    class_acc_path = os.path.join(output_dir, "per_class_accuracy.png")
    plt.savefig(class_acc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy chart saved to {class_acc_path}")

    # 5. Save summary
    summary = {
        "model_path": model_path,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_accuracy": {
            name: float(acc) for name, acc in zip(class_names, per_class_acc)
        },
    }

    import json
    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll evaluation results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PostureLSTM model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data", type=str, default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/figures",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (optional, uses checkpoint config)",
    )
    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.output, args.config)


if __name__ == "__main__":
    main()
