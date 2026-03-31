"""
Training script for PostureLSTM model.

Supports:
- Auto device detection (CUDA / MPS / CPU)
- TensorBoard logging
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Class-weighted loss for imbalanced data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import argparse
import time
from datetime import datetime

from model import PostureLSTM, get_device
from dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_interval: int = 10,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), step)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate the model.

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train(config_path: str, data_dir: str, output_dir: str):
    """Main training function.

    Args:
        config_path: Path to training config YAML
        data_dir: Path to processed data directory
        output_dir: Path to save outputs (models, logs, figures)
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup device
    if config.get("device", "auto") == "auto":
        device = get_device()
    else:
        device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(logs_dir)

    # Data loaders
    data_cfg = config["data"]
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_dir=data_dir,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        batch_size=config["training"]["batch_size"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        num_workers=0,  # macOS compatibility (set >0 for Linux/Windows)
        augment_train=True,
    )

    # Model
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

    print(f"\nModel:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with class weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    train_cfg = config["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Learning rate scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 5),
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        min_delta=es_cfg.get("min_delta", 0.001),
    )

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    log_interval = config.get("logging", {}).get("log_interval", 10)
    save_interval = config.get("logging", {}).get("save_interval", 5)

    print(f"\nStarting training for {train_cfg['epochs']} epochs...\n")
    start_time = time.time()

    for epoch in range(train_cfg["epochs"]):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, log_interval,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to TensorBoard
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Print progress
        print(
            f"Epoch [{epoch+1}/{train_cfg['epochs']}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_path = os.path.join(models_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "config": config,
                },
                best_model_path,
            )
            print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                ckpt_path,
            )

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"TensorBoard logs: {logs_dir}")
    print(f"{'='*60}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)[
            "model_state_dict"
        ]
    )
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    writer.add_hparams(
        {
            "lr": train_cfg["learning_rate"],
            "batch_size": train_cfg["batch_size"],
            "hidden_size": model_cfg["hidden_size"],
            "num_layers": model_cfg["num_layers"],
        },
        {
            "hparam/best_val_loss": best_val_loss,
            "hparam/best_val_acc": best_val_acc,
            "hparam/test_loss": test_loss,
            "hparam/test_acc": test_acc,
        },
    )

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train PostureLSTM model")
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--data", type=str, default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Path to output directory",
    )
    args = parser.parse_args()

    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
