"""
PostureLSTM - Bidirectional LSTM model for sitting posture classification.

Architecture:
    Input (batch, seq_len=30, features=46)
    → Bidirectional LSTM (2 layers, 128 hidden)
    → BatchNorm
    → FC(256→64) → ReLU → Dropout
    → FC(64→6)
    → Output (batch, 6 classes)
"""

import torch
import torch.nn as nn


class PostureLSTM(nn.Module):
    """Bidirectional LSTM for temporal posture classification.

    Takes a sequence of body landmark features across multiple frames
    and classifies the overall sitting posture.

    Args:
        input_size: Number of features per frame (default: 46)
        hidden_size: LSTM hidden state dimension (default: 128)
        num_layers: Number of stacked LSTM layers (default: 2)
        num_classes: Number of posture categories (default: 6)
        dropout: Dropout rate for LSTM layers (default: 0.3)
        fc_dropout: Dropout rate for FC layers (default: 0.4)
        bidirectional: Use bidirectional LSTM (default: True)
    """

    def __init__(
        self,
        input_size: int = 46,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        fc_dropout: float = 0.4,
        bidirectional: bool = True,
    ):
        super(PostureLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output dimension from LSTM
        lstm_output_size = hidden_size * self.num_directions

        # Batch normalization after LSTM
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last timestep output
        last_output = lstm_out[:, -1, :]

        # Batch norm
        normalized = self.batch_norm(last_output)

        # Classification
        logits = self.classifier(normalized)

        return logits

    def predict(self, x: torch.Tensor) -> tuple:
        """Run inference and return class prediction + confidence.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tuple of (predicted_class_indices, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        return predicted, confidence


def get_device() -> torch.device:
    """Auto-detect the best available device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(
    model_path: str,
    device: torch.device = None,
    **model_kwargs,
) -> PostureLSTM:
    """Load a trained PostureLSTM model from a checkpoint.

    Args:
        model_path: Path to the saved model weights (.pth file)
        device: Device to load the model onto (auto-detect if None)
        **model_kwargs: Override default model hyperparameters

    Returns:
        Loaded PostureLSTM model in eval mode
    """
    if device is None:
        device = get_device()

    model = PostureLSTM(**model_kwargs)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle both raw state_dict and checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
