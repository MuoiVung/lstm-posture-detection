"""
PostureLSTM model definition for backend inference.

This is a copy of training/src/model.py to avoid cross-module imports.
Must stay in sync with training module.
"""

import torch
import torch.nn as nn


class PostureLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 49,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        fc_dropout: float = 0.4,
        bidirectional: bool = True,
    ):
        super(PostureLSTM, self).__init__()
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        normalized = self.batch_norm(last_output)
        return self.classifier(normalized)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
