from __future__ import annotations

import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]
        return self.head(last).squeeze(-1)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size: int = 9, conv_channels: int = 32, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]
        return self.head(last).squeeze(-1)


class CNNGRUModel(nn.Module):
    def __init__(self, input_size: int = 9, conv_channels: int = 32, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(conv_channels, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        outputs, _ = self.gru(x)
        last = outputs[:, -1, :]
        return self.head(last).squeeze(-1)


MODEL_REGISTRY = {
    "LSTM": LSTMModel,
    "CNN-LSTM": CNNLSTMModel,
    "CNN-GRU": CNNGRUModel,
}
