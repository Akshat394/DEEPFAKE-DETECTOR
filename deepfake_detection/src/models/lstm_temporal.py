from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemporalConsistencyAnalyzer(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = False
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        output, hidden = self.lstm(x, hidden)
        return output, hidden
