from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, temporal_weight: float = 0.5, regularization_weight: float = 0.01):
        super().__init__()
        self.bce_weight = bce_weight
        self.temporal_weight = temporal_weight
        self.regularization_weight = regularization_weight
        self.bce = nn.BCELoss()

    def bce_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(predictions, targets)

    def temporal_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            return torch.mean(torch.abs(predictions[:, 1:] - predictions[:, :-1]))
        return torch.tensor(0.0, device=predictions.device)

    def regularization_loss(self, model_params: List[torch.Tensor]) -> torch.Tensor:
        reg = torch.tensor(0.0, device=model_params[0].device if model_params else "cpu")
        for p in model_params:
            reg = reg + torch.sum(p ** 2)
        return reg

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        temporal_consistency: Optional[torch.Tensor] = None,
        model_params: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        bce = self.bce_loss(predictions, targets)
        temporal = self.temporal_loss(temporal_consistency if temporal_consistency is not None else predictions)
        reg = self.regularization_loss(model_params) if model_params is not None else torch.tensor(0.0, device=predictions.device)
        return self.bce_weight * bce + self.temporal_weight * temporal + self.regularization_weight * reg
