from __future__ import annotations

import torch


def create_optimizer(model_parameters, lr: float = 1e-4, weight_decay: float = 1e-4):
    return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, epochs: int = 50):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
