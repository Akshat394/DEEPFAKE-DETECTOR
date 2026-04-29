from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetASCIIExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, feature_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize low-res ASCII map for stable EfficientNet feature extraction.
        x = F.interpolate(x, size=(380, 380), mode="bilinear", align_corners=False)
        features = self.backbone(x)
        return self.projection(features)
