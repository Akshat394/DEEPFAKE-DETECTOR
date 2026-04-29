from __future__ import annotations

import torch
import torch.nn as nn
import timm


class InceptionResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, feature_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feature_dim = feature_dim
        self.output_layer = "avgpool_1x1"
        self.projection = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 0.5) / 0.5
        features = self.backbone(x)
        return self.projection(features)