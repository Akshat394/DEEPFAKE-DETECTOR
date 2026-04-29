from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .attention_fusion import AttentionFeatureFusion
from .efficientnet import EfficientNetASCIIExtractor
from .inception_resnet import InceptionResNetFeatureExtractor
from .lstm_temporal import TemporalConsistencyAnalyzer


class ASCIIHybridDeepfakeDetector(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config.get("model", {})
        feature_dim = int(model_cfg.get("feature_dim", 128))

        self.pixel_extractor = InceptionResNetFeatureExtractor(pretrained=model_cfg.get("pretrained", True), feature_dim=feature_dim)
        self.ascii_extractor = EfficientNetASCIIExtractor(pretrained=model_cfg.get("pretrained", True), feature_dim=feature_dim)
        self.temporal_analyzer = TemporalConsistencyAnalyzer(input_dim=feature_dim * 2, hidden_dim=256, num_layers=2, dropout=0.5)
        self.attention_fusion = AttentionFeatureFusion(feature_dim=256, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    def extract_features(self, pixel_input: torch.Tensor, ascii_input: torch.Tensor) -> torch.Tensor:
        b, t = pixel_input.shape[:2]
        pixel_flat = pixel_input.view(b * t, *pixel_input.shape[2:])
        ascii_flat = ascii_input.view(b * t, *ascii_input.shape[2:])
        pixel_features = self.pixel_extractor(pixel_flat)
        ascii_features = self.ascii_extractor(ascii_flat)
        combined = torch.cat([pixel_features, ascii_features], dim=1)
        return combined.view(b, t, -1)

    def classify(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_features).squeeze(-1)

    def forward(self, pixel_input: torch.Tensor, ascii_input: torch.Tensor, sequence_length: int = 8) -> torch.Tensor:
        combined_seq = self.extract_features(pixel_input, ascii_input)
        temporal_out, _ = self.temporal_analyzer(combined_seq)
        temporal_features = temporal_out[:, -1, :]
        # Use mean sequence feature as spatial summary for fusion.
        cnn_features = combined_seq.mean(dim=1)
        fused_features = self.attention_fusion(cnn_features, temporal_features)
        return self.classify(fused_features)
