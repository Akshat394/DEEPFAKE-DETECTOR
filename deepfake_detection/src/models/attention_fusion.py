from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFeatureFusion(nn.Module):
    def __init__(self, feature_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.pre_norm = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, cnn_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        # Inputs: (B, 256), (B, 256)
        x = torch.stack([cnn_features, temporal_features], dim=1)  # (B, 2, 256)
        attn_in = self.pre_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out
        x = x + self.ffn(self.post_norm(x))
        return x.mean(dim=1)
