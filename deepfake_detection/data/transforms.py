from __future__ import annotations

import torch
import torchvision.transforms as T


class VideoTransforms:
    def __init__(self):
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (T, 3, H, W)
        out = frames.clone()
        if torch.rand(1).item() < 0.5:
            out = torch.flip(out, dims=[3])

        jittered = []
        for frame in out:
            jittered.append(self.jitter(frame))
        out = torch.stack(jittered, dim=0)

        noise = torch.randn_like(out) * 0.01
        out = torch.clamp(out + noise, 0.0, 1.0)
        return out
