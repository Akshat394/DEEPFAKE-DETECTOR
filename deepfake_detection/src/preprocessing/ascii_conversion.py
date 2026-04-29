from __future__ import annotations

from typing import Tuple

import numpy as np


class ASCIIConverter:
    def __init__(self, grid_size: Tuple[int, int] = (80, 40), ascii_chars: str = ".+=@*%#"):
        self.grid_size = grid_size
        self.ascii_chars = ascii_chars
        self.intensity_bins = self._compute_intensity_bins()

    def _compute_intensity_bins(self) -> np.ndarray:
        return np.linspace(0, 255, len(self.ascii_chars) + 1)

    def _grayscale(self, frame: np.ndarray) -> np.ndarray:
        return np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    def _map_to_ascii(self, mean_intensity: float) -> int:
        idx = int(np.digitize(mean_intensity, self.intensity_bins) - 1)
        return int(np.clip(idx, 0, len(self.ascii_chars) - 1))

    def convert_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = self._grayscale(frame)
        gh, gw = self.grid_size
        h, w = gray.shape

        crop_h = (h // gh) * gh
        crop_w = (w // gw) * gw
        gray = gray[:crop_h, :crop_w]

        cell_h = max(crop_h // gh, 1)
        cell_w = max(crop_w // gw, 1)

        means = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                cell = gray[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                means[i, j] = float(np.mean(cell)) if cell.size else 0.0

        ascii_idx = np.digitize(means, self.intensity_bins) - 1
        ascii_idx = np.clip(ascii_idx, 0, len(self.ascii_chars) - 1)
        chars_ord = np.vectorize(lambda x: ord(self.ascii_chars[int(x)]))(ascii_idx).astype(np.float32)

        # Normalize ASCII ord values to [0, 255]
        min_v, max_v = chars_ord.min(), chars_ord.max()
        if max_v > min_v:
            chars_ord = (chars_ord - min_v) / (max_v - min_v) * 255.0
        ascii_img = np.stack([chars_ord, chars_ord, chars_ord], axis=-1).astype(np.uint8)
        return ascii_img
