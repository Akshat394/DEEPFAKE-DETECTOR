from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class FrameSampler:
    target_fps: float = 8.0
    sampling_strategy: str = "uniform"

    def __init__(self, target_fps: float = 8.0, strategy: str = "uniform"):
        self.target_fps = target_fps
        self.sampling_strategy = strategy

    def get_frame_indices(self, total_frames: int, video_fps: float) -> List[int]:
        if total_frames <= 0:
            return []
        if video_fps <= 0:
            return list(range(total_frames))
        step = max(video_fps / self.target_fps, 1.0)
        indices = np.arange(0, total_frames, step).astype(int)
        return np.unique(np.clip(indices, 0, total_frames - 1)).tolist()

    def sample_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        indices = set(self.get_frame_indices(total_frames, fps))

        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            idx += 1

        cap.release()
        return frames
