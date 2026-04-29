from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from .frame_sampler import FrameSampler


class MTCNNFaceDetector:
    """Compatibility wrapper name preserved; implementation uses OpenCV Haar cascade."""

    def __init__(self, device: Optional[torch.device] = None, output_size: int = 256):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _detect_single(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        crop = frame_rgb[y : y + h, x : x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

    def detect_faces(self, frames: torch.Tensor) -> Optional[torch.Tensor]:
        if frames.numel() == 0:
            return None

        faces = []
        for frame in frames:
            np_frame = frame.detach().cpu().numpy().astype(np.uint8)
            face = self._detect_single(np_frame)
            if face is not None:
                faces.append(face)

        if not faces:
            return None

        face_tensor = torch.from_numpy(np.stack(faces)).permute(0, 3, 1, 2).float() / 255.0
        mean = self.mean.to(face_tensor.device)
        std = self.std.to(face_tensor.device)
        face_tensor = (face_tensor - mean) / std
        return face_tensor.permute(0, 2, 3, 1).contiguous()

    def process_video(self, video_path: str, fps: float = 8.0) -> Optional[torch.Tensor]:
        sampler = FrameSampler(target_fps=fps)
        frames = sampler.sample_frames(video_path)
        if not frames:
            return None
        tensor = torch.from_numpy(np.stack(frames, axis=0))
        return self.detect_faces(tensor)
