from __future__ import annotations

from typing import Optional

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .frame_sampler import FrameSampler

try:
    from facenet_pytorch import MTCNN  # type: ignore
except Exception:
    MTCNN = None


class MTCNNFaceDetector:
    """Primary detector uses facenet-pytorch MTCNN; falls back to OpenCV Haar cascade."""

    def __init__(self, device: Optional[torch.device] = None, output_size: int = 256):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_size = output_size
        use_mtcnn = os.getenv("USE_MTCNN", "1").strip() not in {"0", "false", "False"}
        self.mtcnn = (
            MTCNN(
                image_size=output_size,
                margin=0,
                keep_all=False,
                post_process=False,
                device=self.device,
            )
            if (MTCNN is not None and use_mtcnn)
            else None
        )
        self.haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _detect_single(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
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

        # Preferred path: MTCNN with face alignment
        if self.mtcnn is not None:
            rgb_frames = [frame.detach().cpu().numpy().astype(np.uint8) for frame in frames]
            mtcnn_faces = self.mtcnn(rgb_frames)  # (N, 3, H, W) or None
            if mtcnn_faces is not None:
                if mtcnn_faces.ndim == 3:
                    mtcnn_faces = mtcnn_faces.unsqueeze(0)
                mtcnn_faces = mtcnn_faces.float()
                # facenet-pytorch may output [0,255] if post_process=False
                if mtcnn_faces.max() > 1.5:
                    mtcnn_faces = mtcnn_faces / 255.0
                mtcnn_faces = torch.clamp(mtcnn_faces, 0.0, 1.0)
                if mtcnn_faces.shape[-1] != self.output_size or mtcnn_faces.shape[-2] != self.output_size:
                    mtcnn_faces = F.interpolate(
                        mtcnn_faces, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
                    )
                return mtcnn_faces.permute(0, 2, 3, 1).contiguous()

        # Fallback path: OpenCV Haar detector
        faces = []
        for frame in frames:
            np_frame = frame.detach().cpu().numpy().astype(np.uint8)
            face = self._detect_single(np_frame)
            if face is not None:
                faces.append(face)

        if not faces:
            return None

        face_tensor = torch.from_numpy(np.stack(faces)).permute(0, 3, 1, 2).float() / 255.0
        face_tensor = torch.clamp(face_tensor, 0.0, 1.0)
        return face_tensor.permute(0, 2, 3, 1).contiguous()

    def process_video(self, video_path: str, fps: float = 8.0) -> Optional[torch.Tensor]:
        sampler = FrameSampler(target_fps=fps)
        frames = sampler.sample_frames(video_path)
        if not frames:
            return None
        tensor = torch.from_numpy(np.stack(frames, axis=0))
        return self.detect_faces(tensor)
