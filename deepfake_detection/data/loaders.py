from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.ascii_conversion import ASCIIConverter
from src.preprocessing.face_detection import MTCNNFaceDetector


class DeepfakeDataset(Dataset):
    def __init__(self, video_paths: List[str], labels: List[int], transform: Callable = None, sequence_length: int = 8):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length
        self.detector = MTCNNFaceDetector()
        self.converter = ASCIIConverter(grid_size=(80, 40), ascii_chars=".+=@*%#")

    def __len__(self) -> int:
        return len(self.video_paths)

    def load_video_sequence(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        faces = self.detector.process_video(video_path, fps=8.0)
        if faces is None or faces.shape[0] == 0:
            pixel_seq = torch.zeros(self.sequence_length, 3, 256, 256, dtype=torch.float32)
            ascii_seq = torch.zeros(self.sequence_length, 3, 80, 40, dtype=torch.float32)
            return pixel_seq, ascii_seq

        faces = faces[: self.sequence_length]
        if faces.shape[0] < self.sequence_length:
            pad = self.sequence_length - faces.shape[0]
            faces = torch.cat([faces, faces[-1:].repeat(pad, 1, 1, 1)], dim=0)

        pixel_seq = faces.permute(0, 3, 1, 2).contiguous().float()
        # faces currently normalized around imagenet stats; bring to 0..1-ish for augmentation path
        pixel_seq = (pixel_seq - pixel_seq.min()) / (pixel_seq.max() - pixel_seq.min() + 1e-6)

        ascii_frames = []
        for i in range(self.sequence_length):
            frame = (pixel_seq[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            ascii_img = self.converter.convert_frame(frame)
            ascii_frames.append(torch.from_numpy(ascii_img).permute(2, 0, 1).float() / 255.0)

        ascii_seq = torch.stack(ascii_frames, dim=0)
        return pixel_seq, ascii_seq

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pixel_sequence, ascii_sequence = self.load_video_sequence(self.video_paths[idx])
        if self.transform is not None:
            pixel_sequence = self.transform(pixel_sequence)
        return pixel_sequence, ascii_sequence, int(self.labels[idx])


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int = 32, num_workers: int = 4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
