from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.hybrid_model import ASCIIHybridDeepfakeDetector
from src.preprocessing.ascii_conversion import ASCIIConverter
from src.preprocessing.face_detection import MTCNNFaceDetector
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cpu"))

    model = ASCIIHybridDeepfakeDetector(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    detector = MTCNNFaceDetector(device=device)
    converter = ASCIIConverter(grid_size=(80, 40), ascii_chars=".+=@*%#")

    faces = detector.process_video(args.video, fps=cfg["data"].get("target_fps", 8.0))
    if faces is None or faces.shape[0] == 0:
        print({"error": "No faces detected."})
        return

    seq_len = cfg["data"].get("sequence_length", 8)
    faces = faces[:seq_len]
    if faces.shape[0] < seq_len:
        pad = seq_len - faces.shape[0]
        faces = torch.cat([faces, faces[-1:].repeat(pad, 1, 1, 1)], dim=0)

    pixel_seq = faces.permute(0, 3, 1, 2).contiguous().float()
    pixel_seq = (pixel_seq - pixel_seq.min()) / (pixel_seq.max() - pixel_seq.min() + 1e-6)

    ascii_frames = []
    for i in range(seq_len):
        frame = (pixel_seq[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
        ascii_img = converter.convert_frame(frame)
        ascii_frames.append(torch.from_numpy(ascii_img).permute(2, 0, 1).float() / 255.0)

    ascii_seq = torch.stack(ascii_frames, dim=0)

    with torch.no_grad():
        score = model(pixel_seq.unsqueeze(0).to(device), ascii_seq.unsqueeze(0).to(device)).item()

    print({"deepfake_probability": float(score), "label": "fake" if score >= 0.5 else "real"})


if __name__ == "__main__":
    main()
