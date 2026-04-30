from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import DeepfakeDataset, create_dataloaders
from src.evaluation.benchmark import BenchmarkEvaluator
from src.models.hybrid_model import ASCIIHybridDeepfakeDetector
from src.utils.config import load_config


def _validate_eval_config(cfg):
    data_cfg = cfg.get("data", {})
    videos = data_cfg.get("test_videos", [])
    labels = data_cfg.get("test_labels", [])
    if not videos or not labels:
        raise ValueError(
            "Config field data.test_videos / data.test_labels is empty. "
            "Populate test dataset paths and labels in configs/default.yaml before evaluation."
        )
    if len(videos) != len(labels):
        raise ValueError(
            f"Mismatched lengths for data.test_videos ({len(videos)}) and data.test_labels ({len(labels)})."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _validate_eval_config(cfg)
    device = torch.device(cfg.get("device", "cpu"))

    cache_dir = cfg.get("data", {}).get("cache_dir")
    test_dataset = DeepfakeDataset(
        cfg["data"]["test_videos"],
        cfg["data"]["test_labels"],
        sequence_length=cfg["data"].get("sequence_length", 8),
        cache_dir=cache_dir,
    )
    dummy_dataset = DeepfakeDataset([], [], sequence_length=cfg["data"].get("sequence_length", 8), cache_dir=cache_dir)
    _, _, test_loader = create_dataloaders(dummy_dataset, dummy_dataset, test_dataset, batch_size=cfg["training"].get("batch_size", 32), num_workers=cfg["training"].get("num_workers", 4))

    model = ASCIIHybridDeepfakeDetector(cfg).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    evaluator = BenchmarkEvaluator()
    results = evaluator.evaluate_on_dataset(model, test_loader, dataset_name="test")
    print("Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
