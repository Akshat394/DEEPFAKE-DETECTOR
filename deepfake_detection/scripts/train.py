from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import DeepfakeDataset, create_dataloaders
from data.transforms import VideoTransforms
from src.models.hybrid_model import ASCIIHybridDeepfakeDetector
from src.training.losses import CompositeLoss
from src.training.optimizers import create_optimizer, create_scheduler
from src.training.trainer import ModelTrainer
from src.utils.config import load_config
from src.utils.helpers import seed_everything


def _validate_data_config(cfg):
    data_cfg = cfg.get("data", {})
    required = [
        ("train_videos", "train_labels"),
        ("val_videos", "val_labels"),
        ("test_videos", "test_labels"),
    ]
    for videos_key, labels_key in required:
        videos = data_cfg.get(videos_key, [])
        labels = data_cfg.get(labels_key, [])
        if not videos or not labels:
            raise ValueError(
                f"Config field data.{videos_key} / data.{labels_key} is empty. "
                "Populate dataset paths and labels in configs/default.yaml before training."
            )
        if len(videos) != len(labels):
            raise ValueError(
                f"Mismatched lengths for data.{videos_key} ({len(videos)}) and "
                f"data.{labels_key} ({len(labels)})."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _validate_data_config(cfg)
    seed_everything(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cpu"))

    train_dataset = DeepfakeDataset(cfg["data"]["train_videos"], cfg["data"]["train_labels"], transform=VideoTransforms(), sequence_length=cfg["data"].get("sequence_length", 8))
    val_dataset = DeepfakeDataset(cfg["data"]["val_videos"], cfg["data"]["val_labels"], transform=None, sequence_length=cfg["data"].get("sequence_length", 8))
    test_dataset = DeepfakeDataset(cfg["data"]["test_videos"], cfg["data"]["test_labels"], transform=None, sequence_length=cfg["data"].get("sequence_length", 8))

    train_loader, val_loader, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg["training"].get("batch_size", 32),
        num_workers=cfg["training"].get("num_workers", 4),
    )

    model = ASCIIHybridDeepfakeDetector(cfg).to(device)
    optimizer = create_optimizer(model.parameters(), lr=cfg["training"].get("learning_rate", 1e-4))
    scheduler = create_scheduler(optimizer, epochs=cfg["training"].get("epochs", 50))
    criterion = CompositeLoss(
        bce_weight=cfg["training"].get("bce_weight", 1.0),
        temporal_weight=cfg["training"].get("temporal_weight", 0.5),
        regularization_weight=cfg["training"].get("regularization_weight", 0.01),
    )

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 10),
    )
    history = trainer.train(train_loader, val_loader, epochs=cfg["training"].get("epochs", 50), save_path=cfg["training"].get("checkpoint_dir", "checkpoints"))
    print("Training complete.", history)


if __name__ == "__main__":
    main()
