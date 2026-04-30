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
    val_count = len(data_cfg.get("val_videos", []))
    if val_count < 20:
        print(
            f"Warning: validation set is very small ({val_count} videos). "
            "Metrics may be noisy; consider larger val split or cross-validation."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _validate_data_config(cfg)
    seed_everything(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cpu"))
    cache_dir = cfg.get("data", {}).get("cache_dir")

    train_dataset = DeepfakeDataset(
        cfg["data"]["train_videos"],
        cfg["data"]["train_labels"],
        transform=VideoTransforms(),
        sequence_length=cfg["data"].get("sequence_length", 8),
        cache_dir=cache_dir,
    )
    val_dataset = DeepfakeDataset(
        cfg["data"]["val_videos"],
        cfg["data"]["val_labels"],
        transform=None,
        sequence_length=cfg["data"].get("sequence_length", 8),
        cache_dir=cache_dir,
    )
    test_dataset = DeepfakeDataset(
        cfg["data"]["test_videos"],
        cfg["data"]["test_labels"],
        transform=None,
        sequence_length=cfg["data"].get("sequence_length", 8),
        cache_dir=cache_dir,
    )

    train_loader, val_loader, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg["training"].get("batch_size", 32),
        num_workers=cfg["training"].get("num_workers", 4),
    )

    model = ASCIIHybridDeepfakeDetector(cfg).to(device)
    optimizer = create_optimizer(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
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
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        use_amp=cfg["training"].get("use_amp", True),
        use_amp_eval=cfg["training"].get("use_amp_eval", False),
    )
    checkpoint_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    start_epoch = 0
    total_epochs = cfg["training"].get("epochs", 50)
    if args.resume and os.path.exists(checkpoint_path):
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint_path} (next epoch: {start_epoch + 1})")
    elif args.resume:
        print(f"No checkpoint found at {checkpoint_path}. Starting fresh training.")

    if start_epoch >= total_epochs:
        extra_epochs = int(cfg["training"].get("resume_extra_epochs", 10))
        total_epochs = start_epoch + extra_epochs
        print(
            f"Checkpoint epoch already reached configured epochs. "
            f"Extending training to epochs={total_epochs} (extra={extra_epochs})."
        )

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=total_epochs,
        save_path=checkpoint_dir,
        start_epoch=start_epoch,
    )
    print("Training complete.", history)


if __name__ == "__main__":
    main()
