from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import DeepfakeDataset
from src.utils.config import load_config


def warm_cache(dataset: DeepfakeDataset, name: str):
    total = len(dataset)
    for i in range(total):
        _ = dataset[i]
        if (i + 1) % max(total // 10, 1) == 0 or (i + 1) == total:
            print(f"{name}: cached {i + 1}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.get("data", {}).get("cache_dir")
    if not cache_dir:
        raise ValueError("Set data.cache_dir in config before precomputing cache.")

    sequence_length = cfg["data"].get("sequence_length", 8)
    train_dataset = DeepfakeDataset(
        cfg["data"]["train_videos"],
        cfg["data"]["train_labels"],
        transform=None,
        sequence_length=sequence_length,
        cache_dir=cache_dir,
    )
    val_dataset = DeepfakeDataset(
        cfg["data"]["val_videos"],
        cfg["data"]["val_labels"],
        transform=None,
        sequence_length=sequence_length,
        cache_dir=cache_dir,
    )
    test_dataset = DeepfakeDataset(
        cfg["data"]["test_videos"],
        cfg["data"]["test_labels"],
        transform=None,
        sequence_length=sequence_length,
        cache_dir=cache_dir,
    )

    warm_cache(train_dataset, "train")
    warm_cache(val_dataset, "val")
    warm_cache(test_dataset, "test")
    print("Cache precompute finished.")


if __name__ == "__main__":
    main()
