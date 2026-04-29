from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import load_dataset


FREE_DATASETS = [
    # Public previews with downloadable samples
    "ud-biometrics/deepfake-dataset",
    "UniDataPro/deepfake-videos-dataset",
]


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def normalize_label(raw) -> int:
    s = str(raw).lower()
    if any(x in s for x in ["deepfake", "fake", "1", "true"]):
        return 1
    return 0


def download_repo(repo_id: str, out_root: Path, max_items: int | None = None) -> List[Tuple[str, int]]:
    local_dir = out_root / safe_name(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, split="train")
    if max_items is not None:
        ds = ds.select(range(min(max_items, len(ds))))

    pairs: List[Tuple[str, int]] = []
    for i, row in enumerate(ds):
        # Most of these datasets expose `video` and `label` columns
        video_obj = row.get("video")
        label_raw = row.get("label", 0)
        label = normalize_label(label_raw)

        video_path = None
        if isinstance(video_obj, dict):
            # datasets Video feature typically yields dict with path/bytes
            if video_obj.get("path"):
                src = Path(video_obj["path"])
                ext = src.suffix if src.suffix else ".mp4"
                dst = local_dir / f"sample_{i:05d}{ext}"
                if src.exists():
                    dst.write_bytes(src.read_bytes())
                    video_path = dst
            elif video_obj.get("bytes"):
                dst = local_dir / f"sample_{i:05d}.mp4"
                dst.write_bytes(video_obj["bytes"])
                video_path = dst
        elif isinstance(video_obj, str) and Path(video_obj).exists():
            src = Path(video_obj)
            dst = local_dir / f"sample_{i:05d}{src.suffix or '.mp4'}"
            dst.write_bytes(src.read_bytes())
            video_path = dst

        if video_path is not None:
            pairs.append((str(video_path.resolve()), label))

    with open(local_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump([{"video_path": p, "label": l} for p, l in pairs], f, indent=2)

    return pairs


def split_pairs(pairs: List[Tuple[str, int]]):
    n = len(pairs)
    n_train = max(int(0.7 * n), 1 if n > 0 else 0)
    n_val = max(int(0.15 * n), 1 if n > 2 else 0)
    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    if not test and len(pairs) > 2:
        test = [pairs[-1]]
    return train, val, test


def write_config(root: Path, train, val, test):
    cfg_path = root / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg.setdefault("data", {})
    cfg["data"]["train_videos"] = [p for p, _ in train]
    cfg["data"]["train_labels"] = [l for _, l in train]
    cfg["data"]["val_videos"] = [p for p, _ in val]
    cfg["data"]["val_labels"] = [l for _, l in val]
    cfg["data"]["test_videos"] = [p for p, _ in test]
    cfg["data"]["test_labels"] = [l for _, l in test]

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-per-dataset", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = root / "datasets" / "free_hf"
    out_root.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Tuple[str, int]] = []
    for repo_id in FREE_DATASETS:
        try:
            pairs = download_repo(repo_id, out_root, max_items=args.max_per_dataset)
            print(f"Downloaded {len(pairs)} samples from {repo_id}")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"Skipping {repo_id}: {e}")

    if not all_pairs:
        raise RuntimeError("No downloadable free samples found.")

    train, val, test = split_pairs(all_pairs)
    write_config(root, train, val, test)

    print(f"Training ready with {len(all_pairs)} videos total")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print("Updated configs/default.yaml")


if __name__ == "__main__":
    main()
