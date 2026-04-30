from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedKFold


def read_pairs(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    videos = data.get("train_videos", []) + data.get("val_videos", [])
    labels = data.get("train_labels", []) + data.get("val_labels", [])
    if not videos or not labels or len(videos) != len(labels):
        raise ValueError("Need non-empty train+val videos/labels with matching lengths.")
    return videos, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    videos, labels = read_pairs(cfg_path)
    out_dir = cfg_path.parent.parent / "data" / "manifests" / "kfold"
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(videos, labels), start=1):
        train_rows = [{"video_path": videos[i], "label": int(labels[i])} for i in train_idx]
        val_rows = [{"video_path": videos[i], "label": int(labels[i])} for i in val_idx]
        (out_dir / f"fold_{fold}_train.json").write_text(json.dumps(train_rows, indent=2), encoding="utf-8")
        (out_dir / f"fold_{fold}_val.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")
        print(f"fold {fold}: train={len(train_rows)} val={len(val_rows)}")

    print(f"Wrote stratified k-fold manifests to {out_dir}")


if __name__ == "__main__":
    main()
