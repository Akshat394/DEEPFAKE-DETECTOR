from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml
from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "bitmind/dataset"
REAL_PREFIX = "sdfvd/videos_real/"
FAKE_PREFIX = "sdfvd/videos_fake/"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def natural_key(path: str) -> tuple:
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (Path(path).parent.as_posix(), int(digits) if digits else stem)


def list_sdfvd_files() -> tuple[list[str], list[str]]:
    files = list_repo_files(REPO_ID, repo_type="dataset")
    real = sorted(
        [f for f in files if f.startswith(REAL_PREFIX) and Path(f).suffix.lower() in VIDEO_EXTS],
        key=natural_key,
    )
    fake = sorted(
        [f for f in files if f.startswith(FAKE_PREFIX) and Path(f).suffix.lower() in VIDEO_EXTS],
        key=natural_key,
    )
    if not real or not fake:
        raise RuntimeError("Could not find both real and fake SDFVD videos in the public dataset.")
    return real, fake


def download_file(repo_path: str, target_dir: Path) -> Path:
    source = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=repo_path,
        )
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return target.resolve()


def download_sdfvd(root: Path, limit_per_class: int | None = None) -> list[tuple[str, int]]:
    real_files, fake_files = list_sdfvd_files()
    if limit_per_class is not None:
        real_files = real_files[:limit_per_class]
        fake_files = fake_files[:limit_per_class]

    dataset_root = root / "datasets" / "sdfvd"
    real_dir = dataset_root / "original_sequences" / "youtube" / "c23" / "videos"
    fake_dir = dataset_root / "manipulated_sequences" / "SDFVD" / "c23" / "videos"

    pairs: list[tuple[str, int]] = []
    for repo_path in real_files:
        pairs.append((str(download_file(repo_path, real_dir)), 0))
    for repo_path in fake_files:
        pairs.append((str(download_file(repo_path, fake_dir)), 1))

    source_manifest = {
        "dataset": "SDFVD",
        "source_repo": REPO_ID,
        "license": "MIT",
        "real_count": len(real_files),
        "fake_count": len(fake_files),
        "layout": {
            "real": str(real_dir.resolve()),
            "fake": str(fake_dir.resolve()),
        },
    }
    (dataset_root / "manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    return pairs


def split_balanced(pairs: Iterable[Tuple[str, int]], seed: int = 42):
    by_label = {0: [], 1: []}
    for path, label in pairs:
        by_label[int(label)].append((path, int(label)))

    rng = random.Random(seed)
    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []

    for items in by_label.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def write_manifests(root: Path, train, val, test) -> None:
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: List[Tuple[str, int]]) -> None:
        payload = [{"video_path": path, "label": label} for path, label in rows]
        (manifest_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dump("sdfvd_train", train)
    dump("sdfvd_val", val)
    dump("sdfvd_test", test)
    dump("train", train)
    dump("val", val)
    dump("test", test)


def update_config(root: Path, train, val, test) -> None:
    cfg_path = root / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = "sdfvd_public"
    cfg["data"]["source_repo"] = REPO_ID
    cfg["data"]["train_videos"] = [path for path, _ in train]
    cfg["data"]["train_labels"] = [label for _, label in train]
    cfg["data"]["val_videos"] = [path for path, _ in val]
    cfg["data"]["val_labels"] = [label for _, label in val]
    cfg["data"]["test_videos"] = [path for path, _ in test]
    cfg["data"]["test_labels"] = [label for _, label in test]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public no-auth SDFVD deepfake dataset subset.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pairs = download_sdfvd(root, limit_per_class=args.limit_per_class)
    train, val, test = split_balanced(pairs, seed=args.seed)
    write_manifests(root, train, val, test)
    update_config(root, train, val, test)

    print("Downloaded public SDFVD dataset from Hugging Face without auth.")
    print(f"videos={len(pairs)} train={len(train)} val={len(val)} test={len(test)}")
    print(f"dataset_root={root / 'datasets' / 'sdfvd'}")
    print("Updated configs/default.yaml and data/manifests/*.json")


if __name__ == "__main__":
    main()
