from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run(cmd: List[str], cwd: Path | None = None) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def ensure_dirs(root: Path):
    for p in [
        root / "datasets" / "faceforensicspp",
        root / "datasets" / "celebdf",
        root / "datasets" / "dfdc",
        root / "data" / "manifests",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def clone_or_pull(repo_url: str, target: Path):
    if target.exists() and (target / ".git").exists():
        run(["git", "pull"], cwd=target)
    elif not target.exists():
        run(["git", "clone", repo_url, str(target)])


def try_faceforensics_download(root: Path):
    scripts_dir = root / "external" / "FaceForensics"
    scripts_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_or_pull("https://github.com/ondyari/FaceForensics.git", scripts_dir)

    downloader = scripts_dir / "download_videos.py"
    if downloader.exists():
        out_dir = root / "datasets" / "faceforensicspp"
        print("Attempting FaceForensics++ download (official script)...")
        # Official script may still require dataset terms/credentials.
        run([
            "python",
            str(downloader),
            "--output_path",
            str(out_dir),
            "--dataset",
            "original_youtube",
            "--compression",
            "c23",
        ])
    else:
        print("FaceForensics downloader script not found; please inspect repository updates.")


def try_celebdf_download(root: Path):
    scripts_dir = root / "external" / "celeb-deepfakeforensics"
    scripts_dir.parent.mkdir(parents=True, exist_ok=True)
    clone_or_pull("https://github.com/yuezunli/celeb-deepfakeforensics.git", scripts_dir)

    print(
        "Celeb-DF repository cloned. Dataset download is typically gated by external links/terms; "
        "place files under datasets/celebdf/{Celeb-real,Celeb-synthesis}."
    )


def try_dfdc_download(root: Path):
    if shutil.which("kaggle") is None:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Missing ~/.kaggle/kaggle.json. Create Kaggle API credentials first.")
        return

    out_dir = root / "datasets" / "dfdc"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Attempting DFDC download via Kaggle API...")
    run(["kaggle", "competitions", "download", "-c", "deepfake-detection-challenge", "-p", str(out_dir)])


def gather_videos_with_label(dataset_root: Path) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    if not dataset_root.exists():
        return pairs

    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        low = str(path).lower()
        label = None
        if any(k in low for k in ["real", "original", "youtube"]):
            label = 0
        if any(k in low for k in ["fake", "manipulated", "synthesis", "deepfake", "faceswap", "face2face", "neuraltextures"]):
            label = 1
        if label is not None:
            pairs.append((str(path.resolve()), label))
    return pairs


def gather_dfdc_with_metadata(dataset_root: Path) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    if not dataset_root.exists():
        return pairs

    meta_files = list(dataset_root.rglob("metadata.json"))
    if not meta_files:
        return pairs

    for meta_file in meta_files:
        base = meta_file.parent
        try:
            data: Dict[str, Dict[str, str]] = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for name, info in data.items():
            label = info.get("label", "").upper()
            if label not in {"REAL", "FAKE"}:
                continue
            vid = base / name
            if vid.exists():
                pairs.append((str(vid.resolve()), 0 if label == "REAL" else 1))
    return pairs


def split_pairs(pairs: List[Tuple[str, int]], seed: int = 42):
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    return train, val, test


def write_manifest(root: Path, train, val, test):
    mdir = root / "data" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, items: List[Tuple[str, int]]):
        with open(mdir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump([{"video_path": p, "label": l} for p, l in items], f, indent=2)

    dump("train", train)
    dump("val", val)
    dump("test", test)


def update_default_yaml(root: Path, train, val, test):
    cfg_path = root / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg["data"]["train_videos"] = [p for p, _ in train]
    cfg["data"]["train_labels"] = [l for _, l in train]
    cfg["data"]["val_videos"] = [p for p, _ in val]
    cfg["data"]["val_labels"] = [l for _, l in val]
    cfg["data"]["test_videos"] = [p for p, _ in test]
    cfg["data"]["test_labels"] = [l for _, l in test]

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ensure_dirs(root)

    if not args.skip_download:
        try_faceforensics_download(root)
        try_celebdf_download(root)
        try_dfdc_download(root)

    all_pairs: List[Tuple[str, int]] = []
    all_pairs.extend(gather_videos_with_label(root / "datasets" / "faceforensicspp"))
    all_pairs.extend(gather_videos_with_label(root / "datasets" / "celebdf"))
    dfdc_pairs = gather_dfdc_with_metadata(root / "datasets" / "dfdc")
    if dfdc_pairs:
        all_pairs.extend(dfdc_pairs)
    else:
        all_pairs.extend(gather_videos_with_label(root / "datasets" / "dfdc"))

    if not all_pairs:
        print("No labeled videos discovered yet.")
        print("Place datasets under:")
        print("- datasets/faceforensicspp")
        print("- datasets/celebdf")
        print("- datasets/dfdc")
        print("Then rerun: python scripts/setup_datasets.py --skip-download")
        return

    train, val, test = split_pairs(all_pairs, seed=args.seed)
    write_manifest(root, train, val, test)
    update_default_yaml(root, train, val, test)

    print(f"Prepared dataset config: train={len(train)}, val={len(val)}, test={len(test)}")
    print("Updated configs/default.yaml and wrote data/manifests/*.json")


if __name__ == "__main__":
    main()
