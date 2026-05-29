#!/usr/bin/env python3
"""Visualize 20 pseudo-labeled samples: RGB | GT depth | pseudo depth."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TRAIN_DIR = Path("/cluster/courses/cil/monocular-depth-estimation/train")
PSEUDO_DIR = Path("/work/scratch/nmeurer/outputs/baseline3/pseudo_labels_DA3-GIANT-1.1")
OUT_DIR = Path("/work/scratch/cdeubel/outputs/baseline3/pseudo_preview_20")
NUM = 20


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [1, 99])
    depth_norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    # RdYlBu_r: red=close, blue=far
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color=(0, 0, 0, 1))
    depth_norm = np.where(valid, depth_norm, np.nan)
    return (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)


def save_trio(rgb_path: Path, gt_path: Path, pseudo_path: Path, out_dir: Path) -> None:
    stem = rgb_path.stem.replace("_rgb", "")
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    gt_rgb = depth_to_rgb(np.load(gt_path).astype(np.float32))
    pseudo_rgb = depth_to_rgb(np.load(pseudo_path).astype(np.float32))

    h, w = rgb.shape[:2]
    if gt_rgb.shape[:2] != (h, w):
        gt_rgb = np.array(Image.fromarray(gt_rgb).resize((w, h), Image.NEAREST))
    if pseudo_rgb.shape[:2] != (h, w):
        pseudo_rgb = np.array(Image.fromarray(pseudo_rgb).resize((w, h), Image.NEAREST))

    Image.fromarray(rgb).save(out_dir / f"{stem}_rgb.png")
    Image.fromarray(gt_rgb).save(out_dir / f"{stem}_gt.png")
    Image.fromarray(pseudo_rgb).save(out_dir / f"{stem}_pseudo.png")
    print(f"  saved {stem}_{{rgb,gt,pseudo}}.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(TRAIN_DIR.glob("*_rgb.png"))[:NUM]
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png in {TRAIN_DIR}")

    written = 0
    for rgb_path in rgb_files:
        gt_path = TRAIN_DIR / rgb_path.name.replace("_rgb.png", "_depth.npy")
        pseudo_path = PSEUDO_DIR / rgb_path.name.replace("_rgb.png", "_depth.npy")
        if not gt_path.exists() or not pseudo_path.exists():
            print(f"  skip {rgb_path.name} (missing gt or pseudo)")
            continue
        save_trio(rgb_path, gt_path, pseudo_path, OUT_DIR)
        written += 1

    print(f"\nWrote {written * 3} images ({written} samples) to {OUT_DIR}")
    print("Files: <stem>_rgb.png | <stem>_gt.png | <stem>_pseudo.png")


if __name__ == "__main__":
    main()
