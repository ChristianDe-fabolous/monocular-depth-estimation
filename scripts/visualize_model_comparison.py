#!/usr/bin/env python3
"""Visualize 20 test samples across multiple submission CSVs.

Saves per sample:
  <stem>_rgb.png          — test RGB image
  <stem>_<model>.png      — depth map from each model (RdYlBu_r, red=close blue=far)

Usage:
  python scripts/visualize_model_comparison.py \
      --csvs baseline1:/path/to/bs1.csv baseline2:/path/to/bs2.csv \
      [--test-dir /cluster/courses/cil/monocular-depth-estimation/test] \
      [--out-dir /work/scratch/cdeubel/outputs/model_comparison] \
      [--num 20] \
      [--depth-shape 560 560]
"""
from pathlib import Path
import argparse
import base64
import zlib

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TEST_DIR = Path("/cluster/courses/cil/monocular-depth-estimation/test")
OUT_DIR = Path("/work/scratch/cdeubel/outputs/model_comparison")
DEPTH_SHAPE = (560, 560)
NUM = 20


def decode_depth(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    compressed = base64.b64decode(encoded)
    raw = zlib.decompress(compressed)
    return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [1, 99])
    depth_norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color=(0, 0, 0, 1))
    depth_norm = np.where(valid, depth_norm, np.nan)
    return (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help="Model CSVs as name:path pairs, e.g. baseline1:/path/bs1.csv",
    )
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num", type=int, default=NUM)
    parser.add_argument("--depth-shape", type=int, nargs=2, default=list(DEPTH_SHAPE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = tuple(args.depth_shape)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse name:path pairs
    models: list[tuple[str, Path]] = []
    for entry in args.csvs:
        if ":" not in entry:
            raise ValueError(f"Expected NAME:PATH, got: {entry}")
        name, path = entry.split(":", 1)
        models.append((name, Path(path)))

    # Load CSVs lazily into dicts: idx -> encoded string
    import pandas as pd
    model_data: dict[str, dict[str, str]] = {}
    for name, csv_path in models:
        df = pd.read_csv(csv_path)
        # id format: test_{idx}_depth
        model_data[name] = {row["id"].split("_")[1]: row["Depths"] for _, row in df.iterrows()}
        print(f"Loaded {len(model_data[name])} entries from {name} ({csv_path.name})")

    # Pick test images — take first --num that all models have
    rgb_files = sorted(args.test_dir.glob("*_rgb.png"))
    selected: list[tuple[str, Path]] = []  # (idx, rgb_path)
    for rgb_path in rgb_files:
        # test_{idx}_rgb.png -> idx
        idx = rgb_path.stem.replace("_rgb", "").split("_")[-1]
        if all(idx in model_data[name] for name, _ in models):
            selected.append((idx, rgb_path))
        if len(selected) >= args.num:
            break

    if not selected:
        raise RuntimeError("No test images found that are present in all CSVs.")

    print(f"\nProcessing {len(selected)} samples...")
    for idx, rgb_path in selected:
        stem = rgb_path.stem.replace("_rgb", "")

        # Save RGB
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        Image.fromarray(rgb).save(args.out_dir / f"{stem}_rgb.png")

        # Save one depth image per model
        for name, _ in models:
            depth = decode_depth(model_data[name][idx], shape)
            depth_img = depth_to_rgb(depth)
            h, w = rgb.shape[:2]
            if depth_img.shape[:2] != (h, w):
                depth_img = np.array(Image.fromarray(depth_img).resize((w, h), Image.NEAREST))
            Image.fromarray(depth_img).save(args.out_dir / f"{stem}_{name}.png")

        print(f"  {stem}: rgb + {[n for n, _ in models]}")

    print(f"\nDone. {len(selected) * (1 + len(models))} images in {args.out_dir}")


if __name__ == "__main__":
    main()
