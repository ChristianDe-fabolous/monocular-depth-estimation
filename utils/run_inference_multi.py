"""
Multi-model test-set inference script.
Run on a Jupyter/compute node with GPU. Fill in MODELS config below.
Creates one submission CSV per model + a zip of all CSVs at the end.
"""

# ── FILL IN YOUR MODELS HERE ─────────────────────────────────────────────────
# mode:
#   "zero_shot"     – load DA3MONO-LARGE as-is, no checkpoint
#   "full_head"     – load DA3MONO-LARGE + load state_dict from ckpt_path (head only trained)
#   "surface_head"  – same as full_head
#   "lora_dpt_blocks" – wrap with PEFT LoRA first, then load state_dict from ckpt_path
# denormalize:
#   True  if model was trained with depth normalized to [0,1] (bs2 training scripts)
#   False for zero-shot (DA3 returns metric depth directly)

MODELS = [
    {
        "name": "baseline1_zeroshot",
        "mode": "zero_shot",
        "ckpt_path": None,
        "denormalize": False,
    },
    {
        "name": "baseline2_full_head",
        "mode": "full_head",
        "ckpt_path": "/work/scratch/cdeubel/outputs/baseline2/baseline2-full_head/checkpoints/best.pth",
        "denormalize": True,
    },
    {
        "name": "model3",
        "mode": "full_head",
        "ckpt_path": "/work/scratch/cdeubel/FILL_IN/best.pth",
        "denormalize": True,
    },
]

TEST_DATA_ROOT = "/cluster/courses/cil/monocular-depth-estimation/test"
OUTPUT_ROOT    = "/work/scratch/cdeubel/submissions"
INFER_BATCH    = 32
# ─────────────────────────────────────────────────────────────────────────────

import base64
import gc
import zipfile
import zlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

_DEPTH_MIN_M = 0.001
_DEPTH_MAX_M = 80.0

_DPT_LORA_TARGETS = r"model\.backbone\.pretrained\.blocks\.(4|11|17|23)\.(attn\.(qkv|proj)|mlp\.fc[12])"


def denormalize_depth(d: np.ndarray) -> np.ndarray:
    return d * (_DEPTH_MAX_M - _DEPTH_MIN_M) + _DEPTH_MIN_M


def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    return base64.b64encode(zlib.compress(depth.tobytes(), level=9)).decode("utf-8")


def load_model(mode: str, ckpt_path: Optional[str], device: torch.device) -> DepthAnything3:
    print(f"  Loading DA3MONO-LARGE base model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

    if mode == "lora_dpt_blocks":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            init_lora_weights="gaussian",
            lora_dropout=0.0,
            bias="none",
            target_modules=_DPT_LORA_TARGETS,
        )
        model = get_peft_model(model, lora_config)

    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        epoch_info = f" (epoch {ckpt.get('epoch', '?')}, val_si_rmse={ckpt.get('val_si_rmse', '?')})"
        print(f"  Loaded checkpoint:{epoch_info}")

    return model.to(device).eval()


def run_inference(model: DepthAnything3, image_paths: List[Path], pred_dir: Path,
                  denormalize: bool, device: torch.device) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(image_paths), INFER_BATCH):
        batch = image_paths[i : i + INFER_BATCH]
        with torch.no_grad():
            predictions = model.inference(
                image=[str(p) for p in batch],
                process_res=560,
                process_res_method="upper_bound_resize",
            )
        for p, depth in zip(batch, predictions.depth):
            assert depth.shape == (560, 560), f"Unexpected depth shape {depth.shape}"
            depth = depth.astype(np.float32)
            if denormalize:
                depth = denormalize_depth(depth)
            valid = np.isfinite(depth) & (depth > 0)
            if not np.all(valid):
                fill = float(np.median(depth[valid])) if np.any(valid) else 1.0
                depth = np.where(valid, depth, fill).astype(np.float32)
            depth = np.clip(depth, _DEPTH_MIN_M if denormalize else 1e-6, None)
            np.save(pred_dir / (p.stem.replace("_rgb", "") + ".npy"), depth)

        done = min(i + INFER_BATCH, len(image_paths))
        print(f"  [{done}/{len(image_paths)}] done", end="\r")
    print()


def build_submission_csv(pred_dir: Path, out_csv: Path) -> None:
    rows = []
    for pred_path in sorted(pred_dir.glob("test_*.npy")):
        idx = pred_path.stem.split("_")[-1]
        rows.append({"id": f"test_{idx}_depth", "Depths": encode_depth(np.load(pred_path))})
    import pandas as pd
    pd.DataFrame(rows, columns=["id", "Depths"]).to_csv(out_csv, index=False)
    print(f"  Submission CSV: {out_csv}  ({len(rows)} predictions)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available, running on CPU (will be slow)")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    test_dir = Path(TEST_DATA_ROOT)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    image_paths: List[Path] = sorted(test_dir.glob("*_rgb.png"))
    print(f"Found {len(image_paths)} test images\n")

    output_root = Path(OUTPUT_ROOT)
    csv_paths = []

    for cfg in MODELS:
        name = cfg["name"]
        print(f"=== {name} ===")
        model_dir = output_root / name
        pred_dir  = model_dir / "preds"
        out_csv   = model_dir / f"{name}_submission.csv"

        model = load_model(cfg["mode"], cfg["ckpt_path"], device)
        print(f"  Running inference ({len(image_paths)} images, batch={INFER_BATCH})...")
        run_inference(model, image_paths, pred_dir, cfg["denormalize"], device)

        print(f"  Building submission CSV...")
        build_submission_csv(pred_dir, out_csv)
        csv_paths.append(out_csv)

        # Free GPU memory before next model
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU memory cleared\n")

    # Zip all CSVs for easy download
    zip_path = output_root / "all_submissions.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv in csv_paths:
            zf.write(csv, csv.name)
    print(f"\n=== Done ===")
    print(f"Zip with all submissions: {zip_path}")
    for csv in csv_paths:
        print(f"  {csv}")


if __name__ == "__main__":
    main()
