"""
Sanity check: which parameters are trainable for a given training mode?
Mirrors the freeze logic in src/bs2_train_new_head_DA3.py without importing it.

Usage:
    python utils/sanity_check_trainable.py --mode scratch_head
    python utils/sanity_check_trainable.py --mode full_head
    python utils/sanity_check_trainable.py --mode lora_dpt_blocks
    python utils/sanity_check_trainable.py  # head inspection only
"""

import argparse
import re
from collections import defaultdict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from depth_anything_3.api import DepthAnything3

_DPT_LORA_TARGETS = r"model\.backbone\.pretrained\.blocks\.(4|11|17|23)\.(attn\.(qkv|proj)|mlp\.fc[12])"
_LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=16,
    init_lora_weights="gaussian",
    lora_dropout=0.0,
    bias="none",
    target_modules=_DPT_LORA_TARGETS,
)

_EXPECTED_PATTERNS = {
    "full_head":    re.compile(r"^model\.head\."),
    "scratch_head": re.compile(r"^model\.head\."),
    "lora_dpt_blocks": re.compile(r"lora_"),
}


def apply_mode(model: DepthAnything3, mode: str) -> DepthAnything3:
    if mode in ("full_head", "scratch_head"):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.head.parameters():
            p.requires_grad = True
    elif mode == "lora_dpt_blocks":
        model = get_peft_model(model, _LORA_CONFIG)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def prefix_summary(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        if p.requires_grad:
            prefix = name.split(".")[0] if "lora_" not in name else "lora_*"
            counts[prefix] += p.numel()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full_head", "scratch_head", "lora_dpt_blocks"], default=None)
    args = parser.parse_args()

    print("Loading model...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

    print(f"\nHead type : {type(model.model.head)}")
    print(f"Head      :\n{model.model.head}\n")

    if args.mode is None:
        return

    model = apply_mode(model, args.mode)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"Mode      : {args.mode}")
    print(f"Trainable : {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)\n")

    print("Trainable param names (first 30):")
    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}  {tuple(p.shape)}")
            shown += 1
            if shown >= 30:
                remaining = sum(1 for _, p in model.named_parameters() if p.requires_grad) - 30
                if remaining > 0:
                    print(f"  ... ({remaining} more)")
                break

    pattern = _EXPECTED_PATTERNS[args.mode]
    violations = [
        name for name, p in model.named_parameters()
        if p.requires_grad and not pattern.search(name)
    ]

    print()
    if violations:
        print(f"[FAIL] {len(violations)} trainable param(s) don't match expected pattern '{pattern.pattern}':")
        for v in violations[:10]:
            print(f"  {v}")
    else:
        print(f"[OK] All trainable params match expected pattern for mode '{args.mode}'")


if __name__ == "__main__":
    main()
