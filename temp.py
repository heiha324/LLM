#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def _collect_masks(mask_root: Path) -> List[Path]:
    return sorted(mask_root.rglob("*.npy"))


def _scale_to_uint8(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _save_visuals(img_path: Path, mask_path: Path, out_prefix: str) -> None:
    img = np.load(img_path)
    rgb = img[..., [2, 1, 0]].astype(np.float32)
    rgb_u8 = np.stack([_scale_to_uint8(rgb[..., c]) for c in range(3)], axis=-1)
    Image.fromarray(rgb_u8).save(f"{out_prefix}_rgb.png")

    mask = np.load(mask_path)
    mask_vis = np.zeros_like(mask, dtype=np.uint8)
    mask_vis[mask == 254] = 128
    mask_vis[mask == 255] = 255
    Image.fromarray(mask_vis).save(f"{out_prefix}_mask.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/data/cloud_detec_dataset/GF_1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--find-sample", action="store_true")
    parser.add_argument("--min-ratio", type=float, default=0.1)
    parser.add_argument("--out-prefix", default="sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mask_root = Path(args.data_root) / args.split / "mask"
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {mask_root}")

    paths = _collect_masks(mask_root)
    if not paths:
        raise FileNotFoundError(f"No mask files under: {mask_root}")

    if args.max_samples > 0 and len(paths) > args.max_samples:
        random.seed(args.seed)
        paths = random.sample(paths, args.max_samples)

    allowed = {0, 254, 255}
    value_counts: Dict[int, int] = {}
    bad_shapes: List[str] = []
    unknown_values: Dict[int, int] = {}
    cloud_ratios: List[float] = []

    for path in paths:
        mask = np.load(path)
        if mask.shape != (512, 512):
            bad_shapes.append(str(path))
            continue

        uniq, counts = np.unique(mask, return_counts=True)
        total = mask.size
        cloud_ratio = float(counts[uniq == 255][0]) / total if 255 in uniq else 0.0
        cloud_ratios.append(cloud_ratio)

        for val, cnt in zip(uniq.tolist(), counts.tolist()):
            value_counts[val] = value_counts.get(val, 0) + int(cnt)
            if val not in allowed:
                unknown_values[val] = unknown_values.get(val, 0) + int(cnt)

    print("Mask check summary")
    print(f"- split: {args.split}")
    print(f"- samples checked: {len(paths)}")
    print(f"- value counts: {dict(sorted(value_counts.items()))}")
    print(f"- allowed values: {sorted(allowed)} (254 treated as non-cloud)")
    if cloud_ratios:
        print(f"- mean cloud ratio (255 only): {float(np.mean(cloud_ratios)):.4f}")
    if bad_shapes:
        print(f"- bad shapes: {len(bad_shapes)} (showing first 5)")
        for p in bad_shapes[:5]:
            print(f"  - {p}")
    if unknown_values:
        print(f"- unknown values: {dict(sorted(unknown_values.items()))}")

    if bad_shapes or unknown_values:
        raise SystemExit(1)

    if args.find_sample:
        found = None
        best_254 = (None, 0.0)
        full_paths = _collect_masks(mask_root)
        for path in full_paths:
            mask = np.load(path)
            if mask.shape != (512, 512):
                continue
            counts = np.bincount(mask.ravel(), minlength=256)
            total = mask.size
            r0 = counts[0] / total
            r254 = counts[254] / total
            r255 = counts[255] / total
            if r254 > best_254[1]:
                best_254 = (path, r254, r0, r255)
            if r0 >= args.min_ratio and r254 >= args.min_ratio and r255 >= args.min_ratio:
                found = (path, r0, r254, r255)
                break

        if not found:
            if best_254[0] is None:
                print("No valid masks found.")
                raise SystemExit(1)
            mask_path, r254, r0, r255 = best_254
            rel = mask_path.relative_to(mask_root)
            img_path = mask_root.parent / "img" / rel
            _save_visuals(img_path, mask_path, args.out_prefix)
            print("No sample found with ratios >= min-ratio; using max-254 sample.")
            print(f"- mask: {mask_path}")
            print(f"- img: {img_path}")
            print(f"- ratios: 0={r0:.4f}, 254={r254:.6f}, 255={r255:.4f}")
            print(f"- saved: {args.out_prefix}_rgb.png, {args.out_prefix}_mask.png")
            return

        mask_path, r0, r254, r255 = found
        rel = mask_path.relative_to(mask_root)
        img_path = mask_root.parent / "img" / rel
        _save_visuals(img_path, mask_path, args.out_prefix)
        print("Found sample:")
        print(f"- mask: {mask_path}")
        print(f"- img: {img_path}")
        print(f"- ratios: 0={r0:.4f}, 254={r254:.4f}, 255={r255:.4f}")
        print(f"- saved: {args.out_prefix}_rgb.png, {args.out_prefix}_mask.png")


if __name__ == "__main__":
    main()
