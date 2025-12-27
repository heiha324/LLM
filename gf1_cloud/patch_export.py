from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .utils import ensure_dir, scale_to_uint8


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    return scale_to_uint8(img)


def make_views(
    img: np.ndarray,
    mask_cloud: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    rgb = _to_uint8_rgb(img[..., [2, 1, 0]])
    nirrg = _to_uint8_rgb(img[..., [3, 2, 1]])

    overlay = rgb.astype(np.float32)
    if mask_cloud is not None:
        overlay[mask_cloud] = 0.6 * overlay[mask_cloud] + 0.4 * np.array([255, 0, 0])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return {"rgb": rgb, "nirrg": nirrg, "overlay": overlay}


def save_patch_views(
    out_dir: str,
    patch_id: str,
    img_patch: np.ndarray,
    mask_cloud: np.ndarray | None = None,
) -> Dict[str, str]:
    ensure_dir(out_dir)
    views = make_views(img_patch, mask_cloud=mask_cloud)
    paths: Dict[str, str] = {}
    for key, arr in views.items():
        path = str(Path(out_dir) / f"{patch_id}_{key}.png")
        Image.fromarray(arr).save(path)
        paths[key] = path
    return paths


def extract_patch(
    img: np.ndarray,
    bounds: Tuple[int, int, int, int],
) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    return img[y0:y1, x0:x1]
