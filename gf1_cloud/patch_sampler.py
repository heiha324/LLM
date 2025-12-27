from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage

from .morphology import ring_mask


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _topk_points(score: np.ndarray, k: int) -> List[Tuple[int, int]]:
    if k <= 0:
        return []
    flat = score.ravel()
    if flat.size == 0:
        return []
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    ys, xs = np.unravel_index(idx, score.shape)
    return list(zip(ys.tolist(), xs.tolist()))


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return labeled == largest


def _select_centers(
    candidates: List[Tuple[int, int]], k: int, min_dist: int
) -> List[Tuple[int, int]]:
    if not candidates:
        return []
    selected: List[Tuple[int, int]] = []
    min_dist_sq = min_dist * min_dist
    for y, x in candidates:
        if all((y - yy) ** 2 + (x - xx) ** 2 >= min_dist_sq for yy, xx in selected):
            selected.append((y, x))
            if len(selected) >= k:
                break
    return selected


def _crop_bounds(center: Tuple[int, int], shape: Tuple[int, int], patch: int) -> Tuple[int, int, int, int]:
    h, w = shape
    half = patch // 2
    y, x = center
    y0 = max(0, y - half)
    x0 = max(0, x - half)
    y1 = min(h, y0 + patch)
    x1 = min(w, x0 + patch)
    y0 = max(0, y1 - patch)
    x0 = max(0, x1 - patch)
    return y0, y1, x0, x1


def sample_patches(
    p_cloud: np.ndarray,
    mask_cloud: np.ndarray,
    mask_cloud_2: np.ndarray | None = None,
    k: int = 12,
    patch: int = 256,
    ring_radius: int = 5,
) -> List[Tuple[int, int, int, int]]:
    entropy = _entropy(p_cloud)
    ring = ring_mask(mask_cloud, radius=ring_radius)

    k1 = k // 2
    k2 = max(1, int(k * 0.3))
    k3 = max(0, k - k1 - k2)

    cand1 = _topk_points(entropy * ring, k1)

    largest = _largest_cc(mask_cloud)
    edge = ring_mask(largest, radius=2)
    cand2 = _topk_points(entropy * edge, k2)

    cand3: List[Tuple[int, int]] = []
    if mask_cloud_2 is not None and k3 > 0:
        disagree = np.logical_xor(mask_cloud, mask_cloud_2)
        cand3 = _topk_points(entropy * disagree, k3)

    merged = cand1 + cand2 + cand3
    centers = _select_centers(merged, k=k, min_dist=patch // 3)
    bounds = [_crop_bounds(c, p_cloud.shape, patch) for c in centers]
    return bounds
