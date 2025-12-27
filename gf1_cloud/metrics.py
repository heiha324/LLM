from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .morphology import connected_components, ring_mask


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _safe_mean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else 0.0


def _quality_scores(img: np.ndarray) -> Dict[str, float]:
    rgb = img[..., [2, 1, 0]]
    brightness = rgb.mean(axis=-1)
    std_all = float(np.std(brightness))
    row_std = float(np.std(brightness.mean(axis=1)))
    col_std = float(np.std(brightness.mean(axis=0)))
    stripe_score = 0.0
    if std_all > 0:
        stripe_score = float((row_std + col_std) / (std_all + 1e-6))

    p99 = np.percentile(brightness, 99)
    overexposure_ratio = float(np.mean(brightness >= p99))

    nir = img[..., 3]
    p95 = np.percentile(brightness, 95)
    p50_nir = np.percentile(nir, 50)
    glare_like_ratio = float(np.mean((brightness >= p95) & (nir <= p50_nir)))

    return {
        "stripe_score": stripe_score,
        "overexposure_ratio": overexposure_ratio,
        "glare_like_ratio": glare_like_ratio,
    }


def compute_features(
    p_cloud: np.ndarray,
    thresholds: Dict[str, float],
    img: np.ndarray | None = None,
    ring_radius: int = 5,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    t_cloud = float(thresholds.get("t_cloud", 0.5))
    mask_cloud = p_cloud > t_cloud

    stats: Dict[str, float] = {}
    stats["cloud_frac_full"] = float(mask_cloud.mean())
    stats["cloud_conf_mean"] = _safe_mean(p_cloud[mask_cloud])

    ent = _entropy(p_cloud)
    stats["entropy_mean"] = float(ent.mean())
    ring = ring_mask(mask_cloud, radius=ring_radius)
    stats["boundary_uncertainty"] = _safe_mean(ent[ring])

    num_cc, areas = connected_components(mask_cloud)
    stats["num_cloud_cc"] = float(num_cc)
    total = mask_cloud.size
    if areas.size:
        stats["largest_cloud_cc_frac"] = float(areas.max() / total)
        stats["cc_area_p90"] = float(np.percentile(areas, 90))
        stats["cc_area_max"] = float(areas.max())
    else:
        stats["largest_cloud_cc_frac"] = 0.0
        stats["cc_area_p90"] = 0.0
        stats["cc_area_max"] = 0.0
    stats["fragmentation"] = float(num_cc / (stats["cloud_frac_full"] + 1e-6))

    quality = _quality_scores(img) if img is not None else {}

    features = {
        "stats": stats,
        "quality": quality,
        "thresholds": {"t_cloud": t_cloud},
    }
    return features, mask_cloud
