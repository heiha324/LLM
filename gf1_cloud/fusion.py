from __future__ import annotations

from typing import Dict

from .utils import load_json


def load_fusion_config(path: str) -> Dict:
    return load_json(path)


def fuse_decision(
    features: Dict,
    fusion_cfg: Dict,
    second_check: Dict | None = None,
    patch_stats: Dict | None = None,
) -> Dict:
    weights = fusion_cfg.get("weights", {})
    thresholds = fusion_cfg.get("thresholds", {})

    stats = features.get("stats", {})
    score = 0.0
    score += weights.get("cloud_frac_full", 0.0) * stats.get("cloud_frac_full", 0.0)
    score += weights.get("boundary_uncertainty", 0.0) * stats.get("boundary_uncertainty", 0.0)

    if second_check:
        agreement = second_check.get("agreement_iou_cloud", 1.0)
        score += weights.get("disagreement", 0.0) * (1.0 - agreement)

    if patch_stats:
        score += weights.get("patch_cloud_like_mean", 0.0) * patch_stats.get(
            "patch_cloud_like_mean", 0.0
        )
        score += weights.get("patch_confusing_surface_mean", 0.0) * patch_stats.get(
            "patch_confusing_surface_mean", 0.0
        )

    t_accept = thresholds.get("accept", 0.3)
    t_reject = thresholds.get("reject", 0.6)
    if score < t_accept:
        decision = "ACCEPT"
    elif score >= t_reject:
        decision = "REJECT"
    else:
        decision = "REJECT_SAFE"

    return {"score": float(score), "decision": decision}
