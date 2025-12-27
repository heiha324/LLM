from __future__ import annotations

from typing import Dict

from .utils import load_json


def load_policy(path: str) -> Dict:
    return load_json(path)


def route_scene(features: Dict, policy: Dict) -> Dict:
    stats = features.get("stats", {})
    why = []

    fa = policy.get("fast_accept", {})
    fr = policy.get("fast_reject", {})
    esc = policy.get("escalate", {})

    if (
        stats.get("cloud_frac_full", 0.0) <= fa.get("cloud_frac_full_max", 0.15)
        and stats.get("boundary_uncertainty", 0.0) <= fa.get("boundary_uncertainty_max", 0.06)
        and stats.get("entropy_mean", 0.0) <= fa.get("entropy_mean_max", 0.12)
    ):
        route = "FAST_ACCEPT"
        why.append("low cloud fraction and low uncertainty")
    elif stats.get("cloud_frac_full", 0.0) >= fr.get("cloud_frac_full_min", 0.35):
        route = "FAST_REJECT"
        why.append("high cloud fraction")
    else:
        route = "ESCALATE"
        why.append("in gray zone; need second-check/patches")

    next_step = {
        "run_second_check": bool(esc.get("run_second_check", False)),
        "sample_patches": esc.get("sample_patches", {"enabled": False}),
    }

    return {"route": route, "why": why, "next": next_step}
