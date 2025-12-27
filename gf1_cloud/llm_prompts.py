from __future__ import annotations

import json
from typing import Dict


def router_prompt(features: Dict, policy_id: str = "global_screening_v1") -> str:
    payload = json.dumps(features, ensure_ascii=True)
    return (
        "You are an automatic routing module for GF-1 cloud screening.\n"
        "Input is a JSON with scene features, thresholds, and policy_id.\n"
        "You must output ONLY JSON with schema:\n"
        '{"route":"FAST_ACCEPT|FAST_REJECT|ESCALATE","why":["..."],"next":{...}}\n'
        "Do not output extra text. Do not change thresholds.\n\n"
        f"Policy {policy_id}:\n"
        "- FAST_ACCEPT: cloud_frac_full<=0.15 AND boundary_uncertainty<=0.06 AND entropy_mean<=0.12\n"
        "- FAST_REJECT: cloud_frac_full>=0.35\n"
        "- otherwise: ESCALATE with run_second_check=true and sample_patches enabled (k=12)\n\n"
        "Input JSON:\n"
        f"{payload}"
    )


def patch_verifier_prompt() -> str:
    return (
        "You are an automatic patch verifier for cloud vs non-cloud.\n"
        "You will receive RGB, NIRRG, and overlay images.\n"
        "Output ONLY JSON:\n"
        '{"cloud_like_prob":0.0,"confusing_surface_prob":0.0,"notes":"short"}\n'
        "Do not output extra text."
    )


def judge_explainer_prompt(payload: Dict) -> str:
    data = json.dumps(payload, ensure_ascii=True)
    return (
        "You are a decision explainer.\n"
        "Input includes features, second-check agreement, patch verifier summary, and fusion result.\n"
        "Output ONLY JSON with concise numeric reasons.\n"
        "Do not output ACCEPT/REJECT. Do not change thresholds.\n\n"
        f"Input JSON:\n{data}"
    )
