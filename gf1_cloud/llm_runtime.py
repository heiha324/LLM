from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .llm_client import chat_completion, extract_json, get_message_text
from .llm_prompts import patch_verifier_prompt, router_prompt
from .patch_export import make_views

_VALID_ROUTES = {"FAST_ACCEPT", "FAST_REJECT", "ESCALATE"}


def _encode_png_data_url(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + data


def _normalize_route(route: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    route_val = str(route.get("route", "")).upper().strip()
    if route_val not in _VALID_ROUTES:
        raise ValueError(f"invalid route: {route_val}")

    why = route.get("why", [])
    if not isinstance(why, list):
        why = [str(why)]

    esc = policy.get("escalate", {})
    next_step = route.get("next", {}) if isinstance(route.get("next"), dict) else {}
    if route_val == "ESCALATE":
        if not next_step:
            next_step = dict(esc)
        if "run_second_check" not in next_step:
            next_step["run_second_check"] = bool(esc.get("run_second_check", False))
        sample = next_step.get("sample_patches")
        if not isinstance(sample, dict):
            sample = dict(esc.get("sample_patches", {"enabled": False}))
        if "enabled" not in sample:
            sample["enabled"] = bool(esc.get("sample_patches", {}).get("enabled", False))
        if "k" not in sample:
            sample["k"] = int(esc.get("sample_patches", {}).get("k", 12))
        if "strategy" not in sample and "strategy" in esc.get("sample_patches", {}):
            sample["strategy"] = esc.get("sample_patches", {}).get("strategy")
        next_step["sample_patches"] = sample
    else:
        if not next_step:
            next_step = {"run_second_check": False, "sample_patches": {"enabled": False}}

    return {"route": route_val, "why": why, "next": next_step}


def llm_route(
    features: Dict[str, Any],
    policy: Dict[str, Any],
    api_url: str,
    model: str,
    timeout: int = 60,
    max_tokens: int = 256,
    retries: int = 1,
) -> Dict[str, Any]:
    payload = dict(features)
    policy_id = policy.get("policy_id", "global_screening_v1")
    payload["policy_id"] = policy_id
    prompt = router_prompt(payload, policy_id=policy_id)
    messages = [{"role": "user", "content": prompt}]

    last_err: Exception | None = None
    for _ in range(retries + 1):
        response = chat_completion(
            api_url,
            model,
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        text = get_message_text(response)
        try:
            route = extract_json(text)
            return _normalize_route(route, policy)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"LLM route failed: {last_err}")


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def llm_verify_patch(
    patch_item: Dict[str, Any],
    api_url: str,
    model: str,
    timeout: int = 60,
    max_tokens: int = 256,
    retries: int = 1,
) -> Dict[str, Any]:
    img = patch_item["img"]
    mask_cloud = patch_item.get("mask_cloud")
    views = make_views(img, mask_cloud=mask_cloud)

    content = [
        {"type": "text", "text": patch_verifier_prompt()},
        {"type": "image_url", "image_url": {"url": _encode_png_data_url(views["rgb"])}},
        {"type": "image_url", "image_url": {"url": _encode_png_data_url(views["nirrg"])}},
        {"type": "image_url", "image_url": {"url": _encode_png_data_url(views["overlay"])}},
    ]
    messages = [{"role": "user", "content": content}]

    last_err: Exception | None = None
    for _ in range(retries + 1):
        response = chat_completion(
            api_url,
            model,
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        text = get_message_text(response)
        try:
            result = extract_json(text)
            return {
                "cloud_like_prob": _clamp01(result.get("cloud_like_prob", 0.0)),
                "confusing_surface_prob": _clamp01(result.get("confusing_surface_prob", 0.0)),
                "notes": str(result.get("notes", "")).strip(),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"LLM patch verify failed: {last_err}")


def llm_verify_patches(
    patch_items: List[Dict[str, Any]],
    api_url: str,
    model: str,
    timeout: int = 60,
    max_tokens: int = 256,
    retries: int = 1,
) -> List[Dict[str, Any]]:
    results = []
    for item in patch_items:
        result = llm_verify_patch(
            item,
            api_url=api_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            retries=retries,
        )
        results.append(result)
    return results
