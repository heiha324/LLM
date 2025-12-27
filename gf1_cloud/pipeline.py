from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .fusion import fuse_decision
from .metrics import compute_features
from .patch_sampler import sample_patches
from .router import route_scene


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _predict_probs(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    temperature: float = 1.0,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(img_t)
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.sigmoid(logits)
    p_cloud = _to_numpy(probs[:, 0]).squeeze(0)
    return p_cloud


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def run_pipeline_from_probs(
    img: np.ndarray,
    p_cloud: np.ndarray,
    thresholds: Dict[str, float],
    policy: Dict,
    fusion_cfg: Dict,
    second_probs: Optional[np.ndarray] = None,
    router: Optional[Callable[[Dict, Dict], Dict]] = None,
    patch_verifier: Optional[Callable[[List[Dict]], List[Dict]]] = None,
    patch_size: int = 256,
    return_masks: bool = False,
    require_llm: bool = False,
) -> Dict:
    features, mask_cloud = compute_features(p_cloud, thresholds=thresholds, img=img)
    features["policy_id"] = policy.get("policy_id", "global_screening_v1")

    if router is None:
        if require_llm:
            raise RuntimeError("LLM router is required but not provided.")
        route = route_scene(features, policy)
    else:
        route = router(features, policy)

    second_check = None
    mask_cloud_2 = None
    if route["route"] == "ESCALATE" and second_probs is not None:
        mask_cloud_2 = second_probs > thresholds.get("t_cloud", 0.5)
        second_check = {
            "cloud_frac_full_2": float(mask_cloud_2.mean()),
            "agreement_iou_cloud": _iou(mask_cloud, mask_cloud_2),
            "disagreement_area": float(np.logical_xor(mask_cloud, mask_cloud_2).mean()),
        }

    patch_bounds: List[Tuple[int, int, int, int]] = []
    patch_results: List[Dict] = []
    sample_cfg = route.get("next", {}).get("sample_patches", {})
    if route.get("route") == "ESCALATE" and sample_cfg.get("enabled"):
        patch_bounds = sample_patches(
            p_cloud,
            mask_cloud,
            mask_cloud_2=mask_cloud_2,
            k=int(sample_cfg.get("k", 12)),
            patch=patch_size,
        )
        if patch_verifier is None:
            if require_llm:
                raise RuntimeError("LLM patch verifier is required but not provided.")
        else:
            patch_items = []
            for y0, y1, x0, x1 in patch_bounds:
                patch_items.append(
                    {
                        "img": img[y0:y1, x0:x1],
                        "mask_cloud": mask_cloud[y0:y1, x0:x1],
                    }
                )
            patch_results = patch_verifier(patch_items)

    patch_stats = None
    if patch_results:
        cloud_probs = [p.get("cloud_like_prob", 0.0) for p in patch_results]
        confuse_probs = [p.get("confusing_surface_prob", 0.0) for p in patch_results]
        patch_stats = {
            "patch_cloud_like_mean": float(np.mean(cloud_probs)),
            "patch_confusing_surface_mean": float(np.mean(confuse_probs)),
        }

    decision = fuse_decision(features, fusion_cfg, second_check=second_check, patch_stats=patch_stats)

    output = {
        "features": features,
        "route": route,
        "second_check": second_check,
        "patch_bounds": patch_bounds,
        "patch_results": patch_results,
        "patch_stats": patch_stats,
        "decision": decision,
    }
    if return_masks:
        return output, mask_cloud
    return output


def run_pipeline_on_sample(
    img: np.ndarray,
    model: torch.nn.Module,
    thresholds: Dict[str, float],
    policy: Dict,
    fusion_cfg: Dict,
    device: str = "cpu",
    second_model: Optional[torch.nn.Module] = None,
    router: Optional[Callable[[Dict, Dict], Dict]] = None,
    patch_verifier: Optional[Callable[[List[Dict]], List[Dict]]] = None,
    patch_size: int = 256,
    return_masks: bool = False,
    require_llm: bool = False,
) -> Dict:
    img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    p_cloud = _predict_probs(
        model,
        img_t,
        temperature=float(thresholds.get("prob_temperature", 1.0)),
    )
    second_probs = None
    if second_model is not None:
        second_probs = _predict_probs(second_model, img_t)

    return run_pipeline_from_probs(
        img,
        p_cloud,
        thresholds=thresholds,
        policy=policy,
        fusion_cfg=fusion_cfg,
        second_probs=second_probs,
        router=router,
        patch_verifier=patch_verifier,
        patch_size=patch_size,
        return_masks=return_masks,
        require_llm=require_llm,
    )
