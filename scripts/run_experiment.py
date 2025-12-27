#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gf1_cloud.data import GF1CloudDataset, collate_with_meta
from gf1_cloud.fusion import load_fusion_config
from gf1_cloud.logging_utils import setup_logging
from gf1_cloud.llm_runtime import llm_route, llm_verify_patches
from gf1_cloud.models import CloudSegNet
from gf1_cloud.pipeline import run_pipeline_from_probs
from gf1_cloud.progress import tqdm
from gf1_cloud.router import load_policy
from gf1_cloud.utils import ensure_dir, load_json, save_json


def _metrics(pred_cloud: np.ndarray, gt_cloud: np.ndarray) -> dict:
    """Compute OA/Precision/Recall/FAR/F1/IoU for cloud vs non-cloud."""
    tp = np.logical_and(pred_cloud, gt_cloud).sum()
    fp = np.logical_and(pred_cloud, ~gt_cloud).sum()
    fn = np.logical_and(~pred_cloud, gt_cloud).sum()
    tn = np.logical_and(~pred_cloud, ~gt_cloud).sum()

    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    far = fp / (fp + tn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "oa": float(oa),
        "precision": float(precision),
        "recall": float(recall),
        "far": float(far),
        "f1": float(f1),
        "iou": float(iou),
    }


def _aggregate(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {k: 0.0 for k in ["oa", "precision", "recall", "far", "f1", "iou"]}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def _load_model(ckpt: str, device: str) -> torch.nn.Module:
    model = CloudSegNet()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default="configs/calibration.json")
    parser.add_argument("--policy", default="configs/policy_v1.json")
    parser.add_argument("--fusion", default="configs/fusion_weights.json")
    parser.add_argument("--out-dir", default="outputs/experiment")
    parser.add_argument("--llm-api-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", default="logs/experiment.log")
    args = parser.parse_args()

    setup_logging(args.log_file if args.log_file else None)
    thresholds = load_json(args.calibration)
    policy = load_policy(args.policy)
    fusion_cfg = load_fusion_config(args.fusion)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认驱动/CUDA 是否安装正确。")
    device = args.device
    model = _load_model(args.checkpoint, device)

    ds = GF1CloudDataset(args.data_root, args.split, return_meta=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_with_meta,
    )

    ensure_dir(args.out_dir)

    def router_fn(features: dict, policy_cfg: dict) -> dict:
        return llm_route(
            features,
            policy_cfg,
            api_url=args.llm_api_url,
            model=args.llm_model,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            retries=args.llm_retries,
        )

    def patch_verifier_fn(patch_items: list) -> list:
        return llm_verify_patches(
            patch_items,
            api_url=args.llm_api_url,
            model=args.llm_model,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            retries=args.llm_retries,
        )

    metrics_baseline = []
    metrics_pipeline = []

    with torch.inference_mode():
        for imgs, masks, metas in tqdm(loader, desc="experiment", total=len(loader), leave=True):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            imgs_cpu = imgs.detach().cpu().numpy()
            probs_cpu = probs.detach().cpu().numpy()
            gt_cpu = masks.cpu().numpy()

            for i in range(imgs.size(0)):
                p_cloud = probs_cpu[i, 0]
                gt_cloud = gt_cpu[i, 0] > 0.5

                # Baseline UNet thresholding
                mask_cloud_base = p_cloud > thresholds.get("t_cloud", 0.5)
                metrics_baseline.append(_metrics(mask_cloud_base, gt_cloud))

                # Pipeline with LLM router + patch verifier
                result, mask_cloud_pipe = run_pipeline_from_probs(
                    imgs_cpu[i].transpose(1, 2, 0),
                    p_cloud,
                    thresholds=thresholds,
                    policy=policy,
                    fusion_cfg=fusion_cfg,
                    second_probs=None,
                    router=router_fn,
                    patch_verifier=patch_verifier_fn,
                    patch_size=256,
                    return_masks=True,
                    require_llm=True,
                )

                # Optionally adjust thresholds on REJECT (same逻辑：当前保持原样，仅统计 base)
                metrics_pipeline.append(_metrics(mask_cloud_pipe, gt_cloud))

    summary = {
        "baseline": _aggregate(metrics_baseline),
        "pipeline": _aggregate(metrics_pipeline),
    }

    out_path = Path(args.out_dir) / "experiment_metrics.json"
    save_json(summary, str(out_path))
    logging.info("Experiment metrics saved to %s", out_path)
    logging.info(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
