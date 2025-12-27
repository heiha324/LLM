#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gf1_cloud.data import GF1CloudDataset, collate_with_meta
from gf1_cloud.fusion import load_fusion_config
from gf1_cloud.llm_runtime import llm_route, llm_verify_patches
from gf1_cloud.models import CloudSegNet
from gf1_cloud.patch_export import extract_patch, save_patch_views
from gf1_cloud.pipeline import run_pipeline_from_probs
from gf1_cloud.progress import tqdm
from gf1_cloud.router import load_policy
from gf1_cloud.utils import ensure_dir, load_json, save_json
from gf1_cloud.logging_utils import setup_logging


def _save_mask(path: Path, mask_cloud: np.ndarray) -> None:
    mask = np.zeros_like(mask_cloud, dtype=np.uint8)
    mask[mask_cloud] = 255
    np.save(path, mask)


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
    parser.add_argument("--second-checkpoint")
    parser.add_argument("--calibration", default="configs/calibration.json")
    parser.add_argument("--policy", default="configs/policy_v1.json")
    parser.add_argument("--fusion", default="configs/fusion_weights.json")
    parser.add_argument("--out-dir", default="outputs/pipeline")
    parser.add_argument("--patch-dir", default="outputs/patches")
    parser.add_argument("--mask-dir", default="outputs/masks")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--llm-api-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--log-file", default="logs/pipeline.log")
    args = parser.parse_args()

    setup_logging(args.log_file if args.log_file else None)
    logging.info("Pipeline start: split=%s", args.split)
    thresholds = load_json(args.calibration)
    policy = load_policy(args.policy)
    fusion_cfg = load_fusion_config(args.fusion)

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

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认驱动/CUDA 是否安装正确。")
    device = args.device
    model = _load_model(args.checkpoint, device)
    second_model = _load_model(args.second_checkpoint, device) if args.second_checkpoint else None

    ds = GF1CloudDataset(args.data_root, args.split, return_meta=True)
    ensure_dir(args.out_dir)
    ensure_dir(args.patch_dir)
    ensure_dir(args.mask_dir)

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

    count = 0
    with torch.inference_mode():
        for imgs, masks, metas in tqdm(
            loader, desc="pipeline", total=len(loader), leave=True
        ):
            imgs = imgs.to(device, non_blocking=True)
            probs = torch.sigmoid(model(imgs))
            probs2 = None
            if second_model is not None:
                probs2 = torch.sigmoid(second_model(imgs))

            imgs_cpu = imgs.detach().cpu().numpy()
            probs_cpu = probs.detach().cpu().numpy()
            probs2_cpu = probs2.detach().cpu().numpy() if probs2 is not None else None

            for i, meta in enumerate(metas):
                img = imgs_cpu[i].transpose(1, 2, 0)
                p_cloud = probs_cpu[i, 0]
                second_probs = None
                if probs2_cpu is not None:
                    second_probs = probs2_cpu[i, 0]

                result, pred_cloud, final_cloud = run_pipeline_from_probs(
                    img,
                    p_cloud,
                    thresholds=thresholds,
                    policy=policy,
                    fusion_cfg=fusion_cfg,
                    second_probs=second_probs,
                    router=router_fn,
                    patch_verifier=patch_verifier_fn,
                    patch_size=args.patch_size,
                    return_masks=True,
                    require_llm=True,
                )

                base_mask_path = Path(args.mask_dir) / (meta.rel_path.replace("/", "_") + "_base.npy")
                _save_mask(base_mask_path, pred_cloud)
                final_mask_path = Path(args.mask_dir) / (meta.rel_path.replace("/", "_") + "_final.npy")
                _save_mask(final_mask_path, final_cloud)

                decision = result.get("decision", {})
                decision_label = decision.get("decision", "")
                adjusted_masks = []
                if decision_label == "REJECT":
                    t_cloud = float(thresholds.get("t_cloud", 0.5))
                    delta = 0.05
                    low_cloud = p_cloud > max(0.0, t_cloud - delta)
                    high_cloud = p_cloud > min(1.0, t_cloud + delta)

                    low_path = Path(args.mask_dir) / (meta.rel_path.replace("/", "_") + "_low.npy")
                    high_path = Path(args.mask_dir) / (meta.rel_path.replace("/", "_") + "_high.npy")
                    _save_mask(low_path, low_cloud)
                    _save_mask(high_path, high_cloud)
                    adjusted_masks = [str(low_path), str(high_path)]

                if result.get("patch_bounds"):
                    for patch_idx, bounds in enumerate(result["patch_bounds"]):
                        patch = extract_patch(img, bounds)
                        patch_cloud = extract_patch(pred_cloud, bounds)
                        patch_id = f"{meta.rel_path.replace('/', '_')}_{patch_idx:02d}"
                        save_patch_views(
                            args.patch_dir, patch_id, patch, patch_cloud
                        )

                out_path = Path(args.out_dir) / (meta.rel_path.replace("/", "_") + ".json")
                save_json(
                    {
                        "scene_id": meta.scene_id,
                        "rel_path": meta.rel_path,
                        "mask_base": str(base_mask_path),
                        "mask_final": str(final_mask_path),
                        "mask_adjusted": adjusted_masks,
                        **result,
                    },
                    str(out_path),
                )

                count += 1
                if args.limit and count >= args.limit:
                    logging.info("Reached limit=%s, stopping early.", args.limit)
                    return
    logging.info("Pipeline done. total=%s, out_dir=%s", count, args.out_dir)


if __name__ == "__main__":
    main()
