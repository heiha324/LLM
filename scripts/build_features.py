#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gf1_cloud.data import GF1CloudDataset, collate_with_meta
from gf1_cloud.llm_runtime import llm_route
from gf1_cloud.metrics import compute_features
from gf1_cloud.logging_utils import setup_logging
from gf1_cloud.models import CloudSegNet
from gf1_cloud.progress import tqdm
from gf1_cloud.router import load_policy
from gf1_cloud.utils import ensure_dir, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", default="configs/calibration.json")
    parser.add_argument("--policy", default="configs/policy_v1.json")
    parser.add_argument("--out-dir", default="outputs/features")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-api-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--log-file", default="logs/build_features.log")
    args = parser.parse_args()

    setup_logging(args.log_file if args.log_file else None)
    logging.info("Build features start: split=%s", args.split)
    thresholds = load_json(args.calibration)
    policy = load_policy(args.policy)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认驱动/CUDA 是否安装正确。")
    device = args.device
    model = CloudSegNet()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()

    ds = GF1CloudDataset(args.data_root, args.split, return_meta=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_meta,
    )

    ensure_dir(args.out_dir)
    count = 0
    with torch.no_grad():
        for imgs, masks, metas in tqdm(
            loader, desc="features", total=len(loader), leave=True
        ):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            for i in range(imgs.size(0)):
                p_cloud = probs[i, 0].cpu().numpy()
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                features, _ = compute_features(p_cloud, thresholds, img=img_np)
                features["policy_id"] = policy.get("policy_id", "global_screening_v1")
                route = llm_route(
                    features,
                    policy,
                    api_url=args.llm_api_url,
                    model=args.llm_model,
                    timeout=args.llm_timeout,
                    max_tokens=args.llm_max_tokens,
                    retries=args.llm_retries,
                )
                payload = {
                    "scene_id": metas[i].scene_id,
                    "rel_path": metas[i].rel_path,
                    "features": features,
                    "route": route,
                }
                out_path = Path(args.out_dir) / (metas[i].rel_path.replace("/", "_") + ".json")
                save_json(payload, str(out_path))
                count += 1
                if args.limit and count >= args.limit:
                    logging.info("Reached limit=%s, stopping early.", args.limit)
                    return
    logging.info("Build features done. total=%s, out_dir=%s", count, args.out_dir)


if __name__ == "__main__":
    main()
