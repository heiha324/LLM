#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gf1_cloud.calibration import best_threshold
from gf1_cloud.data import GF1CloudDataset, build_index, split_indices_by_scene
from gf1_cloud.logging_utils import setup_logging
from gf1_cloud.models import CloudSegNet
from gf1_cloud.progress import tqdm
from gf1_cloud.utils import ensure_dir, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/calibration.json")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", default="logs/calibrate.log")
    args = parser.parse_args()

    setup_logging(args.log_file if args.log_file else None)
    metas = build_index(args.data_root, args.split)
    _, val_idx = split_indices_by_scene(metas, args.val_ratio, args.seed)
    val_ds = GF1CloudDataset(args.data_root, args.split, indices=val_idx)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认驱动/CUDA 是否安装正确。")
    device = args.device
    model = CloudSegNet()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()

    probs_cloud = []
    gts_cloud = []

    with torch.no_grad():
        for imgs, masks in tqdm(
            loader, desc="calibrate", total=len(loader), leave=True
        ):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            probs_cloud.append(probs[:, 0].cpu().numpy().squeeze(0))
            gts_cloud.append((masks[:, 0] > 0.5).cpu().numpy().squeeze(0))

    grid = np.linspace(0.1, 0.9, 17)
    t_cloud, f1_cloud = best_threshold(probs_cloud, gts_cloud, grid)

    ensure_dir(str(Path(args.out).parent))
    save_json(
        {
            "t_cloud": t_cloud,
            "prob_temperature": 1.0,
            "f1_cloud": f1_cloud,
        },
        args.out,
    )
    logging.info("Saved calibration to %s", args.out)


if __name__ == "__main__":
    main()
