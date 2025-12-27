#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gf1_cloud.data import GF1CloudDataset, build_index, split_indices_by_scene
from gf1_cloud.logging_utils import setup_logging
from gf1_cloud.losses import cloud_loss
from gf1_cloud.models import CloudSegNet
from gf1_cloud.progress import tqdm
from gf1_cloud.utils import ensure_dir, save_json, set_seed


def _metrics(pred_cloud: np.ndarray, gt_cloud: np.ndarray) -> dict:
    """Compute OA, precision, recall, FAR, F1, IoU for cloud vs non-cloud."""
    tp = np.logical_and(pred_cloud, gt_cloud).sum()
    fp = np.logical_and(pred_cloud, ~gt_cloud).sum()
    fn = np.logical_and(~pred_cloud, gt_cloud).sum()
    tn = np.logical_and(~pred_cloud, ~gt_cloud).sum()

    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    far = fp / (fp + tn + eps)  # false alarm rate
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


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    metric_list = []
    with torch.no_grad():
        for imgs, masks in tqdm(
            loader, desc="val", total=len(loader), leave=True
        ):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            pred_cloud = probs[:, 0] > 0.5
            gt_cloud = masks[:, 0] > 0.5
            for i in range(imgs.size(0)):
                metric_list.append(
                    _metrics(
                        pred_cloud[i].cpu().numpy(),
                        gt_cloud[i].cpu().numpy(),
                    )
                )
    if not metric_list:
        return {k: 0.0 for k in ["oa", "precision", "recall", "far", "f1", "iou"]}
    mean_metrics = {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0].keys()}
    return mean_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default="/home/data/KXShen/model/gf1_cloud/seg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", default="logs/train_seg.log")
    args = parser.parse_args()

    setup_logging(args.log_file if args.log_file else None)
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    metas = build_index(args.data_root, args.split)
    train_idx, val_idx = split_indices_by_scene(metas, args.val_ratio, args.seed)

    train_ds = GF1CloudDataset(args.data_root, args.split, indices=train_idx)
    val_ds = GF1CloudDataset(args.data_root, args.split, indices=val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认驱动/CUDA 是否安装正确。")
    device = args.device
    model = CloudSegNet(base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(
            train_loader, desc=f"train {epoch}", total=len(train_loader), leave=True
        ):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = cloud_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["iou"] > best:
            best = val_metrics["iou"]
            ckpt_path = Path(args.out_dir) / "cloudsegnet_best.pt"
            torch.save({"model": model.state_dict()}, ckpt_path)
        log = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, len(train_loader)),
            **val_metrics,
        }
        save_json(log, str(Path(args.out_dir) / f"metrics_epoch_{epoch:03d}.json"))
        logging.info(log)


if __name__ == "__main__":
    main()
