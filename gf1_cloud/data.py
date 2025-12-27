from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleMeta:
    scene_id: str
    rel_path: str
    img_path: str
    mask_path: str


def _collect_image_paths(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npy"))


def _scene_id(img_root: Path, img_path: Path) -> str:
    return str(img_path.parent.relative_to(img_root))


def build_index(data_root: str, split: str) -> List[SampleMeta]:
    img_root = Path(data_root) / split / "img"
    mask_root = Path(data_root) / split / "mask"
    if not img_root.exists():
        raise FileNotFoundError(f"Missing img root: {img_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Missing mask root: {mask_root}")

    metas: List[SampleMeta] = []
    for img_path in _collect_image_paths(img_root):
        rel = img_path.relative_to(img_root)
        mask_path = mask_root / rel
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_path}: {mask_path}")
        metas.append(
            SampleMeta(
                scene_id=_scene_id(img_root, img_path),
                rel_path=str(rel),
                img_path=str(img_path),
                mask_path=str(mask_path),
            )
        )
    return metas


def collate_with_meta(batch):
    imgs, masks, metas = zip(*batch)
    return torch.stack(list(imgs), dim=0), torch.stack(list(masks), dim=0), list(metas)


def split_indices_by_scene(
    metas: Sequence[SampleMeta], val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    scene_to_indices: Dict[str, List[int]] = {}
    for i, meta in enumerate(metas):
        scene_to_indices.setdefault(meta.scene_id, []).append(i)
    scenes = list(scene_to_indices.keys())
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * val_ratio))
    val_scenes = set(scenes[:n_val])
    train_idx: List[int] = []
    val_idx: List[int] = []
    for scene, idxs in scene_to_indices.items():
        if scene in val_scenes:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return train_idx, val_idx


class GF1CloudDataset(Dataset):
    """GF-1 WFV 4-band patch dataset with cloud vs non-cloud mask."""

    def __init__(
        self,
        data_root: str,
        split: str,
        indices: Optional[Sequence[int]] = None,
        normalize: bool = True,
        max_value: float = 1023.0,
        return_meta: bool = False,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.metas = build_index(data_root, split)
        if indices is not None:
            self.metas = [self.metas[i] for i in indices]
        self.normalize = normalize
        self.max_value = max_value
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.metas)

    def __getitem__(self, idx: int):
        meta = self.metas[idx]
        img = np.load(meta.img_path).astype(np.float32)
        mask = np.load(meta.mask_path).astype(np.uint8)
        if self.normalize:
            img = img / self.max_value
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        cloud = (mask == 255).astype(np.float32)
        mask_t = torch.from_numpy(cloud[None, ...])
        if self.return_meta:
            return img_t, mask_t, meta
        return img_t, mask_t
