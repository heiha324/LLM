import json
import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may be optional in some scripts
    torch = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, sort_keys=True)


def scale_to_uint8(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Scale array to uint8 using percentile stretch."""
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)
