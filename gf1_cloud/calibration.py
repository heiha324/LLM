from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def best_threshold(
    probs: Iterable[np.ndarray],
    gts: Iterable[np.ndarray],
    thresholds: Iterable[float],
) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        f1s = []
        for p, g in zip(probs, gts):
            f1s.append(_f1_score(p > t, g))
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_t = float(t)
    return best_t, best_f1
