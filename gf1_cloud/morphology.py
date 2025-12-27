from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


def ring_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    return np.logical_xor(dilated, eroded)


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    labeled, n = ndimage.label(mask.astype(np.uint8))
    if n == 0:
        return 0, np.array([], dtype=np.int64)
    counts = np.bincount(labeled.ravel())
    areas = counts[1:]
    return n, areas
