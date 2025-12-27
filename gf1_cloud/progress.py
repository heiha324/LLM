from __future__ import annotations

from typing import Iterable, TypeVar

import sys

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

T = TypeVar("T")


def tqdm(iterable: Iterable[T], **kwargs):  # type: ignore[override]
    if _tqdm is None:
        return iterable
    kwargs.setdefault("ncols", 60)
    kwargs.setdefault("leave", True)
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("disable", False)
    kwargs.setdefault("unit", "batch")
    return _tqdm(iterable, **kwargs)
