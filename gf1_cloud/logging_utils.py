from __future__ import annotations

import logging
import sys
from pathlib import Path

from .utils import ensure_dir


def setup_logging(log_path: str | None) -> logging.Logger:
    """Configure root logger with stdout and optional file handler."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs when scripts rerun.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        path = Path(log_path)
        ensure_dir(str(path.parent))
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
