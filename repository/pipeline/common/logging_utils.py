"""Logging setup for pipeline stages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(*, name: str, level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _parse_level(level: str) -> int:
    parsed = getattr(logging, level.upper(), None)
    if isinstance(parsed, int):
        return parsed
    raise ValueError(f"Invalid log level: {level}")
