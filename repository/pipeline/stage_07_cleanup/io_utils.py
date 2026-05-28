from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


class Stage7IOError(RuntimeError):
    """Raised for Stage 7 filesystem contract errors."""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        if tmp.exists():
            if tmp.is_dir():
                shutil.rmtree(tmp)
            else:
                tmp.unlink()
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            if tmp.is_dir():
                shutil.rmtree(tmp)
            else:
                tmp.unlink()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if hasattr(value, "as_posix"):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def clean_dir_guarded(path: Path, *, force: bool, required_token: str, logger: logging.Logger | None = None) -> None:
    if not force or not path.exists():
        return
    resolved = path.resolve()
    parts = set(resolved.parts)
    if required_token not in parts:
        raise Stage7IOError(f"Refusing to remove {resolved}; path does not contain guard token '{required_token}'.")
    if str(resolved) in {"/", "/workspace", "/workspace/data"}:
        raise Stage7IOError(f"Refusing to remove unsafe path: {resolved}")
    if logger:
        logger.info("Removing Stage 7 workspace: %s", resolved)
    shutil.rmtree(resolved)


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def path_for_report(path: Path) -> str:
    return path.as_posix()
