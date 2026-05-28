from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Stage77IOError(RuntimeError):
    """Raised for Stage 7.7 filesystem errors."""


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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Stage77IOError(f"Invalid JSON: {path}") from exc


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0
