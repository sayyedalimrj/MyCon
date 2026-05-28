"""File IO helpers for Stage 4 refinement."""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Write to a same-directory temporary file and atomically replace target."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing directory without --force: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
