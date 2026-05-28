from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Stage76IOError(RuntimeError):
    """Raised for Stage 7.6 filesystem errors."""


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


def clean_dir_guarded(path: Path, *, force: bool, required_token: str = "viewer") -> None:
    if not force or not path.exists():
        return
    resolved = path.resolve()
    if str(resolved) in {"/", "/workspace", "/workspace/data", "/workspace/exports"}:
        raise Stage76IOError(f"Refusing to remove unsafe path: {resolved}")
    if required_token not in set(resolved.parts):
        raise Stage76IOError(f"Refusing to remove {resolved}; missing guard token {required_token!r}")
    shutil.rmtree(resolved)


def copy_file(src: Path, dst: Path) -> int:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return int(dst.stat().st_size)
