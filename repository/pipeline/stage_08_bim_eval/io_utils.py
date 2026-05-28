"""File IO helpers for Stage 8 BIM extraction and registration."""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Write to a same-directory temporary path and atomically replace target."""
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
        tmp.write_text(
            json.dumps(_json_sanitize(payload), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    with atomic_output_path(path) as tmp:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_sanitize(row), ensure_ascii=False, sort_keys=True) + "\n")


def clean_dir(path: Path, force: bool, *, expected_leaf: str | None = None) -> None:
    if expected_leaf and expected_leaf not in path.parts:
        raise RuntimeError(f"Refusing to clean unsafe Stage 8 path outside expected leaf {expected_leaf!r}: {path}")
    if path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing directory without --force: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
