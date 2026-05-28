from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_output_path(final_path: Path) -> Iterator[Path]:
    """Yield a temporary path in the same directory and atomically replace final_path.

    Keeping the temp file in the same directory avoids cross-device rename
    problems on Docker/WSL mounts.
    """
    ensure_parent(final_path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=str(final_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with atomic_output_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clean_dir_guarded(path: Path, *, force: bool, required_token: str, logger: logging.Logger) -> None:
    """Safely clean a generated directory.

    Stage 6 only cleans paths that look like a DA3/generated depth workspace. This
    prevents accidental deletion of upstream SfM/dense workspaces if YAML paths
    are misconfigured.
    """
    if not path.exists():
        return
    norm = path.as_posix().lower()
    if required_token not in norm:
        raise ValueError(f"Refusing to clean unsafe path without '{required_token}' token: {path}")
    if not force:
        raise FileExistsError(f"Output directory already exists: {path}. Use --force to overwrite generated files.")
    logger.warning("Removing existing generated directory: %s", path)
    shutil.rmtree(path)
