"""Root-relative pathlib helpers and atomic output utilities."""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pipeline.common.config import ConfigError, PipelineConfig


def resolve_project_path(cfg: PipelineConfig, path_value: str | Path) -> Path:
    raw = str(path_value)
    if _looks_like_windows_path(raw):
        raise ConfigError(f"Windows/UNC paths are not allowed in YAML: {raw}")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else cfg.project_root / path


def input_path(cfg: PipelineConfig, key: str) -> Path:
    return resolve_project_path(cfg, cfg.require(f"inputs.{key}"))


def output_path(cfg: PipelineConfig, key: str) -> Path:
    return resolve_project_path(cfg, cfg.require(f"paths.{key}"))


def run_reports_dir(cfg: PipelineConfig) -> Path:
    return cfg.project_root / "runs" / cfg.run_id / "reports"


def run_logs_dir(cfg: PipelineConfig) -> Path:
    return cfg.project_root / "runs" / cfg.run_id / "logs"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with atomic_output_path(path) as tmp_path:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


@contextmanager
def atomic_output_path(final_path: Path) -> Iterator[Path]:
    """Yield a temp path and publish it atomically when the context succeeds.

    The temporary file is created in the final file's parent directory so the
    normal path is same-filesystem and safe for ``os.replace``. A guarded
    ``shutil.move`` fallback is included for unusual mount behavior or future
    refactors that might place temp files on a different filesystem.
    """
    ensure_parent_dir(final_path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=f".tmp{final_path.suffix}",
        dir=str(final_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        _publish_tmp_file(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _publish_tmp_file(tmp_path: Path, final_path: Path) -> None:
    try:
        os.replace(tmp_path, final_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(tmp_path), str(final_path))


def _looks_like_windows_path(value: str) -> bool:
    return value.startswith("\\\\") or (len(value) >= 3 and value[1:3] == ":\\")
