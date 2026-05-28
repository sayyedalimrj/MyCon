"""File IO helpers for Stage 5 dense stereo."""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class DenseWorkspaceSafetyError(RuntimeError):
    """Raised when a dense workspace path is unsafe to delete."""


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


def _resolve_for_safety(path: Path) -> Path:
    """Resolve a path for containment checks without requiring it to exist."""
    return path.expanduser().absolute()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_safe_dense_workspace(workspace: Path, project_root: Path) -> None:
    """Ensure a --force delete can only target the Stage 5 dense workspace area.

    This prevents a bad config such as ``dense.workspace_dir: data/sfm`` from
    deleting Stage 3 databases/models. A workspace is considered safe when it is
    below ``<project_root>/data/dense`` or when an existing directory contains a
    Stage 5 lock file created by this module.
    """
    workspace_abs = _resolve_for_safety(workspace)
    root_abs = _resolve_for_safety(project_root)
    dense_root = root_abs / "data" / "dense"

    forbidden = {
        root_abs,
        root_abs / "data",
        root_abs / "data" / "sfm",
        root_abs / "data" / "sparse",
        root_abs / "data" / "sparse_refined",
        root_abs / "data" / "frames",
        root_abs / "runs",
        root_abs / "exports",
    }
    if workspace_abs in forbidden:
        raise DenseWorkspaceSafetyError(f"Unsafe dense workspace path refuses deletion: {workspace_abs}")

    if workspace.exists() and (workspace / ".dense_workspace_lock").exists():
        return

    if not _is_relative_to(workspace_abs, dense_root):
        raise DenseWorkspaceSafetyError(
            "Unsafe dense workspace path. Stage 5 may only overwrite directories under "
            f"{dense_root} or an existing directory containing .dense_workspace_lock; got {workspace_abs}"
        )


def clean_dense_workspace(path: Path, project_root: Path, force: bool) -> None:
    """Safely create or overwrite a dense workspace and write a lock marker."""
    assert_safe_dense_workspace(path, project_root)
    if path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing dense workspace without --force: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".dense_workspace_lock").write_text(
        "Stage 5 dense workspace. Safe to overwrite with --force.\n",
        encoding="utf-8",
    )


def clean_dir(path: Path, force: bool) -> None:
    """Generic cleaner retained for tests/backward compatibility.

    New Stage 5 workspace deletion should use ``clean_dense_workspace``.
    """
    if path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing directory without --force: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total
