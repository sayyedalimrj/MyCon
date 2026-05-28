"""COLMAP database path management for Stage 3 attempts."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Stage3AttemptWorkspace:
    name: str
    feature_type: str
    matcher_type: str
    attempt_dir: Path
    database_path: Path
    sparse_attempt_dir: Path


def make_attempt_workspace(
    sfm_dir: Path,
    attempt_index: int,
    feature_type: str,
    matcher_type: str,
    force: bool,
) -> Stage3AttemptWorkspace:
    safe_feature = feature_type.lower().replace("/", "_")
    safe_matcher = matcher_type.lower().replace("/", "_")
    name = f"attempt_{attempt_index:02d}_{safe_feature}_{safe_matcher}"
    attempt_dir = sfm_dir / name
    if attempt_dir.exists() and force:
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    return Stage3AttemptWorkspace(
        name=name,
        feature_type=feature_type,
        matcher_type=matcher_type,
        attempt_dir=attempt_dir,
        database_path=attempt_dir / "database.db",
        sparse_attempt_dir=attempt_dir / "sparse",
    )
