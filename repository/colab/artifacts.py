"""Discover, list, and bundle pipeline artifacts for download.

The pipeline writes outputs into the configured ``project.root``:

- ``data/normalized/...mp4`` (Stage 1)
- ``data/frames/key/...`` (Stage 2)
- ``data/sparse/`` and ``data/sparse_refined/`` (Stages 3, 4)
- ``data/dense/.../fused.ply`` (Stage 5)
- ``data/da3/...``  (Stage 6)
- ``data/clean/...`` (Stage 7)
- ``data/bim/...`` (Stages 8/9)
- ``runs/<run_id>/reports/...json`` (every stage's summary)
- ``exports/viewer/...``, ``exports/cams_gs/...`` (Stages 7.6, 7.7)

This module knows the rough layout and provides a single ``collect_artifacts``
entry-point the Gradio UI calls to render the artifact list and to build a
zip bundle.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Categorised glob patterns used by collect_artifacts. Each pattern is
# evaluated with rglob() relative to project_root.
ARTIFACT_PATTERNS: dict[str, list[str]] = {
    "video": [
        "data/normalized/*.mp4",
        "data/normalized/*.json",
        "data/normalized/*.csv",
    ],
    "keyframes": [
        "data/frames/key/*.csv",
        "data/frames/key/*.jpg",
    ],
    "sparse": [
        "data/sparse/**/*.bin",
        "data/sparse/**/*.txt",
        "data/sparse_refined/**/*.bin",
        "data/sparse_refined/**/*.txt",
        "data/sfm/**/database.db",
    ],
    "dense": [
        "data/dense/**/fused.ply",
        "data/dense/**/*.json",
    ],
    "da3": [
        "data/da3/**/*.ply",
        "data/da3/**/*.json",
        "data/da3/**/*.csv",
    ],
    "cleanup": [
        "data/clean/**/*.ply",
        "data/clean/**/*.json",
    ],
    "bim": [
        "data/bim/**/*.ply",
        "data/bim/**/*.json",
        "data/bim/**/*.jsonl",
        "data/bim/metrics/**/*.csv",
    ],
    "vlm_qa": [
        "data/vlm_qa/**/*.json",
        "data/vlm_qa/**/*.png",
    ],
    "exports_viewer": [
        "exports/viewer/**/*",
    ],
    "exports_cams_gs": [
        "exports/cams_gs/**/*",
    ],
    "reports": [
        "runs/**/reports/*.json",
    ],
    "logs": [
        "runs/**/logs/*.log",
    ],
}


@dataclass
class ArtifactEntry:
    category: str
    relative_path: str
    absolute_path: str
    bytes: int

    def to_row(self) -> list[str]:
        return [
            self.category,
            self.relative_path,
            f"{self.bytes / 1024:.1f} KB" if self.bytes < 1024 * 1024 else f"{self.bytes / 1024 / 1024:.2f} MB",
        ]


def collect_artifacts(project_root: Path | str) -> list[ArtifactEntry]:
    project_root = Path(project_root)
    out: list[ArtifactEntry] = []
    seen: set[Path] = set()
    if not project_root.exists():
        return out
    for category, patterns in ARTIFACT_PATTERNS.items():
        for pat in patterns:
            for path in project_root.glob(pat):
                if not path.is_file():
                    continue
                if path in seen:
                    continue
                seen.add(path)
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                rel = path.relative_to(project_root).as_posix()
                out.append(
                    ArtifactEntry(
                        category=category,
                        relative_path=rel,
                        absolute_path=str(path),
                        bytes=size,
                    )
                )
    out.sort(key=lambda e: (e.category, e.relative_path))
    return out


def to_table_rows(entries: Iterable[ArtifactEntry]) -> list[list[str]]:
    return [e.to_row() for e in entries]


# ---------------------------------------------------------------------------
# Bundle for download
# ---------------------------------------------------------------------------


def build_artifact_bundle(
    *,
    project_root: Path,
    exports_dir: Path,
    categories: Iterable[str] | None = None,
    bundle_name: str | None = None,
) -> Path:
    """Zip a curated subset of artifacts into ``exports_dir`` and return the path."""
    project_root = Path(project_root)
    exports_dir = Path(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    selected = set(categories) if categories else set(ARTIFACT_PATTERNS.keys())
    entries = [e for e in collect_artifacts(project_root) if e.category in selected]

    name = bundle_name or f"mycon_bundle_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    out_path = exports_dir / name

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            zf.write(entry.absolute_path, arcname=entry.relative_path)
    return out_path


def copy_artifact_to_exports(*, src: Path, exports_dir: Path) -> Path:
    """Copy a single artifact into ``exports_dir`` for one-click download."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    exports_dir = Path(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    dest = exports_dir / src.name
    shutil.copy2(src, dest)
    return dest
