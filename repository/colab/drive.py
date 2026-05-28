"""Google Drive mounting + persistent project tree management.

The Drive layout we materialise (relative to ``--drive-root``) is:

    MyCon_Colab/
        projects/<run_id>/
            data/                # videos, frames, sfm, dense, da3, clean...
            runs/<run_id>/       # reports + logs (matches pipeline contract)
            configs/             # active.yaml = effective config for this run
            uploads/             # raw user uploads (videos, IFC, schedule)
            exports/             # zipped artifact bundles ready for download

The pipeline's ``project.root`` is set to that ``projects/<run_id>/``
directory, so every Stage writes directly to Drive — no extra sync step.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from colab.log_capture import LogBuffer


DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_BASE = "MyDrive/MyCon_Colab"


@dataclass
class ProjectPaths:
    """Concrete on-disk paths for a single Colab run."""

    drive_mount: Path
    drive_base: Path  # MyCon_Colab root (under MyDrive)
    project_root: Path  # = drive_base / projects / run_id  (== config project.root)
    run_id: str
    uploads_dir: Path
    configs_dir: Path
    exports_dir: Path
    runs_dir: Path  # = project_root / runs / run_id
    reports_dir: Path
    logs_dir: Path
    active_config_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "drive_mount": str(self.drive_mount),
            "drive_base": str(self.drive_base),
            "project_root": str(self.project_root),
            "run_id": self.run_id,
            "uploads_dir": str(self.uploads_dir),
            "configs_dir": str(self.configs_dir),
            "exports_dir": str(self.exports_dir),
            "runs_dir": str(self.runs_dir),
            "reports_dir": str(self.reports_dir),
            "logs_dir": str(self.logs_dir),
            "active_config_path": str(self.active_config_path),
        }


def mount_drive(
    *, mount_point: Path = DEFAULT_DRIVE_MOUNT, log: Optional[LogBuffer] = None
) -> bool:
    """Mount Google Drive in Colab. Returns True if mounted."""
    mount_point = Path(mount_point)
    if (mount_point / "MyDrive").exists():
        if log is not None:
            log.append(f"[drive] already mounted at {mount_point}")
        return True
    try:
        from google.colab import drive as colab_drive  # type: ignore
    except Exception as exc:  # pragma: no cover - non-Colab fallback
        if log is not None:
            log.append(f"[drive] google.colab not available ({exc}); skipping mount")
        return False
    try:
        colab_drive.mount(str(mount_point), force_remount=False)
    except Exception as exc:  # pragma: no cover
        if log is not None:
            log.append(f"[drive] mount failed: {exc}")
        return False
    if log is not None:
        log.append(f"[drive] mounted at {mount_point}")
    return (mount_point / "MyDrive").exists()


def setup_project_tree(
    *,
    run_id: str,
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    drive_base: str = DEFAULT_DRIVE_BASE,
    fallback_root: Path = Path("/content/MyCon_Colab"),
    log: Optional[LogBuffer] = None,
) -> ProjectPaths:
    """Create the on-Drive (or local fallback) project tree for ``run_id``."""
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    safe_run_id = "".join(c for c in run_id.strip() if c.isalnum() or c in ("-", "_", "."))
    if not safe_run_id:
        raise ValueError(f"run_id contains no usable characters: {run_id!r}")

    drive_mount = Path(drive_mount)
    if (drive_mount / "MyDrive").exists():
        base = drive_mount / drive_base
    else:
        if log is not None:
            log.append(
                f"[drive] {drive_mount} not mounted; using local fallback {fallback_root}"
            )
        base = fallback_root

    project_root = base / "projects" / safe_run_id
    uploads = project_root / "uploads"
    configs_dir = project_root / "configs"
    exports = project_root / "exports"
    runs_dir = project_root / "runs" / safe_run_id
    reports = runs_dir / "reports"
    logs = runs_dir / "logs"

    for d in (project_root, uploads, configs_dir, exports, runs_dir, reports, logs):
        d.mkdir(parents=True, exist_ok=True)

    # Standard data directories the pipeline writes into.
    for sub in (
        "data/raw",
        "data/normalized",
        "data/frames/key",
        "data/sfm",
        "data/sparse",
        "data/sparse_refined",
        "data/dense",
        "data/da3",
        "data/clean",
        "data/bim/design",
        "data/bim/aligned",
        "data/bim/metrics",
        "data/cams_gs",
        "data/semantics",
        "data/masks",
        "data/vlm_qa",
        "exports/viewer",
        "exports/cams_gs",
        "model_cache",
    ):
        (project_root / sub).mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(
        drive_mount=drive_mount,
        drive_base=base,
        project_root=project_root,
        run_id=safe_run_id,
        uploads_dir=uploads,
        configs_dir=configs_dir,
        exports_dir=exports,
        runs_dir=runs_dir,
        reports_dir=reports,
        logs_dir=logs,
        active_config_path=configs_dir / "active.yaml",
    )
    if log is not None:
        log.banner("Project tree ready")
        for k, v in paths.as_dict().items():
            log.append(f"  {k:20s} = {v}")
    return paths


def stage_upload_to_drive(
    *,
    src: Path | str,
    dest_dir: Path,
    new_name: Optional[str] = None,
    log: Optional[LogBuffer] = None,
) -> Path:
    """Copy a Colab/Gradio upload into the Drive ``uploads/`` directory."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"upload source not found: {src}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = new_name or src.name
    dest = dest_dir / name
    if dest.resolve() == src.resolve():
        if log is not None:
            log.append(f"[drive] upload already at destination: {dest}")
        return dest
    shutil.copy2(src, dest)
    if log is not None:
        log.append(f"[drive] copied upload {src} -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest
