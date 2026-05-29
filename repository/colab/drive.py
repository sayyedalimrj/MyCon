"""Google Drive mounting + persistent project tree management.

The Drive layout we materialise (relative to ``--drive-base``) is:

    MyCon_Colab/
        projects/<run_id>/
            data/                # videos, frames, sfm, dense, da3, clean...
            runs/<run_id>/       # reports + logs (matches pipeline contract)
                reports/run_state.json   # checkpoint/resume manifest
            configs/             # active.yaml = effective config for this run
            uploads/             # raw user uploads (videos, IFC, schedule)
            exports/             # zipped artifact bundles ready for download
            model_cache/         # persistent model/binary cache (DA3, VLM, ...)
            hf_cache/            # Hugging Face hub cache (mirrored from local)

The pipeline's ``project.root`` is set to that ``projects/<run_id>/``
directory, so every Stage writes directly to Drive — no extra sync step.
Heavy caches (Hugging Face, models) are written to a fast *local* scratch
dir and mirrored onto Drive by :class:`colab.sync.DriveSyncManager`.

This module is resilient to the realities of the Colab Drive FUSE mount:
mount is retried, health is verified by an actual write probe, and a stale
mount is force-remounted on demand.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from colab.log_capture import LogBuffer
from colab.sync import copy_file, verify_writable

DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_BASE = "MyDrive/MyCon_Colab"
DEFAULT_LOCAL_FALLBACK = Path("/content/MyCon_Colab")
# Fast local scratch root for caches that are too slow to write straight to
# the Drive FUSE mount. Mirrored onto Drive by the sync manager.
DEFAULT_LOCAL_SCRATCH = Path("/content/mycon_scratch")


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
    run_state_path: Path
    model_cache_dir: Path  # persistent (on Drive) model cache
    hf_cache_dir: Path  # persistent (on Drive) Hugging Face cache
    local_scratch_dir: Path  # fast local scratch (mirrored to Drive)
    local_hf_cache_dir: Path  # fast local HF cache (mirrored to Drive)
    on_drive: bool

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
            "run_state_path": str(self.run_state_path),
            "model_cache_dir": str(self.model_cache_dir),
            "hf_cache_dir": str(self.hf_cache_dir),
            "local_scratch_dir": str(self.local_scratch_dir),
            "local_hf_cache_dir": str(self.local_hf_cache_dir),
            "on_drive": "yes" if self.on_drive else "no (local fallback)",
        }


# ---------------------------------------------------------------------------
# Mount / health
# ---------------------------------------------------------------------------


def is_mounted(mount_point: Path = DEFAULT_DRIVE_MOUNT) -> bool:
    return (Path(mount_point) / "MyDrive").exists()


def drive_health(mount_point: Path = DEFAULT_DRIVE_MOUNT) -> bool:
    """Mounted *and* writable (probe write). Detects stale FUSE mounts."""
    mydrive = Path(mount_point) / "MyDrive"
    return mydrive.exists() and verify_writable(mydrive)


def mount_drive(
    *,
    mount_point: Path = DEFAULT_DRIVE_MOUNT,
    force_remount: bool = False,
    attempts: int = 3,
    log: Optional[LogBuffer] = None,
) -> bool:
    """Mount Google Drive in Colab with retries. Returns True if mounted.

    When ``force_remount`` is True (or the existing mount fails its write
    probe) we re-issue ``drive.mount(force_remount=True)`` to recover from a
    stale mount after a runtime reconnect.
    """
    mount_point = Path(mount_point)

    if not force_remount and drive_health(mount_point):
        if log is not None:
            log.append(f"[drive] healthy mount at {mount_point}")
        return True

    try:
        from google.colab import drive as colab_drive  # type: ignore
    except Exception as exc:  # pragma: no cover - non-Colab fallback
        if log is not None:
            log.append(f"[drive] google.colab not available ({exc}); skipping mount")
        return False

    need_remount = force_remount or (is_mounted(mount_point) and not drive_health(mount_point))
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            colab_drive.mount(str(mount_point), force_remount=need_remount)
        except Exception as exc:  # pragma: no cover - Colab-only path
            last_exc = exc
            if log is not None:
                log.append(f"[drive] mount attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(min(8.0, 1.5 * attempt))
            need_remount = True
            continue
        if drive_health(mount_point):
            if log is not None:
                log.append(f"[drive] mounted + writable at {mount_point}")
            return True
        need_remount = True
    if log is not None and last_exc is not None:
        log.append(f"[drive] giving up after {attempts} attempts: {last_exc}")
    return is_mounted(mount_point)


def remount_drive(
    *, mount_point: Path = DEFAULT_DRIVE_MOUNT, log: Optional[LogBuffer] = None
) -> bool:
    """Force a remount (used as the DriveSyncManager recovery callback)."""
    return mount_drive(mount_point=mount_point, force_remount=True, log=log)


def free_space_gb(path: Path) -> float:
    """Best-effort free space in GiB for the filesystem holding ``path``."""
    try:
        usage = shutil.disk_usage(str(Path(path)))
        return usage.free / (1024 ** 3)
    except OSError:
        return -1.0


# ---------------------------------------------------------------------------
# Project tree
# ---------------------------------------------------------------------------


def sanitize_run_id(run_id: str) -> str:
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    safe = "".join(c for c in run_id.strip() if c.isalnum() or c in ("-", "_", "."))
    if not safe:
        raise ValueError(f"run_id contains no usable characters: {run_id!r}")
    return safe


def setup_project_tree(
    *,
    run_id: str,
    drive_mount: Path = DEFAULT_DRIVE_MOUNT,
    drive_base: str = DEFAULT_DRIVE_BASE,
    fallback_root: Path = DEFAULT_LOCAL_FALLBACK,
    local_scratch: Path = DEFAULT_LOCAL_SCRATCH,
    auto_mount: bool = True,
    log: Optional[LogBuffer] = None,
) -> ProjectPaths:
    """Create the on-Drive (or local fallback) project tree for ``run_id``."""
    safe_run_id = sanitize_run_id(run_id)
    drive_mount = Path(drive_mount)

    if auto_mount and not drive_health(drive_mount):
        mount_drive(mount_point=drive_mount, log=log)

    on_drive = drive_health(drive_mount)
    if on_drive:
        base = drive_mount / drive_base
    else:
        if log is not None:
            log.append(
                f"[drive] {drive_mount} not mounted/writable; using local fallback {fallback_root}"
            )
        base = Path(fallback_root)

    project_root = base / "projects" / safe_run_id
    uploads = project_root / "uploads"
    configs_dir = project_root / "configs"
    exports = project_root / "exports"
    runs_dir = project_root / "runs" / safe_run_id
    reports = runs_dir / "reports"
    logs = runs_dir / "logs"
    model_cache = project_root / "model_cache"
    hf_cache = project_root / "hf_cache"

    for d in (project_root, uploads, configs_dir, exports, runs_dir, reports, logs,
              model_cache, hf_cache):
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
    ):
        (project_root / sub).mkdir(parents=True, exist_ok=True)

    # Fast local scratch (per run). Mirrored onto Drive by the sync manager.
    local_scratch_dir = Path(local_scratch) / safe_run_id
    local_hf_cache = local_scratch_dir / "hf_cache"
    for d in (local_scratch_dir, local_hf_cache):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

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
        run_state_path=reports / "run_state.json",
        model_cache_dir=model_cache,
        hf_cache_dir=hf_cache,
        local_scratch_dir=local_scratch_dir,
        local_hf_cache_dir=local_hf_cache,
        on_drive=on_drive,
    )
    if log is not None:
        log.banner("Project tree ready")
        for k, v in paths.as_dict().items():
            log.append(f"  {k:20s} = {v}")
        fg = free_space_gb(base)
        if fg >= 0:
            log.append(f"  {'free_space_gb':20s} = {fg:.1f}")
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
    # Atomic + retrying copy so a Drive hiccup mid-upload does not corrupt.
    copy_file(src, dest, log=log)
    if log is not None:
        log.append(f"[drive] copied upload {src} -> {dest} ({os.path.getsize(dest)} bytes)")
    return dest
