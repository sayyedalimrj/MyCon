"""Metadata + manifest exporter.

We emit several JSON / CSV files to make this dataset directly
consumable by the rest of the MyCon pipeline:

Per-stage:
  * ``stage_<id>_metadata.json`` - resolution, fps, frame count,
    intrinsics, file paths, list of elements with status.
  * ``stage_<id>_camera_path.json`` - all 4x4 cam-to-world matrices.
  * ``stage_<id>_element_metrics.csv`` - the canonical Stage 9 schema
    used by ``pipeline.stage_11_schedule_variance``.

Per-dataset:
  * ``manifest.json`` - global index of every file with SHA-256 hash.
  * ``schedule.csv`` - synthetic 7-step construction schedule mapped
    to the seven stages.
  * ``bim_schedule_mapping.csv`` - mapping of every IFC GlobalId to
    the activity that owns it.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .camera_path import Pose, intrinsics_for
from .layout import Element
from .scene_spec import SceneSpec
from .stage_controller import StagedElement


SCHEMA_VERSION_STAGE = "synthetic_floor_stage_metadata.v1"
SCHEMA_VERSION_MANIFEST = "synthetic_floor_manifest.v1"
SCHEMA_VERSION_CAMERA = "synthetic_floor_camera_path.v1"


# ---------------------------------------------------------------------
# Per-stage outputs
# ---------------------------------------------------------------------


def write_stage_metadata(
    spec: SceneSpec,
    stage_id: int,
    staged: Sequence[StagedElement],
    files: dict,
    out_path: Path,
) -> Path:
    cam = spec.camera
    intr = intrinsics_for(cam)
    n_frames = int(round(cam.duration_per_stage_sec * cam.fps))
    payload = {
        "schema_version": SCHEMA_VERSION_STAGE,
        "project": spec.project_name,
        "run_id": spec.run_id,
        "stage": {
            "id": stage_id,
            "name": spec.stages[stage_id - 1].name,
            "description": spec.stages[stage_id - 1].description,
        },
        "video": {
            "width_px": cam.width_px,
            "height_px": cam.height_px,
            "fps": cam.fps,
            "frame_count": n_frames,
            "duration_sec": float(n_frames / cam.fps),
            "aspect_ratio": cam.aspect,
        },
        "intrinsics": intr,
        "files": {k: str(v) for k, v in files.items() if v is not None},
        "elements_present": [
            {
                "id": s.element.id,
                "ifc_global_id": s.element.ifc_global_id,
                "name": s.element.name,
                "category": s.element.category,
                "completion": s.completion,
                "finishing": s.finishing,
                "status": s.status,
            }
            for s in staged if s.completion >= 0.5
        ],
        "elements_missing": [
            {
                "id": s.element.id,
                "ifc_global_id": s.element.ifc_global_id,
                "category": s.element.category,
                "status": s.status,
            }
            for s in staged if s.completion < 0.5
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def write_camera_path(
    spec: SceneSpec,
    stage_id: int,
    poses: Sequence[Pose],
    out_path: Path,
) -> Path:
    cam = spec.camera
    payload = {
        "schema_version": SCHEMA_VERSION_CAMERA,
        "project": spec.project_name,
        "stage_id": stage_id,
        "fps": cam.fps,
        "intrinsics": intrinsics_for(cam),
        "frames": [
            {
                "frame_index": i,
                "timestamp_sec": p.timestamp,
                "eye_world_m": p.eye.tolist(),
                "target_world_m": p.target.tolist(),
                "up_world": p.up.tolist(),
                "cam_to_world_4x4": p.cam_to_world.tolist(),
            }
            for i, p in enumerate(poses)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_element_metrics_csv(
    staged: Sequence[StagedElement],
    out_path: Path,
) -> Path:
    """Emit the Stage-9 canonical ``element_metrics.csv`` for one stage.

    Columns follow ``pipeline/stage_11_schedule_variance/activity_rollup.py``:

        global_id, name, status

    where ``status`` is one of ``likely_completed``, ``partially_observed``
    or ``not_evidenced``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("global_id", "name", "status"))
        for s in staged:
            w.writerow((s.element.ifc_global_id, s.element.name, s.status))
    return out_path


# ---------------------------------------------------------------------
# Dataset-level outputs
# ---------------------------------------------------------------------


def write_dataset_schedule(spec: SceneSpec, all_elements: Sequence[Element], out_path: Path) -> Path:
    """Emit a 7-row canonical schedule CSV (one activity per stage).

    The format matches ``pipeline/common/schedule_io.py`` (``schedule.v1``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.date(2026, 3, 1)
    activity_duration = datetime.timedelta(days=14)
    cols = (
        "activity_id", "activity_name", "wbs_code",
        "planned_start_iso", "planned_finish_iso",
        "percent_complete", "predecessors", "trade", "location",
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        prev = ""
        for i, stage in enumerate(spec.stages):
            aid = f"A{stage.id:04d}"
            sd = start + i * activity_duration
            fd = sd + activity_duration
            w.writerow((
                aid,
                stage.name,
                f"1.{stage.id}",
                sd.isoformat(),
                fd.isoformat(),
                "",
                prev,
                "structural" if stage.id <= 4 else "finishing",
                "Floor 01",
            ))
            prev = aid
    return out_path


def write_bim_schedule_mapping(
    spec: SceneSpec,
    all_elements: Sequence[Element],
    out_path: Path,
) -> Path:
    """Map every element to the LAST stage in which it gets fully built.

    That activity (``A000<stage_id>``) is its "owner" for schedule
    variance. Weights default to 1.0.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Determine the first stage that fully builds each category.
    cat_first_full: dict[str, int] = {}
    for cat in {e.category for e in all_elements}:
        for stage in spec.stages:
            info = stage.elements.get(cat, {})
            if float(info.get("completion", 0.0)) >= 0.999:
                cat_first_full[cat] = stage.id
                break
        else:
            cat_first_full[cat] = spec.stages[-1].id
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("activity_id", "ifc_global_id", "weight"))
        for e in all_elements:
            sid = cat_first_full.get(e.category, spec.stages[-1].id)
            aid = f"A{sid:04d}"
            w.writerow((aid, e.ifc_global_id, "1.0"))
    return out_path


def write_manifest(
    spec: SceneSpec,
    files_by_stage: dict[int, dict[str, Path]],
    extra_files: dict[str, Path],
    out_path: Path,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION_MANIFEST,
        "project": spec.project_name,
        "run_id": spec.run_id,
        "description": spec.description,
        "random_seed": spec.random_seed,
        "config_path": str(spec.config_path),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "video": {
            "width_px": spec.camera.width_px,
            "height_px": spec.camera.height_px,
            "fps": spec.camera.fps,
            "frame_count_per_stage": int(spec.camera.duration_per_stage_sec * spec.camera.fps),
            "duration_sec_per_stage": spec.camera.duration_per_stage_sec,
        },
        "intrinsics": intrinsics_for(spec.camera),
        "stage_files": {
            str(sid): {k: _entry(v) for k, v in files.items()}
            for sid, files in files_by_stage.items()
        },
        "dataset_files": {k: _entry(v) for k, v in extra_files.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _entry(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "sha256": _sha256(p),
    }


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()
