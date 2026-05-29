"""Stage-level resume/checkpoint logic.

When a long pipeline run dies halfway through (e.g. Colab disconnects,
GPU OOM at stage 5/7) we want a way to pick up where we left off
**without** rerunning the stages that already produced valid outputs.

This module defines the policy the runners use to decide:

* Is stage *N* already complete on disk?
* Should we skip / overwrite / rerun it?

The decision is based on a fixed list of artefacts produced per stage
(via :mod:`paths`) plus a tiny ``.done`` marker file the runner writes
after a successful stage.

Public surface
--------------
- ``StageStatus``: enum-ish dataclass returned by :func:`stage_status`
- ``stage_status(spec, stage_id, gpu=False)``: inspect disk
- ``filter_stages(spec, stage_ids, mode, gpu=False)``: apply a
  resume policy and return the stages the runner should actually run
- ``write_done_marker(spec, stage_id, payload, gpu=False)``: write
  the ``.done`` file with run metadata
- ``read_done_marker(...)``: read it back

Resume modes
------------
``run``      Always run every requested stage (legacy default).
``resume``   Skip stages whose artefacts already exist and are
             consistent. Run the rest.
``force``    Run every requested stage, deleting prior outputs first.
``redo``     Synonym for ``force``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from . import paths as P


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ResumeMode = Literal["run", "resume", "force", "redo"]
RESUME_MODES: tuple[str, ...] = ("run", "resume", "force", "redo")
DEFAULT_RESUME_MODE: ResumeMode = "run"


#: Artefacts that MUST exist for a CPU stage to be considered complete.
#: Per-frame outputs (depth/seg npz) are optional because the CPU
#: pipeline only writes them when not in --skip-render mode; we do not
#: want to penalise a legitimate "BIM only" run.
CPU_REQUIRED_ARTIFACTS = (
    "ifc",
    "mesh_obj", "mesh_glb", "mesh_ply",
    "elements_json",
    "camera_path",
    "metadata_json",
    "element_metrics_csv",
)

CPU_RENDER_ARTIFACTS = ("video", "video_clean", "keyframe")

#: Artefacts that MUST exist for a GPU stage to be considered complete.
#: Note: the per-frame PNG/EXR/seg dirs are validated separately
#: (presence + at least 1 frame).
GPU_REQUIRED_ARTIFACTS = (
    "elements_json",
    "mesh_glb",
    "blender_dir",
    "blender_rgb_dir",
    "camera_path",
    "blender_done_marker",
)


# ---------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StageStatus:
    """Result of inspecting a stage on disk."""

    stage_id: int
    complete: bool
    missing: tuple[str, ...]
    extras: dict = field(default_factory=dict)

    def reason(self) -> str:
        if self.complete:
            return "complete"
        return "missing: " + ", ".join(self.missing)


def stage_status(spec, stage_id: int, *, gpu: bool = False) -> StageStatus:
    """Inspect disk and report whether ``stage_id`` is complete."""
    inv = P.stage_artifact_inventory(spec, stage_id, gpu=gpu)
    required = GPU_REQUIRED_ARTIFACTS if gpu else CPU_REQUIRED_ARTIFACTS
    missing: list[str] = []
    extras: dict = {}

    for key in required:
        path = inv.get(key)
        if path is None:
            missing.append(key)
            continue
        if not Path(path).exists():
            missing.append(key)

    if gpu:
        rgb_dir = inv.get("blender_rgb_dir")
        if rgb_dir and Path(rgb_dir).exists():
            n_pngs = len(list(Path(rgb_dir).glob("frame_*.png")))
            extras["rgb_frame_count"] = n_pngs
            if n_pngs == 0 and "blender_rgb_dir" not in missing:
                missing.append("blender_rgb_frames")
    else:
        # For CPU, render artefacts are optional but we record them.
        for key in CPU_RENDER_ARTIFACTS:
            path = inv.get(key)
            extras[f"{key}_present"] = bool(path and Path(path).exists())

    return StageStatus(
        stage_id=stage_id,
        complete=not missing,
        missing=tuple(missing),
        extras=extras,
    )


# ---------------------------------------------------------------------
# Resume policy
# ---------------------------------------------------------------------


def _delete_stage_outputs(spec, stage_id: int, *, gpu: bool, log: logging.Logger | None = None) -> int:
    """Remove everything produced by a previous run of ``stage_id``.

    Returns the number of files / directories removed. Never raises:
    if a path is locked we just log and continue.
    """
    inv = P.stage_artifact_inventory(spec, stage_id, gpu=gpu)
    removed = 0
    for key, path in inv.items():
        p = Path(path)
        if not p.exists():
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except OSError as e:  # pragma: no cover - depends on FS
            if log:
                log.warning("could not remove %s for stage %d: %s", p, stage_id, e)
    return removed


def filter_stages(
    spec,
    stage_ids: Iterable[int],
    mode: str = DEFAULT_RESUME_MODE,
    *,
    gpu: bool = False,
    log: logging.Logger | None = None,
) -> tuple[list[int], list[StageStatus]]:
    """Apply a resume policy to a list of stage ids.

    Parameters
    ----------
    stage_ids: requested stages from the CLI.
    mode: one of :data:`RESUME_MODES`.
    gpu: True for the Blender pipeline, False for the CPU pipeline.
    log: optional logger; when given, decisions are written to it.

    Returns
    -------
    to_run, statuses
        ``to_run`` is the subset the caller should actually execute.
        ``statuses`` is the per-stage diagnostic report (one entry per
        input stage_id, even those that will be skipped).
    """
    if mode not in RESUME_MODES:
        raise ValueError(f"Unknown resume mode {mode!r}; valid: {RESUME_MODES}")
    requested = list(stage_ids)
    statuses: list[StageStatus] = []
    to_run: list[int] = []

    for sid in requested:
        s = stage_status(spec, sid, gpu=gpu)
        statuses.append(s)

        if mode == "run":
            to_run.append(sid)
            if log:
                log.info("[stage %d] mode=run -> rerunning regardless of disk state", sid)
        elif mode == "resume":
            if s.complete:
                if log:
                    log.info("[stage %d] mode=resume -> skip (already complete)", sid)
            else:
                to_run.append(sid)
                if log:
                    log.info("[stage %d] mode=resume -> rerun (%s)", sid, s.reason())
        elif mode in ("force", "redo"):
            removed = _delete_stage_outputs(spec, sid, gpu=gpu, log=log)
            to_run.append(sid)
            if log:
                log.info("[stage %d] mode=%s -> rerun (deleted %d prior artefact(s))",
                         sid, mode, removed)

    return to_run, statuses


# ---------------------------------------------------------------------
# .done marker
# ---------------------------------------------------------------------


def _done_marker_path(spec, stage_id: int, *, gpu: bool) -> Path:
    if gpu:
        return P.stage_blender_subdirs(spec, stage_id)["done_marker"]
    # CPU: drop the marker next to the metadata json.
    return spec.output.manifests / f"{P.stage_tag(stage_id)}.done"


def write_done_marker(spec, stage_id: int, payload: dict, *, gpu: bool = False) -> Path:
    """Write a JSON ``.done`` marker recording how the stage finished.

    The payload is augmented with a ``finished_at`` epoch timestamp
    plus the schema version. The caller is free to add fields like
    elapsed seconds, frame counts, preset name, git commit, etc.
    """
    marker = _done_marker_path(spec, stage_id, gpu=gpu)
    marker.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "synthetic_floor_stage_done.v1",
        "stage_id": int(stage_id),
        "gpu": bool(gpu),
        "finished_at": time.time(),
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    marker.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return marker


def read_done_marker(spec, stage_id: int, *, gpu: bool = False) -> dict | None:
    """Return the parsed ``.done`` payload, or ``None`` if absent / unreadable."""
    marker = _done_marker_path(spec, stage_id, gpu=gpu)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------
# Portable run-state manifest (cross-device / cross-Drive resume)
# ---------------------------------------------------------------------


def run_state_path(spec, *, gpu: bool = False) -> Path:
    """Single JSON manifest summarising the whole run's progress.

    Unlike the per-stage ``.done`` markers, this is one file a user can copy
    to another machine / Drive account to see (and resume) exactly how far a
    run got. It lives under ``output/manifests/`` so it is part of the Drive
    mirror.
    """
    name = "run_state_blender_gpu.json" if gpu else "run_state_cpu.json"
    return spec.output.manifests / name


def write_run_state(
    spec,
    stage_ids: Iterable[int],
    *,
    gpu: bool = False,
    extra: dict | None = None,
) -> Path:
    """Aggregate per-stage status + ``.done`` markers into one portable JSON.

    Written atomically (temp file + ``os.replace``) so a reader never sees a
    half-written manifest even if the process / Colab session dies mid-write.
    """
    stages: dict[str, dict] = {}
    for sid in stage_ids:
        s = stage_status(spec, sid, gpu=gpu)
        stages[str(int(sid))] = {
            "complete": bool(s.complete),
            "missing": list(s.missing),
            "extras": s.extras,
            "done_marker": read_done_marker(spec, sid, gpu=gpu),
        }
    body = {
        "schema_version": "synthetic_floor_run_state.v1",
        "project_name": getattr(spec, "project_name", ""),
        "run_id": getattr(spec, "run_id", ""),
        "gpu": bool(gpu),
        "updated_at": time.time(),
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_root": str(spec.output.root),
        "stages": stages,
    }
    if extra:
        body.update(extra)

    path = run_state_path(spec, gpu=gpu)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_run_state(spec, *, gpu: bool = False) -> dict | None:
    path = run_state_path(spec, gpu=gpu)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
