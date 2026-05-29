"""Centralised path helpers for the synthetic_floor_7stage example.

The CPU and GPU pipelines historically built artefact paths inline,
which led to small mismatches (e.g. ``output/renders/stage_07_keyframe.png``
versus ``output/blender_renders/stage_07/...``). This module is the
single source of truth so every module agrees on where things live.

All helpers take a ``SceneSpec`` (or its ``output`` ``OutputPaths``
sub-object) plus a stage id, and return a ``pathlib.Path``. They
**never** create directories on their own; call ``spec.output.ensure()``
once at startup, then use these helpers freely.

Naming convention
-----------------
Every per-stage artefact uses the prefix ``stage_NN_`` where ``NN`` is
zero-padded to two digits. Dataset-level artefacts (manifests, schedule
CSV, etc.) live under ``output/manifests/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .scene_spec import SceneSpec, OutputPaths


# ---------------------------------------------------------------------
# Naming helpers (no I/O, no scene_spec needed)
# ---------------------------------------------------------------------


def stage_tag(stage_id: int) -> str:
    """Return the canonical zero-padded stage tag, e.g. ``stage_07``."""
    return f"stage_{int(stage_id):02d}"


# ---------------------------------------------------------------------
# OutputPaths-based helpers (work whether you pass spec or spec.output)
# ---------------------------------------------------------------------


def _out(spec_or_output: "SceneSpec | OutputPaths") -> "OutputPaths":
    """Accept either a ``SceneSpec`` or its ``OutputPaths``."""
    return getattr(spec_or_output, "output", spec_or_output)


# --- BIM (IFC) --------------------------------------------------------------


def stage_ifc_path(spec, stage_id: int) -> Path:
    """``output/bim/stage_07.ifc``"""
    return _out(spec).bim / f"{stage_tag(stage_id)}.ifc"


# --- Mesh -------------------------------------------------------------------


def stage_mesh_dir(spec, stage_id: int) -> Path:
    """Directory where stage meshes live (currently shared with all stages)."""
    return _out(spec).mesh


def stage_mesh_paths(spec, stage_id: int) -> dict[str, Path]:
    """Return all mesh artefacts for one stage keyed by extension."""
    base = _out(spec).mesh / stage_tag(stage_id)
    return {
        "obj": base.with_suffix(".obj"),
        "glb": base.with_suffix(".glb"),
        "ply": base.with_suffix(".ply"),
        "elements_json": base.parent / f"{base.name}_elements.json",
    }


def stage_glb_path(spec, stage_id: int) -> Path:
    return stage_mesh_paths(spec, stage_id)["glb"]


def stage_obj_path(spec, stage_id: int) -> Path:
    return stage_mesh_paths(spec, stage_id)["obj"]


def stage_ply_path(spec, stage_id: int) -> Path:
    return stage_mesh_paths(spec, stage_id)["ply"]


def stage_elements_json_path(spec, stage_id: int) -> Path:
    """Sidecar JSON describing element_id -> category mapping."""
    return stage_mesh_paths(spec, stage_id)["elements_json"]


# --- Renders / Video / Depth / Segmentation --------------------------------


def stage_render_dir(spec, stage_id: int) -> Path:
    """Where the CPU pipeline writes per-frame PNGs (and the keyframe).

    For the GPU/Blender pipeline use :func:`stage_blender_render_dir` instead.
    """
    return _out(spec).renders


def stage_keyframe_path(spec, stage_id: int) -> Path:
    """``output/renders/stage_07_keyframe.png`` -- single inspection PNG."""
    return _out(spec).renders / f"{stage_tag(stage_id)}_keyframe.png"


def stage_video_path(spec, stage_id: int, *, clean: bool = False, gpu: bool = False) -> Path:
    """Per-stage MP4 path.

    Parameters
    ----------
    clean:
        If True, return the un-postprocessed ``..._clean.mp4`` produced
        by the CPU pipeline.
    gpu:
        If True, return the Blender-rendered ``..._blender.mp4`` produced
        by the GPU pipeline.
    """
    if gpu:
        suffix = "_blender"
    elif clean:
        suffix = "_clean"
    else:
        suffix = ""
    return _out(spec).video / f"{stage_tag(stage_id)}{suffix}.mp4"


def stage_depth_npz_path(spec, stage_id: int) -> Path:
    return _out(spec).depth / f"{stage_tag(stage_id)}_depth.npz"


def stage_seg_npz_path(spec, stage_id: int) -> Path:
    return _out(spec).segmentation / f"{stage_tag(stage_id)}_seg.npz"


# --- GPU/Blender outputs ----------------------------------------------------


def blender_renders_root(spec) -> Path:
    """Root directory for the GPU pipeline's per-stage subfolders."""
    return _out(spec).root / "blender_renders"


def stage_blender_render_dir(spec, stage_id: int) -> Path:
    """``output/blender_renders/stage_07/`` -- holds rgb/, depth/, seg/, etc."""
    return blender_renders_root(spec) / stage_tag(stage_id)


def stage_blender_subdirs(spec, stage_id: int) -> dict[str, Path]:
    base = stage_blender_render_dir(spec, stage_id)
    return {
        "root": base,
        "rgb": base / "rgb",
        "depth": base / "depth",
        "seg": base / "seg",
        "camera_path": base / "camera_path.json",
        "log": base / "blender_render.log",
        "stdout": base / "blender_stdout.log",
        "stderr": base / "blender_stderr.log",
        "done_marker": base / ".done",
    }


# --- Camera path ------------------------------------------------------------


def stage_camera_path(spec, stage_id: int) -> Path:
    """``output/camera/stage_07_camera_path.json``"""
    return _out(spec).camera / f"{stage_tag(stage_id)}_camera_path.json"


# --- Manifests / metadata ---------------------------------------------------


def stage_metadata_path(spec, stage_id: int) -> Path:
    return _out(spec).manifests / f"{stage_tag(stage_id)}_metadata.json"


def stage_element_metrics_csv(spec, stage_id: int) -> Path:
    return _out(spec).manifests / f"{stage_tag(stage_id)}_element_metrics.csv"


def dataset_schedule_csv(spec) -> Path:
    return _out(spec).manifests / "schedule.csv"


def dataset_bim_schedule_mapping_csv(spec) -> Path:
    return _out(spec).manifests / "bim_schedule_mapping.csv"


def dataset_manifest_path(spec, *, gpu: bool = False) -> Path:
    """``output/manifests/manifest.json`` for CPU,
    ``output/manifests/manifest_blender_gpu.json`` for GPU."""
    name = "manifest_blender_gpu.json" if gpu else "manifest.json"
    return _out(spec).manifests / name


# ---------------------------------------------------------------------
# Convenience: full inventory for a stage
# ---------------------------------------------------------------------


def stage_artifact_inventory(spec, stage_id: int, *, gpu: bool = False) -> dict[str, Path]:
    """Return every canonical artefact path for a single stage.

    This is what the resume/checkpoint logic uses to decide whether a
    stage is already complete on disk.
    """
    inv: dict[str, Path] = {
        "ifc": stage_ifc_path(spec, stage_id),
        "mesh_obj": stage_obj_path(spec, stage_id),
        "mesh_glb": stage_glb_path(spec, stage_id),
        "mesh_ply": stage_ply_path(spec, stage_id),
        "elements_json": stage_elements_json_path(spec, stage_id),
        "camera_path": stage_camera_path(spec, stage_id),
        "metadata_json": stage_metadata_path(spec, stage_id),
        "element_metrics_csv": stage_element_metrics_csv(spec, stage_id),
    }
    if gpu:
        sub = stage_blender_subdirs(spec, stage_id)
        inv.update({
            "blender_dir": sub["root"],
            "blender_rgb_dir": sub["rgb"],
            "blender_done_marker": sub["done_marker"],
            "video": stage_video_path(spec, stage_id, gpu=True),
        })
    else:
        inv.update({
            "video": stage_video_path(spec, stage_id),
            "video_clean": stage_video_path(spec, stage_id, clean=True),
            "keyframe": stage_keyframe_path(spec, stage_id),
            "depth_npz": stage_depth_npz_path(spec, stage_id),
            "seg_npz": stage_seg_npz_path(spec, stage_id),
        })
    return inv
