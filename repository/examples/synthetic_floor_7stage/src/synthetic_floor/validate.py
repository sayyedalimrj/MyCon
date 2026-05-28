"""Sanity checks on the generated dataset.

Each function returns ``(ok: bool, messages: list[str])``. The runner
runs all checks and aborts only if a *critical* check fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .layout import Element
from .scene_spec import SceneSpec
from .stage_controller import StagedElement


def check_geometry_dimensions(spec: SceneSpec, elements: Sequence[Element]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    floor = spec.floor
    # Slab and ceiling must lie inside the building footprint
    L, W, H = floor.length_m, floor.width_m, floor.height_m
    for e in elements:
        if e.category in ("slab", "ceiling"):
            x0, y0, _ = e.box_min
            x1, y1, _ = e.box_max
            if not (0.0 <= x0 < x1 <= L + 1e-6) or not (0.0 <= y0 < y1 <= W + 1e-6):
                msgs.append(f"{e.id}: slab/ceiling outside footprint")
    if msgs:
        return False, msgs
    msgs.append(f"footprint OK: {L:.2f}m x {W:.2f}m x {H:.2f}m, {len(elements)} elements")
    return True, msgs


def check_stage_progression(staged_per_stage: Mapping[int, Sequence[StagedElement]]) -> tuple[bool, list[str]]:
    """The number of kept elements should be monotonically non-decreasing
    across stages 1..7.
    """
    msgs: list[str] = []
    counts = []
    for sid in sorted(staged_per_stage):
        kept = sum(1 for s in staged_per_stage[sid] if s.completion >= 0.5)
        counts.append((sid, kept))
        msgs.append(f"stage {sid}: {kept} kept elements")
    for (sa, ca), (sb, cb) in zip(counts[:-1], counts[1:]):
        if cb < ca - 1:
            return False, msgs + [f"FAIL: stage {sb} kept ({cb}) < stage {sa} kept ({ca}) - 1"]
    return True, msgs


def check_unique_ifc_guids(elements: Sequence[Element]) -> tuple[bool, list[str]]:
    seen: set[str] = set()
    for e in elements:
        if e.ifc_global_id in seen:
            return False, [f"duplicate IFC GlobalId: {e.ifc_global_id}"]
        seen.add(e.ifc_global_id)
    return True, [f"all {len(elements)} IFC GlobalIds are unique"]


def check_camera_path(poses: Sequence) -> tuple[bool, list[str]]:
    if len(poses) < 2:
        return False, ["camera path has fewer than 2 frames"]
    eyes = np.array([p.eye for p in poses])
    finite = np.isfinite(eyes).all()
    if not finite:
        return False, ["non-finite values in camera path"]
    speed = np.linalg.norm(np.diff(eyes, axis=0), axis=1).max()
    if speed > 2.0:
        return False, [f"FAIL: implausibly large per-frame translation ({speed:.2f} m)"]
    return True, [f"camera path: {len(poses)} frames, max per-frame step = {speed:.3f} m"]


def check_outputs_exist(files: Mapping[str, Path]) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    for k, p in files.items():
        if p is None:
            continue
        path = Path(p)
        if not path.exists():
            ok = False
            msgs.append(f"MISSING: {k} -> {path}")
        else:
            msgs.append(f"  OK    {k} -> {path.name} ({path.stat().st_size:,} bytes)")
    return ok, msgs


def check_manifest_consistency(manifest_path: Path) -> tuple[bool, list[str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    msgs: list[str] = []
    missing: list[str] = []
    for sid, files in payload.get("stage_files", {}).items():
        for k, info in files.items():
            if not info.get("exists", False):
                missing.append(f"stage {sid}: {k}")
    for k, info in payload.get("dataset_files", {}).items():
        if not info.get("exists", False):
            missing.append(f"dataset: {k}")
    if missing:
        return False, ["manifest references missing files:"] + missing
    msgs.append(f"manifest OK ({manifest_path})")
    return True, msgs
