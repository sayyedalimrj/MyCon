"""Repo-level coverage for the synthetic_floor GPU renderer fixes.

Ensures the laptop/CI suite guards:
  * the coordinate-frame alignment that fixes the "nothing but light" render
    (trimesh Z-up GLB vs Blender's glTF Y-up import), and
  * the Drive sync + portable run-state resume helpers.

Pure NumPy / stdlib only (no Blender, no GPU), so it runs in the default
lightweight suite. The example lives under examples/, so we add its src to
sys.path explicitly.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SRC = REPO_ROOT / "examples" / "synthetic_floor_7stage" / "src"
EXAMPLE_CFG = REPO_ROOT / "examples" / "synthetic_floor_7stage" / "config" / "scene.yaml"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))

from synthetic_floor.geometry_align import compute_alignment, _corners, _rot_x  # noqa: E402


AUTHOR_MIN = [0.0, 0.0, -0.2]
AUTHOR_MAX = [22.0, 14.0, 3.4]


def _simulate_blender_yup_import(amin, amax):
    corners = _corners(np.array(amin), np.array(amax))
    rotated = corners @ _rot_x(90.0).T  # Blender glTF Y-up -> Z-up (+90 about X)
    return rotated.min(0), rotated.max(0), rotated


def test_alignment_recovers_author_frame_from_gltf_yup_import() -> None:
    bmin, bmax, bc = _simulate_blender_yup_import(AUTHOR_MIN, AUTHOR_MAX)
    res = compute_alignment(bmin.tolist(), bmax.tolist(),
                            author_min=AUTHOR_MIN, author_max=AUTHOR_MAX)
    assert res["mode"] == "rot_x_-90"
    assert res["needs_change"] is True
    M = np.array(res["matrix"])
    homog = np.hstack([bc, np.ones((len(bc), 1))])
    recovered = (homog @ M.T)[:, :3]
    assert np.allclose(recovered.min(0), AUTHOR_MIN, atol=1e-6)
    assert np.allclose(recovered.max(0), AUTHOR_MAX, atol=1e-6)


def test_alignment_noop_when_already_aligned() -> None:
    res = compute_alignment(AUTHOR_MIN, AUTHOR_MAX,
                            author_min=AUTHOR_MIN, author_max=AUTHOR_MAX)
    assert res["mode"] == "identity"
    assert res["needs_change"] is False


def test_alignment_heuristic_without_author_bbox() -> None:
    bmin, bmax, _ = _simulate_blender_yup_import(AUTHOR_MIN, AUTHOR_MAX)
    res = compute_alignment(bmin.tolist(), bmax.tolist())
    assert res["mode"] in ("rot_x_-90", "rot_x_+90")
    assert res["needs_change"] is True


def _spec_with_output(tmp: Path):
    from synthetic_floor.scene_spec import load_scene_spec, OutputPaths

    spec = load_scene_spec(EXAMPLE_CFG)
    out = OutputPaths(
        root=tmp, bim=tmp / "bim", mesh=tmp / "mesh", renders=tmp / "renders",
        video=tmp / "video", depth=tmp / "depth", segmentation=tmp / "seg",
        camera=tmp / "camera", manifests=tmp / "manifests", logs=tmp / "logs",
    )
    spec = replace(spec, output=out)
    spec.output.ensure()
    return spec


def test_run_state_manifest_roundtrip() -> None:
    from synthetic_floor import checkpoint as ck

    with tempfile.TemporaryDirectory() as d:
        spec = _spec_with_output(Path(d))
        ck.write_run_state(spec, [1, 2, 3], gpu=True, extra={"in_progress": True})
        body = ck.read_run_state(spec, gpu=True)
        assert body["schema_version"] == "synthetic_floor_run_state.v1"
        assert set(body["stages"]) == {"1", "2", "3"}
        assert body["in_progress"] is True


def test_drive_mirror_push_pull_roundtrip() -> None:
    from synthetic_floor import colab_sync as CS

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        mount = root / "drive"
        (mount / "MyDrive").mkdir(parents=True)
        local = root / "local"
        drive_root = mount / "MyDrive" / "out"
        (local / "manifests").mkdir(parents=True)
        (local / "manifests" / "m.json").write_text("{}", encoding="utf-8")

        m = CS.DriveMirror(local, drive_root, mount=mount, interval=999)
        push = m.push()
        assert push["copied"] >= 1
        assert (drive_root / "manifests" / "m.json").exists()

        local2 = root / "local2"
        CS.DriveMirror(local2, drive_root, mount=mount, interval=999).pull()
        assert (local2 / "manifests" / "m.json").exists()


# ---------------------------------------------------------------------------
# Detailed example scenes (office / loft / warehouse) — construction logic
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

_DETAILED_CONFIGS = ["scene_office.yaml", "scene_loft.yaml", "scene_warehouse.yaml"]
_CONFIG_DIR = REPO_ROOT / "examples" / "synthetic_floor_7stage" / "config"


def _cats_at(spec, by_cat, stage_id):
    from synthetic_floor.stage_controller import select_for_stage, kept_only

    staged = select_for_stage(spec.stages[stage_id - 1], by_cat)
    return {s.element.category for s in kept_only(staged)}


@pytest.mark.parametrize("cfg", _DETAILED_CONFIGS)
def test_detailed_example_has_rich_geometry(cfg: str) -> None:
    from synthetic_floor.scene_spec import load_scene_spec
    from synthetic_floor.layout import build_layout, elements_by_category

    spec = load_scene_spec(_CONFIG_DIR / cfg)
    assert len(spec.stages) == 7
    cats = set(elements_by_category(build_layout(spec)))
    for need in ("site_ground", "foundation", "beams", "window_frame", "floor_finish"):
        assert need in cats, f"{cfg}: missing {need}"


@pytest.mark.parametrize("cfg", _DETAILED_CONFIGS)
def test_detailed_example_construction_sequence(cfg: str) -> None:
    """Doors/windows installed late; final floor only at the end; beams by S3."""
    from synthetic_floor.scene_spec import load_scene_spec
    from synthetic_floor.layout import build_layout, elements_by_category

    spec = load_scene_spec(_CONFIG_DIR / cfg)
    by = elements_by_category(build_layout(spec))

    c1 = _cats_at(spec, by, 1)
    assert "foundation" in c1 and "site_ground" in c1
    for absent in ("slab", "columns", "windows", "door"):
        assert absent not in c1, f"{cfg}: {absent} should not exist at stage 1"

    assert "beams" in _cats_at(spec, by, 3), f"{cfg}: beams missing by stage 3"

    c4 = _cats_at(spec, by, 4)
    assert "north_wall" in c4
    for absent in ("windows", "window_frame", "door"):
        assert absent not in c4, f"{cfg}: {absent} installed too early"

    c6 = _cats_at(spec, by, 6)
    for present in ("windows", "window_frame", "door"):
        assert present in c6, f"{cfg}: {present} missing at stage 6"

    c7 = _cats_at(spec, by, 7)
    assert "floor_finish" not in c6 and "floor_finish" in c7
    assert "ceiling_lights" in c7



# ---------------------------------------------------------------------------
# Config-driven lighting, parallel workers, delta-sync (this PR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", _DETAILED_CONFIGS + ["scene.yaml"])
def test_renderer_lighting_and_parallel_config_exposed(cfg: str) -> None:
    """HDR/AO/exposure + parallel worker params must be read from the YAML."""
    from synthetic_floor.scene_spec import load_scene_spec

    spec = load_scene_spec(_CONFIG_DIR / cfg)
    r = spec.renderer
    assert 0.0 < r.world_strength <= 1.0          # low ambient fill -> deep shadows
    assert r.sun_energy > 0.0
    assert r.exposure <= 0.0                        # never the old blown-out boost
    assert 0.0 <= r.ao_factor <= 1.0
    assert r.parallel_workers_count >= 1
    assert r.view_transform == "Filmic"
    # Human eye-level walking height (not floating near the ceiling).
    assert 1.5 <= spec.camera.hold_height_m <= 1.75


def test_worker_frame_partition_is_disjoint_and_complete() -> None:
    from synthetic_floor.blender_gpu_renderer import _worker_frames

    todo = list(range(1, 23))
    for n in (1, 2, 3, 5):
        parts = [_worker_frames(todo, i, n) for i in range(n)]
        assert sorted(sum(parts, [])) == todo, f"workers={n} not a partition"
        assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1


def test_delta_sync_skips_identical_content() -> None:
    import os
    import time as _time

    from synthetic_floor import colab_sync as CS

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src, dst = root / "s", root / "t"
        src.mkdir()
        dst.mkdir()
        (src / "f.bin").write_bytes(b"x" * 4096)
        assert CS.mirror_tree(src, dst, use_hash=True)["copied"] == 1
        old = _time.time() - 50000
        os.utime(dst / "f.bin", (old, old))  # drift mtime, identical content
        out = CS.mirror_tree(src, dst, use_hash=True)
        assert out["copied"] == 0 and out["skipped"] == 1
