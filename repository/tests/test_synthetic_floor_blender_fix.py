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
