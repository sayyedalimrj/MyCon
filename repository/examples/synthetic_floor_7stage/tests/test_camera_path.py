"""Tests for the hyper-realistic procedural camera operator.

Pure NumPy (no Blender), so these run in the laptop-safe suite. They assert
the qualities the overhaul promised: decoupled look-at, full coverage,
collision-safety (no wall clipping), 6-DOF verticality, active looking-around
(including ~180 deg turns), physical smoothness, and config-driven behaviour.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CFG = HERE.parent / "config"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from synthetic_floor import camera_path as CP  # noqa: E402
from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402

CONFIGS = ["scene.yaml", "scene_office.yaml", "scene_loft.yaml", "scene_warehouse.yaml"]


def _interior(spec):
    f = spec.floor
    m = spec.camera.motion
    wt = f.exterior_wall_thickness_m
    return (wt + m.collision_margin_m, f.length_m - wt - m.collision_margin_m,
            wt + m.collision_margin_m, f.width_m - wt - m.collision_margin_m)


class TestCameraOperator(unittest.TestCase):
    def _poses(self, cfg, n=150):
        spec = load_scene_spec(CFG / cfg)
        poses = CP.plan_camera_path(spec, stage_id=7, n_frames=n)
        eyes = np.array([p.eye for p in poses])
        tgts = np.array([p.target for p in poses])
        return spec, poses, eyes, tgts

    def test_no_nan_and_count(self):
        for cfg in CONFIGS:
            _, poses, eyes, tgts = self._poses(cfg)
            self.assertEqual(len(poses), 150)
            self.assertTrue(np.isfinite(eyes).all())
            self.assertTrue(np.isfinite(tgts).all())

    def test_collision_no_wall_clipping(self):
        for cfg in CONFIGS:
            spec, _, eyes, _ = self._poses(cfg)
            x0, x1, y0, y1 = _interior(spec)
            self.assertGreaterEqual(eyes[:, 0].min(), x0 - 1e-6, cfg)
            self.assertLessEqual(eyes[:, 0].max(), x1 + 1e-6, cfg)
            self.assertGreaterEqual(eyes[:, 1].min(), y0 - 1e-6, cfg)
            self.assertLessEqual(eyes[:, 1].max(), y1 + 1e-6, cfg)
            # never below the floor or above the ceiling
            self.assertGreater(eyes[:, 2].min(), 0.2, cfg)
            self.assertLess(eyes[:, 2].max(), spec.floor.height_m - 0.1, cfg)

    def test_full_coverage_of_interior(self):
        for cfg in CONFIGS:
            spec, _, eyes, _ = self._poses(cfg)
            x0, x1, y0, y1 = _interior(spec)
            covered = (eyes[:, 0].max() - eyes[:, 0].min()) * (eyes[:, 1].max() - eyes[:, 1].min())
            room = (x1 - x0) * (y1 - y0)
            self.assertGreater(covered / room, 0.6, f"{cfg}: only {100*covered/room:.0f}% covered")

    def test_six_dof_verticality(self):
        for cfg in CONFIGS:
            _, _, eyes, _ = self._poses(cfg)
            z_range = float(eyes[:, 2].max() - eyes[:, 2].min())
            self.assertGreater(z_range, 0.5, f"{cfg}: camera Z barely moves ({z_range:.2f}m)")

    def test_look_at_is_decoupled_and_scans(self):
        for cfg in CONFIGS:
            _, _, eyes, tgts = self._poses(cfg)
            # Heading from eye->target sweeps a wide arc (looks left/right + ~180 turns).
            head = np.unwrap(np.arctan2(tgts[:, 1] - eyes[:, 1], tgts[:, 0] - eyes[:, 0]))
            yaw_span = float(np.degrees(head.max() - head.min()))
            self.assertGreater(yaw_span, 180.0, f"{cfg}: gaze only sweeps {yaw_span:.0f} deg")
            # The gaze must NOT be locked to the velocity vector (decoupling):
            # measure that the look direction often differs a lot from motion dir.
            vel = np.diff(eyes[:, :2], axis=0)
            look = (tgts[:, :2] - eyes[:, :2])[:-1]
            vn = np.linalg.norm(vel, axis=1)
            ln = np.linalg.norm(look, axis=1)
            ok = (vn > 1e-4) & (ln > 1e-4)
            cos = np.sum(vel[ok] * look[ok], axis=1) / (vn[ok] * ln[ok])
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            self.assertGreater(float(ang.max()), 90.0, f"{cfg}: gaze never decoupled from motion")

    def test_inspects_floor_and_ceiling(self):
        for cfg in CONFIGS:
            _, _, eyes, tgts = self._poses(cfg)
            d = tgts - eyes
            pitch = np.degrees(np.arcsin(np.clip(d[:, 2] / np.linalg.norm(d, axis=1), -1, 1)))
            self.assertLess(float(pitch.min()), -10.0, f"{cfg}: never looks down at the floor")
            self.assertGreater(float(pitch.max()), 8.0, f"{cfg}: never looks up at the ceiling")

    def test_motion_is_smooth_no_teleports(self):
        for cfg in CONFIGS:
            _, _, eyes, _ = self._poses(cfg)
            step = np.linalg.norm(np.diff(eyes, axis=0), axis=1)
            # No single-frame jump larger than ~1.2 m (no teleports / snapping).
            self.assertLess(float(step.max()), 1.2, f"{cfg}: max per-frame jump {step.max():.2f}m")

    def test_config_driven_disables_verticality(self):
        # Turning vertical inspection off via the spec must flatten Z (proves
        # the behaviour is read from config, not hardcoded).
        from dataclasses import replace
        spec = load_scene_spec(CFG / "scene_office.yaml")
        motion = replace(spec.camera.motion, vertical_inspect_enabled=False)
        cam = replace(spec.camera, motion=motion, hold_height_m=1.55)
        spec2 = replace(spec, camera=cam)
        eyes = np.array([p.eye for p in CP.plan_camera_path(spec2, stage_id=1, n_frames=120)])
        # Only micro gait/breathing remains -> small Z range.
        self.assertLess(float(eyes[:, 2].max() - eyes[:, 2].min()), 0.35)

    def test_serialization_roundtrip(self):
        import json
        import tempfile
        spec = load_scene_spec(CFG / "scene_office.yaml")
        poses = CP.plan_camera_path(spec, stage_id=3, n_frames=60)
        with tempfile.TemporaryDirectory() as d:
            p = CP.write_camera_poses(poses, spec.camera, Path(d) / "cp.json", stage_id=3)
            data = json.loads(p.read_text())
            self.assertEqual(data["schema_version"], "synthetic_floor_camera_poses.v1")
            self.assertEqual(len(data["frames"]), 60)
            self.assertIn("eye", data["frames"][0])
            self.assertIn("target", data["frames"][0])
            self.assertEqual(list(Path(d).glob(".campose.*")), [])


if __name__ == "__main__":
    unittest.main()
