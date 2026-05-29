"""Smoke tests for the path helpers.

Run from the repository root::

    PYTHONPATH=examples/synthetic_floor_7stage/src \\
        python3 -m unittest examples.synthetic_floor_7stage.tests.test_paths

Or, if the ``examples/`` directory is not a package on your shell::

    PYTHONPATH=examples/synthetic_floor_7stage/src \\
        python3 examples/synthetic_floor_7stage/tests/test_paths.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor import paths as P  # noqa: E402


class TestStageTag(unittest.TestCase):
    def test_zero_padded(self):
        self.assertEqual(P.stage_tag(1), "stage_01")
        self.assertEqual(P.stage_tag(7), "stage_07")
        self.assertEqual(P.stage_tag(10), "stage_10")

    def test_accepts_ints_only(self):
        # str input is a programming bug; we don't pretend to support it
        with self.assertRaises((TypeError, ValueError)):
            P.stage_tag("seven")  # type: ignore[arg-type]


class TestPathHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = HERE.parent / "config" / "scene.yaml"
        cls.spec = load_scene_spec(config)

    def test_basic_paths(self):
        s = self.spec
        self.assertEqual(P.stage_ifc_path(s, 7).name, "stage_07.ifc")
        self.assertEqual(P.stage_glb_path(s, 7).name, "stage_07.glb")
        self.assertEqual(P.stage_obj_path(s, 7).name, "stage_07.obj")
        self.assertEqual(P.stage_ply_path(s, 7).name, "stage_07.ply")
        self.assertEqual(P.stage_keyframe_path(s, 7).name, "stage_07_keyframe.png")
        self.assertEqual(P.stage_video_path(s, 7).name, "stage_07.mp4")
        self.assertEqual(P.stage_video_path(s, 7, clean=True).name, "stage_07_clean.mp4")
        self.assertEqual(P.stage_video_path(s, 7, gpu=True).name, "stage_07_blender.mp4")
        self.assertEqual(P.stage_camera_path(s, 7).name, "stage_07_camera_path.json")

    def test_all_paths_under_output_root(self):
        """Every helper must produce a path under spec.output.root."""
        s = self.spec
        root = s.output.root
        helpers = [
            P.stage_ifc_path(s, 1),
            P.stage_glb_path(s, 1),
            P.stage_keyframe_path(s, 1),
            P.stage_video_path(s, 1),
            P.stage_video_path(s, 1, gpu=True),
            P.stage_metadata_path(s, 1),
            P.stage_element_metrics_csv(s, 1),
            P.stage_camera_path(s, 1),
            P.stage_blender_render_dir(s, 1),
            P.dataset_manifest_path(s),
            P.dataset_manifest_path(s, gpu=True),
        ]
        for p in helpers:
            self.assertTrue(
                str(p).startswith(str(root)),
                f"{p} is not under {root}",
            )

    def test_inventory_keys(self):
        cpu_inv = P.stage_artifact_inventory(self.spec, 4, gpu=False)
        gpu_inv = P.stage_artifact_inventory(self.spec, 4, gpu=True)
        # Every inventory has at least the BIM + mesh + metadata triple
        for inv in (cpu_inv, gpu_inv):
            for key in ("ifc", "mesh_glb", "elements_json", "metadata_json"):
                self.assertIn(key, inv)
        # CPU has video/keyframe/depth
        self.assertIn("keyframe", cpu_inv)
        self.assertIn("video_clean", cpu_inv)
        self.assertIn("depth_npz", cpu_inv)
        # GPU has blender_dir / done_marker
        self.assertIn("blender_dir", gpu_inv)
        self.assertIn("blender_done_marker", gpu_inv)

    def test_manifest_paths_distinct(self):
        cpu = P.dataset_manifest_path(self.spec)
        gpu = P.dataset_manifest_path(self.spec, gpu=True)
        self.assertNotEqual(cpu.name, gpu.name)


if __name__ == "__main__":
    unittest.main()
