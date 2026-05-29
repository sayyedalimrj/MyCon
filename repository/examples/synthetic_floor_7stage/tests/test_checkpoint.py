"""Smoke tests for the resume/checkpoint module."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor import checkpoint as ck  # noqa: E402
from synthetic_floor import paths as P  # noqa: E402


class TestStageStatusOnEmptyDisk(unittest.TestCase):
    """A pristine output directory means every stage is incomplete."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.spec = _load_spec_with_temp_output(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cpu_status_missing_everything(self):
        s = ck.stage_status(self.spec, 1, gpu=False)
        self.assertFalse(s.complete)
        self.assertGreater(len(s.missing), 0)

    def test_gpu_status_missing_everything(self):
        s = ck.stage_status(self.spec, 1, gpu=True)
        self.assertFalse(s.complete)
        self.assertGreater(len(s.missing), 0)


class TestFilterStagesPolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.spec = _load_spec_with_temp_output(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_mode_runs_everything(self):
        to_run, _ = ck.filter_stages(self.spec, [1, 2, 3], "run", gpu=False)
        self.assertEqual(to_run, [1, 2, 3])

    def test_resume_mode_runs_incomplete_stages(self):
        to_run, _ = ck.filter_stages(self.spec, [1, 2], "resume", gpu=False)
        # Empty disk -> resume must run everything
        self.assertEqual(to_run, [1, 2])

    def test_force_mode_returns_everything(self):
        to_run, _ = ck.filter_stages(self.spec, [1, 2], "force", gpu=False)
        self.assertEqual(to_run, [1, 2])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            ck.filter_stages(self.spec, [1], "blast-everything")  # type: ignore[arg-type]


class TestDoneMarkerRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.spec = _load_spec_with_temp_output(Path(cls.tmp.name))
        cls.spec.output.ensure()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cpu_done_marker_roundtrip(self):
        ck.write_done_marker(self.spec, 3, payload={"preset": "debug", "elapsed_sec": 1.2}, gpu=False)
        body = ck.read_done_marker(self.spec, 3, gpu=False)
        self.assertIsNotNone(body)
        self.assertEqual(body["stage_id"], 3)
        self.assertEqual(body["preset"], "debug")
        self.assertEqual(body["gpu"], False)
        self.assertIn("finished_at", body)

    def test_gpu_done_marker_roundtrip(self):
        # Need to create the blender_dir before writing the marker
        sub = P.stage_blender_subdirs(self.spec, 5)
        sub["root"].mkdir(parents=True, exist_ok=True)
        ck.write_done_marker(self.spec, 5, payload={"rgb_frame_count": 30}, gpu=True)
        body = ck.read_done_marker(self.spec, 5, gpu=True)
        self.assertIsNotNone(body)
        self.assertEqual(body["stage_id"], 5)
        self.assertEqual(body["gpu"], True)
        self.assertEqual(body["rgb_frame_count"], 30)

    def test_missing_marker_returns_none(self):
        body = ck.read_done_marker(self.spec, 999, gpu=False)
        self.assertIsNone(body)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _load_spec_with_temp_output(tmp_root: Path):
    """Load the canonical config but redirect every output path under ``tmp_root``."""
    config = HERE.parent / "config" / "scene.yaml"
    spec = load_scene_spec(config)
    # The OutputPaths is a frozen dataclass, but we can rebuild it.
    from dataclasses import replace
    from synthetic_floor.scene_spec import OutputPaths
    new_out = OutputPaths(
        root=tmp_root,
        bim=tmp_root / "bim",
        mesh=tmp_root / "mesh",
        renders=tmp_root / "renders",
        video=tmp_root / "video",
        depth=tmp_root / "depth",
        segmentation=tmp_root / "segmentation",
        camera=tmp_root / "camera",
        manifests=tmp_root / "manifests",
        logs=tmp_root / "logs",
    )
    return replace(spec, output=new_out)


if __name__ == "__main__":
    unittest.main()
