"""Tests for the example's Google Drive sync + portable run-state manifest."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor import colab_sync as CS  # noqa: E402
from synthetic_floor import checkpoint as ck  # noqa: E402
from synthetic_floor.scene_spec import load_scene_spec, OutputPaths  # noqa: E402


def _spec_with_output(tmp: Path):
    spec = load_scene_spec(HERE.parent / "config" / "scene.yaml")
    out = OutputPaths(
        root=tmp, bim=tmp / "bim", mesh=tmp / "mesh", renders=tmp / "renders",
        video=tmp / "video", depth=tmp / "depth", segmentation=tmp / "seg",
        camera=tmp / "camera", manifests=tmp / "manifests", logs=tmp / "logs",
    )
    spec = replace(spec, output=out)
    spec.output.ensure()
    return spec


class TestDriveMirror(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Simulate a mounted Drive.
        self.mount = self.root / "drive"
        (self.mount / "MyDrive").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_push_then_pull_roundtrip(self):
        local = self.root / "local"
        drive_root = self.mount / "MyDrive" / "out"
        (local / "manifests").mkdir(parents=True)
        (local / "manifests" / "m.json").write_text("{}", encoding="utf-8")
        (local / "blender_renders" / "stage_07" / "rgb").mkdir(parents=True)
        (local / "blender_renders" / "stage_07" / "rgb" / "frame_0001.png").write_bytes(b"x" * 64)

        m = CS.DriveMirror(local, drive_root, mount=self.mount, interval=999)
        push = m.push()
        self.assertGreaterEqual(push["copied"], 2)
        self.assertTrue((drive_root / "manifests" / "m.json").exists())

        # Fresh "device": pull restores everything.
        local2 = self.root / "local2"
        m2 = CS.DriveMirror(local2, drive_root, mount=self.mount, interval=999)
        m2.pull()
        self.assertTrue((local2 / "manifests" / "m.json").exists())
        self.assertTrue((local2 / "blender_renders" / "stage_07" / "rgb" / "frame_0001.png").exists())

    def test_default_drive_root_sanitizes(self):
        root = CS.default_drive_root("my run/2026", mount=Path("/m"))
        self.assertTrue(str(root).endswith("myrun2026"))


class TestRunStateManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = _spec_with_output(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_read_run_state(self):
        p = ck.write_run_state(self.spec, [1, 2, 3], gpu=True,
                               extra={"preset": "debug", "in_progress": True})
        self.assertTrue(p.exists())
        body = ck.read_run_state(self.spec, gpu=True)
        self.assertEqual(body["schema_version"], "synthetic_floor_run_state.v1")
        self.assertTrue(body["gpu"])
        self.assertEqual(set(body["stages"].keys()), {"1", "2", "3"})
        self.assertTrue(body["in_progress"])
        # Empty disk -> nothing complete.
        self.assertTrue(all(not s["complete"] for s in body["stages"].values()))

    def test_run_state_atomic_no_temp_left(self):
        ck.write_run_state(self.spec, [1], gpu=True)
        leftovers = list(self.spec.output.manifests.glob(".run_state_blender_gpu.json.*"))
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
