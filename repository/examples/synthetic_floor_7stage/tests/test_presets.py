"""Smoke tests for the quality presets module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor import presets as pr  # noqa: E402


class TestPresetCatalogue(unittest.TestCase):
    def test_three_named_presets_exist(self):
        for name in ("debug", "balanced", "hq"):
            self.assertIn(name, pr.CPU_PRESETS)
            self.assertIn(name, pr.GPU_PRESETS)
        self.assertEqual(set(pr.PRESET_NAMES), {"debug", "balanced", "hq"})

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            pr.get_cpu_preset("unknown")
        with self.assertRaises(ValueError):
            pr.get_gpu_preset("unknown")

    def test_preset_ordering(self):
        # Each preset should produce more pixels than the previous one
        cpu_pixels = [pr.CPU_PRESETS[n].width_px * pr.CPU_PRESETS[n].height_px
                      for n in ("debug", "balanced", "hq")]
        self.assertEqual(cpu_pixels, sorted(cpu_pixels))
        gpu_pixels = [pr.GPU_PRESETS[n].width_px * pr.GPU_PRESETS[n].height_px
                      for n in ("debug", "balanced", "hq")]
        self.assertEqual(gpu_pixels, sorted(gpu_pixels))
        # And HQ should have more samples than balanced, debug less
        self.assertLess(pr.GPU_PRESETS["debug"].samples, pr.GPU_PRESETS["balanced"].samples)
        self.assertLess(pr.GPU_PRESETS["balanced"].samples, pr.GPU_PRESETS["hq"].samples)


class TestApplyCPUPreset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_scene_spec(HERE.parent / "config" / "scene.yaml")

    def test_cli_overrides_win(self):
        out = pr.apply_cpu_preset(self.spec, "balanced", width=999, height=111)
        self.assertEqual(out.camera.width_px, 999)
        self.assertEqual(out.camera.height_px, 111)
        # duration not overridden, so it falls back to preset default
        self.assertEqual(out.camera.duration_per_stage_sec,
                         pr.CPU_PRESETS["balanced"].duration_per_stage_sec)

    def test_immutable_input(self):
        # SceneSpec is frozen; apply_cpu_preset must return a new instance.
        out = pr.apply_cpu_preset(self.spec, "debug")
        self.assertIsNot(out, self.spec)


class TestApplyGPUPreset(unittest.TestCase):
    def test_explicit_overrides_win(self):
        p = pr.apply_gpu_preset(
            "balanced",
            resolution=(1920, 1080),
            samples=256,
            frames=240,
            fps=60,
            motion_blur=False,
        )
        self.assertEqual(p.width_px, 1920)
        self.assertEqual(p.height_px, 1080)
        self.assertEqual(p.samples, 256)
        self.assertEqual(p.frames_per_stage, 240)
        self.assertEqual(p.fps, 60)
        self.assertFalse(p.motion_blur)

    def test_no_overrides_returns_preset_values(self):
        p = pr.apply_gpu_preset("hq")
        ref = pr.GPU_PRESETS["hq"]
        self.assertEqual(p, ref)


class TestDescribePreset(unittest.TestCase):
    def test_human_readable_one_liners(self):
        cpu_line = pr.describe_preset("balanced", gpu=False)
        gpu_line = pr.describe_preset("hq", gpu=True)
        self.assertIn("balanced", cpu_line)
        self.assertIn("hq", gpu_line)
        self.assertIn("samples", gpu_line)


if __name__ == "__main__":
    unittest.main()
