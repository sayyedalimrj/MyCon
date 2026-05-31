"""Tests for the renderer's frame-level resume + progress helpers.

These import ``blender_gpu_renderer`` without Blender (the helpers are pure
Python; ``bpy`` is only touched inside the render functions), so they run in
the laptop-safe suite.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor import blender_gpu_renderer as R  # noqa: E402


class TestFrameResume(unittest.TestCase):
    def test_existing_frame_numbers_skips_tiny_and_parses_index(self):
        with tempfile.TemporaryDirectory() as d:
            rgb = Path(d) / "rgb"
            rgb.mkdir()
            for i in (1, 2, 3, 17):
                (rgb / f"frame_{i:04d}.png").write_bytes(b"x" * 5000)
            (rgb / "frame_0099.png").write_bytes(b"x" * 100)  # too small -> ignored
            done = R._existing_frame_numbers(rgb)
            self.assertEqual(done, {1, 2, 3, 17})

    def test_existing_frame_numbers_missing_dir(self):
        self.assertEqual(R._existing_frame_numbers(Path("/no/such/dir")), set())

    def test_atomic_write_json_no_temp_leftover(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "render_progress.json"
            R._atomic_write_json(p, {"frames_done": 5, "frames_total": 10})
            self.assertEqual(json.loads(p.read_text())["frames_done"], 5)
            self.assertEqual(list(Path(d).glob(".render_progress.json.*")), [])

    def test_resume_recomputes_remaining_frames(self):
        # Simulate the resume arithmetic: with frames 1..10 and 1..6 done,
        # the last done frame (6) is re-rendered (possible partial) so the
        # remaining work is frames 6..10.
        with tempfile.TemporaryDirectory() as d:
            rgb = Path(d) / "rgb"
            rgb.mkdir()
            for i in range(1, 7):
                (rgb / f"frame_{i:04d}.png").write_bytes(b"x" * 5000)
            done = R._existing_frame_numbers(rgb)
            if done:
                done.discard(max(done))
            todo = [f for f in range(1, 11) if f not in done]
            self.assertEqual(todo, [6, 7, 8, 9, 10])


    def test_worker_frames_partition_is_disjoint_and_complete(self):
        import synthetic_floor.blender_gpu_renderer as R
        todo = list(range(1, 26))
        for n in (1, 2, 3, 4):
            parts = [R._worker_frames(todo, i, n) for i in range(n)]
            flat = sorted(sum(parts, []))
            self.assertEqual(flat, todo, f"workers={n}: not a partition")
            # roughly balanced
            self.assertLessEqual(max(len(p) for p in parts) - min(len(p) for p in parts), 1)
        self.assertEqual(R._worker_frames(todo, 0, 1), todo)


if __name__ == "__main__":
    unittest.main()
