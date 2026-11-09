"""Regression checks for MyCON G1 temporal coverage and safe quality manifest."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mycon_scene_qa import temporal_coverage, frame_quality_candidates


def write_colmap_pose_rows(path: Path, frame_ids: list[int]) -> None:
    # COLMAP images.txt: one camera-pose header, then observations line.
    path.write_text(
        "# Image list\n" + "".join(
            f"{i} 1 0 0 0 0 0 0 1 frame_{frame:06d}.jpg\n\n"
            for i, frame in enumerate(frame_ids, 1)),
        encoding="utf-8")


class SceneQAGates(unittest.TestCase):
    def test_good_distribution_passes_but_never_proves_metric(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "images.txt"
            write_colmap_pose_rows(path, [i for i in range(1, 181) if i % 6])
            out = temporal_coverage(path, 180)
            self.assertEqual(out["status"], "PASS")
            self.assertTrue(all(x["passed"] for x in out["segments"]))
            self.assertFalse(out["spatial_connectivity_verified"])
            self.assertFalse(out["bim_progress_authorized"])

    def test_global_seventy_percent_can_hide_last_room(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "images.txt"
            write_colmap_pose_rows(path, list(range(1, 131)))
            out = temporal_coverage(path, 180)
            self.assertGreater(out["registered_fraction"], .65)
            self.assertEqual(out["status"], "FAIL")
            self.assertIn("UNDERCOVERED_SEGMENT_8", out["fail_reasons"])

    def test_gap_fail_even_when_bins_have_some_registered(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "images.txt"
            write_colmap_pose_rows(path, list(range(1, 76)) + list(range(109, 181)))
            out = temporal_coverage(path, 180)
            self.assertEqual(out["status"], "FAIL")
            self.assertIn("UNREGISTERED_CONTIGUOUS_GAP", out["fail_reasons"])

    def test_unknown_frame_name_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "images.txt"
            write_colmap_pose_rows(path, [1, 2, 200])
            with self.assertRaisesRegex(RuntimeError, "OUTSIDE_VERIFIED_SOURCE"):
                temporal_coverage(path, 180)

    def test_duplicate_registered_name_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "images.txt"
            write_colmap_pose_rows(path, [1, 1, 2])
            with self.assertRaisesRegex(RuntimeError, "DUPLICATED"):
                temporal_coverage(path, 180)

    def test_image_analysis_never_modifies_source_or_calls_it_metric(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and numpy are optional for CPU image QA fixture")
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for i in range(1, 17):
                # Distinct scenes with identifiable alternating patches.
                frame = np.full((96, 160), 45 + i * 7, dtype=np.uint8)
                frame[:, i % 13::13] = 255
                assert cv2.imwrite(str(folder / f"frame_{i:06d}.jpg"), frame)
            original = {p.name: p.read_bytes() for p in folder.glob("*.jpg")}
            qa = frame_quality_candidates(folder, segments=8, max_candidates=10)
            self.assertEqual(qa["source_frame_count"], 16)
            self.assertEqual(qa["candidate_count"], 10)
            self.assertEqual(len(qa["frames"]), 16)
            self.assertFalse(qa["native_pose_or_translation_verified"])
            self.assertTrue(qa["all_original_frames_preserved"])
            self.assertEqual(original, {p.name: p.read_bytes() for p in folder.glob("*.jpg")})


if __name__ == "__main__":
    unittest.main()
