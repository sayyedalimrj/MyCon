"""LongSplat derived data branch: no source mutation, no GPU and hashed resume."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mycon_longsplat_data as m


class LongSplatDataSafety(unittest.TestCase):
    def test_8fps_dataset_is_independent_hashed_reusable_and_not_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "source.mp4"
            video.write_bytes(b"verified captured video" * 200)
            old_video = video.read_bytes()
            dataset = root / "derived_longsplat"
            (dataset / "old").mkdir(parents=True)
            (dataset / "old" / "partial.txt").write_text("old artifact protected")
            def fake_ffmpeg(cmd, log, **kwargs):
                self.assertEqual(cmd[cmd.index("-vf") + 1],
                                 "transpose=clock,fps=8,scale=512:-2:flags=lanczos")
                folder = Path(cmd[-1]).parent
                for i in range(1, 81):
                    (folder / f"frame_{i:06d}.jpg").write_bytes(bytes([i])*1300)
            with patch.object(m, "run_logged", side_effect=fake_ffmpeg):
                receipt = m.prepare_longsplat_frames(
                    video, dataset, root / "log", capture_sha256="a"*64,
                    duration_s=10.0)
            self.assertEqual(receipt["frame_count"], 80)
            self.assertFalse(receipt["full_gpu_training_executed"])
            self.assertEqual(video.read_bytes(), old_video)
            self.assertEqual(len(list(root.glob("derived_longsplat.preserved_*"))), 1)
            saved = json.loads((dataset / "dataset_receipt.json").read_text())
            self.assertFalse(saved["metric_scale_verified"])
            self.assertFalse(saved["training_executed"])
            with patch.object(m, "run_logged",
                              side_effect=AssertionError("FFMPEG MUST NOT REPEAT")):
                reuse = m.prepare_longsplat_frames(
                    video, dataset, root / "log", capture_sha256="a"*64,
                    duration_s=10.0)
            self.assertIn("VERIFIED", reuse["status"])
            (dataset / "images/frame_000042.jpg").write_bytes(b"altered")
            with self.assertRaisesRegex(RuntimeError, "FRAME_HASH_INTEGRITY_FAILED"):
                m.prepare_longsplat_frames(
                    video, dataset, root / "log", capture_sha256="a"*64,
                    duration_s=10.0)
            self.assertEqual(video.read_bytes(), old_video)

    def test_wrong_duration_and_bad_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "video.mp4"
            video.write_bytes(b"x" * 3000)
            with patch.object(m, "run_logged", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "FRAME_COUNT_INVALID"):
                    m.prepare_longsplat_frames(video, root / "derived",
                                               root / "log",
                                               capture_sha256="a"*64,
                                               duration_s=10.0)
            self.assertFalse((root / "derived/dataset_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
