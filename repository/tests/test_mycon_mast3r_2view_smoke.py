"""Non-GPU safety gates for real NVIDIA MASt3R inference preflight."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mycon_mast3r_2view_smoke as smoke


class TestNvidiaMast3rSmoke(unittest.TestCase):
    def test_full_frame_inventory_selects_spaced_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(30):
                (root / f"frame_{i:06d}.jpg").write_bytes(b"safe-fake-frame")
            first, second, count = smoke.choose_pair(root)
            self.assertEqual(count, 30)
            self.assertEqual(first.name, "frame_000000.jpg")
            self.assertEqual(second.name, "frame_000008.jpg")

    def test_short_video_is_rejected_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(29):
                (root / f"frame_{i:06d}.jpg").write_bytes(b"fake-frame")
            with self.assertRaisesRegex(RuntimeError, "COMPLETE_PREPARED_FRAMES"):
                smoke.choose_pair(root)

    def test_zero_byte_source_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(30):
                (root / f"frame_{i:06d}.jpg").write_bytes(b"fake-frame")
            (root / "frame_000008.jpg").write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "FRAME_PAIR_INVALID"):
                smoke.choose_pair(root)

    def test_unverified_source_fails_before_loading_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(30):
                (root / f"frame_{i:06d}.jpg").write_bytes(b"fake-frame")
            with self.assertRaisesRegex(RuntimeError, "OFFICIAL_NVIDIA"):
                smoke.run(root / "bad_source", root)


if __name__ == "__main__":
    unittest.main()
