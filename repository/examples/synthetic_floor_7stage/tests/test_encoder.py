"""Smoke tests for the robust MP4 encoder in ``run_blender_gpu.encode_mp4``.

We do **not** rely on ffmpeg actually being available here; we only
test the frame-loading and normalisation logic, which is the part that
broke in production (mixed sizes + corrupt frames).
"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import importlib.util  # noqa: E402

# Load encode_mp4 as a regular module attribute even though it lives in
# scripts/run_blender_gpu.py (not a package).
_spec = importlib.util.spec_from_file_location(
    "run_blender_gpu", SCRIPTS / "run_blender_gpu.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
encode_mp4 = _mod.encode_mp4


def _silent_logger() -> logging.Logger:
    log = logging.getLogger("encode_test")
    log.setLevel(logging.CRITICAL)
    return log


class TestEncodeMP4Robustness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_pngs_returns_none(self):
        # Empty dir -> the encoder must NOT crash, must return None.
        out = encode_mp4(self.dir, self.dir / "out.mp4", fps=30, log=_silent_logger())
        self.assertIsNone(out)

    def test_mixed_size_and_corrupt_frames(self):
        try:
            from PIL import Image
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("Pillow + NumPy not available")

        # 3 valid frames, 2 different sizes, 1 corrupt file.
        Image.new("RGB", (64, 36), color=(10, 20, 30)).save(self.dir / "frame_0001.png")
        Image.new("RGB", (32, 18), color=(40, 50, 60)).save(self.dir / "frame_0002.png")
        Image.new("RGBA", (64, 36), color=(70, 80, 90, 255)).save(self.dir / "frame_0003.png")
        # Corrupt: not a valid PNG
        (self.dir / "frame_0004.png").write_bytes(b"not a png file")

        out_path = self.dir / "out.mp4"
        result = encode_mp4(self.dir, out_path, fps=30, log=_silent_logger())

        # If imageio + ffmpeg are available, the encoder should produce a file.
        # Otherwise it returns None (still a clean path -- not a crash).
        # In both cases, no exception must propagate.
        if result is not None:
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
        # Either way the function must have *handled* the corrupt frame
        # rather than crashing.


if __name__ == "__main__":
    unittest.main()
