"""Tests for the physically-plausible concrete material fix.

The old raw-concrete texture clipped toward white (~0.95). It must now be a
mid-dark grey (~0.35 albedo) with real high-frequency detail.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from synthetic_floor import materials as M  # noqa: E402


class TestConcreteMaterial(unittest.TestCase):
    def test_concrete_albedo_is_dark_not_blown_out(self):
        img = M._build_concrete(256, seed=7)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = float(arr.mean())
        # Physically plausible raw concrete: mid-dark grey, well clear of white.
        self.assertGreater(mean, 0.20, f"too dark ({mean:.3f})")
        self.assertLess(mean, 0.45, f"still blown out toward white ({mean:.3f})")
        # Almost nothing should be near-white anymore.
        near_white = float((arr.max(axis=2) > 0.85).mean())
        self.assertLess(near_white, 0.02, f"{near_white:.3f} of pixels near-white")

    def test_concrete_has_high_frequency_detail(self):
        img = M._build_concrete(256, seed=11)
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        # High-frequency content via the gradient magnitude (aggregate/grain).
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        self.assertGreater(gx + gy, 0.5, "concrete texture looks flat (no grain)")

    def test_concrete_deterministic(self):
        a = np.asarray(M._build_concrete(128, seed=3))
        b = np.asarray(M._build_concrete(128, seed=3))
        self.assertTrue((a == b).all())

    def test_library_builds_with_dark_concrete(self):
        lib = M.build_material_library(seed=5, size=128)
        arr = np.asarray(lib["raw_concrete"].image, dtype=np.float32) / 255.0
        self.assertLess(float(arr.mean()), 0.45)


if __name__ == "__main__":
    unittest.main()
