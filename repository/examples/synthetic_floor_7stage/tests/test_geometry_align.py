"""Tests for the Blender import coordinate-frame alignment.

These reproduce the real bug: trimesh writes the GLB Z-up verbatim, Blender's
glTF importer rotates it Y-up->Z-up, and the room ends up rotated out of the
hard-coded camera frame ("nothing but light"). ``compute_alignment`` must
detect this and produce the transform that maps the geometry back.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from synthetic_floor.geometry_align import (  # noqa: E402
    author_bbox_from_elements,
    compute_alignment,
    _corners,
    _rot_x,
)


AUTHOR_MIN = [0.0, 0.0, -0.2]
AUTHOR_MAX = [22.0, 14.0, 3.4]


def _simulate_blender_import(amin, amax):
    """Apply Blender's glTF Y-up->Z-up (+90 deg about X) to author corners."""
    corners = _corners(np.array(amin), np.array(amax))
    rotated = corners @ _rot_x(90.0).T
    return rotated.min(axis=0), rotated.max(axis=0), rotated


class TestComputeAlignment(unittest.TestCase):
    def test_detects_and_fixes_gltf_yup_import(self):
        bmin, bmax, bc = _simulate_blender_import(AUTHOR_MIN, AUTHOR_MAX)
        res = compute_alignment(bmin.tolist(), bmax.tolist(),
                                author_min=AUTHOR_MIN, author_max=AUTHOR_MAX)
        self.assertEqual(res["mode"], "rot_x_-90")
        self.assertTrue(res["needs_change"])
        # Applying the returned 4x4 to the Blender corners recovers the author bbox.
        M = np.array(res["matrix"])
        homog = np.hstack([bc, np.ones((len(bc), 1))])
        recovered = (homog @ M.T)[:, :3]
        self.assertTrue(np.allclose(recovered.min(axis=0), AUTHOR_MIN, atol=1e-6))
        self.assertTrue(np.allclose(recovered.max(axis=0), AUTHOR_MAX, atol=1e-6))

    def test_already_aligned_is_identity_no_change(self):
        res = compute_alignment(AUTHOR_MIN, AUTHOR_MAX,
                                author_min=AUTHOR_MIN, author_max=AUTHOR_MAX)
        self.assertEqual(res["mode"], "identity")
        self.assertFalse(res["needs_change"])

    def test_heuristic_without_author_bbox(self):
        bmin, bmax, _ = _simulate_blender_import(AUTHOR_MIN, AUTHOR_MAX)
        res = compute_alignment(bmin.tolist(), bmax.tolist())
        # The thin (height) axis must be detected as 'up' and rotated onto Z.
        self.assertIn(res["mode"], ("rot_x_-90", "rot_x_+90"))
        self.assertTrue(res["needs_change"])

    def test_author_bbox_from_elements(self):
        payload = {"elements": [
            {"box_min": [0, 0, 0], "box_max": [10, 2, 3]},
            {"box_min": [-1, 5, -0.2], "box_max": [22, 14, 3.4]},
        ]}
        amin, amax = author_bbox_from_elements(payload)
        self.assertEqual(amin, [-1.0, 0.0, -0.2])
        self.assertEqual(amax, [22.0, 14.0, 3.4])

    def test_author_bbox_empty_returns_none(self):
        self.assertIsNone(author_bbox_from_elements({"elements": []}))


if __name__ == "__main__":
    unittest.main()
