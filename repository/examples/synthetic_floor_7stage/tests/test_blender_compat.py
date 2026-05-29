"""Smoke tests for the Blender 4.x compatibility shims.

These tests exercise only the host-side helpers (no ``bpy`` required).
The bpy-dependent helpers are smoke-tested implicitly by the GPU
notebook in ``colab/synthetic_floor_blender_gpu.ipynb``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor import blender_compat as compat  # noqa: E402


class TestSafeColorMode(unittest.TestCase):
    """The BW->RGB regression that broke Blender 4.2 must stay fixed."""

    def test_bw_on_exr_is_rejected(self):
        self.assertEqual(compat.safe_color_mode("BW", "OPEN_EXR"), "RGB")
        self.assertEqual(compat.safe_color_mode("BW", "OPEN_EXR_MULTILAYER"), "RGB")

    def test_bw_on_png_is_kept(self):
        # PNG/JPEG/etc. legitimately accept BW
        self.assertEqual(compat.safe_color_mode("BW", "PNG"), "BW")
        self.assertEqual(compat.safe_color_mode("BW", "JPEG"), "BW")
        self.assertEqual(compat.safe_color_mode("BW", "TIFF"), "BW")

    def test_rgb_passthrough(self):
        self.assertEqual(compat.safe_color_mode("RGB", "OPEN_EXR"), "RGB")
        self.assertEqual(compat.safe_color_mode("RGBA", "PNG"), "RGBA")
        self.assertEqual(compat.safe_color_mode("RGB", "PNG"), "RGB")

    def test_unknown_mode_falls_back_to_rgb(self):
        self.assertEqual(compat.safe_color_mode("XYZ", "PNG"), "RGB")
        self.assertEqual(compat.safe_color_mode("", "OPEN_EXR"), "RGB")
        self.assertEqual(compat.safe_color_mode(None, "PNG"), "RGB")  # type: ignore[arg-type]

    def test_unknown_format_defaults_safely(self):
        self.assertEqual(compat.safe_color_mode("RGB", "AVIF"), "RGB")
        self.assertEqual(compat.safe_color_mode("BW", "AVIF"), "RGB")  # safe fallback


class TestVersionHelpers(unittest.TestCase):
    def test_version_outside_blender(self):
        # Without bpy installed (host side), version is (0, 0, 0)
        v = compat.blender_version()
        self.assertIsInstance(v, tuple)
        self.assertEqual(len(v), 3)
        # Either (0,0,0) on host or (>=4,...) inside Blender 4.x
        if v == (0, 0, 0):
            self.assertFalse(compat.is_blender_4_or_newer())


class TestPrincipledAliases(unittest.TestCase):
    def test_aliases_are_complete(self):
        for logical_name in ("base_color", "roughness", "metallic", "ior",
                              "transmission", "specular", "alpha", "normal"):
            self.assertIn(logical_name, compat.PRINCIPLED_SOCKET_ALIASES)
            self.assertGreater(len(compat.PRINCIPLED_SOCKET_ALIASES[logical_name]), 0)

    def test_transmission_includes_both_names(self):
        # The whole point of this map is to handle the rename.
        names = compat.PRINCIPLED_SOCKET_ALIASES["transmission"]
        self.assertIn("Transmission Weight", names)  # 4.x
        self.assertIn("Transmission", names)  # 3.x

    def test_specular_includes_both_names(self):
        names = compat.PRINCIPLED_SOCKET_ALIASES["specular"]
        self.assertIn("Specular IOR Level", names)  # 4.x
        self.assertIn("Specular", names)  # 3.x


class TestFallbackChain(unittest.TestCase):
    def test_optix_chain_includes_cpu(self):
        chain = compat.GPU_BACKEND_FALLBACKS["OPTIX"]
        self.assertIn("OPTIX", chain)
        self.assertIn("CUDA", chain)
        self.assertIn("CPU", chain)
        # CPU must always be the last resort
        self.assertEqual(chain[-1], "CPU")


if __name__ == "__main__":
    unittest.main()
