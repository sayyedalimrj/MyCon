"""Smoke tests for ``metadata_exporter._entry``.

This is the function that exploded on directories and ``None`` in the
runtime patches; we now have a tiny test that pins down the new
behaviour so we don't regress.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

# Import the private helper for a tighter unit test
from synthetic_floor.metadata_exporter import _entry  # noqa: E402


class TestEntryHandlesAllPathTypes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_none_is_handled(self):
        out = _entry(None)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["type"], "null")

    def test_missing_file_is_handled(self):
        out = _entry(self.tmp_path / "does_not_exist.txt")
        self.assertEqual(out["type"], "missing")
        self.assertIn("path", out)

    def test_directory_is_handled(self):
        sub = self.tmp_path / "some_dir"
        sub.mkdir()
        (sub / "a.txt").write_text("a", encoding="utf-8")
        (sub / "b.txt").write_text("b", encoding="utf-8")
        out = _entry(sub)
        self.assertEqual(out["type"], "directory")
        self.assertEqual(out.get("children_count"), 2)

    def test_regular_file_is_hashed(self):
        f = self.tmp_path / "file.txt"
        f.write_text("hello", encoding="utf-8")
        out = _entry(f)
        self.assertEqual(out["type"], "file")
        self.assertIn("size_bytes", out)
        self.assertIn("sha256", out)
        self.assertEqual(out["size_bytes"], 5)
        # SHA-256 of "hello"
        self.assertTrue(out["sha256"].startswith("2cf24dba5fb0a30e"))


if __name__ == "__main__":
    unittest.main()
