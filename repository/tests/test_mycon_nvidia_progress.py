"""Unit tests: NVIDIA stdout is saved and original errors are visible to Colab."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.stage_04_5_cams_gs.longsplat_research import invoke


class TestNvidiaProgress(unittest.TestCase):
    def test_upstream_output_persists_without_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            log = folder / "nvidia.log"
            invoke([sys.executable, "-c", "print('NVIDIA_PREPARED')"],
                   cwd=folder, log=log)
            body = log.read_text()
            self.assertIn("NVIDIA_PREPARED", body)
            self.assertIn("===", body)

    def test_upstream_failure_retains_source_error(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            log = folder / "nvidia.log"
            with self.assertRaisesRegex(RuntimeError, "NVIDIA_ACTUAL_ERROR"):
                invoke([sys.executable, "-c",
                        "import sys;print('NVIDIA_ACTUAL_ERROR');sys.exit(7)"],
                       cwd=folder, log=log)
            self.assertIn("NVIDIA_ACTUAL_ERROR", log.read_text())


if __name__ == "__main__":
    unittest.main()
