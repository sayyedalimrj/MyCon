"""Prevent an actual Colab MPLBACKEND=inline leak into isolated TorchMetrics.

This reproduces the real v7 Runtime failure in setup.py and checks subprocess
environments without requiring a GPU, runtime downloads or a host restart.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import mycon_colmap_gsplat_env as colmap_env
import mycon_longsplat_autoenv as nvidia_env

BROKEN_COLAB_BACKEND = "module://matplotlib_inline.backend_inline"


class ColabBackendIsolation(unittest.TestCase):
    def test_gsplat_nonlive_worker_uses_agg_but_notebook_env_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            supplied={"MPLBACKEND":BROKEN_COLAB_BACKEND,"EXPERIMENT":"preserve"}
            with patch.dict(os.environ,{"MPLBACKEND":BROKEN_COLAB_BACKEND}):
                with patch.object(colmap_env.subprocess,"run",
                                  return_value=SimpleNamespace(returncode=0)) as invoked:
                    colmap_env.run_logged(["python","-c","pass"],Path(d)/"log.txt",env=supplied)
                passed=invoked.call_args.kwargs["env"]
                self.assertEqual(passed["MPLBACKEND"],"Agg")
                self.assertEqual(passed["EXPERIMENT"],"preserve")
                self.assertEqual(supplied["MPLBACKEND"],BROKEN_COLAB_BACKEND)
                self.assertEqual(os.environ["MPLBACKEND"],BROKEN_COLAB_BACKEND)

    def test_gsplat_implicit_env_uses_agg(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ,{"MPLBACKEND":BROKEN_COLAB_BACKEND}):
                with patch.object(colmap_env.subprocess,"run",
                                  return_value=SimpleNamespace(returncode=0)) as invoked:
                    colmap_env.run_logged(["python","-c","pass"],Path(d)/"log.txt")
                self.assertEqual(invoked.call_args.kwargs["env"]["MPLBACKEND"],"Agg")

    def test_longsplat_installer_child_uses_agg(self):
        with tempfile.TemporaryDirectory() as d:
            supplied={"MPLBACKEND":BROKEN_COLAB_BACKEND}
            with patch.object(nvidia_env.subprocess,"run",
                              return_value=SimpleNamespace(returncode=0)) as invoked:
                nvidia_env.run_logged(["python","-c","pass"],Path(d)/"log.txt",env=supplied)
            self.assertEqual(invoked.call_args.kwargs["env"]["MPLBACKEND"],"Agg")
            self.assertEqual(supplied["MPLBACKEND"],BROKEN_COLAB_BACKEND)


if __name__ == "__main__":
    unittest.main()
