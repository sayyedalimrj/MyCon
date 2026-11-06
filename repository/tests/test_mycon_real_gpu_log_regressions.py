"""Regression for REAL T4 failure observed 2026-11-06 on saved MyCON Drive pipeline.log.

The official gsplat 1.5.3 torch2.4 CUDA12.1 wheel embeds the PEP440
local version +pt24cu121. Test must never mistake it for a broken install.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import mycon_colmap_gsplat_env as environment


class RealGpuLogRegression(unittest.TestCase):
    def test_exact_installed_wheel_version_accepted_in_cuda_smoke(self):
        source=inspect.getsource(environment.gsplat_cuda_smoke)
        self.assertIn('assert gsplat.__version__=="1.5.3+pt24cu121"',source)
        self.assertIn('assert version("gsplat")=="1.5.3+pt24cu121"',source)
        self.assertNotIn('assert gsplat.__version__=="1.5.3",',source)
        compile(source,"<gsplat_smoke_regression>","exec")

    def test_cached_verified_environment_does_not_require_18_gib_free(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            prefix=root/"gsplat_py310_torch24_cu121"
            (prefix/"bin").mkdir(parents=True)
            py=prefix/"bin/python"
            py.write_text("existing already installed python")
            receipt=root/(prefix.name+"_verified.json")
            receipt.write_text(json.dumps({
                "pinned":"gsplat-v1.5.3-torch2.4.1-cu121-t4"}))
            with (patch.object(environment.subprocess,"run",
                               return_value=SimpleNamespace(returncode=0,stdout="7.5\n")),
                  patch.object(environment.shutil,"disk_usage",
                               return_value=SimpleNamespace(free=5*1024**3)),
                  patch.object(environment,"gsplat_cuda_smoke") as smoke,
                  patch.object(environment,"run_logged",
                               side_effect=AssertionError("NO_INSTALL_EXPECTED"))):
                result=environment.setup(prefix,root,root/"gsplat",root/"pipeline.log")
            self.assertEqual(result,py)
            smoke.assert_called_once()
            self.assertTrue(py.is_file())

    def test_new_environment_fails_before_install_if_disk_too_low(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            prefix=root/"new_cuda_env"
            with (patch.object(environment.subprocess,"run",
                               return_value=SimpleNamespace(returncode=0,stdout="7.5\n")),
                  patch.object(environment.shutil,"disk_usage",
                               return_value=SimpleNamespace(free=5*1024**3)),
                  patch.object(environment,"run_logged",
                               side_effect=AssertionError("MUST_FAIL_BEFORE_INSTALL"))):
                with self.assertRaisesRegex(RuntimeError,
                                            "GPU_NEW_INSTALL_LOCAL_DISK_LT_18_GIB"):
                    environment.setup(prefix,root,root/"gsplat",root/"pipeline.log")
            self.assertFalse(prefix.exists())


if __name__=="__main__":
    unittest.main()
