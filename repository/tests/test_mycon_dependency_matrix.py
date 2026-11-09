"""Evidence-based offline ABI guards for NVIDIA LongSplat (no GPU training)."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/"scripts"))
import mycon_gpu_autoprofile as profile
import mycon_longsplat_autoenv as env
import mycon_thesis_one_run as one


class LongSplatSupportedABI(unittest.TestCase):
    def hw(self,sm=75):
        return profile.Hardware(
            "T4" if sm==75 else "L4", 15109 if sm==75 else 23000,
            sm, "570.00", "12.1", 40., 60.)

    def test_supported_t4_l4_profiles_match_official_pytorch3d_v078(self):
        for sm in (75,89):
            data=profile.profile_for(self.hw(sm))
            self.assertEqual(data["torch_version"],"2.4.1")
            self.assertEqual(data["torchvision_version"],"0.19.1")
            self.assertEqual(data["torch_cuda_wheel"],"cu121")
            self.assertFalse(data["native_longsplat_gpu_training_verified"])

    def test_unsupported_old_251_rejected_before_any_package_download(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/"LongSplat"
            (source/"submodules/mast3r").mkdir(parents=True)
            (source/"requirements.txt").write_text("torch\n")
            old=profile.profile_for(self.hw())
            old.update(torch_version="2.5.1",torchvision_version="0.20.1")
            with patch.object(env,"run_logged",side_effect=AssertionError("PIP_CALLED")):
                with self.assertRaisesRegex(RuntimeError,
                                            "UNSUPPORTED_TORCH_ABI"):
                    env.install(old,source,root/"install",root/"build.log")
            self.assertFalse((root/"install").exists())

    def test_new_abi_creates_separate_prefix_preserving_old_install(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            old=root/"mycon_stabilized_sm75_cu121"
            (old/"bin").mkdir(parents=True)
            (old/"bin/python").write_bytes(b"old torch2.5.1 environment")
            actual=profile.profile_for(self.hw())
            with (patch.object(one,"KNOWN_ENV",root/"other"),
                  patch.object(one,"verify_cuda",side_effect=RuntimeError("old ABI"))):
                new_path,status=one.choose_env(root,actual,root/"logs")
            self.assertEqual(status,"NEW_ISOLATED_ENV_NO_OLD_DELETION")
            self.assertEqual(new_path.name,"mycon_stabilized_sm75_cu121_pt241")
            self.assertTrue((old/"bin/python").exists())
            self.assertFalse(new_path.exists())


if __name__=="__main__":
    unittest.main()
