"""Offline bootstrap contract tests; NEVER contact NVIDIA or allocate GPU."""
from pathlib import Path
from unittest.mock import patch
import importlib.util
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"mycon_fetch_nvidia_longsplat.py"
spec = importlib.util.spec_from_file_location("official_longsplat", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_actual_nvidia_url_and_pinned_git_revision():
    assert mod.URL == "https://github.com/NVlabs/LongSplat.git"
    assert len(mod.REV) == 40
    assert len(mod.SUBMODULES) == 4

def test_missing_source_does_not_count_as_installed(tmp_path):
    with pytest.raises(RuntimeError, match="No official"):
        mod.verify(tmp_path/"missing")

def test_fake_source_is_rejected(tmp_path):
    root = tmp_path/"source"
    (root/".git").mkdir(parents=True)
    with patch.object(mod,"git",return_value="https://github.com/fake/LongSplat.git"):
        with pytest.raises(RuntimeError, match="Unrecognized upstream"):
            mod.verify(root)

def test_wrong_revision_rejected(tmp_path):
    root=tmp_path/"source"
    (root/".git").mkdir(parents=True)
    with patch.object(mod,"git",side_effect=[mod.URL,"f"*40]):
        with pytest.raises(RuntimeError, match="Revision mismatch"):
            mod.verify(root)

def test_does_not_overwrite_an_existing_checkout(tmp_path):
    root=tmp_path/"source"
    root.mkdir()
    with patch.object(mod,"verify",return_value={"status":"SOURCE_VERIFIED_NOT_CUDA_TESTED"}), patch.object(mod,"git") as git:
        assert mod.verify(root)["status"]=="SOURCE_VERIFIED_NOT_CUDA_TESTED"
        git.assert_not_called()
