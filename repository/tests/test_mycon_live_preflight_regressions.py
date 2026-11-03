"""Regressions from the actual 2026-11-03 Colab failures. No training/network."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mycon_fetch_nvidia_longsplat as fetch
import mycon_longsplat_autoenv as env
from test_longsplat_research_backend import prepare,bridge


def test_git_keeps_clean_submodule_prefix():
    clean=' '+ 'a'*40+' submodules/example (heads/main)\n'
    with patch.object(fetch.subprocess,'run',return_value=subprocess.CompletedProcess([],0,clean,'')):
        assert fetch._git('submodule','status').startswith(' ')
        assert not fetch._git('submodule','status').endswith('\n')


@pytest.mark.parametrize('version,needs_install',[('13.0',True),('12.1',False),('12.4',True)])
def test_toolkit_matches_torch_even_if_host_is_newer(tmp_path,version,needs_install):
    source=tmp_path/'source';(source/'submodules/mast3r').mkdir(parents=True)
    (source/'requirements.txt').write_text('')
    profile={'torch_version':'2.5.1','torchvision_version':'0.20.1','torch_cuda_wheel':'cu121',
             'torch_cuda_arch':'7.5','gpu':{'sm':75}}
    with patch.object(env,'micromamba',return_value=tmp_path/'micromamba'), patch.object(env,'nvcc_info',return_value=(version,tmp_path/'host')), patch.object(env,'run_logged') as run, patch.object(env,'verify_cuda'):
        env.install(profile,source,tmp_path/'newenv',tmp_path/'log')
    toolkit=[c.args[0] for c in run.call_args_list if 'cuda-toolkit=12.1' in c.args[0]]
    assert bool(toolkit)==needs_install


def test_new_training_needs_explicit_budget_approval(tmp_path):
    a=prepare(tmp_path);a.phase='train'
    with patch.object(bridge,'invoke') as trainer:
        with pytest.raises(RuntimeError,match='NEW_TRAINING_APPROVAL_REQUIRED'):
            bridge.run(a)
        trainer.assert_not_called()
    experiment=a.workspace/bridge.capture_key(a.video,bridge.digest_file(a.video))
    assert not (experiment/'longsplat_model').exists()
