"""Offline, no GPU or network: tests for user-reported restart failures."""
from pathlib import Path
from unittest.mock import patch
import json
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from pipeline.stage_04_5_cams_gs import longsplat_research as bridge
import mycon_longsplat_autoenv as installer
from test_longsplat_research_backend import prepare


def test_partial_derived_frame_extraction_recovers_without_reprocessing_other_stages(tmp_path):
    from test_longsplat_research_backend import fixture,make_frames
    a=fixture(tmp_path);a.phase='prepare';a.accept_research_license=True
    experiment=a.workspace/bridge.capture_key(a.video,bridge.digest_file(a.video))
    images=experiment/'dataset'/'images';images.mkdir(parents=True)
    (images/'frame_000001.jpg').write_bytes(b'partial\n')
    with patch.object(bridge,'invoke',side_effect=make_frames), patch.object(bridge.shutil,'which',return_value='/usr/bin/ffmpeg'):
        assert bridge.run(a)==0
    backups=list(images.parent.glob('interrupted_images_*'))
    assert len(backups)==1 and (backups[0]/'frame_000001.jpg').read_bytes()==b'partial\n'
    assert len(list(images.glob('frame_*.jpg')))==35


def test_does_not_accept_old_incomplete_training_checkpoint(tmp_path):
    a=prepare(tmp_path);a.phase='train'
    experiment=a.workspace/bridge.capture_key(a.video,bridge.digest_file(a.video))
    model=experiment/'longsplat_model'
    checkpoint=model/'point_cloud/iteration_2000';checkpoint.mkdir(parents=True)
    (checkpoint/'point_cloud.ply').write_bytes(b'p'*1024)
    (model/'cameras_all_train.json').write_text('[{}]'*100)
    with pytest.raises(RuntimeError,match='PARTIAL_TRAIN_NO_OFFICIAL_MIDRUN_RESUME'):
        bridge.run(a)
    assert (checkpoint/'point_cloud.ply').exists()


def test_completed_training_without_receipt_can_be_recovered(tmp_path):
    a=prepare(tmp_path);a.phase='train'
    experiment=a.workspace/bridge.capture_key(a.video,bridge.digest_file(a.video))
    model=experiment/'longsplat_model'
    checkpoint=model/'point_cloud/iteration_50000';checkpoint.mkdir(parents=True)
    (checkpoint/'point_cloud.ply').write_bytes(b'p'*4096)
    (model/'cameras_all_train.json').write_text('[{}]'*100)
    with patch.object(bridge,'invoke') as invoke:
        assert bridge.run(a)==0
        invoke.assert_not_called()
    state=json.loads((experiment/'research_state.json').read_text())
    assert state['trained'] is True and state['recovered_final_model'] is True


def test_partial_cuda_env_is_preserved(tmp_path):
    root=tmp_path/'env'; (root/'lib').mkdir(parents=True)
    artifact=root/'lib'/'broken.so';artifact.write_bytes(b'preserve')
    source=tmp_path/'NVIDIA';(source/'submodules/mast3r').mkdir(parents=True)
    (source/'requirements.txt').write_text('')
    profile={'torch_version':'2.5.1','torchvision_version':'0.20.1','torch_cuda_wheel':'cu121',
             'torch_cuda_arch':'7.5','gpu':{'sm':75}}
    with patch.object(installer,'micromamba') as download:
        with pytest.raises(RuntimeError,match='PARTIAL_ENV_PRESERVED'):
            installer.install(profile,source,root,tmp_path/'log')
        download.assert_not_called()
    assert artifact.read_bytes()==b'preserve'
