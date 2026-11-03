"""Deterministic/offline smoke for the new Colab hardware and capture contracts."""
from __future__ import annotations
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
import mycon_gpu_autoprofile as gpu
import mycon_capture_safe as capture
import mycon_longsplat_auto_run as runner
from pipeline.stage_04_5_cams_gs import longsplat_research as bridge


def hardware(model='T4',vram=15109,sm=75,nvcc='12.4',disk=110):
    return gpu.Hardware(model,vram,sm,'570.00',nvcc,30.,disk)

@pytest.mark.parametrize('name,mb,sm,expected_fps,expected_width,wheel',[
    ('Tesla T4',15109,75,8,512,'cu121'),
    ('NVIDIA L4',23034,89,10,640,'cu121'),
    ('A100-SXM4-40GB',40960,80,10,768,'cu121'),
    ('A100-SXM4-80GB',81920,80,12,896,'cu121'),
    ('H100 80GB',81920,90,12,896,'cu121'),
    ('NVIDIA RTX PRO 6000 Blackwell',96000,120,12,896,'cu128'),
])
def test_profiles_cover_colab_gpu_options(name,mb,sm,expected_fps,expected_width,wheel):
    nvcc='12.8' if sm>=100 else '12.4'
    p=gpu.profile_for(hardware(name,mb,sm,nvcc))
    assert p['fps']==expected_fps and p['width']==expected_width
    assert p['torch_cuda_wheel']==wheel
    assert not p['metric_accuracy_guaranteed']


def test_blackwell_prepares_toolkit_128_without_lie():
    p=gpu.profile_for(hardware('G4',96000,120,'12.4',disk=50))
    assert p['needs_local_cuda_12_8_install']
    with pytest.raises(RuntimeError,match='30 GiB'):
        gpu.profile_for(hardware('G4',96000,120,'12.4',disk=27))


def test_no_gpu_or_tiny_gpu_is_refused():
    with pytest.raises(RuntimeError,match='BELOW_RESEARCH_MINIMUM'):
        gpu.profile_for(hardware('P100',16000,60))


def test_changes_to_gpu_parameters_create_separate_experiments():
    t=gpu.profile_for(hardware());a=gpu.profile_for(hardware('A100',40960,80))
    assert runner.profile_key(t)!=runner.profile_key(a)
    assert runner.profile_key(t)==runner.profile_key(dict(t))


def build_recorder_zip(path: Path,wrong_hash=False,unsafe=False):
    vid=b'A'*2_000_100
    rows={
        'arcore_recording.mp4':vid,
        'pose.csv':b'timestamp_ns,tx_m,ty_m,tz_m\n100,0,0,0\n',
        'imu.csv':b'timestamp_ns,ax,ay,az\n100,1,0,0\n',
        'session.json':json.dumps({'video_dataset':'arcore_recording.mp4','app_version':'1.2.1'}).encode(),
        'qr_events.jsonl':b'',
        'r4_camera.json':b'{"schema":"mycon.r4.camera.v1"}',
        'r4_controls.json':b'{"schema":"mycon.r4.capture_controls.v1"}',
        'r4_compatibility.json':b'{"schema":"mycon.r4.capture_compatibility.v1"}',
        'R4_IMPORT_README.txt':b'Native Recorder capture',
        'tools/mycon_r4_bridge.py':b'# bridge included in native package\n',
    }
    manifest={'schema':'mycon.capture.integrity.v1','files':[{'path':k,'hash_policy':'SHA256',
        'size_bytes':len(v),'sha256':('0'*64 if wrong_hash and k=='arcore_recording.mp4' else hashlib.sha256(v).hexdigest())} for k,v in rows.items()]}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for n,v in rows.items():z.writestr('Capture/'+n,v)
        z.writestr('Capture/integrity_sha256.json',json.dumps(manifest))
        if unsafe:z.writestr('../bad.txt','evil')
    return rows


def test_original_recorder_zip_manifest_paths_and_sha(tmp_path):
    path=tmp_path/'original.zip';build_recorder_zip(path)
    with zipfile.ZipFile(path) as z:
        prefix,infos,rows,record=capture.source_members(z)
        assert record['video_name']=='arcore_recording.mp4'
        assert 'pose.csv' in rows
        dest=tmp_path/'video.mp4'
        h=capture._extract_verified(z,record['video_name'],infos[record['video_full']],rows[record['video_name']],dest)
        assert h==hashlib.sha256(dest.read_bytes()).hexdigest()
        assert dest.stat().st_size>2_000_000


def test_tampered_original_zip_refused(tmp_path):
    path=tmp_path/'bad.zip';build_recorder_zip(path,wrong_hash=True)
    with zipfile.ZipFile(path) as z:
        _,infos,rows,record=capture.source_members(z)
        with pytest.raises(RuntimeError,match='SHA256'):
            capture._extract_verified(z,record['video_name'],infos[record['video_full']],rows[record['video_name']],tmp_path/'bad.mp4')
        assert not (tmp_path/'bad.mp4').exists()


def test_zip_path_traversal_rejected(tmp_path):
    path=tmp_path/'unsafe.zip';build_recorder_zip(path,unsafe=True)
    with zipfile.ZipFile(path) as z:
        with pytest.raises(ValueError,match='traversal'):
            capture.source_members(z)


def test_frame_rotation_explicit_and_no_double_autorotate(tmp_path):
    args=dict(video=tmp_path/'raw.mp4',source=tmp_path/'nvidia',scene=tmp_path/'scene',
              model=tmp_path/'model',fps=10.,width=640,python='python',prune_ratio=.6,post_iter=20000)
    row=bridge.commands(**args,rotation_cw=90)['prepare']
    assert '-noautorotate' in row
    assert 'transpose=clock' in row[row.index('-vf')+1]
    assert 'fps=10' in row[row.index('-vf')+1]
    assert 'scale=640:-2' in row[row.index('-vf')+1]
    assert bridge.commands(**args,rotation_cw=0)['prepare'][0]=='ffmpeg'


def test_parameter_guard_preserves_nonmatching_stage_settings():
    a={'schema':bridge.VERSION,'video_sha256':'a','fps':10,'width':512,'post_iter':20000,'prune_ratio':.6,'rotation_cw':90}
    bridge.require_matching_params(a,dict(a))
    with pytest.raises(RuntimeError,match='different rotation_cw'):
        bridge.require_matching_params(a,{**a,'rotation_cw':0})


def test_training_resume_not_faked(tmp_path):
    assert not runner.profile_for(hardware())['metric_accuracy_guaranteed']
    assert bridge.outputs(tmp_path/'missing')['iteration'] is None


def test_environment_install_errors_never_pass_as_completed():
    from mycon_longsplat_autoenv import install
    with pytest.raises(RuntimeError,match='OFFICIAL_NVIDIA_SOURCE_NOT_VERIFIED'):
        install({},Path('/does-not-exist'),Path('/tmp/nonexistent_mycon_cuda'),Path('/tmp/mycon_test_cuda.log'))


def test_deferred_large_video_policy_computes_sha256_when_unpacked(tmp_path):
    path=tmp_path/'deferred.zip'
    rows=build_recorder_zip(path)
    with zipfile.ZipFile(path) as z:
        contents={x.filename:z.read(x.filename) for x in z.infolist()}
    key='Capture/integrity_sha256.json'
    manifest=json.loads(contents[key])
    entry=next(x for x in manifest['files'] if x['path']=='arcore_recording.mp4')
    entry.update(hash_policy='DEFERRED_LARGE_BINARY',sha256=None)
    contents[key]=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w') as z:
        for name,body in contents.items():z.writestr(name,body)
    with zipfile.ZipFile(path) as z:
        _,infos,rows,record=capture.source_members(z)
        out=tmp_path/'native.mp4'
        computed=capture._extract_verified(z,record['video_name'],infos[record['video_full']],rows[record['video_name']],out)
        assert computed==hashlib.sha256(out.read_bytes()).hexdigest()

