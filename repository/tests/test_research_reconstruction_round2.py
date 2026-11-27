"""Round-2 synthetic-only safety and coverage tests."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from reconstruction.coverage import coverage
from reconstruction.preflight import audit
from reconstruction.visual_review import export
from reconstruction.workflow import Blocked, report_poses, sha256, score
from test_research_reconstruction_workflow import select_run, write_model


def prepare_poses(tmp_path):
    run, m = select_run(tmp_path)
    tr = [Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split'] == 'train']
    te = [Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split'] == 'test']
    write_model(run/'train_txt',tr)
    write_model(run/'all_txt',tr+te)
    report_poses(run, run/'train_txt', run/'all_txt')
    return run,m


def test_coverage_registered_angles_unscaled_only(tmp_path):
    run,m=prepare_poses(tmp_path)
    r=coverage(run)
    assert r['registered_pose_count']==len(m['frames'])
    assert r['metric_scale_verified'] is False
    assert r['angular_coverage']['room_wall_floor_ceiling_door_coverage'] is None
    assert r['angular_coverage']['max_pairwise_look_direction_deg']==pytest.approx(0)
    assert r['angular_coverage']['pair_count']>0
    assert r['camera_poses_sha256']==sha256(run/'camera_poses.json')
    with pytest.raises(Blocked):coverage(run)


def test_coverage_before_colmap_marks_angles_unknown(tmp_path):
    run,_=select_run(tmp_path)
    r=coverage(run)
    assert r['registered_pose_count']==0
    assert r['angular_coverage']['state'].startswith('not_executed')
    assert r['camera_poses_sha256'] is None


def test_coverage_rejects_mismatched_pose_identity(tmp_path):
    run,m=prepare_poses(tmp_path)
    p=run/'camera_poses.json'; j=json.loads(p.read_text()); j['poses'][0]['image_sha256']='0'*64;p.write_text(json.dumps(j))
    with pytest.raises(Blocked,match='SHA/split mismatch'):coverage(run)
    assert not (run/'coverage_report.json').exists()


def test_preflight_is_inspection_only(tmp_path):
    run,_=select_run(tmp_path)
    r=audit(run)
    assert r['run_id']==run.name and r['gs_gpu_train_executed'] is False
    assert r['dataset_manifest_sha256']==sha256(run/'dataset_manifest.json')
    assert not (run/'runtime_preflight.json').exists()


def test_verified_visual_review_creates_video_but_not_claim_geometry(tmp_path):
    run,m=prepare_poses(tmp_path)
    ckpt=tmp_path/'synthetic_ckpt.pt';ckpt.write_bytes(b'fake')
    renders=tmp_path/'renders';renders.mkdir()
    tst=next(x for x in m['frames'] if x['split']=='test')
    predicted=renders/'test.png';shutil.copy2(run/tst['image_relpath'],predicted)
    manifest={'schema_version':'mycon.reconstruction.render_manifest.v1','run_id':run.name,
              'checkpoint_path':str(ckpt),'checkpoint_sha256':sha256(ckpt),
              'camera_poses_sha256':sha256(run/'camera_poses.json'),
              'renders':[{'view_id':tst['view_id'],'image_sha256':tst['image_sha256'],
                          'render_relpath':'test.png','render_sha256':sha256(predicted)}]}
    render_manifest=renders/'render_manifest.json';render_manifest.write_text(json.dumps(manifest))
    score(run,render_manifest)
    r=export(run,render_manifest)
    assert r['view_count']==1 and (run/r['preview_video_relpath']).stat().st_size>0
    assert r['metric_scale_verified'] is False
    with pytest.raises(Blocked):export(run,render_manifest)


def test_visual_review_blocks_tampered_render(tmp_path):
    run,m=prepare_poses(tmp_path)
    (run/'per_view_metrics.csv').write_text('view_id\n')
    tst=next(x for x in m['frames'] if x['split']=='test')
    renders=tmp_path/'renders';renders.mkdir()
    p=renders/'t.png';p.write_bytes(b'tampered')
    rm=renders/'renders.json'
    rm.write_text(json.dumps({'schema_version':'mycon.reconstruction.render_manifest.v1',
            'run_id':run.name,'camera_poses_sha256':sha256(run/'camera_poses.json'),
            'renders':[{'view_id':tst['view_id'],'image_sha256':tst['image_sha256'],
                        'render_relpath':'t.png','render_sha256':'1'*64}]}))
    with pytest.raises(Blocked, match='tampered'):export(run,rm)
    assert not (run/'visual_review').exists()


def test_colmap_cpu_option_probe_supports_two_generations(monkeypatch):
    import subprocess
    from reconstruction.workflow import _colmap_cpu_options
    def fake(argv,**kwargs):
        return subprocess.CompletedProcess(argv,0,stdout='--FeatureExtraction.use_gpu' if argv[1]=='feature_extractor' else '--FeatureMatching.use_gpu',stderr='')
    monkeypatch.setattr(subprocess,'run',fake)
    assert _colmap_cpu_options('colmap')==('--FeatureExtraction.use_gpu','--FeatureMatching.use_gpu')
    def legacy(argv,**kwargs):
        return subprocess.CompletedProcess(argv,0,stdout='--SiftExtraction.use_gpu' if argv[1]=='feature_extractor' else '--SiftMatching.use_gpu',stderr='')
    monkeypatch.setattr(subprocess,'run',legacy)
    assert _colmap_cpu_options('colmap')==('--SiftExtraction.use_gpu','--SiftMatching.use_gpu')


def test_gs_dataset_blocks_before_fresh_colmap(tmp_path):
    from reconstruction.gs_experiment import prepare_gs_dataset
    run,_=select_run(tmp_path)
    with pytest.raises(Blocked,match='Fresh COLMAP'):prepare_gs_dataset(run)
    assert not (run/'gs_train_dataset').exists()


def test_gs_dataset_train_only_and_refuse_repeat(tmp_path):
    from reconstruction.gs_experiment import prepare_gs_dataset
    run,m=prepare_poses(tmp_path)
    model=run/'colmap'/'sparse'/'0';model.mkdir(parents=True)
    hashes={}
    for k in ('cameras.bin','images.bin','points3D.bin'):
        p=model/k;p.write_bytes(k.encode());hashes[k]=sha256(p)
    plan={'schema_version':'mycon.reconstruction.colmap_plan.v1','run_id':run.name,
          'status':'colmap_executed','train_model_relpath':'colmap/sparse/0','train_model_sha256':hashes}
    (run/'colmap'/'colmap_execution.json').write_text(json.dumps(plan))
    (run/'colmap'/'train_text').mkdir()
    for f in ['cameras.txt','images.txt','points3D.txt']:
        shutil.copy2(run/'train_txt'/f,run/'colmap'/'train_text'/f)
    r=prepare_gs_dataset(run)
    assert r['test_views_copied']==0
    assert r['train_view_count']==sum(x['split']=='train' for x in m['frames'])
    imgs=list((run/'gs_train_dataset'/'images').rglob('*.png'))
    assert len(imgs)==r['train_view_count']
    assert len({x['image_sha256'] for x in r['image_receipts']})==len(imgs)
    assert r['command_execution'].startswith('NOT_EXECUTED')
    with pytest.raises(Blocked,match='exists'):prepare_gs_dataset(run)
