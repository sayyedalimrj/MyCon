"""Synthetic-only tests; never mistaken for a real-site/GPU validation."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reconstruction.workflow import (Blocked, checked_run, compare_6999, parse_colmap_text, receipt,
                                      prepare, report_poses, run_colmap, score, sha256, train_plan)


def make_video(path, *, seed=1):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (128,96))
    assert writer.isOpened()
    for i in range(90):
        img = rng.integers(32,220,size=(96,128,3),dtype=np.uint8)
        cv2.rectangle(img,(i % 55,12),(60+(i%55),40),(0,155,200),2)
        writer.write(img)
    writer.release()


def select_run(tmp_path, cross=True):
    a = tmp_path/'train.avi'; make_video(a,seed=11)
    b = tmp_path/'test.avi'
    if cross: make_video(b,seed=12)
    out = tmp_path/'runs'; out.mkdir()
    res = prepare(run_root=out,run_id='recon-test-20261113', train_video=a,train_session='session_A',
                  test_video=b if cross else None,test_session='session_B' if cross else None,
                  sample_hz=5,max_frames=35,min_blur=5,min_mean=15,max_mean=240,max_clip=.9)
    return out/'recon-test-20261113', res


def write_model(path, names):
    path.mkdir(parents=True)
    (path/'cameras.txt').write_text('# Number of cameras: 1\n1 PINHOLE 128 96 90 90 64 48\n')
    lines=['# images records']
    for j, name in enumerate(names,1):
        lines.append(f'{j} 1 0 0 0 0 0 0 1 {name}')
        lines.append('20 20 10 25 25 -1')
    (path/'images.txt').write_text('\n'.join(lines)+'\n')
    (path/'points3D.txt').write_text('# comment\n1 0 0 0 255 255 255 0.3 1 0\n')


def test_selected_manifest_has_genuine_disjoint_sessions_and_hashes(tmp_path):
    run, m = select_run(tmp_path)
    assert m['counts']['train']>=5 and m['counts']['test']>=2
    assert m['train_test_protocol']=='capture_session_heldout'
    assert {x['capture_session_id'] for x in m['frames'] if x['split']=='train'}=={'session_A'}
    assert {x['capture_session_id'] for x in m['frames'] if x['split']=='test'}=={'session_B'}
    assert checked_run(run)['run_id']==run.name
    assert len({x['image_sha256'] for x in m['frames']})==len(m['frames'])
    with pytest.raises(FileExistsError):
        prepare(run_root=tmp_path/'runs',run_id=run.name,train_video=tmp_path/'train.avi',train_session='session_A',
                test_video=tmp_path/'test.avi',test_session='session_B',sample_hz=5,max_frames=35,
                min_blur=5,min_mean=15,max_mean=240,max_clip=.9)


def test_heldout_within_capture_is_explicitly_exploratory(tmp_path):
    run, m=select_run(tmp_path, cross=False)
    assert 'exploratory' in m['train_test_protocol']
    train_times=[x['timestamp_sec'] for x in m['frames'] if x['split']=='train']
    test_times=[x['timestamp_sec'] for x in m['frames'] if x['split']=='test']
    assert all(abs(a-b)>1 for a in train_times for b in test_times)


def test_bad_input_and_tampered_image_fail_closed(tmp_path):
    run,m=select_run(tmp_path)
    f=run/m['frames'][0]['image_relpath']
    f.write_bytes(b'bad')
    with pytest.raises(Blocked,match='SHA mismatch'):
        checked_run(run)


def test_report_poses_includes_all_selected_views(tmp_path):
    run, m=select_run(tmp_path)
    tr=[Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split']=='train']
    te=[Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split']=='test']
    train_txt, all_txt=run/'train_text',run/'all_text'
    write_model(train_txt,tr)
    write_model(all_txt,tr+te[:2])
    result=report_poses(run,train_txt,all_txt)
    assert result['registered_train']==len(tr)
    assert result['registered_test']==2
    entries=json.loads((run/'camera_poses.json').read_text())['poses']
    assert len(entries)==len(m['frames'])
    assert all(p['metric_scale_verified'] is False for p in entries)
    assert entries[0]['camera']['model']=='PINHOLE'
    assert np.allclose(np.array(entries[0]['world_to_camera_4x4']),np.eye(4))
    assert result['per_view'][0]['reprojection_error_px'] is None


def test_mapping_test_leakage_is_rejected(tmp_path):
    run,m=select_run(tmp_path)
    tr=[Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split']=='train']
    te=[Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split']=='test']
    train_txt, all_txt=run/'train_text',run/'all_text'
    write_model(train_txt,tr+te[:1]);write_model(all_txt,tr+te)
    with pytest.raises(Blocked,match='TEST LEAKAGE'):
        report_poses(run,train_txt,all_txt)
    assert not (run/'camera_poses.json').exists()


def test_holdout_scores_are_matched_by_id_and_hash(tmp_path):
    run,m=select_run(tmp_path)
    tr=[Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split']=='train']
    te=[Path(x['image_relpath']).relative_to('images').as_posix() for x in m['frames'] if x['split']=='test']
    write_model(run/'train_text',tr);write_model(run/'all_text',tr+te)
    report_poses(run,run/'train_text',run/'all_text')
    ckpt=tmp_path/'ckpt_fresh.pt';ckpt.write_bytes(b'synthetic_test_not_actual_checkpoint')
    rend=tmp_path/'renders';rend.mkdir()
    test=m['frames'][0]
    test=next(x for x in m['frames'] if x['split']=='test')
    pred=rend/'view.png';pred.write_bytes((run/test['image_relpath']).read_bytes())
    config={"schema_version":"mycon.reconstruction.render_manifest.v1","run_id":run.name,
            "checkpoint_path":str(ckpt),"checkpoint_sha256":sha256(ckpt),
            "camera_poses_sha256":sha256(run/"camera_poses.json"),
            "renders":[{"view_id":test['view_id'],"image_sha256":test['image_sha256'],
                        "render_relpath":"view.png","render_sha256":sha256(pred)}]}
    (rend/'render_manifest.json').write_text(json.dumps(config))
    result=score(run,rend/'render_manifest.json')
    assert result['evaluated_views']==1
    assert result['psnr_mean_db'] is None # Identical image => infinite PSNR encoded as null, not fabricated
    assert result['ssim_mean']==pytest.approx(1)
    assert result['metric_scale_verified'] is False
    with (run/'per_view_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert rows[0]['split']=='test' and rows[0]['psnr_state']=='identical_inf'


def test_comparison_6999_not_claimed_comparable(tmp_path):
    run,_=select_run(tmp_path)
    r=compare_6999(run)
    assert r['comparable'] is False and r['delta'] is None
    assert r['historical_dataset_and_split_verified'] is False


def test_plan_gs_never_trains_without_opt_in(tmp_path):
    run,_=select_run(tmp_path)
    r=train_plan(run)
    assert r['trained_checkpoint'] is None and r['training_status'].startswith('not_executed')
    assert not (run/'colmap').exists()
    assert run_colmap(run,'colmap')['status']=='not_executed'


def test_colmap_parser_refuses_invalid_quaternion(tmp_path):
    md=tmp_path/'bad';write_model(md,['test.png'])
    p=md/'images.txt';p.write_text(p.read_text().replace('1 0 0 0 0 0 0 1','0 0 0 0 0 0 0 1'))
    with pytest.raises(Blocked,match='quaternion'):
        parse_colmap_text(md)


def test_create_only_sha_receipt(tmp_path):
    run,_ = select_run(tmp_path)
    r = receipt(run)
    assert r['file_count'] >= 3
    rows = json.loads((run/'artifact_receipt.json').read_text())['artifacts']
    assert any(x['relative_path']=='dataset_manifest.json' for x in rows)
    assert all(len(x['sha256'])==64 for x in rows)
    with pytest.raises(FileExistsError): receipt(run)
