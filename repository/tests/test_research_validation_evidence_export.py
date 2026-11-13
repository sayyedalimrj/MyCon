"""Synthetic-only correctness tests, NEVER publication-grade experimental outcomes."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.mycon_2026.evidence_export import (
    SCHEMA, digest, secure_path, leakage_guard, score_geometry,
    score_corners, score_planes, score_image, score_reprojection,
    export, _site_ci)


def test_3d_identity_and_tolerance():
    a=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
    m=score_geometry(a,a,.01)
    assert m['accuracy_m']==0 and m['completeness_m']==0
    assert m['chamfer_l1_m']==0 and m['f1_at_0.01m']==1


def test_3d_noise_decreases_f1():
    gt=np.array([[0.,0,0],[1,0,0]])
    pr=gt+np.array([0,.04,0])
    assert score_geometry(pr,gt,.01)['f1_at_0.01m']==0
    assert score_geometry(pr,gt,.05)['f1_at_0.05m']==1


def test_point_to_plane():
    gt=np.array([[0.,0.,0.,0,0,1],[1.,0.,0.,0,0,1.]])
    pred=np.array([[0,0,.02],[1,0,.02]])
    assert score_geometry(pred,gt,.05)['point_to_plane_mae_m']==pytest.approx(.02)


def test_no_leakage_site():
    a=dict(case_id='a',site_id='X',building_id='b1',session_id='s1',method='a',variant='v',split='train')
    b=dict(a,case_id='b',building_id='b2',session_id='s2',split='test')
    with pytest.raises(ValueError,match='leakage'):leakage_guard([a,b])


def test_no_leakage_capture():
    a=dict(case_id='a',site_id='A',building_id='b',session_id='s',split='train',method='x',capture_sha256='a'*64)
    b=dict(a,case_id='b',site_id='B',split='test')
    with pytest.raises(ValueError,match='leakage'):leakage_guard([a,b])


def test_safe_path_sha_and_traversal(tmp_path):
    (tmp_path/'x.txt').write_text('a')
    ref={'path':'x.txt','sha256':digest(tmp_path/'x.txt')}
    assert secure_path(tmp_path,ref)==tmp_path/'x.txt'
    with pytest.raises(ValueError,match='sha256'):secure_path(tmp_path,dict(ref,sha256='0'*64))
    with pytest.raises(ValueError,match='unsafe'):secure_path(tmp_path,dict(ref,path='../x.txt'))


def test_corner_extrapolation_not_claimed(tmp_path):
    def write(name,rows):
        with (tmp_path/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['feature_id','x','y','z','state']);w.writeheader();w.writerows(rows)
    write('p.csv',[dict(feature_id='p0',x=0,y=0,z=0,state='candidate_extrapolated'),dict(feature_id='p1',x=1,y=1,z=1,state='observed_supported')]);write('g.csv',[dict(feature_id='g0',x=1,y=1,z=1,state='observed_supported')])
    out=score_corners(tmp_path/'p.csv',tmp_path/'g.csv',.01)
    assert out['corner_precision']==1 and out['corner_recall']==1 and out['corner_mae_m']==0


def test_plane_finite_review_required(tmp_path):
    fields=['feature_id','nx','ny','nz','finite_support','reviewed']
    for name,nx in [('p.csv','1'),('g.csv','0')]:
        with (tmp_path/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(feature_id='wall',nx=nx,ny=1 if nx=='0' else 0,nz=0,finite_support='true',reviewed='true'))
    assert score_planes(tmp_path/'p.csv',tmp_path/'g.csv')['plane_angular_mae_deg']==pytest.approx(90)


def test_heldout_site_ci_not_frames():
    row=dict(status='evaluated',method='a',variant='old',metric='ssim',unit='fraction')
    one=_site_ci([dict(row,site_id='A',value=.8),dict(row,site_id='A',value=.9)],'validation-c-test')
    assert one[0]['status']=='not_evaluated' and one[0]['ci95_lower'] is None and one[0]['n_sites']==1
    two=_site_ci([dict(row,site_id='A',value=.8),dict(row,site_id='B',value=.9)],'validation-c-test',draws=200)
    assert two[0]['status']=='evaluated' and .8<=two[0]['ci95_lower']<=two[0]['ci95_upper']<=.9


def test_image_metrics_identical(tmp_path):
    from PIL import Image
    im=Image.new('RGB',(16,16),(30,80,120));im.save(tmp_path/'a.png')
    scores=score_image(tmp_path/'a.png',tmp_path/'a.png')
    assert scores['psnr_db'] is None and scores['ssim']==pytest.approx(1)


def test_reprojection_coverage_and_rmse(tmp_path):
    with (tmp_path/'r.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['registered','u_gt','v_gt','u_pred','v_pred']);w.writeheader();w.writerow(dict(registered='true',u_gt=0,v_gt=0,u_pred=3,v_pred=4));w.writerow(dict(registered='false',u_gt='',v_gt='',u_pred='',v_pred=''))
    scores=score_reprojection(tmp_path/'r.csv')
    assert scores['reprojection_rmse_px']==pytest.approx(5)
    assert scores['registered_camera_coverage']==pytest.approx(.5)


def test_no_data_exports_unmeasured_and_immutable(tmp_path):
    root=tmp_path/'incoming';root.mkdir();m={'schema_version':SCHEMA,'run_id':'validation-c-synthetic-empty','cases':[]}
    (root/'manifest.json').write_text(json.dumps(m)); before=digest(root/'manifest.json')
    out=export(root/'manifest.json',tmp_path/'results')
    assert digest(root/'manifest.json')==before
    assert json.loads((out/'experiment_registry.json').read_text())['scientific_evaluated_experiments']==0
    assert json.loads((out/'reproducibility_manifest.json').read_text())['n_real_metric_cells']==0
    assert (out/'methodology_flow.png').exists()
    for key in ('per_view_metrics.csv','geometry_metrics.csv','ablation_results.csv','confidence_intervals.csv','failure_cases.csv'):
        assert len((out/key).read_text().splitlines())==1
    with pytest.raises(FileExistsError):export(root/'manifest.json',tmp_path/'results')


def test_synthetic_case_never_scientific(tmp_path):
    root=tmp_path/'incoming';root.mkdir()
    c=dict(case_id='one',site_id='synthetic_site',building_id='b',session_id='s',method='MyCON',variant='new',split='test',data_kind='synthetic',independent_gt_verified=True,gt_reviewed=True,independent_gt_reviewer='fixture')
    m={'schema_version':SCHEMA,'run_id':'validation-c-synthetic-only','cases':[c]};(root/'manifest.json').write_text(json.dumps(m))
    out=export(root/'manifest.json',tmp_path/'results')
    assert json.loads((out/'experiment_registry.json').read_text())['scientific_evaluated_experiments']==0
    assert 'not_evaluated' in (out/'failure_cases.csv').read_text()
    assert len((out/'geometry_metrics.csv').read_text().splitlines())==1


def test_claimed_real_without_survey_hashed_is_blocked(tmp_path):
    root=tmp_path/'incoming';root.mkdir()
    c=dict(case_id='one',site_id='X',building_id='b',session_id='s',method='MyCON',variant='new',split='test',data_kind='real',independent_gt_verified=True,gt_reviewed=True,independent_gt_reviewer='fixture',units='m',metric_scale_verified=True)
    m={'schema_version':SCHEMA,'run_id':'validation-c-unaudited','cases':[c]};(root/'manifest.json').write_text(json.dumps(m))
    out=export(root/'manifest.json',tmp_path/'results')
    assert json.loads((out/'experiment_registry.json').read_text())['scientific_evaluated_experiments']==0


def test_malformed_run_rejected_before_write(tmp_path):
    m={'schema_version':SCHEMA,'run_id':'../../unsafe','cases':[]};(tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='run_id'):export(tmp_path/'manifest.json',tmp_path/'results')
    assert not (tmp_path/'results').exists()


def test_isolated_fake_real_gate_fixture_only(tmp_path):
    """Simulated real flags exercise gates; NEVER ship fixture scores as field evidence."""
    from PIL import Image
    r=tmp_path/'manifest_root';r.mkdir()
    (r/'capture.txt').write_text('independent test captures (fictional)')
    (r/'gt.csv').write_text('x,y,z\n0,0,0\n1,0,0\n0,1,0\n')
    (r/'pred.csv').write_text('x,y,z\n0,0,0\n1,0,0\n0,1,0\n')
    (r/'scale.json').write_text(json.dumps({'fit_anchor_ids':['a','b','c'],'heldout_anchor_ids':['d','e'],'independent_survey_reviewer':'fixture-surveyor'}))
    Image.new('RGB',(10,10),(30,80,90)).save(r/'gt.png')
    Image.new('RGB',(10,10),(40,80,90)).save(r/'pr.png')
    def ref(name):return {'path':name,'sha256':digest(r/name)}
    c=dict(experiment_id='fake-001',case_id='fixture-room',site_id='FICTIONAL',building_id='b',session_id='s',split='test',method='old',variant='baseline',data_kind='real',units='m',coordinate_frame='synthetic_local',capture_source=ref('capture.txt'),capture_sha256=digest(r/'capture.txt'),independent_gt_source=ref('gt.csv'),independent_gt_verified=True,independent_gt_reviewer='fixture-surveyor',independent_gt_review_date_utc='2026-11-13T00:00:00Z',ground_truth_source_type='independent_survey',gt_reviewed=True,metric_scale_verified=True,metric_scale_evidence=ref('scale.json'),geometry={'gt':ref('gt.csv'),'prediction':ref('pred.csv')},views=[{'view_id':'heldout-test-only','reference_rgb':ref('gt.png'),'prediction_rgb':ref('pr.png')}])
    m={'schema_version':SCHEMA,'run_id':'validation-c-test-simulated-real','cases':[c]};(r/'manifest.json').write_text(json.dumps(m))
    out=export(r/'manifest.json',tmp_path/'out')
    assert json.loads((out/'experiment_registry.json').read_text())['scientific_evaluated_experiments']==1
    assert len((out/'geometry_metrics.csv').read_text().splitlines())>=8
    assert 'P10' in (out/'view_tail_summary.csv').read_text()
    assert 'fewer_than_two_independent_sites' in (out/'confidence_intervals.csv').read_text()
    # The benchmark uses no training images or checkpoint writes.
    assert digest(r/'gt.csv')==ref('gt.csv')['sha256']


def test_metric_geometry_requires_independent_control_split(tmp_path):
    root=tmp_path/'root';root.mkdir()
    (root/'capture.txt').write_text('not-really-capture')
    (root/'survey.csv').write_text('x,y,z\n0,0,0\n1,0,0\n')
    (root/'model.csv').write_text('x,y,z\n0,0,0\n1,0,0\n')
    (root/'bad_control.json').write_text(json.dumps({'fit_anchor_ids':['a'],'heldout_anchor_ids':['a'],'independent_survey_reviewer':'sim'}))
    def ref(p):return dict(path=p,sha256=digest(root/p))
    c=dict(case_id='fake',site_id='S',building_id='B',session_id='T',split='test',method='A',variant='baseline',data_kind='real',independent_gt_verified=True,gt_reviewed=True,independent_gt_reviewer='sim',independent_gt_review_date_utc='2026-11-13',ground_truth_source_type='independent_survey',units='m',metric_scale_verified=True,metric_scale_evidence=ref('bad_control.json'),independent_gt_source=ref('survey.csv'),capture_source=ref('capture.txt'),capture_sha256=digest(root/'capture.txt'),geometry={'gt':ref('survey.csv'),'prediction':ref('model.csv')})
    (root/'manifest.json').write_text(json.dumps(dict(schema_version=SCHEMA,run_id='validation-c-test-leaking-controls',cases=[c])))
    out=export(root/'manifest.json',tmp_path/'output')
    assert 'invalid_or_leaking_metric_scale_controls' in (out/'failure_cases.csv').read_text()
    assert len((out/'geometry_metrics.csv').read_text().splitlines())==1