"""Synthetic correctness and safety gates; no real surveying claims."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.stage_07_cleanup import structural_review as sr


def fake_source(root: Path, *, with_extrapolated=False):
    source=root/'runs'/'geom-ready-synthetic'/ 'geometry';source.mkdir(parents=True)
    rng=np.random.default_rng(7)
    vals=np.linspace(.02,1.,13)
    a,b=np.meshgrid(vals,vals)
    grid=np.column_stack([a.ravel(),b.ravel()])
    xyz=np.vstack([np.column_stack((np.zeros(len(grid)),grid)),
                   np.column_stack((grid[:,0],np.zeros(len(grid)),grid[:,1])),
                   np.column_stack((grid,np.zeros(len(grid))))])
    xyz+=rng.normal(0,.0005,xyz.shape)
    np.save(source/'clean_cloud.npy',xyz)
    sr._write_ply(source/'clean_cloud.ply',xyz,np.tile([130,180,200],(len(xyz),1)).astype('uint8'))
    planes=[];N=len(grid)
    for i,(norm,u,v) in enumerate([([1.,0,0],[0,1,0],[0,0,1]),([0,1.,0],[1,0,0],[0,0,1]),([0,0,1.],[1,0,0],[0,1,0])]):
        idx=np.arange(i*N,(i+1)*N)
        center=np.mean(xyz[idx],axis=0)
        lo=.02;hi=1.
        uv=np.array([[lo,lo],[hi,lo],[hi,hi],[lo,hi]])
        bound=np.array([np.array(norm)*0+np.array(u)*p[0]+np.array(v)*p[1] for p in uv])
        planes.append({'plane_id':f'pl-{i+1:03d}','support_indices':idx.tolist(),
                       'centroid':center.tolist(),'normal':norm,'basis_u':u,'basis_v':v,
                       'finite_boundary_xyz':bound.tolist(),'fit_state':'candidate_observed_patch',
                       'point_count':N,'residual_p95_units':.001})
    edges=[]
    for i,(p1,p2) in enumerate(((1,2),(1,3),(2,3))):
        edges.append({'edge_id':f'ed-{i+1:03d}','plane_ids':[f'pl-{p1:03d}',f'pl-{p2:03d}'],
                      'endpoints_xyz':[[0,0,0],[1,0,0]], 'fit_state':'locally_supported_candidate',
                      'support_count_per_plane':[13,13]})
    corner={'corner_id':'co-001','plane_ids':[p['plane_id'] for p in planes],
            'edge_ids':[e['edge_id'] for e in edges], 'xyz':[0,0,0],
            'state':'observed_supported','multiscale_confirmed':not with_extrapolated,
            'support_near_corner_per_plane':[4,4,4], 'inside_observed_patch_per_plane':[True]*3}
    for n,d in (('planes',planes),('edges',edges),('corners',[corner])):
        (source/(n+'.json')).write_text(json.dumps({n:d}))
    entries={n:{'path':str((source/n).relative_to(root)),'sha256':sr.sha(source/n)} for n in
             ['clean_cloud.ply','clean_cloud.npy','planes.json','edges.json','corners.json']}
    manifest={'schema_version':sr.BASE_SCHEMA,'run_id':'geom-ready-synthetic','coordinate_frame':'colmap_world',
              'units':'arbitrary_reconstruction_units','metric_scale_verified':False,
              'source_sha256':'synthetic-input-no-real-claim','parameters':{'threshold_reconstruction_units':.02},
              'outputs':entries}
    path=source/'comparison_ready.json';path.write_text(json.dumps(manifest))
    return path


def test_end_to_end_output_manifest_and_human_review(tmp_path):
    base=fake_source(tmp_path)
    result=sr.generate(tmp_path,base,'geom-civil-synthetic-001',input_kind='synthetic')
    out=tmp_path/'runs'/'geom-civil-synthetic-001'/'geometry'
    assert result['schema_version']==sr.SCHEMA
    assert result['metric_scale_verified'] is False
    assert len(result['outputs'])>=13
    for name, spec in result['outputs'].items():
        assert sr.sha(tmp_path/spec['relative_path'])==spec['sha256'],name
    qa=json.loads((out/'geometry_qa.json').read_text())
    assert qa['input_kind']=='synthetic' and qa['confidence_calibrated'] is False
    assert qa['counts']['human_confirmed_corners']==0
    corners=json.loads((out/'corners.json').read_text())['corners']
    assert corners[0]['field_confirmed'] is False and corners[0]['confidence']['calibrated_probability'] is None
    planes=json.loads((out/'planes.json').read_text())['planes']
    assert all(p['semantic_label']=='unknown' for p in planes)
    assert (out/'geometry_review.html').is_file()
    assert all((out/f'shape3_view_{i:02d}.png').is_file() for i in (1,2,3))
    assert sr.sha(base)==result['source_manifest_sha256']


def test_unsupported_candidate_cannot_be_upgraded(tmp_path):
    base=fake_source(tmp_path,with_extrapolated=True)
    sr.generate(tmp_path,base,'geom-civil-extrapolate-001',input_kind='synthetic')
    out=tmp_path/'runs'/'geom-civil-extrapolate-001'/'geometry'
    c=json.loads((out/'corners.json').read_text())['corners'][0]
    assert c['state']=='unknown' and c['field_confirmed'] is False


def test_source_hash_tamper_fails_without_creating_output(tmp_path):
    base=fake_source(tmp_path)
    source=base.parent/'planes.json';source.write_text(source.read_text()+' ')
    with pytest.raises(ValueError,match='hash mismatch'):
        sr.generate(tmp_path,base,'geom-civil-tamper-001')
    assert not (tmp_path/'runs'/'geom-civil-tamper-001').exists()


def test_duplicate_run_refusal_preserves_previous_output(tmp_path):
    base=fake_source(tmp_path)
    sr.generate(tmp_path,base,'geom-civil-duplicate-001',input_kind='synthetic')
    existing=tmp_path/'runs'/'geom-civil-duplicate-001'/'geometry'/'geometry_qa.json'
    original=sr.sha(existing)
    with pytest.raises(FileExistsError):sr.generate(tmp_path,base,'geom-civil-duplicate-001')
    assert sr.sha(existing)==original


def test_no_relabeling_floor_without_gravity_and_survey(tmp_path):
    base=fake_source(tmp_path)
    sr.generate(tmp_path,base,'geom-civil-gravity-001',input_kind='synthetic')
    obj=json.loads((tmp_path/'runs'/'geom-civil-gravity-001'/'geometry'/'planes.json').read_text())
    assert all(p['semantic_label']=='unknown' and p['physical_surface_confirmed'] is False for p in obj['planes'])


def test_refit_noise_density_occlusion_reported_with_denominators(tmp_path):
    base=fake_source(tmp_path)
    sr.generate(tmp_path,base,'geom-civil-stress-001',input_kind='synthetic')
    qa=json.loads((tmp_path/'runs'/'geom-civil-stress-001'/'geometry'/'geometry_qa.json').read_text())
    for per_plane in qa['sensitivity']['by_plane'].values():
        assert set(per_plane)=={'density_half','noise_sigma_source_units','spatial_occlusion_quadrant'}
        assert all(v['sample_count']>=6 and v['status']=='fitted' for v in per_plane.values())
    assert qa['quality_assessment']=='synthetic_algorithm_diagnostic_only'


def test_path_escape_and_metric_assertion_rejected(tmp_path):
    base=fake_source(tmp_path)
    ob=json.loads(base.read_text());ob['metric_scale_verified']=True;base.write_text(json.dumps(ob))
    with pytest.raises(ValueError,match='metric-scale'):
        sr.generate(tmp_path,base,'geom-civil-metric-001')
    ob['metric_scale_verified']=False;ob['outputs']['planes.json']['path']='../../etc/passwd';base.write_text(json.dumps(ob))
    with pytest.raises(ValueError,match='out-of-project'):
        sr.generate(tmp_path,base,'geom-civil-escape-001')
