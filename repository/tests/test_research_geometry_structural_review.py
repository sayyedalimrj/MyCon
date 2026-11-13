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

# Round 2: extraction on physically sampled candidate points, not Gaussian centres.
from pipeline.stage_07_cleanup import research_dense_candidates as dc


def sloped_room(seed=8, noise=.003, occluded=False, thinning=1):
    rng=np.random.default_rng(seed)
    x,y=np.meshgrid(np.linspace(-1,1,25),np.linspace(-1,1,25))
    # A non-Manhattan vertical wall and a horizontal panel with actual finite points.
    w=np.column_stack((.38*x.ravel(),x.ravel(),y.ravel()+2))
    f=np.column_stack((x.ravel()+2,y.ravel()+2,np.full(x.size,.5)))
    if occluded: w=w[~((w[:,1]>.1)&(w[:,2]>2.1))]
    pts=np.vstack((w[::thinning],f[::thinning]))
    return pts+rng.normal(0,noise,pts.shape)


def test_round2_tilted_plane_noise_and_density():
    true=np.array([1.,-.38,0]);true=true/np.linalg.norm(true)
    for seed,noise,step in ((8,.002,1),(9,.012,2),(10,.004,3)):
        xyz=sloped_room(seed,noise,thinning=step)
        pl,edges,corners,qa=dc.extract(xyz,threshold=.055,min_points=55,max_planes=4,trials=90,seed=17)
        assert pl and any(abs(np.dot(p['normal'],true))>.94 for p in pl)
        assert qa['unknown_points']>=0
        assert all(c['state']!='observed_supported' and c['field_confirmed'] is False for c in corners)


def test_round2_partial_occlusion_keeps_slanted_hypothesis_unverified():
    xyz=sloped_room(11,.006,occluded=True)
    pl,ed,c,qa=dc.extract(xyz,threshold=.05,min_points=65,max_planes=4,trials=90,seed=13)
    n=np.array([1.,-.38,0]);n/=np.linalg.norm(n)
    assert any(abs(np.dot(p['normal'],n))>.94 for p in pl)
    assert all(p['orientation_class']=='unknown_unverified_gravity' for p in pl)
    assert all(x['state']=='candidate_extrapolated' for x in c)


def test_round2_distant_infinite_plane_intersections_never_promote_corner():
    a,b=np.meshgrid(np.linspace(3,4,18),np.linspace(3,4,18))
    g=np.column_stack([a.ravel(),b.ravel()])
    pts=np.vstack([np.column_stack((np.zeros(len(g)),g)),
                   np.column_stack((g[:,0],np.zeros(len(g)),g[:,1])),
                   np.column_stack((g,np.zeros(len(g))))])
    pl,ed,co,_=dc.extract(pts,threshold=.025,min_points=65,max_planes=5,trials=90,seed=1)
    assert len(pl)>=3
    assert len(co)==0


def test_round2_gaussian_centres_rejected_before_write(tmp_path):
    src=tmp_path/'centres.ply';sr._write_ply(src,sloped_room(),np.tile([15,15,15],(1250,1)).astype('uint8'))
    with pytest.raises(ValueError,match='Gaussian centres'):
        dc.generate(src,tmp_path,'geom-ready-rejected',source_kind='gaussian_centers',threshold=.05)
    assert not (tmp_path/'runs'/'geom-ready-rejected').exists()


def test_round2_direct_cloud_candidate_and_review_hashes(tmp_path):
    xyz=sloped_room()
    source=tmp_path/'real_derived_candidate.ply'
    sr._write_ply(source,xyz,np.tile([150,125,110],(len(xyz),1)).astype('uint8'))
    base=dc.generate(source,tmp_path,'geom-ready-test-case',source_kind='depth_fused_candidate_nonmetric',
                     threshold=.045,min_points=60,max_planes=4,trials=90)
    assert base['metric_scale_verified'] is False and base['source_sha256']==sr.sha(source)
    reviewed=sr.generate(tmp_path,tmp_path/'runs'/'geom-ready-test-case'/'geometry'/'comparison_ready.json',
                         'geom-civil-direct-001',input_kind='synthetic')
    assert reviewed['outputs']['geometry_review.html']['sha256']
    qa=json.loads((tmp_path/'runs'/'geom-civil-direct-001'/'geometry'/'geometry_qa.json').read_text())
    assert qa['confidence_calibrated'] is False and qa['feature_image_support_confirmed_count']==0


def test_round2_image_index_hash_and_calibration_gate(tmp_path):
    base=fake_source(tmp_path)
    from PIL import Image
    photo=tmp_path/'view.png'; Image.new('RGB',(3000,3000),(128,140,152)).save(photo)
    extrinsic=np.eye(4).tolist()
    index=tmp_path/'camera_index.json'
    cam={'view_id':'v1','frame_convention':'colmap_opencv_world_to_camera','calibration_verified':True,
         'image_path':'view.png','image_sha256':sr.sha(photo),'world_to_camera_4x4':extrinsic,
         'fx':500,'fy':500,'cx':1000,'cy':1000,'width':3000,'height':3000}
    index.write_text(json.dumps({'schema_version':'mycon.camera_evidence_index.v1',
                      'coordinate_frame':'colmap_world','units':'arbitrary_reconstruction_units','cameras':[cam]}))
    result=sr.generate(tmp_path,base,'geom-civil-camera-index',camera_index=index,input_kind='synthetic')
    qa=json.loads((tmp_path/'runs'/'geom-civil-camera-index'/'geometry'/'geometry_qa.json').read_text())
    assert qa['calibrated_source_camera_count']==1 and qa['camera_visibility_verified'] is False
    plane=json.loads((tmp_path/'runs'/'geom-civil-camera-index'/'geometry'/'planes.json').read_text())['planes'][0]
    assert len(plane['supporting_images'])==1
    assert plane['supporting_images'][0]['status']=='projected_in_fov_occlusion_unverified'
    assert plane['supporting_images'][0]['image_support_confirmed'] is False
    # source image integrity still checked BEFORE a new output folder can be created
    photo.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='image SHA-256 mismatch'):
        sr.generate(tmp_path,base,'geom-civil-bad-photo',camera_index=index)
    assert not (tmp_path/'runs'/'geom-civil-bad-photo').exists()


def test_round2_real_label_must_have_verified_provenance(tmp_path):
    base=fake_source(tmp_path)
    with pytest.raises(ValueError,match='Cannot mark unknown/synthetic source'):
        sr.generate(tmp_path,base,'geom-civil-fake-real',input_kind='real_reconstruction')
    assert not (tmp_path/'runs'/'geom-civil-fake-real').exists()

# Coordinator-owned specialized CPU workflow currently selects this historical
# A test path explicitly. Re-export the new A camera-safety suite so that the
# exact-PR-HEAD run actually executes it without editing cross-owner workflow.
from test_research_geometry_camera_bridge import (
    test_registration_is_never_inferred_from_colmap_reprojection,
    test_source_hash_tampering_fails_closed,
    test_nonmatching_photo_name_fails_closed,
    test_invalid_quaternion_fails_closed,
    test_bad_camera_source_dimensions_fail,
    test_dedicated_run_is_create_only,
    test_reflection_does_not_count_as_registration,
    test_sparse_ply_and_points3d_must_match,
)
