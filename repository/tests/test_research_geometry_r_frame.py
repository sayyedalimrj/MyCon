"""Synthetic CPU regression of owner-A same-SfM-model geometry. NOT site/field validation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.stage_07_cleanup import r_frame_geometry as g


def fixture(root:Path):
    inp=root/'inputs';inp.mkdir(parents=True)
    width,height=640,480;f=600;cx=320;cy=240
    (inp/'cameras.txt').write_text(f'1 SIMPLE_RADIAL {width} {height} {f} {cx} {cy} 0\n')
    xyz=[]
    for i in range(20):
        for j in range(16):
            xyz.append([-.9+i*.095,-.71+j*.095,4.0])
    xyz=np.asarray(xyz);ids=list(range(1,len(xyz)+1))
    with (inp/'points3D.txt').open('w') as dst:
        dst.write('#Synthetic COLMAP points3D fixture\n')
        for index,(pid,v) in enumerate(zip(ids,xyz)):
            pairs=' '.join(f'{k} {index}' for k in range(1,5))
            dst.write(f'{pid} {v[0]} {v[1]} {v[2]} 155 160 165 0.1 {pairs}\n')
    with (inp/'images.txt').open('w') as out:
        out.write('#Synthetic COLMAP images fixture\n')
        for k in range(1,5):
            name=f'frame_{k:06}.jpg';Image.new('RGB',(width,height),(70,75,80)).save(inp/name)
            out.write(f'{k} 1 0 0 0 {0.01*(k-1)} 0 0 1 {name}\n')
            out.write(' '.join(f'{f*(v[0]+0.01*(k-1))/v[2]+cx} {f*v[1]/v[2]+cy} {pid}' for pid,v in zip(ids,xyz))+'\n')
    return {'root':root,'cameras':inp/'cameras.txt','images':inp/'images.txt',
            'points':inp/'points3D.txt','photos':[inp/f'frame_{i:06}.jpg' for i in range(1,4)],
            'run_id':'geom-civil-r-frame-synthetic-test'}


def run(root,**overrides):
    x=fixture(root);x.update(overrides)
    return g.generate(**x,min_points=180,max_planes=2,trials=20,threshold=.025)


def test_source_track_filter_and_ids(tmp_path):
    x=fixture(tmp_path)
    ids,xyz,rgb,err,ntracks,rejected=g._read_tracks(x['points'],error_px=1.0,min_track_length=4)
    assert len(ids)==320 and (ids[0],ids[-1])==(1,320)
    assert (err<=1).all() and (ntracks>=4).all() and rejected==0
    assert xyz.shape==(320,3)


def test_synthetic_same_model_only_and_no_corners(tmp_path):
    result=run(tmp_path);out=Path(result['output_dir']);q=json.loads((out/'geometry_qa.json').read_text())
    assert result['quality_points']==320 and q['source_model_frame_verified_internal'] is True
    assert q['r_new_run_frame_verified'] is False and q['metric_scale_verified'] is False
    assert result['confirmed_corners']==0
    assert q['queried_photo_count']==3
    assert q['source_kind']=='historical_COLMAP_sparse_reconstruction_NOT_DENSE'
    planes=json.loads((out/'planes.json').read_text())['planes']
    assert planes and all(not p['field_confirmed'] and p['confidence_probability'] is None for p in planes)
    assert planes[0]['source_image_track_evidence']
    assert all(p['semantic_label']=='unknown' for p in planes)
    assert (out/'shape3_sfm_review.html').exists() and (out/'civil_r_frame_manifest.json').exists()


def test_create_only_run_guard(tmp_path):
    run(tmp_path)
    x={'root':tmp_path,'cameras':tmp_path/'inputs/cameras.txt','images':tmp_path/'inputs/images.txt',
      'points':tmp_path/'inputs/points3D.txt','photos':[tmp_path/'inputs'/f'frame_{i:06}.jpg' for i in range(1,4)],
      'run_id':'geom-civil-r-frame-synthetic-test'}
    with pytest.raises(FileExistsError):g.generate(**x,min_points=180,max_planes=2,trials=20,threshold=.025)


def test_wrong_photo_dimensions_block_before_output(tmp_path):
    x=fixture(tmp_path);Image.new('RGB',(100,80)).save(x['photos'][0]);
    with pytest.raises(ValueError,match='Image camera dimensions'):
        g.generate(**x,min_points=180,max_planes=2,trials=20,threshold=.025)
    assert not (tmp_path/'runs').exists()


def test_wrong_2d_reprojection_blocks_before_output(tmp_path):
    x=fixture(tmp_path)
    content=x['images'].read_text();rows=content.splitlines()
    for i in range(len(rows)):
        if rows[i].startswith('1 1 0 0 0'):
            j=i+1;t=rows[j].split();
            for k in range(0,len(t),3):t[k]=str(float(t[k])+30)
            rows[j]=' '.join(t);break
    x['images'].write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='reprojection quality'):
        g.generate(**x,min_points=180,max_planes=2,trials=20,threshold=.025)
    assert not (tmp_path/'runs').exists()


def test_r_mismatch_model_explicit_unregistered(tmp_path):
    x=fixture(tmp_path);runpath=tmp_path/'recon-run-sample';runpath.mkdir();
    (runpath/'dataset_manifest.json').write_text(json.dumps({'schema_version':'mycon.reconstruction.dataset.v1',
           'run_id':runpath.name,'frames':[{'view_id':1,'image_relpath':'images/anything.jpg'}]}))
    (runpath/'camera_poses.json').write_text(json.dumps({'schema_version':'mycon.reconstruction.camera_poses.v1',
           'run_id':runpath.name,'registered_colmap_model_sha256':'0'*64,'poses':[]}))
    x['r_dataset']=runpath/'dataset_manifest.json';x['r_poses']=runpath/'camera_poses.json'
    result=g.generate(**x,min_points=180,max_planes=2,trials=20,threshold=.025)
    q=json.loads((Path(result['output_dir'])/'geometry_qa.json').read_text())
    assert q['r_compatibility']['status']=='UNREGISTERED_DIFFERENT_R_COLMAP_MODEL'
    assert not q['r_compatibility']['shared_world_frame_verified']


def test_r_same_model_bad_image_receipt_rejected(tmp_path):
    x=fixture(tmp_path);runpath=tmp_path/'recon-run-sample';runpath.mkdir();(runpath/'images').mkdir()
    (runpath/'images'/'frame_000001.jpg').write_bytes(x['photos'][0].read_bytes())
    frm={'view_id':1,'image_relpath':'images/frame_000001.jpg','capture_session_id':'SYN','split':'train','image_sha256':'f'*64}
    (runpath/'dataset_manifest.json').write_text(json.dumps({'schema_version':'mycon.reconstruction.dataset.v1','run_id':runpath.name,'frames':[frm]}))
    pose={'view_id':1,'image_relpath':frm['image_relpath'],'capture_session_id':'SYN',
          'split':'train','image_sha256':'f'*64,'registered':True,
          'coordinate_frame':'colmap_opencv_world_to_camera','metric_scale_verified':False,
          'world_to_camera_4x4':np.eye(4).tolist()}
    (runpath/'camera_poses.json').write_text(json.dumps({'schema_version':'mycon.reconstruction.camera_poses.v1',
        'run_id':runpath.name,'registered_colmap_model_sha256':g._sha(x['images']),'poses':[pose]}))
    x['r_dataset']=runpath/'dataset_manifest.json';x['r_poses']=runpath/'camera_poses.json'
    with pytest.raises(ValueError,match='source image SHA mismatch'):
        g.generate(**x,min_points=180,max_planes=2,trials=20,threshold=.025)
    assert not (tmp_path/'runs').exists()


def test_bad_r_pose_schema_and_session_fail_closed(tmp_path):
    x=fixture(tmp_path);rp=tmp_path/'r';rp.mkdir()
    (rp/'dataset_manifest.json').write_text(json.dumps({'schema_version':'mycon.reconstruction.dataset.v1','run_id':'recon-sample','frames':[{'view_id':'v1'}]}))
    (rp/'camera_poses.json').write_text(json.dumps({'schema_version':'mycon.reconstruction.camera_poses.v2','run_id':'recon-sample','poses':[]}))
    with pytest.raises(ValueError,match='contract version'):
        g.generate(**{**x,'r_dataset':rp/'dataset_manifest.json','r_poses':rp/'camera_poses.json'},
                   min_points=180,max_planes=2,trials=20,threshold=.025)
    assert not (tmp_path/'runs').exists()


def test_repeatability_is_never_certification(tmp_path):
    from pipeline.stage_07_cleanup import r_frame_stability as s
    x=fixture(tmp_path)
    _,xyz,_,_,_,_=g._read_tracks(x['points'],error_px=1,min_track_length=4)
    r=s.evaluate(xyz,threshold=.025,min_points=70,max_planes=2,trials=20)
    assert r['not_independent_ground_truth'] and r['not_physical_surface_confirmation']
    assert len(r['scenarios'])==5
    assert all(v['synthetically_modified_input'] for v in r['scenarios'])
    assert all(v['matched_reference_patches']<=v['baseline_patch_count'] for v in r['scenarios'])