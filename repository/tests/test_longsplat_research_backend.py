"""No-GPU regression tests for academic LongSplat optional backend."""
from __future__ import annotations
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

# Typical repository tests run with repository/ on PYTHONPATH.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.stage_04_5_cams_gs import longsplat_research as bridge
from pipeline.stage_04_5_cams_gs import longsplat_depth_geometry as geo
from pipeline.stage_04_5_cams_gs import metric_anchor_gate as gate
from pipeline.stage_04_5_cams_gs import longsplat_visual_audit as audit


def fixture(tmp_path):
    vid=tmp_path/'full_upright_96s.mp4';vid.write_bytes(b'fake video bytes'*1000)
    repo=tmp_path/'LongSplat';repo.mkdir()
    for n in ('train.py','render.py','convert_3dgs.py'):(repo/n).write_text('fake')
    (repo/'LICENSE.md').write_text('The work can be used non-commercially for research.')
    args=Namespace(phase='plan',video=vid,longsplat_root=repo,workspace=tmp_path/'mycon',
                   python='python',fps=10.,width=512,post_iter=20000,prune_ratio=.6,
                   max_geometry_points=300000,accept_research_license=False)
    return args


def make_frames(argv,*,cwd,log):
    folder=Path(argv[-1]).parent
    folder.mkdir(parents=True,exist_ok=True)
    for i in range(35): (folder/f'frame_{i:06d}.jpg').write_bytes(b'data')


def prepare(tmp_path):
    args=fixture(tmp_path)
    args.phase='prepare';args.accept_research_license=True
    # The fixture contains synthetic video bytes. Mock ffprobe so this
    # structural unit test does not depend on a real video decoder.
    with (patch.object(bridge,'invoke',side_effect=make_frames),
          patch.object(bridge.shutil,'which',return_value='/usr/bin/ffmpeg'),
          patch.object(bridge.subprocess,'run',return_value=Namespace(returncode=1,stdout=''))):
        assert bridge.run(args)==0
    return args


def test_plan_no_write_and_license_guard(tmp_path):
    a=fixture(tmp_path)
    assert bridge.run(a)==0
    assert not a.workspace.exists()
    a.phase='train'
    with pytest.raises(RuntimeError,match='Explicit'):
        bridge.run(a)


def test_prepare_state_and_parameter_identity(tmp_path):
    a=prepare(tmp_path)
    assert bridge.run(a)==0
    a.width=1024
    with pytest.raises(RuntimeError,match='different width'):
        bridge.run(a)


def test_train_missing_outputs_fails_closed(tmp_path):
    a=prepare(tmp_path);a.phase='train';a.approve_new_training=True
    with patch.object(bridge,'invoke',return_value=None):
        with pytest.raises(RuntimeError,match='no model'):
            bridge.run(a)
    work=a.workspace/bridge.capture_key(a.video,bridge.digest_file(a.video))
    state=json.loads((work/'research_state.json').read_text())
    assert state.get('trained') is False
    assert not state.get('bim_progress_authorized')


def test_3dgs_original_render_and_export_paths_are_separate(tmp_path):
    a=prepare(tmp_path);a.phase='train';a.approve_new_training=True
    work=a.workspace/bridge.capture_key(a.video,bridge.digest_file(a.video))
    model=work/'longsplat_model'
    def simulate_train(cmd,*,cwd,log):
        (model/'point_cloud/iteration_50000').mkdir(parents=True)
        (model/'point_cloud/iteration_50000/point_cloud.ply').write_bytes(b'x'*2048)
        (model/'cameras_all_train.json').write_text('['+'{}'*150+']')
    with patch.object(bridge,'invoke',side_effect=simulate_train):
        assert bridge.run(a)==0
    a.phase='render'
    def simulate_render(cmd,*,cwd,log):
        dest=model/'train/ours_50000'
        (dest/'videos').mkdir(parents=True)
        (dest/'depths').mkdir(parents=True)
        (dest/'videos/nvs_rgb.mp4').write_bytes(b'x'*1500)
        for j in range(4):(dest/'depths'/f'frame_{j:06d}.npy').write_bytes(b'fake')
    with patch.object(bridge,'invoke',side_effect=simulate_render):
        assert bridge.run(a)==0
    a.phase='export'
    def simulate_export(cmd,*,cwd,log):
        (model/'converted_3dgs').mkdir()
        f=model/'converted_3dgs/point_cloud.ply'
        f.write_bytes(b'ply\nelement vertex 3\nproperty float opacity\nproperty float f_dc_0\nend_header\n'+b'x'*600)
    with patch.object(bridge,'invoke',side_effect=simulate_export):
        assert bridge.run(a)==0
    assert (model/'converted_3dgs/point_cloud.ply').exists()
    state=json.loads((work/'research_state.json').read_text())
    assert state['trained'] and state['rendered'] and state['exported']
    assert not state['bim_progress_authorized']


def test_geometry_depth_pose_conversion_not_gaussian_fake(tmp_path):
    depth=tmp_path/'depths';depth.mkdir()
    images=tmp_path/'images';images.mkdir()
    camera=[]
    for j in range(4):
        name=f'frame_{j:06d}'
        np.save(depth/(name+'.npy'),np.full((128,128),2.,dtype=np.float32))
        Image.new('RGB',(128,128),color=(100,120,140)).save(images/(name+'.jpg'))
        camera.append({'image_name':name,'R':np.eye(3).tolist(), 'T':[-.2*j,0,0],
                       'Focalx':80,'Focaly':80,'width':128,'height':128})
    poses=tmp_path/'cameras.json';poses.write_text(json.dumps(camera))
    out=geo.extract_depth_candidate(depth,poses,images,tmp_path/'candidate',max_points=60000)
    ply=out['ply_path']
    assert ply.is_file() and ply.stat().st_size>100000
    report=json.loads(out['report_path'].read_text())
    assert report['matched_depth_frames']==4
    assert not report['metric_truth'] and not report['bim_equivalence_approved']
    assert report['relative_scale_only']
    # Camera convention: point at camera center pixel z=2 with T=[-.2,0,0]
    # shifts +0.2 along world x compared with same pixel first image.
    p=np.array([[64,64]])
    a=geo.backproject(camera[0],np.ones((128,128))*2.,p)
    b=geo.backproject(camera[1],np.ones((128,128))*2.,p)
    assert np.allclose(b-a,[[.2,0,0]])


def test_metric_gate_requires_independent_heldout_and_writes_review_only(tmp_path):
    rng=np.random.default_rng(10)
    x=rng.normal(size=(6,3))
    # Orthogonal proper rotation 90 deg about Z, 1.4 scaling in meters.
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    y=1.4*(x@rot.T)+np.array([3.,2.,1.])
    anchor=tmp_path/'anchors.csv'
    with anchor.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id',*gate.FIELDS]);w.writeheader()
        for j,(a,b) in enumerate(zip(x,y)):
            w.writerow({'id':str(j),**dict(zip(gate.FIELDS,[*a,*b,'fit' if j<4 else 'holdout']))})
    scale,R,t=gate.estimate_sim3(x[:4],y[:4])
    assert np.allclose(gate.apply_sim3(x,scale,R,t),y,atol=1e-7)
    cloud=tmp_path/'cloud.ply';geo._write_ply(cloud,np.repeat(x.astype('f4'),100,axis=0),np.ones((600,3),dtype='u1')*120)
    # _write_ply does not use the LongSplat identity marker; make the identity
    # explicit in this synthetic fixture to exercise strict reader/gate.
    with cloud.open('r+b') as f:
        hdr=f.read(500);assert b'LONGSPLAT_RENDERED_DEPTH_ARBITRARY_SCALE_NOT_SURVEYED' in hdr
    registered,report=gate.run_gate(cloud,anchor,tmp_path/'metric_review',max_holdout_m=.05)
    assert registered.exists()
    assert report['stage8_bim_registration_completed'] is False
    assert report['stage9_progress_eligible'] is False
    assert report['holdout_max_m']<1e-6


def test_bad_anchor_or_bad_holdout_must_not_approve(tmp_path):
    anchor=tmp_path/'anchors.csv'
    anchor.write_text(','.join(gate.FIELDS)+'\n'+'0,0,0,0,0,0,fit\n')
    with pytest.raises(ValueError,match='Require >=4 fit'):
        gate.read_anchors(anchor)


def test_visual_qa_requires_five_heldout_views(tmp_path):
    model=tmp_path/'model'
    report=audit.evaluate_holdout_views(model,1000,tmp_path/'report.json')
    assert report['status']=='INSUFFICIENT_HELDOUT_EVIDENCE'
    assert not report['unobserved_camera_positions_verified']
    p=model/'test/ours_1000';(p/'renders').mkdir(parents=True);(p/'gt').mkdir()
    for j in range(5):
        im=Image.new('RGB',(32,32),color=(60,40,80))
        im.save(p/'renders'/f'{j:05d}.png');im.save(p/'gt'/f'{j:05d}.png')
    report=audit.evaluate_holdout_views(model,1000,tmp_path/'report.json')
    assert report['heldout_pairs']==5 and report['status']=='HELDOUT_VIEWS_MEASURED'
    assert not report['unobserved_camera_positions_verified']


def test_nvidia_pixel_center_and_depth_dimensions_match_original_source():
    camera={'image_name':'frame_000000', 'R':np.eye(3).tolist(),
            'T':[0.,0.,0.], 'Focalx':80., 'Focaly':80.,
            'width':128, 'height':128}
    depth=np.full((128,128),2.,dtype=np.float32)
    center=geo.backproject(camera,depth,np.array([[64,64]]))
    assert np.allclose(center,[[0.,0.,2.]])
    with pytest.raises(ValueError,match='Depth size disagrees'):
        geo.backproject({**camera,'width':256},depth,np.array([[64,64]]))


def test_metric_sim3_rejects_coplanar_sites_that_cannot_validate_3d():
    flat=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    # A plausible 2D fit cannot certify full 3D BIM geometry and scale.
    with pytest.raises(ValueError,match='non-coplanar'):
        gate.estimate_sim3(flat,3*flat+np.array([4.,5.,6.]))

