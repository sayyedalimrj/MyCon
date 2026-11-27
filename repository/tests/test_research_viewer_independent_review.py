"""Synthetic, adversarial A handoff and independent V viewport tests."""
import json
from pathlib import Path
import pytest
from viewer_civil.core import ViewerInputError, load_bundle, sha256
from viewer_civil.independent_review import create_workspace


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(data,dict):path.write_text(json.dumps(data))
    elif isinstance(data,bytes):path.write_bytes(data)
    else:path.write_text(data)
    return path


def rec(root,p):return {'path_relative_to_project':p.relative_to(root).as_posix(),'sha256':sha256(p),'bytes':p.stat().st_size}

@pytest.fixture
def src(tmp_path):
    Image=pytest.importorskip('PIL.Image');root=tmp_path/'source';root.mkdir();g=root/'geometry';a=root/'audit';g.mkdir();a.mkdir()
    photo=root/'inputs/captured_images/frame_a.jpg';photo.parent.mkdir(parents=True);Image.new('RGB',(16,12),(30,70,100)).save(photo)
    copied=write(a/'frame_a.jpg',photo.read_bytes())
    native={n:write(root/'inputs'/('colmap_text' if n.endswith('.txt') else '')/n,v) for n,v in {
        'cameras.txt':'1 SIMPLE_RADIAL 16 12 20 8 6 0.01\n',
        'images.txt':'1 1 0 0 0 0 0 0 1 frame_a.jpg\n\n2 1 0 0 0 -1 0 0 1 frame_b.jpg\n\n',
        'points3D.txt':'1 0 0 2 0 0 0 0\n2 1 0 2 0 0 0 0\n',
        'colmap_sparse_unscaled.ply':b'fixture sparse only'}.items()}
    common={'schema_version':'mycon.civil_geometry_review.v1','run_id':'fixture-geom','coordinate_frame':'depth_local','units':'arbitrary_reconstruction_units','metric_scale_verified':False}
    o=write(g/'viewer_overlay.json',{**common,'points':[[1,2,3]],'planes':[{'plane_id':'p1','finite_boundary_xyz':[[0,0,0],[1,0,0],[0,1,0]]}],'edges':[{'edge_id':'e1','endpoints_xyz':[[0,0,0],[1,0,0]]}],'corners':[{'corner_id':'c1','xyz':[0,0,0],'state':'candidate_extrapolated'}]})
    outs={p.name:{'relative_path':p.relative_to(root).as_posix(),'sha256':sha256(p)} for p in [o,*[write(g/(name+'.json'),{}) for name in ('planes','edges','corners')]]}
    civil=write(g/'civil_geometry_manifest.json',{**common,'registration_verified':False,'outputs':outs})
    identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    model={'view_id':1,'name':'frame_a.jpg','original_source_path':photo.relative_to(root).as_posix(),'source_image_sha256':sha256(photo),'depth_fusion_projection_status':'WITHHELD_UNREGISTERED','camera':{'model':'SIMPLE_RADIAL','width':16,'height':12,'fx':20,'fy':20,'cx':8,'cy':6,'k1':.01},'world_to_camera_4x4':identity,'camera_to_world_4x4':identity}
    q=write(a/'registration_qa.json',{'schema_version':'mycon.civil_camera_geometry_bridge.v1','run_id':'fixture-audit','status':'UNREGISTERED','registration_verified':False,'metric_scale_verified':False,'can_project_depth_features_to_images':False,'camera_pose_frame':'COLMAP_sfm_world_unscaled','source_depth_geometry_frame':'depth_local','civil_manifest_sha256':sha256(civil),'source_file_sha256':{k:sha256(p) for k,p in native.items()},'registered_colmap_view_count':2,'reviewed_photo_count':1,'reviewed_photo_models':[model],'sparse_ply_to_track_consistency':{'points':2}})
    h=write(a/'viewer_handoff.json',{'schema_version':'mycon.civil_geometry_viewer_handoff.v1','run_id':'fixture-audit','registration_status':'UNREGISTERED','camera_and_depth_co_registered':False,'metric_scale_verified':False,'feature_photo_projection_allowed':False,'heatmap_enabled':False,'coordinate_frame_colmap':'COLMAP_sfm_world_unscaled','coordinate_frame_depth':'depth_local','units_camera':'arbitrary_COLMAP_units','units_depth':'arbitrary_reconstruction_units','source_civil_run_id':'fixture-geom','source_civil_manifest_sha256':sha256(civil),'registration_qa':rec(root,q),'geometry_viewer_overlay':rec(root,o),'source_camera_photos':[{'view_id':1,'name':'frame_a.jpg','file':rec(root,copied)}],**{key:rec(root,write(a/(key+'.json'),{})) for key in ('feature_evidence','unknown_regions','unknown_point_indices')}})
    return root,h


def run(src,tmp_path,**kwargs):
    root,h=src;return create_workspace(root,h.relative_to(root).as_posix(),sha256(h),tmp_path/'new-viewer',preview=False,**kwargs)


def test_independent_viewports_and_missing_photo(src,tmp_path):
    r=run(src,tmp_path);out=tmp_path/'new-viewer';c=load_bundle(out/'historical_cameras');g=load_bundle(out/'shape3_depth')
    assert r['displayed_camera_count']==2 and r['displayed_photo_camera_count']==1
    assert c['assets']['camera_poses']['cameras'][1]['frame_id'] is None
    assert c['assets']['camera_poses']['cameras'][1]['camera_to_world'][0][3]==1
    assert c['assets']['planes']['planes']==[] and g['assets']['camera_poses']['cameras']==[]
    assert g['assets']['corners']['corners'][0]['confidence'] is None
    assert g['assets']['planes']['planes'][0]['vertices'][0]==[0,0,0]
    assert c['manifest']['coordinate_frame']!=g['manifest']['coordinate_frame']
    assert not c['deviation_enabled'] and not g['deviation_enabled']


def test_missing_fresh_R_pose_does_not_substitute_history(src,tmp_path):
    rr=tmp_path/'fresh-r';rr.mkdir();r=run(src,tmp_path,r_run_dir=rr)
    assert r['fresh_r_status']=='blocked_missing_registered_R_poses'
    assert 'fresh_r_cameras' not in r['views']

@pytest.mark.parametrize('which',['photo','native','overlay'])
def test_tampering_blocks_before_output(src,tmp_path,which):
    root,h=src;file={'photo':root/'inputs/captured_images/frame_a.jpg','native':root/'inputs/colmap_text/images.txt','overlay':root/'geometry/viewer_overlay.json'}[which];file.write_bytes(file.read_bytes()+b'x')
    with pytest.raises(ViewerInputError,match='SHA'):run(src,tmp_path)
    assert not (tmp_path/'new-viewer').exists()


def test_no_claimed_alignment(src,tmp_path):
    root,h=src;data=json.loads(h.read_text());data['camera_and_depth_co_registered']=True;write(h,data)
    with pytest.raises(ViewerInputError,match='prohibit co-registration'):run(src,tmp_path)


def test_external_handoff_sha_required(src,tmp_path):
    root,h=src
    with pytest.raises(ViewerInputError,match='externally supplied SHA256'):create_workspace(root,h.relative_to(root).as_posix(),'0'*64,tmp_path/'new-viewer',preview=False)


def test_output_cannot_be_inside_source_tree(src,tmp_path):
    root,h=src
    with pytest.raises(ViewerInputError,match='outside source tree'):create_workspace(root,h.relative_to(root).as_posix(),sha256(h),root/'new',preview=False)
