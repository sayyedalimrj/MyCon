"""Owner-A camera/depth registration safety gates; synthetic fixtures are NOT field data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from pipeline.stage_07_cleanup import camera_geometry_bridge as g


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf8")
    return p


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def fixture(root):
    import trimesh
    from PIL import Image
    c=root/'runs'/'geom-civil-old'/'geometry';c.mkdir(parents=True)
    xyz=np.array([[i/20.,j/20.,4.] for i in range(5) for j in range(6)],dtype=float)
    np.save(c/'clean_cloud.npy',xyz)
    cloud=trimesh.points.PointCloud(xyz)
    (c/'clean_cloud.ply').write_bytes(trimesh.exchange.ply.export_ply(cloud))
    fp={
        'planes':[{'plane_id':'P01','support_indices':list(range(30)), 'residual_p95_units':.01,'observed_support_qa':{'occupied_fraction_of_hull':.6}}],
        'edges':[{'edge_id':'E01','support_indices_per_plane':[[0,2],[3,4]],'support_count_per_plane':[2,2]}],
        'corners':[{'corner_id':'C01','support_near_corner_per_plane':[0,0,0],'state':'candidate_extrapolated'}]}
    outputs={}
    for name,obj in fp.items():
        p=write(c/(name+'.json'),json.dumps({'coordinate_frame':'COLMAP_UNVERIFIED','units':'arbitrary','source_sha256':'a'*64,name:obj}))
        outputs[p.name]={'relative_path':p.relative_to(root).as_posix(),'sha256':sha(p)}
    overlay=write(c/'viewer_overlay.json','{}')
    src=write(c/'geometry_review.html','<html>geometry only</html>')
    for p in [c/'clean_cloud.npy',overlay]:outputs[p.name]={'relative_path':p.relative_to(root).as_posix(),'sha256':sha(p)}
    manifest=write(c/'civil_geometry_manifest.json',json.dumps({'schema_version':'mycon.civil_geometry_review.v1',
        'metric_scale_verified':False,'coordinate_frame':'COLMAP_UNVERIFIED','units':'arbitrary',
        'run_id':'geom-civil-old','outputs':outputs,'provenance':{'original_source_sha256':'b'*64}}))
    sparse=root/'input'/'sparse.ply';sparse.parent.mkdir(exist_ok=True)
    sparse.write_bytes(trimesh.exchange.ply.export_ply(cloud))
    cameras=write(root/'input'/'cameras.txt','1 SIMPLE_RADIAL 100 100 100 50 50 0\n')
    lines=[]
    points=write(root/'input'/'points3D.txt',''.join(f'{i+1} {v[0]} {v[1]} {v[2]} 100 100 100 0.2\n' for i,v in enumerate(xyz)))
    photos=[]
    for j in range(1,4):
        name=f'frame_{j:06}.jpg'; p=root/'input'/name
        Image.new('RGB',(100,100),(150,100,100)).save(p)
        photos.append(p)
        coords=[]
        for i,v in enumerate(xyz):
            coords.extend([str(50+100*v[0]/v[2]),str(50+100*v[1]/v[2]),str(i+1)])
        lines.extend([f'{j} 1 0 0 0 0 0 0 1 {name}\n',' '.join(coords)+'\n'])
    images=write(root/'input'/'images.txt',''.join(lines))
    return dict(root=root,civil_manifest=manifest,sparse_ply=sparse,cameras_txt=cameras,images_txt=images,
                points3d_txt=points,photo_files=photos,run_id='geom-civil-camera-synthetic-01')


def test_registration_is_never_inferred_from_colmap_reprojection(tmp_path):
    args=fixture(tmp_path);b=g.make_review(**args)
    out=tmp_path/'runs'/args['run_id']/'geometry'
    assert b['registration_status']=='UNREGISTERED' and b['metric_scale_verified'] is False
    qa=json.loads((out/'registration_qa.json').read_text())
    assert qa['registered_colmap_view_count']==3
    assert all(v['sparse_reprojection']['median_px']<1e-10 for v in qa['reviewed_photo_models'])
    assert all(v['depth_fusion_projection_status']=='WITHHELD_UNREGISTERED' for v in qa['reviewed_photo_models'])
    assert len(list(out.glob('real_photo_shape3_review_*.png')))==3
    assert (out/'round3_shape3_photo_review.html').exists()
    f=json.loads((out/'feature_evidence.json').read_text())['features']
    assert len(f)==3 and all(x['image_support_confirmed_count']==0 for x in f)
    assert json.loads((out/'unknown_regions.json').read_text())['point_count']==0
    assert f[0]['sampled_source_support_points'][0]['point_index']==0


def test_source_hash_tampering_fails_closed(tmp_path):
    x=fixture(tmp_path)
    p=x['civil_manifest'].parent/'planes.json';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA-256'):
        g.make_review(**x)
    assert not (tmp_path/'runs'/x['run_id']).exists()


def test_nonmatching_photo_name_fails_closed(tmp_path):
    x=fixture(tmp_path);x['photo_files'][0].rename(x['photo_files'][0].with_name('not_registered.jpg'))
    x['photo_files'][0]=x['photo_files'][0].with_name('not_registered.jpg')
    with pytest.raises(ValueError,match='explicitly indexed COLMAP'):
        g.make_review(**x)
    assert not (tmp_path/'runs'/x['run_id']).exists()


def test_invalid_quaternion_fails_closed(tmp_path):
    x=fixture(tmp_path)
    p=x['images_txt'];p.write_text(p.read_text().replace('1 1 0 0 0 0 0 0 1','1 0 0 0 0 0 0 0 1',1))
    with pytest.raises(ValueError,match='quaternion'):
        g.make_review(**x)
    assert not (tmp_path/'runs'/x['run_id']).exists()


def test_bad_camera_source_dimensions_fail(tmp_path):
    x=fixture(tmp_path)
    p=x['cameras_txt'];p.write_text('1 SIMPLE_RADIAL 90 90 100 50 50 0\n')
    with pytest.raises(ValueError,match='dimension'):
        g.make_review(**x)


def test_dedicated_run_is_create_only(tmp_path):
    x=fixture(tmp_path);g.make_review(**x)
    with pytest.raises(FileExistsError):
        g.make_review(**x)


def test_reflection_does_not_count_as_registration(tmp_path):
    x=fixture(tmp_path);g.make_review(**x)
    data=json.loads((tmp_path/'runs'/x['run_id']/'geometry'/'registration_qa.json').read_text())
    p=data['sparse_depth_geometric_proximity']
    assert 'reflection_x_control' in p and data['registration_verified'] is False
    assert data['can_project_depth_features_to_images'] is False


def test_sparse_ply_and_points3d_must_match(tmp_path):
    x=fixture(tmp_path)
    p=x['points3d_txt']; lines=p.read_text().splitlines()
    a=lines[0].split();a[3]='40.0';lines[0]=' '.join(a)
    p.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='Sparse PLY is not'):
        g.make_review(**x)
    assert not (tmp_path/'runs'/x['run_id']).exists()