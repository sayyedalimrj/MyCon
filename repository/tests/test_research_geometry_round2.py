from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pytest
from pipeline.stage_08_bim_eval.bim_geometry_evidence import (
    SCHEMA, sample_ifc_triangles, verify_independent_controls,
    parse_camera_manifest, classify_visibility, create_bim_geometry_evidence,
)

# Two genuine IFC-owning triangles: front z=2 and back z=4, no AABB inference.
def geometry():
    vertices=np.asarray([[-1,-1,2],[1,-1,2],[0,1,2],[-1,-1,4],[1,-1,4],[0,1,4]],float)
    faces=np.asarray([[0,1,2],[3,4,5]],int)
    elements=[dict(global_id="WALL_001",ifc_class="IfcWall",vertex_start=0,vertex_count=3,face_count=1),
              dict(global_id="COLUMN_001",ifc_class="IfcColumn",vertex_start=3,vertex_count=3,face_count=1)]
    return vertices,faces,elements

def inputs(root:Path, *,control=True, with_camera=True):
    root.mkdir(parents=True,exist_ok=True)
    ifc=root/'orig.ifc';ifc.write_text('fixture model',encoding='utf-8')
    scan=root/'orig_scan.ply';scan.write_bytes(b'fixture scan')
    report=root/'registration_report.json';report.write_text(json.dumps({'quality_gate':{'status':'pass','passed':True}}))
    anchors=root/'independent_survey.json'
    anchors.write_text(json.dumps({'source':'independent_survey','units':'m','survey_id':'independent-2026',
      'controls':[{'id':k,'scan_xyz':p,'bim_xyz':p,'role':'validation','used_for_registration':False}
      for k,p in [('c1',[0,0,0]),('c2',[3,0,0]),('c3',[0,4,0])]]}))
    camera=root/'cameras.json'
    img=root/'original_capture.png';img.write_bytes(b'fixture camera bytes')
    camera.write_text(json.dumps({'coordinate_frame':'bim_world','units':'m','cameras':[{
      'view_id':'v0001','capture_session_id':'session-01','image_path':str(img),
      'width':640,'height':480,'intrinsics':[[300,0,320],[0,300,240],[0,0,1]],
      'world_to_camera_4x4':np.eye(4).tolist(),
      'qa':{'calibrated':True,'frame_verified':True}}]}))
    return ifc,scan,report,anchors if control else None,camera if with_camera else None

def test_independent_controls_require_unused_survey_and_bim_meters(tmp_path):
    _,_,_,anchors,_=inputs(tmp_path)
    assert verify_independent_controls(anchors,np.eye(4))['passed']
    m=np.eye(4);m[0,3]=0.5
    bad=verify_independent_controls(anchors,m)
    assert not bad['passed'];assert bad['rmse_m']>0.4
    data=json.loads(anchors.read_text());data['controls'][0]['used_for_registration']=True
    anchors.write_text(json.dumps(data))
    assert not verify_independent_controls(anchors,np.eye(4))['passed']

def test_samples_are_stable_triangle_owned_no_bbox_assignment():
    v,f,e=geometry()
    a,qa=sample_ifc_triangles(v,f,e,'abc')
    b,qb=sample_ifc_triangles(v,f,e,'abc')
    assert a==b and qa==qb
    assert len(a)==2
    assert {x['global_id'] for x in a}=={'WALL_001','COLUMN_001'}
    assert all(x['normal']==[0.0,0.0,1.0] for x in a)
    assert len({x['sample_id'] for x in a})==2

def test_overlap_coincidence_ambiguous_never_double_owned():
    v,f,e=geometry();v[3:]=v[:3]
    a,_=sample_ifc_triangles(v,f,e,'abc')
    assert len(a)==2
    assert all(x['global_id'] is None and x['ownership']=='ambiguous_coincident_ifc_surfaces' for x in a)
    assert all(x['candidate_global_ids']==['COLUMN_001','WALL_001'] for x in a)

def test_shared_centroid_but_distinct_triangles_keep_source_ifc_ownership():
    v,f,e=geometry()
    # A vertical triangle and horizontal triangle share centroid (0,0,2),
    # but not geometry. Centroid-only overlap detection is a false positive.
    v[3:]=np.array([[0,-2,2],[0,1,1],[0,1,3]],dtype=float)
    a,_=sample_ifc_triangles(v,f,e,'abc')
    assert len(a)==2
    assert {x['global_id'] for x in a}=={'WALL_001','COLUMN_001'}
    assert all(x['ownership']=='ifc_triangle_face' for x in a)
    assert len({x['sample_id'] for x in a})==2

def test_visibility_real_rays_visible_and_occluded_on_cpu_fixture(tmp_path):
    v,f,e=geometry();samples,_=sample_ifc_triangles(v,f,e,'abc')
    *_,camera=inputs(tmp_path)
    cams=parse_camera_manifest(camera,tmp_path)
    records=classify_visibility(samples,cams,v,f,metric_ready=True)
    states={r['global_id']:r['state'] for r in records}
    assert states=={'WALL_001':'visible','COLUMN_001':'occluded'}
    assert all(r['method']=='raycast_zbuffer' for r in records)
    unknown=classify_visibility(samples,cams,v,f,metric_ready=False)
    assert all(r['state']=='unknown' and r['visible_sample_count'] is None for r in unknown)

def test_manifest_a_to_b_hashes_stable_and_fail_closed(tmp_path):
    v,f,e=geometry();ifc,scan,report,controls,camera=inputs(tmp_path)
    result=create_bim_geometry_evidence(project_root=tmp_path,run_id='geom-test-real-control',
      project_id='site01',ifc_path=ifc,scan_path=scan,registration_report_path=report,
      transform=np.eye(4),vertices=v,triangles=f,elements=e,ifc_units='METRE',
      extraction_source='ifcopenshell',camera_path=camera,independent_controls_path=controls,code_commit="0"*40)
    assert result['schema_version']==SCHEMA and result['metric_scale_verified'] is True
    assert result['status']=='geometry_and_visibility_evaluable'
    for key in ('surface_samples','camera_poses','element_visibility'):
      p=tmp_path/result[key]['path'];assert p.is_file()
      assert hashlib.sha256(p.read_bytes()).hexdigest()==result[key]['sha256']
    with pytest.raises(FileExistsError):
      create_bim_geometry_evidence(project_root=tmp_path,run_id='geom-test-real-control',
        project_id='site01',ifc_path=ifc,scan_path=scan,registration_report_path=report,
        transform=np.eye(4),vertices=v,triangles=f,elements=e,ifc_units='METRE',
        extraction_source='ifcopenshell',camera_path=camera,independent_controls_path=controls,code_commit="0"*40)
    no_control=create_bim_geometry_evidence(project_root=tmp_path,run_id='geom-test-without-control',
      project_id='site01',ifc_path=ifc,scan_path=scan,registration_report_path=report,
      transform=np.eye(4),vertices=v,triangles=f,elements=e,ifc_units='METRE',
      extraction_source='ifcopenshell',camera_path=camera,independent_controls_path=None,write=False)
    assert not no_control['manifest']['metric_scale_verified']
    assert all(row['state']=='unknown' for row in no_control['records']['element_visibility'])

def test_nonmetric_ifc_refuses_sample_coordinates(tmp_path):
    v,f,e=geometry();ifc,scan,report,controls,camera=inputs(tmp_path)
    result=create_bim_geometry_evidence(project_root=tmp_path,run_id='geom-mm-rejected',
      project_id='site01',ifc_path=ifc,scan_path=scan,registration_report_path=report,
      transform=np.eye(4),vertices=v,triangles=f,elements=e,ifc_units='MILLI METRE',
      extraction_source='ifcopenshell',camera_path=camera,independent_controls_path=controls,write=False)
    assert not result['manifest']['metric_scale_verified']
    assert not result['records']['surface_samples']

pytestmark=pytest.mark.lightweight

from pipeline.stage_08_bim_eval.bim_geometry_contract import validate_manifest

def test_validator_accepts_correct_fixture_and_rejects_tampering(tmp_path):
    v,f,e=geometry();ifc,scan,report,controls,camera=inputs(tmp_path)
    m=create_bim_geometry_evidence(project_root=tmp_path,run_id='geom-validation-001',
      project_id='site01',ifc_path=ifc,scan_path=scan,registration_report_path=report,
      transform=np.eye(4),vertices=v,triangles=f,elements=e,ifc_units='METRE',
      extraction_source='ifcopenshell',camera_path=camera,independent_controls_path=controls,code_commit="0"*40)
    path=Path(m['manifest_path'])
    assert validate_manifest(path,tmp_path)['valid']
    with (tmp_path/m['surface_samples']['path']).open('a') as fp:fp.write('garbage\n')
    result=validate_manifest(path,tmp_path)
    assert not result['valid'] and 'surface_samples_hash_mismatch' in result['reasons']

def test_validator_fail_closed_unknown_metric(tmp_path):
    v,f,e=geometry();ifc,scan,report,_,camera=inputs(tmp_path)
    m=create_bim_geometry_evidence(project_root=tmp_path,run_id='geom-validation-002',
      project_id='site01',ifc_path=ifc,scan_path=scan,registration_report_path=report,
      transform=np.eye(4),vertices=v,triangles=f,elements=e,ifc_units='METRE',
      extraction_source='ifcopenshell',camera_path=camera)
    result=validate_manifest(Path(m['manifest_path']),tmp_path)
    assert not result['valid'] and 'metric_registration_not_independently_verified' in result['reasons']

from pipeline.stage_08_bim_eval.research_frames import colmap_camera_to_bim_w2c

def test_colmap_camera_to_bim_keeps_rigid_rotation_with_sim3_scale():
    w2c=np.eye(4);w2c[0,3]=-3.0 # center at COLMAP x=3
    sim=np.eye(4);sim[:3,:3]*=2.0;sim[:3,3]=[10,20,30]
    result=colmap_camera_to_bim_w2c(w2c,sim)
    assert np.allclose(result[:3,:3],np.eye(3))
    assert np.allclose(result[:3,3],[-16,-20,-30])
    p_sfm=np.array([3.0,0,5.0])
    p_bim=sim[:3,:3]@p_sfm+sim[:3,3]
    p_cam=result[:3,:3]@p_bim+result[:3,3]
    assert np.allclose(p_cam,[0,0,10.0]) # metric 2x depth, same projection
    assert np.allclose(p_cam[:2]/p_cam[2],(p_sfm-np.array([3,0,0]))[:2]/5)


def test_colmap_axis_reflection_is_rejected():
    w2c=np.eye(4);w2c[0,0]=-1.0
    with pytest.raises(ValueError):
        colmap_camera_to_bim_w2c(w2c,np.eye(4))


def test_cli_cpu_mesh_json_real_subprocess(tmp_path):
    import subprocess, sys, os
    v,f,e=geometry();ifc,scan,report,controls,camera=inputs(tmp_path)
    mesh=tmp_path/'mesh_fixture.json';mesh.write_text(json.dumps({'vertices':v.tolist(),'triangles':f.tolist()}))
    elements=tmp_path/'elements.jsonl';elements.write_text(''.join(json.dumps(x)+'\n' for x in e))
    transform=tmp_path/'transform.json';transform.write_text(json.dumps({'matrix_4x4_scan_to_bim':np.eye(4).tolist()}))
    cmd=[sys.executable,'-m','pipeline.stage_08_bim_eval.bim_geometry_evidence',
         '--project-root',str(tmp_path),'--run-id','geom-cli-smoke-001',
         '--project-id','synthetic-site','--ifc',str(ifc),'--scan',str(scan),
         '--registration-report',str(report),'--transform-json',str(transform),
         '--mesh-json',str(mesh),'--elements-jsonl',str(elements),'--ifc-units','METRE',
         '--camera-json',str(camera),'--controls-json',str(controls),'--code-commit','0'*40]
    proc=subprocess.run(cmd,capture_output=True,text=True,check=False)
    assert proc.returncode==0,proc.stderr+proc.stdout
    m=tmp_path/'runs/geom-cli-smoke-001/reports/bim_geometry_evidence.v1.json'
    assert m.is_file()
    assert validate_manifest(m,tmp_path)['valid']
    second=subprocess.run(cmd,capture_output=True,text=True,check=False)
    assert second.returncode==2 and 'Refusing to overwrite' in second.stdout
