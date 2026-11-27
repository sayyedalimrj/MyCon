"""Synthetic-only fixture tests for independent real-source intake audit. Not field results."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.mycon_2026.real_intake_audit import (
    safe_member, qrotation, reproj_error, parse_cameras, parse_sparse_points,
    recompute_sparse_reprojection, audit_zip, MANIFEST, REGISTRATION, CAMERAS, IMAGES, POINTS)


def _hash(s):return hashlib.sha256(s).hexdigest()


def _synthetic_pkg(path, *, tamper=False, unregistered=True, skipped_photo=False):
    g='runs/geom-civil-real-depth-20261113-final1/geometry/summary.txt'
    camera=b'1 SIMPLE_RADIAL 640 1138 843.15924740972991 320 569 0.043128330983852707\n'
    image=b'1 1 0 0 0 0 0 0 1 frame_000001.jpg\n320 569 1\n'
    points=b'1 0 0 1 255 255 255 0 1 0\n'
    photo=b'FAKE PHOTO FOR SYNTHETIC TEST ONLY'
    artifact=b'geometry sample'
    gmanifest={'schema_version':'mycon.civil_geometry_review.v1','run_id':'fixture-geom',
               'metric_scale_verified':False,'registration_verified':False,'coordinate_frame':'arbitrary',
               'units':'arbitrary_reconstruction_units','source_manifest_relative_path':'absent.json',
               'outputs':{'summary.txt':{'relative_path':g,'sha256':_hash(artifact),'bytes':len(artifact)}}}
    gb=json.dumps(gmanifest).encode()
    hashes={'civil_geometry_manifest.json':_hash(gb),'cameras.txt':_hash(camera),
            'images.txt':_hash(image),'points3D.txt':_hash(points)}
    reg={'schema_version':'mycon.civil_camera_geometry_bridge.v1','run_id':'fixture-cam',
         'status':'UNREGISTERED' if unregistered else 'REGISTERED',
         'registration_verified':not unregistered,'metric_scale_verified':False,
         'camera_pose_frame':'COLMAP_sfm_world_unscaled','registered_colmap_view_count':1,
         'source_file_sha256':hashes,
         'reviewed_photo_models':[] if skipped_photo else [{'name':'frame_000001.jpg',
            'original_source_path':'inputs/captured_images/frame_000001.jpg','source_image_sha256':_hash(photo),
            'sparse_reprojection':{'num_registered_2d_3d':1,'median_px':0,'p95_px':0}}]}
    entries={MANIFEST:gb,REGISTRATION:json.dumps(reg).encode(),CAMERAS:camera,IMAGES:image,POINTS:points,
             'inputs/captured_images/frame_000001.jpg':photo,g:b'TAMPERED' if tamper else artifact}
    with zipfile.ZipFile(path,'w') as z:
        for k,v in entries.items():z.writestr(k,v)
    return path


def test_projection_zero_residual():
    assert reproj_error((qrotation(1,0,0,0),(0,0,0)),(0,0,1),(843.15924740972991,320,569,.043128330983852707))==(320,569)


def test_projection_rejects_behind_camera():
    assert reproj_error((qrotation(1,0,0,0),(0,0,0)),(0,0,-1),(900,320,569,0)) is None


def test_member_path_rejects_escape():
    for value in ('../etc/passwd','/tmp/x','..\\foo',''):
        with pytest.raises(ValueError):safe_member(value)


def test_synthetic_pkg_hash_and_unregistered_gate(tmp_path):
    pack=_synthetic_pkg(tmp_path/'in.zip')
    report=audit_zip(pack,tmp_path/'results','validation-c-synth-fixture-001')
    assert report['counts']['colmap_image_records']==1
    assert report['counts']['colmap_projected_correspondences']==1
    assert report['metric_scale_verified'] is False
    assert report['depth_camera_registration_verified'] is False
    assert report['independent_field_gt_available'] is False
    assert report['scientific_metrics']['geometry_accuracy_completeness_f1']=='not_evaluated'
    assert report['real_internal_diagnostics_not_for_thesis_comparison']['median_of_per_image_median_reprojection_px']==0
    assert (tmp_path/'results'/'validation-c-synth-fixture-001'/'colmap_internal_reprojection.csv').exists()


def test_reject_tampered_member(tmp_path):
    pack=_synthetic_pkg(tmp_path/'bad.zip',tamper=True)
    with pytest.raises(ValueError,match='sha256|mismatch'):
        audit_zip(pack,tmp_path,'validation-c-invalid-001')


def test_refuse_run_overwrite(tmp_path):
    pack=_synthetic_pkg(tmp_path/'in.zip')
    audit_zip(pack,tmp_path/'res','validation-c-synth-fixture-001')
    with pytest.raises(FileExistsError):audit_zip(pack,tmp_path/'res','validation-c-synth-fixture-001')


def test_reject_expected_package_hash(tmp_path):
    pack=_synthetic_pkg(tmp_path/'in.zip')
    with pytest.raises(ValueError,match='package_sha256_mismatch'):
        audit_zip(pack,tmp_path,'validation-c-invalid-002',expected_package_sha256='0'*64)


def test_reject_unsafe_runid(tmp_path):
    pack=_synthetic_pkg(tmp_path/'in.zip')
    with pytest.raises(ValueError,match='invalid_run_id'):
        audit_zip(pack,tmp_path,'../escape')


def test_reject_incomplete_camera_registration_receipt(tmp_path):
    pack=_synthetic_pkg(tmp_path/'in.zip')
    with zipfile.ZipFile(pack,'a') as z:
        z.writestr('duplicate.txt','synthetic harmless')
    report=audit_zip(pack,tmp_path/'res','validation-c-synth-fixture-002')
    assert report['checks']['source_register_flag']=='UNREGISTERED'
    assert report['eligible_for_scientific_metric_claims'] is False