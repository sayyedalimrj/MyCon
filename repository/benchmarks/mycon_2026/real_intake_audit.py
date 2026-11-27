"""Owner C: non-destructive audit of real upstream civil geometry receipts.

This module independently recomputes COLMAP's *in-sample* sparse reprojection
error and verifies SHA256 receipts. It NEVER estimates metric surface accuracy,
validates depth-to-camera registration, scores a held-out render or claims
physical construction progress. Reads zip members without extraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
import statistics
import zipfile

SCHEMA = 'mycon.validation.real_intake_audit.v1'
MANIFEST = 'runs/geom-civil-real-depth-20261113-final1/geometry/civil_geometry_manifest.json'
REGISTRATION = 'runs/geom-civil-camera-20261114-real-depth-002/geometry/registration_qa.json'
CAMERAS = 'inputs/colmap_text/cameras.txt'
IMAGES = 'inputs/colmap_text/images.txt'
POINTS = 'inputs/colmap_text/points3D.txt'
IMAGE_DIR = 'inputs/captured_images/'


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def safe_member(name):
    if not isinstance(name,str) or not name or '\\' in name or name.startswith('/'):
        raise ValueError('unsafe_member_path')
    parts=PurePosixPath(name).parts
    if '..' in parts or '.' in parts or PurePosixPath(name).is_absolute():
        raise ValueError('unsafe_member_path')
    return name


def member_sha(z, name):
    safe_member(name)
    h=hashlib.sha256()
    with z.open(name) as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def qrotation(w,x,y,z):
    norm=math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm<=0:raise ValueError('bad_quaternion')
    w,x,y,z=(v/norm for v in (w,x,y,z))
    return ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
            (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
            (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))


def reproj_error(xyz, point, camera):
    """COLMAP SIMPLE_RADIAL world-to-camera sparse reprojection in pixels."""
    rot,t=xyz
    coords=[sum(rot[i][k]*point[k] for k in range(3))+t[i] for i in range(3)]
    if coords[2]<=1e-12:return None
    u,v=coords[0]/coords[2], coords[1]/coords[2]
    f,cx,cy,k1=camera
    r2=u*u+v*v;radial=1+k1*r2
    return (f*u*radial+cx,f*v*radial+cy)


def parse_cameras(z):
    ret={}
    for line in z.read(CAMERAS).decode('utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        parts=line.split()
        if len(parts)!=8 or parts[1]!='SIMPLE_RADIAL':raise ValueError('unsupported_intrinsics')
        ret[int(parts[0])]=(float(parts[4]),float(parts[5]),float(parts[6]),float(parts[7]))
    return ret


def parse_sparse_points(z):
    pts={}
    with io.TextIOWrapper(z.open(POINTS),encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():continue
            row=line.split()
            if len(row)<4:raise ValueError('invalid_points3D')
            pid=int(row[0]);pt=tuple(float(v) for v in row[1:4])
            if pid in pts or not all(math.isfinite(v) for v in pt):raise ValueError('invalid_or_duplicate_sparse_point')
            pts[pid]=pt
    return pts


def recompute_sparse_reprojection(z, cameras, pts):
    results=[]
    with io.TextIOWrapper(z.open(IMAGES),encoding='utf-8') as source:
        it=iter(source)
        for header in it:
            if header.startswith('#') or not header.strip():continue
            try:obsline=next(it)
            except StopIteration:raise ValueError('missing_colmap_observations') from None
            tokens=header.split()
            if len(tokens)<10:raise ValueError('bad_colmap_image_header')
            iid=int(tokens[0]);q=tuple(float(x) for x in tokens[1:5]);t=tuple(float(x) for x in tokens[5:8]);cam=int(tokens[8]);name=tokens[9]
            if cam not in cameras:raise ValueError('unknown_camera_id')
            rot=qrotation(*q)
            xyz=(rot,t)
            obs=obsline.split()
            if len(obs)%3:raise ValueError('bad_colmap_observation_triplets')
            errors=[];valid=0
            for i in range(0,len(obs),3):
                pid=int(obs[i+2]);
                if pid == -1:continue
                if pid not in pts:raise ValueError('sparse_point_id_not_found')
                valid+=1
                pred=reproj_error(xyz,pts[pid],cameras[cam])
                if pred is None:continue
                x=float(obs[i]);y=float(obs[i+1]);e=math.hypot(pred[0]-x,pred[1]-y)
                if math.isfinite(e):errors.append(e)
            errors.sort()
            def quantile(p):
                if not errors:return None
                k=(len(errors)-1)*p;lo=math.floor(k);hi=math.ceil(k)
                return errors[lo]+(errors[hi]-errors[lo])*(k-lo)
            results.append({'image_id':iid,'image_name':name,'observations_matched_3d':valid,
                            'valid_projected_observations':len(errors),
                            'median_error_px':quantile(.5),'p95_error_px':quantile(.95),
                            'max_error_px':errors[-1] if errors else None,
                            'evidence_class':'real_internal_colmap_fit_not_heldout',
                            'scientific_eligible':False})
    if len({r['image_name'] for r in results})!=len(results):raise ValueError('duplicate_image_name')
    return results


def audit_zip(package, output_root, run_id, *, expected_package_sha256=None):
    """Writes a new run dir; refuses overwrites and any failed source receipt."""
    if not isinstance(run_id,str) or not re.fullmatch(r'validation-c-[A-Za-z0-9][A-Za-z0-9._-]*',run_id) or '..' in run_id:
        raise ValueError('invalid_run_id')
    path=Path(package).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    sha=digest_file(path)
    if expected_package_sha256 and sha!=expected_package_sha256:raise ValueError('package_sha256_mismatch')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('duplicate_zip_member')
        for name in names:safe_member(name)
        required=[MANIFEST,REGISTRATION,CAMERAS,IMAGES,POINTS]
        if not set(required).issubset(names):raise ValueError('missing_required_artifact')
        manifest=json.loads(z.read(MANIFEST))
        registration=json.loads(z.read(REGISTRATION))
        if manifest.get('schema_version')!='mycon.civil_geometry_review.v1':raise ValueError('unexpected_geometry_schema')
        if registration.get('schema_version')!='mycon.civil_camera_geometry_bridge.v1':raise ValueError('unexpected_registration_schema')
        checked=[];errors=[]
        for label,record in sorted(manifest.get('outputs',{}).items()):
            name=safe_member(record.get('relative_path'))
            present=name in names
            actual=member_sha(z,name) if present else None
            valid=present and actual==record.get('sha256') and z.getinfo(name).file_size==record.get('bytes')
            checked.append({'asset':label,'relative_path':name,'expected_sha256':record.get('sha256'),
                            'actual_sha256':actual,'hash_and_size_ok':bool(valid)})
            if not valid:errors.append(f'civil_output_hash_or_size_mismatch:{label}')
        cross={'civil_geometry_manifest.json':MANIFEST,'colmap_sparse_unscaled.ply':'inputs/colmap_sparse_unscaled.ply',
               'cameras.txt':CAMERAS,'images.txt':IMAGES,'points3D.txt':POINTS}
        for label,expected in registration.get('source_file_sha256',{}).items():
            if label not in cross:continue
            name=cross[label];actual=member_sha(z,name) if name in names else None
            ok=actual==expected
            checked.append({'asset':label,'relative_path':name,'expected_sha256':expected,
                            'actual_sha256':actual,'hash_and_size_ok':bool(ok)})
            if not ok:errors.append(f'camera_source_sha256_mismatch:{label}')
        # Original photographs independently SHA checked, not considered held-out proof.
        for v in registration.get('reviewed_photo_models',[]):
            name=safe_member(v.get('original_source_path'))
            actual=member_sha(z,name) if name in names else None
            ok=actual==v.get('source_image_sha256')
            checked.append({'asset':'source_photo:'+v.get('name',''), 'relative_path':name,
                            'expected_sha256':v.get('source_image_sha256'), 'actual_sha256':actual,'hash_and_size_ok':bool(ok)})
            if not ok:errors.append('photo_sha256_mismatch:'+v.get('name',''))
        if errors:raise ValueError(';'.join(errors))
        cameras=parse_cameras(z);points=parse_sparse_points(z)
        views=recompute_sparse_reprojection(z,cameras,points)
        for view in registration.get('reviewed_photo_models',[]):
            matching=[v for v in views if v['image_name']==view['name']]
            if len(matching)!=1:errors.append('colmap_reviewed_photo_missing:'+view['name']);continue
            a=matching[0];b=view.get('sparse_reprojection',{})
            if (a['valid_projected_observations']!=b.get('num_registered_2d_3d') or
                abs(a['median_error_px']-b.get('median_px',float('inf')))>1e-4 or
                abs(a['p95_error_px']-b.get('p95_px',float('inf')))>1e-4):
                errors.append('recomputed_colmap_sparse_reprojection_mismatch:'+view['name'])
        if errors:raise ValueError(';'.join(errors))
        # Explicit upstream flags remain authoritative; an internal pixel fit cannot unlock them.
        metric=bool(manifest.get('metric_scale_verified')) and bool(registration.get('metric_scale_verified'))
        reg=bool(manifest.get('registration_verified')) and bool(registration.get('registration_verified'))
        medians=[v['median_error_px'] for v in views if v['median_error_px'] is not None]
        p95s=[v['p95_error_px'] for v in views if v['p95_error_px'] is not None]
        if not medians:raise ValueError('no_finite_sparse_reprojection')
        counts={'raw_sparse_points':len(points),'colmap_image_records':len(views),
                'colmap_projected_correspondences':sum(v['valid_projected_observations'] for v in views),
                'input_artifacts_sha256_verified':len(checked)}
        if not views:raise ValueError('no_camera_records')
        if registration.get('registered_colmap_view_count')!=len(views):raise ValueError('colmap_camera_count_mismatch')
        report={
            'schema_version':SCHEMA,'run_id':run_id,'package_sha256':sha,'package_filename':path.name,
            'data_kind':'real_source_diagnostic_only','source_producer':'A/Civil Geometry PR#42',
            'source_schema_versions':[manifest['schema_version'],registration['schema_version']],
            'source_run_ids':[manifest.get('run_id'),registration.get('run_id')],
            'coordinate_frames':{'depth':manifest.get('coordinate_frame'),'COLMAP_cameras':registration.get('camera_pose_frame')},
            'units':{'depth':manifest.get('units'),'COLMAP_cameras':'arbitrary_COLMAP_units'},
            'metric_scale_verified':metric,'depth_camera_registration_verified':reg,
            'independent_field_gt_available':False,'heldout_test_set_verified':False,
            'eligible_for_scientific_metric_claims':False,
            'counts':counts,
            'checks':{'member_sha256':'passed','COLMAP_sparse_reprojection_independent_recalculation':'passed',
                      'source_register_flag':registration.get('status')},
            'real_internal_diagnostics_not_for_thesis_comparison':{
                'median_of_per_image_median_reprojection_px':statistics.median(medians),
                'median_of_per_image_p95_reprojection_px':statistics.median(p95s),
                'n_images_with_valid_sparse_correspondences':len(medians),
                'independent_test_metric':False},
            'upstream_civil_comparison_ready_source_manifest_in_zip':manifest.get('source_manifest_relative_path') in names,
            'scientific_metrics':{'NVS_PSNR_SSIM_LPIPS':'not_evaluated',
                                  'geometry_accuracy_completeness_f1':'not_evaluated',
                                  'metric_registration':'not_evaluated',
                                  'physical_progress':'not_evaluated',
                                  '6999_vs_new':'not_evaluated',
                                  '192_vs_more_frames':'not_evaluated'},
            'scientific_blockers':['no_locked_independent_real_Ground_Truth',
                                   'no_independent_holdout',
                                   'depth_cloud_and_COLMAP_cameras_UNREGISTERED',
                                   'no_verified_metric_scale',
                                   'no_paired_6999_or_new_render_predictions',
                                   'original_comparison_ready_source_receipt_not_in_A_zip'],
            'provenance':'read_only_SHA_verified_authentic_A_package; COLMAP_internal_sparse_SfM_fit_only'
        }
    out=Path(output_root).resolve()/run_id
    out.mkdir(parents=True,exist_ok=False)
    (out/'real_input_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'source_asset_hashes.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['asset','relative_path','expected_sha256','actual_sha256','hash_and_size_ok']);writer.writeheader();writer.writerows(checked)
    for v in views:
        v.update(schema_version='mycon.validation.colmap_internal_reprojection.v1',
                 run_id=run_id,source_package_sha256=sha,
                 coordinate_frame='COLMAP_sfm_world_unscaled',units='px',
                 split='in_sample_optimization_fit_not_independent_test')
    with (out/'colmap_internal_reprojection.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(views[0]));writer.writeheader();writer.writerows(views)
    return report


def main():
    p=argparse.ArgumentParser(description='Read-only real-source intake audit; NOT external GT evaluation')
    p.add_argument('--package',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--expected-package-sha256')
    a=p.parse_args()
    print(json.dumps(audit_zip(a.package,a.output_root,a.run_id,expected_package_sha256=a.expected_package_sha256),indent=2))

if __name__=='__main__':main()
