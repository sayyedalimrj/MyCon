"""Depth-to-geometry side product of LongSplat, deliberately NONMETRIC.

This reads LongSplat's *rendered* camera-Z depth arrays and pose JSON. The raw
renderer depth convention is a hypothesis to verify against independent site
controls; it is not asserted to be survey-quality metric depth.

No writes to MyCON Stage 5/8/9. Outputs are a visual/registration CANDIDATE.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


def _numpy():
    import numpy as np
    return np


def _write_ply(path: Path, xyz, rgb) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.empty(len(xyz), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                       ('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')):
        array[k] = xyz[:, i]
    for i,k in enumerate(('red','green','blue')):
        array[k] = rgb[:, i]
    with path.open('wb') as out:
        out.write((f'ply\nformat binary_little_endian 1.0\n'
                   f'comment LONGSPLAT_RENDERED_DEPTH_ARBITRARY_SCALE_NOT_SURVEYED\n'
                   f'element vertex {len(xyz)}\n'
                   'property float x\nproperty float y\nproperty float z\n'
                   'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                   'end_header\n').encode('ascii'))
        array.tofile(out)


def backproject(camera: dict, depth, pixels):
    """Camera R is c2w (LongSplat camera_to_JSON); T is world-to-camera offset.

    LongSplat render_set: w2c[:3,:3] = camera.R.t(), w2c[:3,3] = camera.T.
    Therefore world_xyz = camera.R @ (camera_xyz - T), not an arbitrary Euler transform.
    """
    import numpy as np
    h,w = depth.shape
    fx=float(camera['Focalx']); fy=float(camera['Focaly'])
    if fx<=0 or fy<=0 or not np.isfinite([fx,fy]).all():
        raise ValueError('Camera focal values invalid')
    rot=np.asarray(camera['R'], dtype=np.float64)
    shift=np.asarray(camera['T'], dtype=np.float64)
    if rot.shape!=(3,3) or shift.shape!=(3,) or abs(np.linalg.det(rot)-1)>0.03:
        raise ValueError('Camera pose invalid')
    y,x=pixels[:,0],pixels[:,1]
    z=depth[y,x]
    view_xyz=np.column_stack(((x-(w-1)/2)/fx*z, (y-(h-1)/2)/fy*z, z))
    return (view_xyz - shift) @ rot.T


def extract_depth_candidate(depth_folder: Path, camera_json: Path, images_folder: Path,
                            output_folder: Path, *, max_points: int=1_500_000) -> dict:
    import numpy as np
    from PIL import Image
    if max_points < 10_000 or max_points > 10_000_000:
        raise ValueError('max_points must be between 10k and 10m')
    cameras=json.loads(camera_json.read_text(encoding='utf-8'))
    if not isinstance(cameras,list) or len(cameras)<3:
        raise RuntimeError('At least three trained camera poses required')
    if output_folder.exists() and any(output_folder.iterdir()):
        existing=output_folder/'geometry_report.json'
        if existing.is_file():
            prev=json.loads(existing.read_text())
            ply=output_folder/'depth_fused_arbitrary_units.ply'
            if prev.get('status')=='DEPTH_CANDIDATE_CREATED' and ply.is_file():
                return {'ply_path':ply,'report_path':existing}
        raise RuntimeError('Partial geometry directory exists: no overwrite of unverified prior results')
    output_folder.mkdir(parents=True, exist_ok=True)
    xyzs=[];rgbs=[];summaries=[]; matched=0
    per_frame_limit=max(100,min(15_000,max_points//len(cameras)))
    for cam in cameras:
        label=str(cam.get('image_name') or '')
        if not label or Path(label).name!=label:
            raise ValueError('Unsafe/missing LongSplat camera image_name')
        stem=Path(label).stem
        depth_path=depth_folder/(label+'.npy')
        if not depth_path.exists():
            depth_path=depth_folder/(stem+'.npy')
        if not depth_path.is_file():
            summaries.append({'image':label,'status':'DEPTH_MISSING'});continue
        d=np.load(depth_path, allow_pickle=False)
        if d.ndim!=2 or d.size<1000:
            summaries.append({'image':label,'status':'DEPTH_SHAPE_INVALID'});continue
        h,w=d.shape
        valid=np.isfinite(d)&(d>0)
        if valid.sum()<100:
            summaries.append({'image':label,'status':'DEPTH_EMPTY'});continue
        # Robust fixed-budget view coverage; unlike 96s DA3 preview, total cap
        # here is configurable and explicitly recorded for scientific audit.
        count=0
        step=max(1,int(np.sqrt(d.size/per_frame_limit)))
        yy,xx=np.mgrid[0:h:step,0:w:step]
        pix=np.column_stack((yy.ravel(),xx.ravel()))
        pix=pix[valid[pix[:,0],pix[:,1]]]
        if len(pix)>per_frame_limit:
            pix=pix[np.linspace(0,len(pix)-1,per_frame_limit,dtype=int)]
        if not len(pix):continue
        depth=np.asarray(d,dtype=np.float64)
        try:
            xyz=backproject(cam,depth,pix)
        except (ValueError,KeyError,TypeError) as exc:
            summaries.append({'image':label,'status':'BAD_CAMERA','error':str(exc)});continue
        source=None
        for suffix in ('.jpg','.jpeg','.png'):
            trial=images_folder/(stem+suffix)
            if trial.is_file():source=trial;break
        if source is None:
            summaries.append({'image':label,'status':'IMAGE_MISSING'});continue
        with Image.open(source) as im:
            colors=np.asarray(im.convert('RGB').resize((w,h),resample=Image.Resampling.BILINEAR),dtype=np.uint8)
        color=colors[pix[:,0],pix[:,1]]
        finite=np.isfinite(xyz).all(axis=1)
        xyzs.append(xyz[finite].astype(np.float32))
        rgbs.append(color[finite]);count=int(finite.sum());matched+=1
        summaries.append({'image':label,'status':'SAMPLED','sample_count':count})
    if matched<3 or sum(len(x) for x in xyzs)<10_000:
        raise RuntimeError(f'Rendered geometry cannot be exported: only {matched} valid views / too few points')
    xyz=np.concatenate(xyzs);rgb=np.concatenate(rgbs)
    # No decimation or smoothing disguised as an accurate surface reconstruction.
    dest=output_folder/'depth_fused_arbitrary_units.ply'
    _write_ply(dest,xyz,rgb)
    report={
        'schema':'mycon.longsplat.depth_candidate.v1','status':'DEPTH_CANDIDATE_CREATED',
        'geometry_representation':'rendered_depth_backprojection_with_train_poses',
        'points':len(xyz),'total_pose_frames':len(cameras),'matched_depth_frames':matched,
        'camera_convention':'LongSplat R=camera_to_world, T=world_to_camera_translation',
        'depth_convention':'ASSUMED_CAMERA_Z; MUST_VERIFY_WITH_PROJECT_REFERENCE',
        'source_depth_folder':str(depth_folder),'source_pose_json':str(camera_json),
        'relative_scale_only':True,'metric_truth':False,'bim_equivalence_approved':False,
        'warning':'Do not use this PLY for dimensions or progress until calibrated and independently validated. '
                  'Novel-view splats are an appearance representation, not certified scan geometry.',
        'per_camera':summaries,
    }
    report_path=output_folder/'geometry_report.json'
    from .longsplat_research import atomic_json
    atomic_json(report_path,report)
    return {'ply_path':dest,'report_path':report_path}