"""Held-out Sim(3) registration of a LongSplat depth candidate to measured site/BIM anchors.

This is an optional geometry adapter, NOT a declaration that an appearance
model alone can be trusted for structural dimensions. Held-out anchors must be
independent of the fit anchors and MUST come from measurements in meters.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

FIELDS=('source_x','source_y','source_z','target_x_m','target_y_m','target_z_m','split')


def estimate_sim3(points_source, points_target):
    import numpy as np
    source=np.asarray(points_source,dtype=np.float64)
    target=np.asarray(points_target,dtype=np.float64)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<4:
        raise ValueError('At least four paired fitting anchors required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Anchor positions must be finite')
    src_mean=source.mean(axis=0);tar_mean=target.mean(axis=0)
    x=source-src_mean;y=target-tar_mean
    if np.linalg.matrix_rank(x,tol=1e-6)<2 or np.linalg.matrix_rank(y,tol=1e-6)<2:
        raise ValueError('Degenerate collinear anchor geometry')
    u,_,vt=np.linalg.svd((x.T@y))
    reflection=np.diag([1.,1.,np.linalg.det(u@vt)])
    rot=(u@reflection@vt).T
    rotated=x@rot.T
    scale=float((rotated*y).sum()/(x*x).sum())
    if scale<=0 or not np.isfinite(scale):raise ValueError('Invalid metric scale')
    translation=tar_mean - scale*(src_mean@rot.T)
    return scale,rot,translation


def apply_sim3(points,scale,rotation,translation):
    return scale*(points@rotation.T)+translation


def read_anchors(path: Path):
    import numpy as np
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in FIELDS):
            raise ValueError('Anchor CSV must include '+', '.join(FIELDS))
        rows=list(reader)
    seen=set(); fit=[];test=[]
    for i,row in enumerate(rows):
        if 'id' in row:
            if row['id'] in seen:raise ValueError('Duplicate anchor ID')
            seen.add(row['id'])
        key=row['split'].strip().lower()
        if key not in ('fit','holdout'):
            raise ValueError('split must be fit or holdout')
        a=[float(row[k]) for k in FIELDS[:3]]
        b=[float(row[k]) for k in FIELDS[3:6]]
        (fit if key=='fit' else test).append((a,b))
    if len(fit)<4 or len(test)<2:
        raise ValueError('Require >=4 fit and >=2 independent holdout site anchors')
    return np.array([x[0] for x in fit]),np.array([x[1] for x in fit]),np.array([x[0] for x in test]),np.array([x[1] for x in test])


def read_ply_binary(path: Path):
    """Accept only controlled binary_xyz_rgb candidate from MyCON, not a Gaussian PLY."""
    import numpy as np
    with path.open('rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line or len(header)>100:raise ValueError('Invalid PLY header')
            header.append(line.strip().decode('ascii'))
            if line.strip()==b'end_header':break
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Only known MyCON binary PLY format accepted')
        if not any('LONGSPLAT_RENDERED_DEPTH_ARBITRARY_SCALE' in x for x in header):
            raise ValueError('Not a recognized depth candidate. Gaussian PLY is not acceptable BIM geometry.')
        n=int(next(s.split()[-1] for s in header if s.startswith('element vertex')))
        if not 100<=n<=12_000_000:raise ValueError('Invalid point cloud size')
        dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
        points=np.fromfile(f,dtype=dtype,count=n)
        if len(points)!=n:raise ValueError('Truncated point cloud')
    xyz=np.column_stack([points[k] for k in ('x','y','z')])
    rgb=np.column_stack([points[k] for k in ('red','green','blue')])
    return xyz,rgb


def run_gate(cloud: Path, anchors: Path, output: Path, max_holdout_m:float=.05):
    import numpy as np
    from .longsplat_depth_geometry import _write_ply
    if max_holdout_m<=0 or max_holdout_m>1:
        raise ValueError('Holdout tolerance must be in (0, 1] meters')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refuse overwrite of existing metric registration output')
    x,y,xt,yt=read_anchors(anchors)
    scale,R,t=estimate_sim3(x,y)
    residual=np.linalg.norm(apply_sim3(xt,scale,R,t)-yt,axis=1)
    holdout_max=float(residual.max())
    holdout_rmse=float(np.sqrt((residual**2).mean()))
    if holdout_max>max_holdout_m or holdout_rmse>max_holdout_m:
        raise RuntimeError(f'Held-out metric anchor validation REFUSED: max={holdout_max:.4f}m, RMSE={holdout_rmse:.4f}m')
    xyz,rgb=read_ply_binary(cloud)
    mapped=apply_sim3(xyz.astype(np.float64),scale,R,t)
    if not np.isfinite(mapped).all():raise ValueError('Invalid registered cloud')
    output.mkdir(parents=True,exist_ok=False)
    registered=output/'depth_candidate_registered_m.ply'
    _write_ply(registered,mapped.astype(np.float32),rgb)
    report={
      'schema':'mycon.longsplat.metric_review.v1',
      'status':'CONTROL_POINTS_PASSED_STAGE8_REVIEW_REQUIRED',
      'scale_model_to_m':scale, 'rotation':R.tolist(),'translation_m':t.tolist(),
      'fit_anchors':len(x),'holdout_anchors':len(xt),'holdout_rmse_m':holdout_rmse,
      'holdout_max_m':holdout_max,'threshold_m':max_holdout_m,
      'stage8_bim_registration_completed':False,
      'stage9_progress_eligible':False,
      'warning':'Passing sparse control anchors does not prove local surface accuracy/visibility throughout the building. Stage 8 geometry alignment and independent surface QA are still mandatory.',
    }
    from .longsplat_research import atomic_json
    atomic_json(output/'metric_registration_review.json',report)
    return registered,report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cloud',type=Path,required=True)
    ap.add_argument('--anchors-csv',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-holdout-m',type=float,default=.05)
    args=ap.parse_args(argv)
    path,report=run_gate(args.cloud,args.anchors_csv,args.output,args.max_holdout_m)
    print('Metric registration candidate for Stage 8 REVIEW ONLY:',path)
    return 0

if __name__=='__main__':
    raise SystemExit(main())