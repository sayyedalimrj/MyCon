"""Non-destructive real-derived cloud -> unscaled, finite geometry hypothesis package.

This is a research candidate adapter, not a replacement for Stage7 or a survey.
It does not accept raw Gaussian centres as a physical surface.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.ndimage import label as cc_label, binary_dilation

SCHEMA = 'mycon.comparison_ready_geometry.v1'
KINDS = {'colmap_dense', 'depth_fused_candidate_nonmetric', 'observed_mesh_sample'}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''): h.update(buf)
    return h.hexdigest()

def _basis(n):
    axis = np.eye(3)[int(np.argmin(abs(n)))]
    u = np.cross(n, axis); u /= np.linalg.norm(u)
    return u, np.cross(n, u)

def _patch(points, ids, threshold, name):
    sub = points[ids]
    c = sub.mean(0)
    try: _, s, vt = np.linalg.svd(sub-c, full_matrices=False)
    except np.linalg.LinAlgError: return None
    if s[1] < 1e-9: return None
    n = vt[-1]
    if n[np.argmax(abs(n))] < 0: n = -n
    d = -float(n @ c)
    u, v = _basis(n)
    uv = np.column_stack(((sub-c) @ u, (sub-c) @ v))
    try: hull = ConvexHull(uv)
    except QhullError: return None
    if hull.volume < threshold**2: return None
    boundary = uv[hull.vertices]
    xyz_boundary = c + boundary[:, 0, None]*u + boundary[:, 1, None]*v
    residual = abs(sub @ n + d)
    return {'plane_id': name, 'coefficients': [*n.tolist(), d], 'normal': n.tolist(),
        'centroid': c.tolist(), 'basis_u': u.tolist(), 'basis_v': v.tolist(),
        'halfspaces_uv': hull.equations.tolist(), 'finite_boundary_xyz': xyz_boundary.tolist(),
        'finite_boundary_uv': boundary.tolist(), 'support_indices': ids.tolist(),
        'point_count': len(ids), 'residual_median_units': float(np.median(residual)),
        'residual_p95_units': float(np.quantile(residual, .95)),
        'area_reconstruction_units_squared': float(hull.volume),
        'fit_state': 'candidate_observed_patch', 'physical_surface_confirmed': False,
        'orientation_class': 'unknown_unverified_gravity', 'surface_island_method': 'occupied_grid',
        'finite_support_only': True, 'threshold_units': threshold}

def _islands(points, idx, normal, threshold, median_spacing, minimum):
    """2D connected occupied cells split disconnected coplanar fragments.

    Dilation creates adjacency only and never adds geometry points.
    """
    p = points[idx]
    ctr = p.mean(0); u, v = _basis(normal)
    xy = np.column_stack(((p-ctr) @ u, (p-ctr) @ v))
    cell = max(2.2*median_spacing, 1.5*threshold, 1e-6)
    mins = xy.min(0); integer = np.floor((xy-mins)/cell).astype(np.int64)
    dims = integer.max(0)+1
    if np.prod(dims) > 2_000_000:
        # no unbounded memory use: preserve as one uncertain finite hypothesis
        return [idx]
    mask = np.zeros(tuple(dims), dtype=bool)
    mask[integer[:,0],integer[:,1]] = True
    conn, count = cc_label(binary_dilation(mask, iterations=1))
    labels = conn[integer[:,0],integer[:,1]]
    out = [idx[labels==n] for n in range(1,count+1) if int((labels==n).sum())>=minimum]
    return out

def extract(points, *, threshold, min_points=250, max_planes=10, trials=160, seed=42, max_sample=24000):
    pts = np.asarray(points, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError('Invalid finite Nx3 source geometry')
    if threshold <= 0 or min_points < 20 or trials < 10 or max_sample < min_points:
        raise ValueError('Invalid plane extraction parameters')
    rng = np.random.default_rng(seed)
    sample_ids = np.linspace(0, len(pts)-1, min(max_sample,len(pts)),dtype=int)
    train = pts[sample_ids]
    nn = cKDTree(train).query(train, k=2)[0][:,1]
    spacing = float(np.median(nn[nn>1e-12])) if (nn>1e-12).any() else threshold
    # Remaining sampling rows: never change source points or source RGB.
    remaining = np.arange(len(train)); assigned = np.zeros(len(pts), dtype=bool)
    patches = []
    for _ in range(max_planes):
        if len(remaining)<min_points: break
        best = np.array([], dtype=int)
        for _trial in range(trials):
            s = rng.choice(remaining, 3, replace=False)
            n = np.cross(train[s[1]]-train[s[0]],train[s[2]]-train[s[0]])
            length=np.linalg.norm(n)
            if length < 1e-10: continue
            n /= length; d = -float(n@train[s[0]])
            cand = remaining[np.abs(train[remaining]@n + d) <= threshold]
            if len(cand)>len(best): best=cand
        if len(best)<min_points: break
        fit = _patch(train,best,threshold,'tmp')
        if fit is None: break
        n=np.array(fit['normal']); d=float(fit['coefficients'][3])
        support = np.flatnonzero((abs(pts@n +d) <= threshold) & ~assigned)
        if len(support)<min_points: break
        for island in _islands(pts,support,n,threshold,spacing,max(150,min_points)):
            patch=_patch(pts,island,threshold,f'pl-{len(patches)+1:03d}')
            if patch is not None and patch['residual_p95_units'] <= threshold*1.7:
                patches.append(patch); assigned[island] = True
        keep = np.abs(train[remaining]@n+d) > threshold
        remaining=remaining[keep]
    # Candidate lines must have local sample support on BOTH finite patches.
    radius=max(2.5*threshold, 3*spacing)
    edges=[]
    for p,q in combinations(patches,2):
        n1,n2=np.array(p['normal']),np.array(q['normal'])
        axis=np.cross(n1,n2); m=np.linalg.norm(axis)
        if m < .20: continue
        axis /= m
        a=np.stack((n1,n2,axis)); b=-np.array([p['coefficients'][3],q['coefficients'][3],0.])
        try: origin=np.linalg.solve(a,b)
        except np.linalg.LinAlgError: continue
        support=[]
        for patch in (p,q):
            indices=np.asarray(patch['support_indices'],int)
            distances=np.linalg.norm(np.cross(pts[indices]-origin,axis),axis=1)
            idx=indices[distances<=radius]
            if len(idx)<8: break
            t=(pts[idx]-origin)@axis
            order=np.argsort(t); t=t[order]; idx=idx[order]
            gaps=np.flatnonzero(np.diff(t)>3*radius)+1
            pieces=np.split(np.arange(len(t)),gaps)
            support.append([(float(t[c].min()),float(t[c].max()),idx[c]) for c in pieces if len(c)>=8])
        if len(support)!=2: continue
        for aa in support[0]:
            for bb in support[1]:
                lo=max(aa[0],bb[0]); hi=min(aa[1],bb[1])
                if hi-lo < 2*radius:continue
                first=aa[2][abs((pts[aa[2]]-origin)@axis -np.clip((pts[aa[2]]-origin)@axis,lo,hi))<radius]
                second=bb[2][abs((pts[bb[2]]-origin)@axis -np.clip((pts[bb[2]]-origin)@axis,lo,hi))<radius]
                if min(len(first),len(second))<8:continue
                edges.append({'edge_id':f'ed-{len(edges)+1:03d}', 'plane_ids':[p['plane_id'],q['plane_id']],
                    'endpoints_xyz':[(origin+lo*axis).tolist(),(origin+hi*axis).tolist()],
                    'support_indices_per_plane':[first.tolist(),second.tolist()],
                    'support_count_per_plane':[len(first),len(second)],'length_reconstruction_units':hi-lo,
                    'support_radius_units':radius,'fit_state':'locally_supported_candidate',
                    'physical_edge_confirmed':False})
    # Trihedral solutions are only hypotheses without independently checked images and GT.
    corners=[]
    for trio in combinations(patches,3):
        a=np.stack([p['normal'] for p in trio]); cond=float(np.linalg.cond(a))
        if not np.isfinite(cond) or cond>65: continue
        b=-np.array([p['coefficients'][3] for p in trio]); xyz=np.linalg.solve(a,b)
        counts=[int(np.sum(np.linalg.norm(pts[np.array(p['support_indices'],int)]-xyz,axis=1)<radius)) for p in trio]
        if min(counts)<8: continue
        local_edges=[]
        for p,q in combinations(trio,2):
            options=[]
            for e in edges:
                if set(e['plane_ids']) != {p['plane_id'],q['plane_id']}: continue
                a,b=np.asarray(e['endpoints_xyz'],float)
                t=float(np.clip(np.dot(xyz-a,b-a)/(np.dot(b-a,b-a)+1e-12),0,1))
                if np.linalg.norm(xyz-(a+t*(b-a)))<=radius: options.append(e['edge_id'])
            if not options: break
            local_edges.append(options[0])
        if len(local_edges)!=3:continue
        corners.append({'corner_id':f'co-{len(corners)+1:03d}', 'xyz':xyz.tolist(),
            'plane_ids':[p['plane_id'] for p in trio], 'edge_ids':local_edges,
            'state':'candidate_extrapolated', 'trihedral_condition':cond,
            'support_near_corner_per_plane':counts,
            'inside_observed_patch_per_plane':[False]*3,
            'multiscale_confirmed':False,'camera_projection_verified':False,
            'field_confirmed':False,'unknown_reason':'no_independent_camera_or_survey_junction_evidence'})
    return patches,edges,corners, {'subsample_count':len(train),'spacing_source_units':spacing,
        'unknown_points':int((~assigned).sum()),'plane_support_points':int(assigned.sum()),
        'sampling':'deterministic_uniform_index','seed':seed,'radius_source_units':radius}

def generate(source:Path, root:Path, run_id:str, *, source_kind:str, threshold:float,
             receipt:Path|None=None, min_points:int=250, max_planes:int=10,
             trials:int=160, seed:int=42, max_sample:int=24000)->dict:
    import trimesh
    if source_kind not in KINDS: raise ValueError('Gaussian centres/unknown source are not admissible physical geometry')
    if not run_id.startswith('geom-ready-') or not all(c.isalnum() or c in '-_.' for c in run_id[11:]):
        raise ValueError('Invalid isolated geom-ready-* run ID')
    source=source.resolve();root=root.resolve();target=root/'runs'/run_id/'geometry'
    if not source.is_file() or source.stat().st_size==0:raise FileNotFoundError(source)
    if target.parent.exists():raise FileExistsError('Create-only run exists: '+str(target.parent))
    if receipt is not None and (not receipt.is_file()):raise FileNotFoundError(receipt)
    cloud=trimesh.load(source,file_type='ply',process=False)
    if not isinstance(cloud,trimesh.points.PointCloud):raise ValueError('Expected point-cloud PLY; mesh requires separately sampled surface')
    xyz=np.asarray(cloud.vertices,dtype=np.float64)
    colors=np.asarray(cloud.colors)[:,:3].astype(np.uint8)
    planes,edges,corners,qa=extract(xyz,threshold=threshold,min_points=min_points,max_planes=max_planes,trials=trials,seed=seed,max_sample=max_sample)
    target.mkdir(parents=True,exist_ok=False)
    # Source point positions and RGB remain exact, non-destructive.
    np.save(target/'clean_cloud.npy',xyz.astype('<f8'),allow_pickle=False)
    clean=trimesh.points.PointCloud(xyz,colors=np.column_stack((colors,np.full(len(xyz),255,np.uint8))))
    (target/'clean_cloud.ply').write_bytes(trimesh.exchange.ply.export_ply(clean,encoding='binary_little_endian'))
    for name,values in [('planes',planes),('edges',edges),('corners',corners)]:
        (target/f'{name}.json').write_text(json.dumps({name:values},indent=2),encoding='utf-8')
    outputs={p.name:{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in target.iterdir() if p.is_file()}
    manifest={'schema_version':SCHEMA,'run_id':run_id,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(source),'source_path':str(source),'source_kind':source_kind,
        'source_receipt_sha256':sha(receipt) if receipt else None,'source_frame':'colmap_world_unverified',
        'coordinate_frame':'colmap_world_unverified','units':'arbitrary_reconstruction_units',
        'metric_scale_verified':False,'registration_verified':False,'source_camera_sha256':None,
        'plane_count':len(planes),'edge_count':len(edges),'corner_count':len(corners),
        'parameters':{'threshold_reconstruction_units':threshold,'seed':seed,'trials':trials,
            'min_plane_points':min_points,'max_planes':max_planes,'max_feature_points':max_sample},
        'qa':qa,'original_points':len(xyz),'clean_points':len(xyz),'outputs':outputs,
        'unknowns':['physical_surface_confirmation','gravity','camera_image_evidence','occlusion','field_scale','mesh_GT'],
        'warnings':['Real-derived candidate only, potential false-positive geometry',
            'All extracted planes/edges/corners are hypotheses without image+survey review'],
        'software':{'python':platform.python_version(),'numpy':np.__version__,'trimesh':trimesh.__version__}}
    (target/'comparison_ready.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description='Read-only real depth/dense cloud -> unscaled finite candidate geometry')
    p.add_argument('--input-ply',required=True,type=Path);p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--run-id',required=True);p.add_argument('--source-kind',required=True)
    p.add_argument('--source-receipt',type=Path);p.add_argument('--threshold',type=float,default=.09)
    p.add_argument('--min-points',type=int,default=250);p.add_argument('--max-planes',type=int,default=10)
    p.add_argument('--trials',type=int,default=160);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-sample',type=int,default=24000)
    a=p.parse_args(argv)
    try:
        m=generate(a.input_ply,a.project_root,a.run_id,source_kind=a.source_kind,
          threshold=a.threshold,receipt=a.source_receipt,min_points=a.min_points,max_planes=a.max_planes,
          trials=a.trials,seed=a.seed,max_sample=a.max_sample)
    except (OSError,ValueError,RuntimeError,ImportError) as e:
        print(json.dumps({'status':'blocked','reason':str(e)}));return 2
    print(json.dumps({'status':'candidate_created','run_id':m['run_id'],'planes':m['plane_count'],
        'edges':m['edge_count'],'corners':m['corner_count'],'metric_scale_verified':False}));return 0

if __name__=='__main__':raise SystemExit(main())
