"""Replay read-only DA3 predictions at fixed spatial detail and lock a short reference."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from time import monotonic

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .workflow import Blocked, emit_json, sha256


PLY = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])


@dataclass(frozen=True)
class Transform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __call__(self, points):
        return self.scale*(points@self.rotation.T)+self.translation

    def as_json(self):
        return dict(scale=self.scale,rotation=self.rotation.tolist(),translation=self.translation.tolist())


def transform(row):
    t = Transform(float(row["scale"]),np.asarray(row["rotation"]),np.asarray(row["translation"]))
    if (not np.isfinite(t.scale) or t.scale <= 0 or t.rotation.shape != (3,3)
            or not np.allclose(t.rotation.T@t.rotation,np.eye(3),atol=1e-5)
            or np.linalg.det(t.rotation)<.999 or t.translation.shape!=(3,) or not np.isfinite(t.translation).all()):
        raise Blocked("Invalid cached similarity transform")
    return t


class Cache:
    """One fingerprint-bound overlap graph and its immutable prediction windows."""
    def __init__(self, manifest, alignment, poses, windows):
        self.manifest = manifest
        self.alignment = alignment
        self.poses = poses
        m = json.loads(manifest.read_text())
        a = json.loads(alignment.read_text())
        p = json.loads(poses.read_text())
        self.frames = [row["name"] for row in m["frames"]]
        self.plan = a["windows"]
        self.edges = a["edges"]
        self.transforms = [transform(t) for t in a["transforms"]]
        self.owners = p["frame_sources"]
        self.poses_data = p["poses"]
        self.windows = windows
        self.fingerprint = p["fingerprint"]
        calculated = hashlib.sha256(json.dumps(m,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
        if (a["status"]!="ALIGNED" or not 0 <= float(a.get("normalized_p95",float("inf"))) < .05
                or calculated != self.fingerprint or len(set(self.frames)) != len(self.frames)
                or set(self.owners)!=set(self.frames) or len(self.plan)!=len(self.transforms)):
            raise Blocked("Historical frame or alignment fingerprint mismatch")

    def load(self,index):
        base = self.windows/f"{index:05d}"
        path = base/"prediction.npz"
        if (base/"complete.json").is_file():
            receipt = json.loads((base/"complete.json").read_text())
            if sha256(path) != receipt["sha256"]["prediction.npz"]:
                raise Blocked(f"Window hash mismatch: {index}")
        with np.load(path,allow_pickle=False) as archive:
            p = {k:archive[k] for k in ("depth","conf","extrinsics","intrinsics","names","keep","rgb")}
        start,end = self.plan[index]
        n = end-start
        shape = p["depth"].shape
        e,k = p["extrinsics"],p["intrinsics"]
        if (p["names"].tolist()!=self.frames[start:end] or len(shape)!=3 or shape[0]!=n
                or e.shape!=(n,3,4) or k.shape!=(n,3,3) or p["rgb"].shape!=(*shape,3)
                or p["conf"].shape!=shape or p["keep"].shape!=shape
                or not np.isfinite(e).all() or not np.isfinite(k).all()
                or (k[:,0,0]<=0).any() or (k[:,1,1]<=0).any()
                or not np.allclose(e[:,:,:3]@e[:,:,:3].transpose(0,2,1),np.eye(3),atol=.015)
                or not np.allclose(np.linalg.det(e[:,:,:3]),1,atol=.015)):
            raise Blocked(f"Prediction shape, orientation or roster mismatch in window {index}")
        return p


def project(p,index,ys,xs):
    depth = p["depth"][index,ys,xs]
    k,e = p["intrinsics"][index],p["extrinsics"][index]
    rays = np.column_stack((xs,ys,np.ones_like(xs)))@np.linalg.inv(k).T
    return (rays*depth[:,None]-e[:,3])@e[:,:3]


def valid_mask(p,i):
    d,c = p["depth"][i],p["conf"][i]
    ok = np.isfinite(d)&(d>0)&np.isfinite(c)&np.asarray(p["keep"][i],bool)
    return ok & (c>=np.quantile(c[ok],.35)) if ok.any() else ok


def sampled(p,i,grid,edges=False):
    d = p["depth"][i]
    h,w = d.shape
    yy,xx = np.meshgrid(np.linspace(0,h-1,min(h,grid),dtype=int),
                        np.linspace(0,w-1,min(w,grid),dtype=int),indexing="ij")
    yy,xx = yy.ravel(),xx.ravel()
    ok = valid_mask(p,i)
    if edges:
        # Pixels whose near-neighbors disagree strongly in depth create flying points at boundaries.
        safe = np.where(ok,d,0).astype(np.float32)
        peak = cv2.dilate(safe,np.ones((3,3),np.uint8))
        minimum = cv2.erode(np.where(safe>0,safe,np.finfo(np.float32).max),np.ones((3,3),np.uint8))
        ok &= (peak-minimum) <= .12*np.maximum(safe,1e-6)
    yes=ok[yy,xx]
    return project(p,i,yy[yes],xx[yes]),p["rgb"][i,yy[yes],xx[yes]]


def ply_write(path,points,colors,ascii_mode=False):
    xyz,rgb = np.asarray(points),np.asarray(colors)
    if len(xyz)<32 or xyz.shape!=rgb.shape or not np.isfinite(xyz).all():
        raise Blocked("Invalid candidate cloud")
    fmt="ascii" if ascii_mode else "binary_little_endian"
    header=(f"ply\nformat {fmt} 1.0\ncomment NONMETRIC_CANDIDATE_NOT_THESIS_EVIDENCE\n"
            f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    if ascii_mode:
        with path.open("x") as out:
            out.write(header)
            np.savetxt(out,np.column_stack((xyz,rgb)),fmt="%.6g %.6g %.6g %d %d %d")
        return
    with path.open("xb") as out:
        out.write(header.encode("ascii"))
        for start in range(0,len(xyz),200_000):
            block=xyz[start:start+200_000]
            color=rgb[start:start+200_000]
            data=np.empty(len(block),dtype=PLY)
            for axis,key in enumerate(PLY.names[:3]):
                data[key]=block[:,axis]
            for axis,key in enumerate(PLY.names[3:]):
                data[key]=color[:,axis]
            out.write(data.tobytes())


def assemble(cache,grid,*,edges=False,registered=None,after_sec=None,source_hz=None,windows_used=None):
    ordered = []
    used = []
    hashes = {}
    for index, current in enumerate(cache.transforms):
        if windows_used is not None and index not in windows_used:
            continue
        p = cache.load(index)
        if cache.plan[index][1]-cache.plan[index][0] < 3:
            raise Blocked("An underconstrained window cannot extend the cloud")
        total = 0
        for i,name in enumerate(p["names"].tolist()):
            if cache.owners[name]!=index:
                continue
            if after_sec is not None and int(re.search(r"(\d+)\.jpg$",name)[1])/source_hz <= after_sec:
                continue
            xyz,rgb=sampled(p,i,grid,edges)
            if len(xyz):
                coordinates=current(xyz)
                if registered is not None:
                    coordinates=registered(coordinates)
                ordered.append((coordinates,rgb))
                total+=len(xyz)
        used.append({"window":index,"points":total})
        hashes[str(index)] = sha256(cache.windows/f"{index:05d}"/"prediction.npz")
    if not ordered:
        raise Blocked("No verified prediction windows produced points")
    return np.concatenate([r[0] for r in ordered]),np.concatenate([r[1] for r in ordered]),used,hashes


def fit_similarity(a,b):
    centered=a-a.mean(0)
    if len(a)<3 or np.linalg.svd(centered,compute_uv=False)[1] < 1e-8:
        raise ValueError("Degenerate Sim(3) controls")
    Q,s,V=np.linalg.svd((b-b.mean(0)).T@centered)
    fix=np.diag([1.,1.,np.sign(np.linalg.det(Q@V))])
    R=Q@fix@V
    scale=float((s*np.diag(fix)).sum()/np.square(centered).sum())
    if scale<=0:
        raise ValueError("Invalid Sim(3) scale")
    return Transform(scale,R,b.mean(0)-scale*(a.mean(0)@R.T))


def reference_views(cache,last_second,hz):
    sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.015)
    views={}
    for index,T in enumerate(cache.transforms):
        start,end=cache.plan[index]
        if int(re.search(r"(\d+)\.jpg$",cache.frames[start])[1])/hz > last_second+.3:
            break
        p=cache.load(index)
        for i,name in enumerate(p["names"].tolist()):
            second=int(re.search(r"(\d+)\.jpg$",name)[1])/hz
            if cache.owners[name]!=index or second>last_second+.3:
                continue
            gray=cv2.cvtColor(p["rgb"][i],cv2.COLOR_RGB2GRAY)
            keys,desc=sift.detectAndCompute(gray,None)
            if desc is None:
                continue
            xy=np.rint([k.pt for k in keys]).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,gray.shape[1]-1)
            xy[:,1]=np.clip(xy[:,1],0,gray.shape[0]-1)
            ok=valid_mask(p,i)[xy[:,1],xy[:,0]]
            views[name]=dict(time=second,xy=xy,desc=desc,ok=ok,cloud=T(project(p,i,xy[:,1],xy[:,0])),rgb=p["rgb"][i])
    return views


def bind_reference(long,short,long_hz,short_hz,overlap):
    source=reference_views(long,overlap,long_hz)
    target=reference_views(short,overlap,short_hz)
    matcher=cv2.BFMatcher()
    observations=[]
    for name,a in sorted(source.items()):
        best=None
        for tname,b in target.items():
            if abs(a["time"]-b["time"])>.22:
                continue
            pairs=[(g[0].queryIdx,g[0].trainIdx) for g in matcher.knnMatch(a["desc"],b["desc"],k=2)
                   if len(g)==2 and g[0].distance<.73*g[1].distance]
            if len(pairs)<16:
                continue
            ii,jj=np.asarray(pairs).T
            _, mask = cv2.findHomography(a["xy"][ii],b["xy"][jj],cv2.RANSAC,2.5)
            if mask is None:
                continue
            keep=mask.ravel().astype(bool)&a["ok"][ii]&b["ok"][jj]
            if keep.sum()>=14 and (best is None or keep.sum()>best[0]):
                best=(int(keep.sum()),tname,a["cloud"][ii[keep]],b["cloud"][jj[keep]])
        if best:
            observations.append((name, *best))
    if len(observations)<6:
        raise Blocked("Insufficient visually linked views for short/long Sim(3)")
    # Hold complete view pairs out of the fit; points within one image are correlated.
    fit=[row for i,row in enumerate(observations) if i%4!=3]
    held=[row for i,row in enumerate(observations) if i%4==3]
    A=np.concatenate([r[3] for r in fit])
    B=np.concatenate([r[4] for r in fit])
    extent=float(np.linalg.norm(np.quantile(B,.875,axis=0)-np.quantile(B,.125,axis=0)))
    limit=.06*extent
    rng=np.random.default_rng(42)
    best=np.zeros(len(A),dtype=bool)
    for _ in range(1000):
        indices=rng.choice(len(A),3,replace=False)
        try:
            T=fit_similarity(A[indices],B[indices])
        except ValueError:
            continue
        good=np.linalg.norm(T(A)-B,axis=1)<limit
        if good.sum()>best.sum():
            best=good
    if best.mean()<.35:
        raise Blocked("Visual correspondences do not support a shared coordinate frame")
    T=fit_similarity(A[best],B[best])
    errors=np.concatenate([np.linalg.norm(T(r[3])-r[4],axis=1) for r in held])
    if np.median(errors)>.08*extent or np.mean(errors<limit)<.40:
        raise Blocked("Reserved visual views do not confirm the short/long transform")
    return T,{
        "method":"SIFT ratio + geometric 2D homography filter + robust 3D Sim(3)",
        "fit_views":len(fit),"reserved_views":len(held),"fit_inlier_fraction":float(best.mean()),
        "heldout_median_over_reference_extent":float(np.median(errors)/extent),
        "heldout_p95_over_reference_extent":float(np.quantile(errors,.95)/extent),
        "heldout_inlier_fraction":float(np.mean(errors<limit)),"transform":T.as_json(),
        "pairs":[{"long":r[0],"short":r[2],"matched":r[1]} for r in observations],
        "claim":"shared exploratory frame for visual comparison; not field control"
    },source,target


def load_ply(path):
    import trimesh
    scene=trimesh.load(path,file_type="ply",process=False)
    return np.asarray(scene.vertices),np.asarray(scene.colors)[:,:3]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    def inputs(prefix):
        for key in ("input","alignment","poses","windows"):
            cli.add_argument(f"--{prefix}{key}",type=Path,required=not prefix)
    inputs("");inputs("short-")
    cli.add_argument("--out",required=True,type=Path)
    cli.add_argument("--grid",type=int,default=48)
    cli.add_argument("--long-hz",type=float,default=29.96666)
    cli.add_argument("--short-hz",type=float,default=30.5136274)
    cli.add_argument("--overlap-seconds",type=float,default=10.2)
    args=cli.parse_args()
    if not 8<=args.grid<=196:
        cli.error("--grid must be between 8 and 196")
    start=monotonic()
    cache=Cache(args.input,args.alignment,args.poses,args.windows)
    args.out.mkdir(parents=True,exist_ok=False)
    legacy=max(8,min(48,int(np.sqrt(250000/len(cache.frames)))))
    xyz,rgb,_,_=assemble(cache,legacy)
    ply_write(args.out/"legacy_replayed.ply",xyz,rgb,ascii_mode=True)
    more,color,windows,hashes=assemble(cache,args.grid)
    ply_write(args.out/"fixed_grid.ply",more,color)
    result=dict(frames=len(cache.frames),windows=len(cache.plan),fingerprint=cache.fingerprint,
                legacy_grid=legacy,legacy_points=len(xyz),fixed_grid=args.grid,fixed_grid_points=len(more),
                window_receipts=hashes,window_point_counts=windows,metric_validated=False)
    if args.short_input:
        short=Cache(args.short_input,args.short_alignment,args.short_poses,args.short_windows)
        try:
            T,binding,_,_=bind_reference(cache,short,args.long_hz,args.short_hz,args.overlap_seconds)
            emit_json(args.out/"short_binding.json",binding)
            ply_write(args.out/"legacy_common_frame.ply",T(xyz),rgb)
            ply_write(args.out/"fixed_grid_common_frame.ply",T(more),color)
            anchor,anchor_color,_,_=assemble(short,48)
            addon,addon_color,_,_=assemble(cache,args.grid,edges=True,registered=T,
                                          after_sec=args.overlap_seconds,source_hz=args.long_hz)
            tree=cKDTree(anchor)
            spacing=float(np.median(tree.query(anchor[::16],k=2)[0][:,1]))
            distance=tree.query(addon,workers=4)[0]
            append=distance>max(3*spacing,1e-6)
            ply_write(args.out/"reference_locked_extension.ply",np.concatenate((anchor,addon[append])),
                      np.concatenate((anchor_color,addon_color[append])))
            result["reference_locked"]={"prefix_vertices":len(anchor),"added_vertices":int(append.sum()),
                       "source_frames_excluded_until_s":args.overlap_seconds,"point_exclusion_radius":3*spacing,
                       "sampled_boundary_filter":True,"global_drift_corrected":False}
        except Blocked as exc:
            result["reference_locked"]={"status":"BLOCKED","reason":str(exc)}
    result["wall_seconds"]=monotonic()-start
    emit_json(args.out/"replay_receipt.json",result)


if __name__ == "__main__":
    main()
