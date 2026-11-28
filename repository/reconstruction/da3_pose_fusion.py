"""Constrain DA3 window drift with visually linked global SfM cameras."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .da3_replay import Cache, Transform, assemble, load_ply, ply_write, project, transform, valid_mask
from .workflow import Blocked, emit_json, parse_colmap_text, sha256


def compose(left,right):
    return Transform(left.scale*right.scale,left.rotation@right.rotation,left(right.translation))


def camera(e):
    R=e[:,:3]
    return -R.T@e[:,3],R


def number(name):
    return int(name.removeprefix("frame_").removesuffix(".jpg"))


def compare_rgb(path,reference):
    # Same view and alignment only. This is a guard for a historical raw/CFR
    # frame-index mapping; COLMAP coordinates are never assigned on filename alone.
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:
        return -1.
    gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY)
    a=cv2.resize(image,(96,128),interpolation=cv2.INTER_AREA).astype(float)[8:-8,8:-8]
    b=cv2.resize(gray,(96,128),interpolation=cv2.INTER_AREA).astype(float)[8:-8,8:-8]
    a-=a.mean();b-=b.mean()
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a*b).sum()/denom) if denom>1e-6 else -1.


def camera_alignment(a,ar,b,br):
    angle=Rotation.from_matrix(br.transpose(0,2,1)@ar)
    Q=angle.mean().as_matrix()
    for _ in range(3):
        residual=(Rotation.from_matrix(Q).inv()*angle).magnitude()
        keep=residual<=max(np.quantile(residual,.75),.02)
        Q=angle[keep].mean().as_matrix()
    X=a@Q.T
    X0=X-X.mean(0);Y0=b-b.mean(0)
    scale=float((X0*Y0).sum()/max(np.square(X0).sum(),1e-12))
    if scale<=0:
        raise Blocked("Visual camera paths do not agree on positive scale")
    return Transform(scale,Q,b.mean(0)-scale*X.mean(0))


def match_cameras(cache,sfm,short_binding):
    _,colmap=parse_colmap_text(sfm/"model_text")
    matching={}
    for index, prior in enumerate(cache.transforms):
        doc=cache.load(index)
        world=compose(short_binding,prior)
        for i,name in enumerate(doc["names"].tolist()):
            if cache.owners[name]!=index or name not in colmap:
                continue
            parity=compare_rgb(sfm/"images"/name,doc["rgb"][i])
            if parity<.83:
                continue
            cp,cr=camera(doc["extrinsics"][i])
            x=colmap[name]
            center=np.asarray(x["camera_center_world"])
            rotation=np.asarray(x["world_to_camera_4x4"])[:3,:3]
            matching[name]=(center,rotation,world(cp),cr@world.rotation.T,parity)
    return matching


def lock_camera_basis(matching,hz,overlap):
    selected=[r for name,r in sorted(matching.items()) if number(name)/hz<=overlap]
    if len(selected)<10:
        raise Blocked("Too few visually checked SfM views in the short reference")
    fit=[r for i,r in enumerate(selected) if i%4!=3]
    check=[r for i,r in enumerate(selected) if i%4==3]
    unpack=lambda rows: tuple(np.asarray([r[i] for r in rows]) for i in range(4))
    A,Ar,B,Br=unpack(fit)
    C,Cr,D,Dr=unpack(check)
    span=float(np.linalg.norm(np.ptp(B,axis=0)))
    if span<1e-4:
        raise Blocked("Short camera baselines too small to verify a similarity")
    T=camera_alignment(A,Ar,B,Br)
    pos=np.linalg.norm(T(C)-D,axis=1)
    rot=Rotation.from_matrix((Cr@T.rotation.T)@Dr.transpose(0,2,1)).magnitude()*180/np.pi
    if np.quantile(pos,.95)>.15*span or np.quantile(rot,.95)>14:
        raise Blocked(f"Reserved cameras disagree with short reference: pos/span {np.quantile(pos,.95)/span:.3f}, rot {np.quantile(rot,.95):.1f}°")
    return T,{"fit":len(fit),"reserved":len(check),"reserved_p95_over_short_camera_span":float(np.quantile(pos,.95)/span),
              "reserved_rotation_p95_deg":float(np.quantile(rot,.95)),"transform":T.as_json()}


def shared_points(a,b):
    ai={name:i for i,name in enumerate(a["names"].tolist())}
    bi={name:i for i,name in enumerate(b["names"].tolist())}
    names=sorted(set(ai)&set(bi))
    if len(names)<2 or a["depth"].shape[1:]!=b["depth"].shape[1:]:
        raise Blocked("Missing compatible shared views in the window graph")
    h,w=a["depth"].shape[1:]
    y,x=np.meshgrid(np.linspace(3,h-4,9,dtype=int),np.linspace(3,w-4,7,dtype=int),indexing="ij")
    ys,xs=y.ravel(),x.ravel()
    result=[[],[]]
    for name in [names[i] for i in sorted({0,len(names)//2,len(names)-1})]:
        one,two=ai[name],bi[name]
        m=valid_mask(a,one)[ys,xs]&valid_mask(b,two)[ys,xs]
        for out,p,i in zip(result,(a,b),(one,two)):
            out.append(project(p,i,ys[m],xs[m]))
    return tuple(np.concatenate(q) for q in result)


def optimize(cache,matching,sfm_to_short,prior,hz,keep_through):
    # Add only validated adjacent overlap edges here. Actual long-range links
    # enter via the single globally adjusted SfM trajectory, not pairwise guesses.
    left=defaultdict(list)
    for edge in cache.edges:
        left[edge["j"]].append(edge["i"])
    cloud=[]
    chunks=[]
    pivots=[]
    fixed=set()
    anchors=[]
    reserved=[]
    queue={}
    for j,T in enumerate(prior):
        p=cache.load(j)
        names=p["names"].tolist()
        is_fixed=number(names[len(names)//2])/hz<=keep_through
        if is_fixed:
            fixed.add(j)
        rows=[camera(e) for e in p["extrinsics"]]
        pivot=T(np.mean([c for c,_ in rows],axis=0))
        pivots.append(pivot)
        group=[(name,c,r,matching[name]) for name,(c,r) in zip(names,rows) if name in matching]
        for name,c,r,record in group:
            C,R=record[:2]
            measured=sfm_to_short(C)
            source=T(c)
            source_rotation=r@T.rotation.T
            target_rotation=R@sfm_to_short.rotation.T
            # One camera contributes once to observation error so that a long
            # overlap cannot multiply an individual view's influence.
            if cache.owners[name]!=j:
                continue
            target=(j,source,measured,source_rotation,target_rotation)
            if number(name)%5==0:
                reserved.append(target)
            else:
                anchors.append(target)
        for i in left[j]:
            a,b=shared_points(queue[i],p)
            if len(a)>=24:
                aa,bb=prior[i](a),T(b)
                d=np.linalg.norm(aa-bb,axis=1)
                good=d<=np.quantile(d,.85)
                cloud.append((i,j,aa[good],bb[good]))
        queue[j]=p
        queue.pop(j-4,None)
        chunks.append(len(group))
    if len(anchors)<30 or len(reserved)<12 or len(cloud)<len(prior)-1:
        raise Blocked("Insufficient validated overlap or global camera anchors")
    control_positions=np.asarray([r[2] for r in anchors])
    extent=float(np.linalg.norm(np.quantile(control_positions,.9,axis=0)-np.quantile(control_positions,.1,axis=0)))
    if extent<1e-5:
        raise Blocked("Global anchors have insufficient translational baseline")
    location={node:i for i,node in enumerate(range(len(prior))) if node not in fixed}
    location={node:i for i,node in enumerate(location)}
    zero=np.zeros(7)
    w_position=max(.004*extent,1e-6)
    w_edge=max(.007*extent,1e-6)
    rotations=lambda old,new: Rotation.from_matrix(old@new.T).as_rotvec()
    size=4*len(location)+6*len(anchors)+sum(len(row[2])*3 for row in cloud)
    structure=lil_matrix((size,7*len(location)),dtype=bool)
    row_index=0
    for i in location:
        col=7*location[i]
        structure[row_index:row_index+4,col:col+7]=True
        row_index+=4
    for j,*_ in anchors:
        if j in location:
            col=7*location[j]
            structure[row_index:row_index+6,col:col+7]=True
        row_index+=6
    for i,j,a,_ in cloud:
        n=len(a)*3
        for node in (i,j):
            if node in location:
                col=7*location[node]
                structure[row_index:row_index+n,col:col+7]=True
        row_index+=n

    def deltas(x):
        out=[]
        for node,pivot in enumerate(pivots):
            v=x[7*location[node]:7*location[node]+7] if node in location else zero
            S=float(np.exp(v[0]))
            Q=Rotation.from_rotvec(v[1:4]).as_matrix()
            out.append(Transform(S,Q,pivot+v[4:] - S*(pivot@Q.T)))
        return out

    def residual(x):
        d=deltas(x)
        out=[x[7*c:7*c+4]*2 for c in location.values()]
        for node,a,b,ar,br in anchors:
            t=d[node]
            out.extend([(t(a)-b)/w_position,
                        rotations(ar@t.rotation.T,br)/.13])
        out.extend(((d[i](a)-d[j](b))/w_edge).ravel()*.28 for i,j,a,b in cloud)
        return np.concatenate(out)

    ini=np.zeros(7*len(location));lo=np.tile([-.7,-.5,-.5,-.5,-extent,-extent,-extent],len(location))
    opt=least_squares(residual,ini,bounds=(lo,-lo),jac_sparsity=structure.tocsr(),
                      loss="soft_l1",f_scale=1.5,max_nfev=50)
    # Select by overlap evidence only. Reserved camera errors are evaluated
    # once after selecting a surface-safe correction.
    def seam_errors(changes):
        return np.array([np.quantile(np.linalg.norm(changes[i](a)-changes[j](b),axis=1),.95)
                         for i,j,a,b in cloud])

    unadjusted=seam_errors(deltas(ini))
    seam_cap=float(np.quantile(unadjusted,.95))
    fraction=0.
    exposed_change=seam_errors(deltas(opt.x))
    for alpha in (1.,.75,.5,.25,.125,.0625,.03125):
        candidate=seam_errors(deltas(alpha*opt.x))
        if np.quantile(candidate,.95)<=seam_cap+1e-10 and np.max(candidate)<=np.max(unadjusted)+1e-10:
            fraction=alpha
            break
    new=deltas(fraction*opt.x)
    before=np.array([np.linalg.norm(a-b) for _,a,b,_,_ in reserved])
    after=np.array([np.linalg.norm(new[j](a)-b) for j,a,b,_,_ in reserved])
    seams=[]
    for i,j,a,b in cloud:
        seams.append([i,j,float(np.quantile(np.linalg.norm(a-b,axis=1),.95)),
                      float(np.quantile(np.linalg.norm(new[i](a)-new[j](b),axis=1),.95))])
    old_seam=np.asarray([r[2] for r in seams]);new_seam=np.asarray([r[3] for r in seams])
    result={"anchor_views":len(anchors),"reserved_views":len(reserved),"fixed_windows":sorted(fixed),
            "verified_edges":len(cloud),"optimization_status":opt.message,"solver_converged":bool(opt.success),
            "before_reserved_median":float(np.median(before)),"after_reserved_median":float(np.median(after)),
            "before_reserved_p95":float(np.quantile(before,.95)),"after_reserved_p95":float(np.quantile(after,.95)),
            "overlap_p95_before":float(np.quantile(old_seam,.95)),"overlap_p95_after":float(np.quantile(new_seam,.95)),
            "source_units":"short DA3 world, no independent metric scale",
            "seam_safe_update_fraction":fraction,"unrestricted_overlap_p95":float(np.quantile(exposed_change,.95)),
            "surface_gate":"No increase in overlap P95 or worst edge; choose using overlap before checking cameras.",
            "window_seams":seams,"windows_with_colmap_views":chunks}
    accepted=(fraction>0 and np.median(after)<.85*np.median(before) and
              np.quantile(after,.95)<np.quantile(before,.95) and
              np.quantile(new_seam,.95)<=np.quantile(old_seam,.95)+1e-10)
    result["status"]="accepted_for_exploratory_render" if accepted else "withheld_due_to_pose_or_overlap_gate"
    return [compose(d,T) for d,T in zip(new,prior)],result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("input","alignment","poses","windows","sfm","short-binding","short-ply","out"):
        p.add_argument(f"--{key}",type=Path,required=True)
    p.add_argument("--hz",type=float,default=29.96666)
    p.add_argument("--short-seconds",type=float,default=10.2)
    a=p.parse_args()
    t0=monotonic()
    cache=Cache(a.input,a.alignment,a.poses,a.windows)
    binding=transform(json.loads(a.short_binding.read_text())["transform"])
    prior=[compose(binding,t) for t in cache.transforms]
    a.out.mkdir(parents=True,exist_ok=False)
    report={"status":"BLOCKED","metric_validated":False}
    accepted=False
    try:
        matches=match_cameras(cache,a.sfm,binding)
        report["sha256_sfm_images"]=sha256(a.sfm/"model_text/images.txt")
        report["visually_linked_sfm_views"]=len(matches)
        sfm_to_short,qa=lock_camera_basis(matches,a.hz,a.short_seconds)
        report["short_camera_basis"]=qa
        new,opt=optimize(cache,matches,sfm_to_short,prior,a.hz,a.short_seconds)
        report["graph"]=opt
        report["status"]=opt["status"]
        accepted=opt["status"]=="accepted_for_exploratory_render"
        if accepted:
            cache.transforms=new
            anchor,colors=load_ply(a.short_ply)
            addon,rgb,_,_=assemble(cache,48,edges=True,after_sec=a.short_seconds,source_hz=a.hz)
            tree=cKDTree(anchor)
            radius=max(3*float(np.median(tree.query(anchor[::16],k=2)[0][:,1])),1e-6)
            good=tree.query(addon,workers=4)[0]>radius
            ply_write(a.out/"reference_locked_global_candidate.ply",np.concatenate((anchor,addon[good])),
                      np.concatenate((colors,rgb[good])))
            report["original_short_sha256"]=sha256(a.short_ply)
            report["protected_reference_vertices"]=len(anchor)
            report["extension_vertices"]=int(good.sum())
    except Blocked as exc:
        report["reason"]=str(exc)
    report["elapsed_seconds"]=monotonic()-t0
    emit_json(a.out/"global_fusion_receipt.json",report)
    if not accepted:
        raise SystemExit(2)


if __name__=="__main__":
    main()
