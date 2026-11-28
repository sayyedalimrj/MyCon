"""Calibrated multi-view depth with cross-neighbor support and resumable reference blocks."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pycolmap as pc
from scipy.spatial import cKDTree

from reconstruction.da3_replay import PLY
from reconstruction.workflow import emit_json, sha256


def undistort(model,frames,work,size,config):
    workspace=work/"workspace"
    pc.undistort_images(workspace,model,frames,output_type="COLMAP",
        undistort_options=pc.UndistortCameraOptions(max_image_size=size),
        num_threads=config["threads"],num_patch_match_src_images=12)
    if not (workspace/"sparse").is_dir():
        raise ValueError("COLMAP undistort produced no sparse camera model")
    n=len(list((workspace/"images").rglob("*.jpg")))
    if n<4:
        raise ValueError("Not enough undistorted cameras for multi-view stereo")
    return {"images":n,"max_side_px":size,"model_frame":"global_colmap_unscaled"}


def camera_matrix(camera):
    return np.array([[camera.focal_length_x,0,camera.principal_point_x],
                     [0,camera.focal_length_y,camera.principal_point_y],[0,0,1.]])


def load_cameras(space, size_limit):
    rec=pc.Reconstruction(space/"sparse")
    records={}
    cloud={key:np.asarray(point.xyz) for key,point in rec.points3D.items()}
    for i in rec.reg_image_ids():
        image=rec.images[i]
        cam=rec.cameras[image.camera_id]
        pose=image.cam_from_world()
        path=space/"images"/image.name
        scale=min(1.,size_limit/max(cam.width,cam.height))
        K=camera_matrix(cam)
        ow,oh=round(cam.width*scale),round(cam.height*scale)
        K[0]*=ow/cam.width;K[1]*=oh/cam.height
        ids={int(v.point3D_id) for v in image.points2D if v.has_point3D() and v.point3D_id in cloud}
        records[image.name]={"name":image.name,"R":pose.rotation.matrix(),"t":np.asarray(pose.translation),
                             "center":np.asarray(image.projection_center()),"K":K,"size":(ow,oh),
                             "file":path,"ids":ids}
    return rec,records,cloud


def stereo_pairs(camera,records,cloud,near):
    order=sorted(near,key=lambda n:(-len(camera["ids"]&records[n]["ids"]),n))
    ranked=[]
    for name in order[:26]:
        other=records[name]
        if other["size"]!=camera["size"]:
            continue
        common=sorted(camera["ids"]&other["ids"])
        if len(common)<25:
            continue
        potential=np.asarray([cloud[k] for k in common[::max(1,len(common)//140)]])
        p=potential-camera["center"];q=potential-other["center"]
        length=np.linalg.norm(p,axis=1)*np.linalg.norm(q,axis=1)
        ang=np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i",p,q)/np.maximum(length,1e-9),-1,1)))
        angle=float(np.median(ang))
        facing=float(np.dot(camera["R"][2],other["R"][2]))
        if 1.15<=angle<=24 and facing>.7:
            rank=len(common)/(1+abs(angle-6)/8)
            ranked.append((rank,name,potential,angle))
    return sorted(ranked,key=lambda r:(-r[0],r[1]))[:4]


def arrange(recs, manifest, scale):
    r={row["name"]:row for row in manifest["frames"]}
    names=sorted(recs)
    distances=np.linalg.norm(np.diff(np.asarray([recs[x]["center"] for x in names]),axis=0),axis=1)
    step=float(np.quantile(distances[distances>0],.5)) if (distances>0).any() else scale*.04
    selected=[]
    last=None
    for name in names:
        a=r[name]
        if last:
            b=r[last]
            if a["group"]==b["group"] and a["pts_s"] is not None and b["pts_s"] is not None:
                dt=a["pts_s"]-b["pts_s"]
                d=float(np.linalg.norm(recs[name]["center"]-recs[last]["center"]))
                dot=float(recs[name]["R"][2]@recs[last]["R"][2])
                if dt<.22 or (dt<.8 and d<1.6*step and dot>.996):
                    continue
        selected.append(name);last=name
    if names[-1]!=selected[-1]:
        selected.append(names[-1])
    return selected


def color_image(camera):
    pixels=cv2.imread(str(camera["file"]),cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"Unreadable rectification input {camera['name']}")
    if pixels.shape[1::-1]!=camera["size"]:
        pixels=cv2.resize(pixels,camera["size"],interpolation=cv2.INTER_AREA)
    return pixels


def matcher(minimum, count):
    return cv2.StereoSGBM_create(minDisparity=minimum,numDisparities=count,blockSize=5,
              P1=8*5**2,P2=32*5**2,disp12MaxDiff=1,uniquenessRatio=12,
              speckleWindowSize=75,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def pair_depth(main,other,tracks,source,mate):
    w,h=main["size"]
    relative=other["R"]@main["R"].T
    offset=other["t"]-relative@main["t"]
    zero=np.zeros(5)
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(main["K"],zero,other["K"],zero,(w,h),relative,offset,
                                            flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    # OpenCV supports vertically rectified rigs. SGBM operates along rows, so
    # transpose the rectified images and retain original coordinates for Q.
    vertical=abs(P2[1,3])>abs(P2[0,3])
    axis=int(vertical)
    projection=P2[axis,3]
    rx=(tracks@main["R"].T+main["t"])@R1.T
    z=rx[:,2]
    z=z[z>0]
    if len(z)<12:
        return None
    d=-projection/z
    low=int(np.floor(np.quantile(d,.06)-8))
    high=int(np.ceil(np.quantile(d,.94)+8))
    count=max(32,16*int(np.ceil((high-low)/16)))
    search_width=h if vertical else w
    if count>min(208,.42*search_width) or abs(low)+count>search_width*.75:
        return None
    def rectify(image,cam,rot,p):
        mx,my=cv2.initUndistortRectifyMap(cam["K"],zero,rot,p,(w,h),cv2.CV_32FC1)
        out=cv2.cvtColor(cv2.remap(image,mx,my,cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
        return out.T.copy() if vertical else out
    a=rectify(source,main,R1,P1);b=rectify(mate,other,R2,P2)
    forward=matcher(low,count).compute(a,b).astype(np.float32)/16
    reverse_min=-low-count+1
    reverse=matcher(reverse_min,count).compute(b,a).astype(np.float32)/16
    yy,xx=np.indices(forward.shape)
    mate_x=np.round(xx-forward).astype(np.int32)
    ins=(mate_x>=2)&(mate_x<b.shape[1]-2)
    reverse_d=reverse[yy,mate_x.clip(0,b.shape[1]-1)]
    good=ins&(forward>low-.5)&(reverse_d>reverse_min-.5)&(abs(forward+reverse_d)<=1.1)
    # Flat areas can satisfy a low disparity cost without real correspondence.
    dx=cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)
    good &= cv2.blur(abs(dx)+abs(dy),(7,7))>=14
    dd=forward[good];x=xx[good];y=yy[good]
    if len(dd)<120:
        return None
    if vertical:
        x,y=y,x
    hp=np.column_stack((x,y,dd,np.ones(len(dd))))@Q.T
    usable=abs(hp[:,3])>1e-8
    raw=(hp[usable,:3]/hp[usable,3,None])@R1
    rz=raw[:,2]
    use=np.isfinite(raw).all(axis=1)&(rz>0)
    pixels=raw[use,:2]/raw[use,2,None] * [main["K"][0,0],main["K"][1,1]] + main["K"][:2,2]
    xy=np.rint(pixels).astype(int)
    mask=(xy[:,0]>=1)&(xy[:,0]<w-1)&(xy[:,1]>=1)&(xy[:,1]<h-1)
    result=np.full((h,w),np.inf,dtype=np.float32)
    np.minimum.at(result,(xy[mask,1],xy[mask,0]),rz[use][mask])
    return result


def depth_block(main,pairs,records):
    source=color_image(main);maps=[];angles=[]
    for _,name,world,angle in pairs:
        other=records[name]
        result=pair_depth(main,other,world,source,color_image(other))
        if result is not None:
            maps.append(result);angles.append(angle)
    if len(maps)<2:
        return None
    stack=np.asarray(maps)
    score=np.zeros(stack.shape,dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        for i in range(len(maps)):
            score[i]=np.sum(np.isfinite(stack)&np.isfinite(stack[i])&
                            (abs(stack-stack[i])<=.023*stack[i]),axis=0)
    support=score.max(axis=0)
    selected=np.argmax(score,axis=0)
    yy,xx=np.nonzero(support>=2)
    if len(yy)<180:
        return None
    dep=stack[selected[yy,xx],yy,xx]
    K=main["K"]
    camera=np.column_stack([(xx-K[0,2])*dep/K[0,0],(yy-K[1,2])*dep/K[1,1],dep])
    world=(camera-main["t"])@main["R"]
    rgb=source[yy,xx,::-1]
    return world.astype("f4"),rgb,support[yy,xx],angles


def ply_from_tiles(path,spool,voxel):
    bins=[]
    total=0
    for fragment in sorted(spool.glob("*.bin")):
        records=np.fromfile(fragment,dtype=PLY)
        xyz=np.column_stack([records[k] for k in ("x","y","z")])
        grid=np.floor(xyz/voxel).astype(np.int64)
        _,first=np.unique(grid,axis=0,return_index=True)
        records=records[first];records.tofile(fragment)
        bins.append(fragment);total+=len(records)
    if total<100:
        return total
    with path.open("xb") as out:
        out.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {total}\n"
                   "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                   "property uchar green\nproperty uchar blue\nend_header\n").encode())
        for fragment in bins:
            with fragment.open("rb") as f:shutil.copyfileobj(f,out)
    return total


def cpu_stereo(space,work,cache,manifest,config,say):
    _,records,cloud=load_cameras(space,config["cpu_size"])
    if len(cloud)<50:
        return {"status":"insufficient_depth_support","engine":"cpu_stereo","reason":"too few sparse scale/neighbor controls"}
    pts=np.asarray(list(cloud.values()))
    nn=cKDTree(pts).query(pts[::max(1,len(pts)//5000)],k=2)[0][:,1]
    voxel=max(float(np.median(nn))*.25,1e-6)
    views=arrange(records,manifest,voxel)
    basis=space.parent/"checkpoint.json"
    key=sha256(basis)[:20]+f"-cpu{config['cpu_size']}-1"
    saved=cache/key;saved.mkdir(parents=True,exist_ok=True)
    spool=work/"tiles";spool.mkdir()
    reports=[]
    own=defaultdict(set)
    for name,c in records.items():
        for index in c["ids"]:
            own[index].add(name)
    tile_size=voxel*96
    cv2.setNumThreads(max(1,min(config["threads"],4)))
    for i,name in enumerate(views):
        c=records[name];dest=saved/f"{i:06d}.npz";receipt=dest.with_suffix(".json")
        pairs=[]
        if receipt.is_file() and dest.is_file():
            prior=json.loads(receipt.read_text())
            if prior.get("name")==name and prior.get("sha256")==sha256(dest):
                with np.load(dest,allow_pickle=False) as value:
                    result=(value["points"],value["rgb"],value["votes"],value["angles"].tolist())
            else:
                raise ValueError(f"Cached depth block integrity failure: {dest.name}")
        elif receipt.is_file():
            prior=json.loads(receipt.read_text())
            if prior.get("status")!="no_consistent_depth" or prior.get("name")!=name:
                raise ValueError("Invalid cached empty-depth receipt")
            result=None
        else:
            possible=set().union(*(own[k] for k in c["ids"]))-{name}
            pairs=stereo_pairs(c,records,cloud,possible)
            try:
                result=depth_block(c,pairs,records)
            except cv2.error:
                result=None
                # Leave transient failures retryable.
                reports.append({"name":name,"pixels":0,"status":"stereo_backend_error"})
                continue
            if result is not None:
                temp=dest.with_suffix(".part")
                with temp.open("wb") as out:
                    np.savez_compressed(out,points=result[0],rgb=result[1],votes=result[2],angles=result[3])
                temp.replace(dest)
                emit_json(receipt,{"name":name,"sha256":sha256(dest),"status":"verified_depth"})
            else:
                emit_json(receipt,{"name":name,"status":"no_consistent_depth"})
        if result is None:
            reports.append({"name":name,"pixels":0,"status":"no_consistent_depth"})
        else:
            xyz,rgb,votes,angles=result
            # Store full validated samples into bounded spatial partitions.
            tiles=np.floor(xyz/tile_size).astype(np.int64)
            keys,inv=np.unique(tiles,axis=0,return_inverse=True)
            order=np.argsort(inv,kind="stable");stops=np.r_[0,np.cumsum(np.bincount(inv.ravel()))]
            for j,key3 in enumerate(keys):
                ind=order[stops[j]:stops[j+1]]
                points=np.empty(len(ind),dtype=PLY)
                for axis,label in enumerate(PLY.names[:3]):points[label]=xyz[ind,axis]
                for axis,label in enumerate(PLY.names[3:]):points[label]=rgb[ind,axis]
                with (spool/("_".join(map(str,key3))+".bin")).open("ab") as stream:
                    points.tofile(stream)
            reports.append({"name":name,"pixels":len(xyz),"status":"independent_neighbor_agreement",
                            "pair_angles_degrees":angles,"minimum_support":int(votes.min())})
        if i%10==0 or i+1==len(views):
            say(f"stereo references {i+1}/{len(views)} · {sum(bool(r['pixels']) for r in reports)} usable")
    count=ply_from_tiles(work/"Dense.ply",spool,voxel)
    shutil.rmtree(spool)
    supported=sum(bool(x["pixels"]) for x in reports)
    return {"status":"dense_candidate" if supported>=3 and count>=1000 else "insufficient_depth_support",
            "engine":"OpenCV calibrated SGBM, left/right and independent neighbor consistency",
            "references_attempted":len(views),"references_with_consistency":supported,
            "vertices":count,"voxel_size_unscaled":voxel,"views":reports,"metric_validated":False}


def cuda_stereo(space,work,config,say):
    errors=[]
    for size in sorted({config["cuda_size"],min(config["cuda_size"],1024),min(config["cuda_size"],768)},reverse=True):
        target=work/f"cuda_{size}"
        shutil.copytree(space,target)
        try:
            say(f"CUDA PatchMatch attempt · max side {size}px")
            pc.patch_match_stereo(target,options=pc.PatchMatchOptions(
                   max_image_size=size,geom_consistency=True,gpu_index="0"))
            target_ply=work/"Dense.ply"
            pc.stereo_fusion(target_ply,target,input_type="geometric")
            from reconstruction.da3_replay import load_ply
            xyz,_=load_ply(target_ply)
            if len(xyz)<1000 or not np.isfinite(xyz).all():
                target_ply.rename(work/f"cuda_{size}_rejected.ply")
                raise ValueError("Fusion produced insufficient supported finite geometry")
            shutil.rmtree(target)
            return {"status":"dense_candidate","engine":"CUDA COLMAP PatchMatch + geometric fusion",
                    "vertices":len(xyz),"max_side_px":size,"previous_attempts":errors,"metric_validated":False}
        except (RuntimeError,OSError,ValueError) as exc:
            errors.append({"max_side_px":size,"error":str(exc)})
            shutil.rmtree(target,ignore_errors=True)
            if (work/"Dense.ply").is_file():
                (work/"Dense.ply").rename(work/f"cuda_{size}_partial.ply")
    return {"status":"cuda_failed","attempts":errors}


def reconstruct(undistorted,work,cache,manifest,config,engine,say):
    workspace=undistorted/"workspace"
    fallback=None
    if engine=="cuda_patchmatch":
        fallback=cuda_stereo(workspace,work,config,say)
        if fallback["status"]=="dense_candidate":
            emit_json(work/"geometry.json",fallback);return fallback
    report=cpu_stereo(workspace,work,cache,manifest,config,say)
    if fallback:report["cuda_fallback"]=fallback
    emit_json(work/"geometry.json",report)
    return report
