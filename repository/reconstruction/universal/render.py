"""Full geometry, tiled review, and source-camera comparison export."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pycolmap as pc
import trimesh

from reconstruction.da3_replay import load_ply, ply_write
from reconstruction.pointcloud_compare import export as comparison, raster, text_panel
from reconstruction.workflow import emit_json


def spatial_sample(xyz, target):
    if len(xyz)<=target:
        return np.arange(len(xyz))
    lo,hi=np.min(xyz,axis=0),np.max(xyz,axis=0)
    scale=max(float(np.linalg.norm(hi-lo)),1e-8)
    result=None
    a,b=0.,scale
    for _ in range(22):
        dim=(a+b)/2
        if dim<=0:break
        grid=np.floor((xyz-lo)/dim).astype(np.int64)
        _,ind=np.unique(grid,axis=0,return_index=True)
        if len(ind)<=target:
            result=ind;b=dim
        else:a=dim
    return result if result is not None and len(result)>=32 else np.linspace(0,len(xyz)-1,target,dtype=int)


def glb(path,points,colors):
    geometry=trimesh.points.PointCloud(points,colors)
    with path.open("xb") as f:
        f.write(geometry.export(file_type="glb"))


def data_blob(xyz,rgb):
    arr=np.empty(len(xyz),dtype=[("p","<f4",3),("c","u1",4)])
    arr["p"]=xyz;arr["c"][:,:3]=rgb;arr["c"][:,3]=255
    return base64.b64encode(arr.tobytes()).decode()


def tile_models(out,xyz,rgb):
    lo=np.quantile(xyz,.02,axis=0);hi=np.quantile(xyz,.98,axis=0)
    count=max(2,int(np.ceil((len(xyz)/120_000)**(1/3))))
    interval=np.maximum((hi-lo)/count,1e-7)
    cells=np.floor((xyz-lo)/interval).astype(int).clip(-1,count)
    keys,ids=np.unique(cells,axis=0,return_inverse=True)
    flat=ids.ravel();order=np.argsort(flat,kind="stable");stops=np.r_[0,np.cumsum(np.bincount(flat))]
    tile_dir=out/"tiles";tile_dir.mkdir()
    result=[]
    for i,key in enumerate(keys):
        r=order[stops[i]:stops[i+1]]
        p=xyz[r];color=rgb[r];name=f"tiles/tile_{i:04d}.js"
        center=(p.min(0)+p.max(0))/2
        span=float(max(np.linalg.norm(np.ptp(p,axis=0))*2.,.01))
        item={"count":len(p),"center":center.tolist(),"span":span,"data":data_blob(p,color)}
        (out/name).write_text("window.myconTile("+json.dumps(name)+","+json.dumps(item)+");")
        result.append({"label":f"Tile {i+1} · {len(p):,} points","script":name})
    return result


def add_inspection(viewer,cameras,tiles):
    text=viewer.read_text()
    mark='<script id="data" type="application/json">'
    before,_,after=text.partition(mark)
    raw,divider,tail=after.partition("</script>")
    data=json.loads(raw)
    data["inspection"]={"cameras":cameras,"tiles":tiles}
    viewer.write_text(before+mark+json.dumps(data,separators=(",",":")).replace("<","\\u003c")+divider+tail)


def export(root,work,manifest,sparse_report,dense_report,seconds,say):
    # Keep the full exported asset independent of the display size.
    source=root/"map/Sparse.ply"
    shutil.copyfile(source,work/"Sparse.ply")
    candidate=root/"dense/Dense.ply"
    dense_usable=dense_report["status"]=="dense_candidate" and candidate.is_file()
    if candidate.is_file():
        shutil.copyfile(candidate,work/"Dense.ply")
    model=candidate if dense_usable else source
    shutil.copyfile(model,work/"Full.ply")
    points,rgb=load_ply(model)
    inds=spatial_sample(points,120_000)
    ply_write(work/"Lite.ply",points[inds],rgb[inds])
    glb(work/"MyCON.glb",points,rgb)
    model_path=root/"map/model"
    undistorted=root/"undistort/workspace"
    use=(undistorted/"sparse").is_dir()
    rec=pc.Reconstruction(undistorted/"sparse" if use else model_path)
    image_root=undistorted/"images" if use else root/"intake/frames"
    refs={}
    cameras=[]
    thumbs=work/"frames";thumbs.mkdir()
    rows={r["name"]:r for r in manifest["frames"]}
    selected=sorted((rec.images[image_id] for image_id in rec.reg_image_ids()),key=lambda x:x.name)
    for no,img in enumerate(selected):
        cp=img.cam_from_world()
        cam=rec.cameras[img.camera_id]
        quat=cp.rotation.quat.tolist()
        w2c=cp.rotation.matrix()
        camera={"eye":img.projection_center().tolist(),"axes":w2c.tolist(),"model":cam.model_name,
                "parameters":cam.params.tolist(),"width":cam.width,"height":cam.height,
                "name":img.name,"pts_s":rows[img.name]["pts_s"],"registered_3d_tracks":img.num_points3D}
        refs[img.name]={"center":camera["eye"],"world_to_camera_q":quat[3:]+quat[:3]}
        jpg=thumbs/f"{no:05d}.jpg"
        image=cv2.imread(str(image_root/img.name),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f"Could not load source image for registered view {img.name}")
        h,w=image.shape[:2];scale=min(1.,480/max(w,h))
        cv2.imwrite(str(jpg),cv2.resize(image,(round(w*scale),round(h*scale))),[cv2.IMWRITE_JPEG_QUALITY,88])
        timestamp=f'{camera["pts_s"]:.2f}s' if camera["pts_s"] is not None else img.name
        cameras.append({**camera,"label":f'{no+1:03d} · {timestamp}',"photo":jpg.relative_to(work).as_posix()})
    emit_json(work/"Camera_Poses_and_Intrinsics.json",{"poses":refs,"cameras":cameras,"metric_validated":False})
    pose=work/"Camera_Poses_and_Intrinsics.json"
    view=work/"review"
    models=[("Sparse SfM",work/"Sparse.ply")]
    if dense_usable:
        models.append(("Dense supported / Lite display",work/"Lite.ply"))
    comparison(models,pose,view,12,seconds)
    viewer=work/"Viewer.html";shutil.move(view/"Viewer.html",viewer)
    tiles=tile_models(work,points,rgb) if len(points)>120_000 else []
    add_inspection(viewer,cameras,tiles)
    xyz_s,rgb_s=load_ply(work/"Sparse.ply")
    evidence=[]
    for i in sorted({int(q*(len(cameras)-1)) for q in (.1,.35,.65,.9)}):
        c=cameras[i];photo=cv2.imread(str(work/c["photo"]),cv2.IMREAD_COLOR)
        height,width=photo.shape[:2]
        K=c["parameters"]
        if c["model"]=="PINHOLE":fx,fy,cx,cy=K
        elif c["model"] in ("SIMPLE_PINHOLE","SIMPLE_RADIAL","RADIAL"):fx,fy,cx,cy=K[0],K[0],K[1],K[2]
        elif c["model"]=="OPENCV":fx,fy,cx,cy=K[:4]
        else:continue
        args=(np.asarray(c["eye"]),np.asarray(c["axes"]),(width,height),
              (fx*width/c["width"],fy*height/c["height"]),(cx*width/c["width"],cy*height/c["height"]))
        sparse_img=raster(xyz_s,rgb_s,*args)
        sections=[text_panel(photo[:,:,::-1],"Original frame"),text_panel(sparse_img,"Sparse")]
        if dense_usable:
            sections.append(text_panel(raster(points,rgb,*args),"Dense candidate"))
        path=view/f"camera_{i:05d}.png"
        cv2.imwrite(str(path),cv2.cvtColor(np.hstack(sections),cv2.COLOR_RGB2BGR))
        evidence.append({"path":str(path.relative_to(work)),"frame":c["name"],"pts_s":c["pts_s"]})
    report={"status":"dense_visual_candidate" if dense_usable else "sparse_visual_candidate",
            "metric_validated":False,"full_geometry":"Dense.ply" if dense_usable else "Sparse.ply",
            "has_mesh":False,"has_texture_mesh":False,"gaussian_splatting":"NOT_RUN",
            "geometry":dense_report,"structure":sparse_report,"visual_comparisons":evidence,
            "full_points":len(points),"lite_points":len(inds),"spatial_tiles":len(tiles),
            "independent_scoring":"NOT_EVALUATED","psnr":None,"ssim":None,"lpips":None,
            "chamfer":None,"accuracy":None,"completeness":None,"f1":None,
            "visibility_note":"PLY and GLB contain colored points. Gaps and low-texture misses remain visible."}
    emit_json(work/"Quality_Report.json",report)
    say(f"visual export · {report['status']} · full point cloud plus independent display levels")
    return {"quality_status":report["status"],"export":"export/Viewer.html","geometry_ply":"export/Full.ply"}
