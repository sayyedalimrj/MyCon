"""Render same-camera evidence and generate a self-contained free-orbit point-cloud viewer."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .da3_replay import load_ply
from .workflow import emit_json


def unit(value):
    return value/np.linalg.norm(value)


def framing(cloud):
    lo,hi=np.quantile(cloud,[.05,.95],axis=0)
    return (hi+lo)/2, max(np.linalg.norm((hi-lo)*.5)*2.5,.05)


def orbit(center,span,up,front,yaw,pitch):
    lateral=unit(np.cross(up,front))
    toward=(front*np.cos(yaw)+lateral*np.sin(yaw))*np.cos(pitch)+up*np.sin(pitch)
    eye=center+span*toward
    f=unit(center-eye)
    r=unit(np.cross(f,up))
    return eye,np.stack((r,np.cross(f,r),f))


def raster(points,colors,eye,R,size,focal,principal=None):
    w,h=size
    if principal is None:
        principal=(w/2,h/2)
    camera=(points-eye)@R.T
    visit=camera[:,2]> max(1e-6,np.max(np.abs(camera[:,2]))*1e-6)
    depth=camera[visit,2]
    xy=np.rint(camera[visit,:2]*np.asarray(focal)/depth[:,None]+principal).astype(int)
    good=(xy[:,0]>=1)&(xy[:,0]<w-1)&(xy[:,1]>=1)&(xy[:,1]<h-1)
    xy=xy[good]; depth=depth[good]
    pix=xy[:,1]*w+xy[:,0]
    order=np.lexsort((depth,pix))
    ind=order[np.r_[True,np.diff(pix[order])!=0]] if len(order) else order
    # The radius is identical for every cloud and camera at this render size.
    zbuffer=np.full((h,w),np.inf,dtype=np.float32)
    layer=np.empty((h,w,3),np.uint8)
    layer[:]=(15,22,31)
    hit=colors[visit][good][ind]
    loc=xy[ind]; d=depth[ind]
    for dy,dx in [(0,0),(-1,0),(1,0),(0,1),(0,-1)]:
        y,x=loc[:,1]+dy,loc[:,0]+dx
        select=zbuffer[y,x]>d
        zbuffer[y[select],x[select]]=d[select]
        layer[y[select],x[select]]=hit[select]
    return layer


def text_panel(image,label):
    top=np.full((36,image.shape[1],3),(26,37,54),np.uint8)
    cv2.putText(top,label,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(225,233,244),1,cv2.LINE_AA)
    return np.vstack((top,image))



def save_png(path, rgb):
    # Keeping the PNG codec out of the PyCOLMAP process avoids incompatible
    # native zlib symbols exposed by some Linux wheel combinations.
    h,w=rgb.shape[:2]
    subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-f","rawvideo",
                    "-pix_fmt","rgb24","-s",f"{w}x{h}","-i","-","-frames:v","1",
                    "-threads","1","-update","1",str(path)],input=rgb.tobytes(),check=True)


def export(models,pose_path,out,fps,seconds):
    out.mkdir(parents=True,exist_ok=False)
    clouds=[(name,*load_ply(path)) for name,path in models]
    first=clouds[0][1]
    poses=json.loads(pose_path.read_text())["poses"]
    ids=list(poses)
    ref=poses[ids[len(ids)//2]]
    q=ref["world_to_camera_q"];R=Rotation.from_quat(q[1:]+q[:1]).as_matrix()
    up=-R[1]
    front=-R[2]
    reference_center,reference_span=framing(first)
    overview_center,overview_span=framing(clouds[-1][1])
    output=dict(center=overview_center.tolist(),span=overview_span,detail_center=reference_center.tolist(),
                detail_span=reference_span,up=up.tolist(),front=front.tolist(),clouds=[])
    for name,xyz,rgb in clouds:
        blob=np.empty(len(xyz),dtype=[("p","<f4",3),("c","u1",4)])
        blob["p"]=xyz;blob["c"][:,:3]=rgb;blob["c"][:,3]=255
        output["clouds"].append(dict(label=name,count=len(xyz),data=base64.b64encode(blob.tobytes()).decode()))
    template=(Path(__file__).parent/"static/pointcloud_compare.html").read_text()
    (out/"Viewer.html").write_text(template.replace("__DATA__",json.dumps(output,separators=(",",":")).replace("<","\\u003c")),encoding="utf-8")
    details=[]
    for index,angle in enumerate((.1,.7,1.45),1):
        eye,orientation=orbit(reference_center,reference_span,up,front,angle,.16)
        panels=[text_panel(raster(xyz,rgb,eye,orientation,(600,570),(570,570)),name)
                for name,xyz,rgb in clouds]
        save_png(out/f"same_view_{index}.png",np.hstack(panels))
        details.append({"filename":f"same_view_{index}.png","center":eye.tolist(),
                        "world_to_camera_rotation":orientation.tolist(),"focal_px":[570,570],"size":[600,570]})
    for index,angle in enumerate((.35,1.65),1):
        eye,orientation=orbit(overview_center,overview_span,up,front,angle,.35)
        panels=[text_panel(raster(xyz,rgb,eye,orientation,(600,570),(570,570)),name)
                for name,xyz,rgb in clouds]
        path=f"overview_{index}.png"
        save_png(out/path,np.hstack(panels))
        details.append({"filename":path,"center":eye.tolist(),
                        "world_to_camera_rotation":orientation.tolist(),"focal_px":[570,570],"size":[600,570]})
    if seconds:
        total=fps*seconds
        width=480*len(clouds);height=420
        ffmpeg=subprocess.Popen(["ffmpeg","-nostdin","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
            "-s",f"{width}x{height}","-r",str(fps),"-i","-","-c:v","libx264","-crf","20",
            "-pix_fmt","yuv420p","-movflags","+faststart",str(out/"free_orbit.mp4")],stdin=subprocess.PIPE)
        try:
            for i,angle in enumerate(np.linspace(.1,2*np.pi+.1,total,endpoint=False)):
                amount=min(1.,max(0.,(i/total-.33)*5.))
                c=reference_center*(1-amount)+overview_center*amount
                span=reference_span*(1-amount)+overview_span*amount
                eye,orientation=orbit(c,span,up,front,angle,.22+.12*np.sin(angle))
                panels=[text_panel(raster(xyz,rgb,eye,orientation,(480,384),(420,420)),name)
                        for name,xyz,rgb in clouds]
                ffmpeg.stdin.write(np.hstack(panels).tobytes())
        finally:
            ffmpeg.stdin.close()
            if ffmpeg.wait():
                raise RuntimeError("Failed encoding free orbit")
    emit_json(out/"render_receipt.json",{"reference_pose":ids[len(ids)//2],"views":details,
                 "labels":[x[0] for x in clouds],"verified_camera_basis":"Shared nonmetric reference crop",
                 "model_point_counts":[len(x[1]) for x in clouds]})


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model",action="append",required=True,help="Label=path.ply; clouds must already share a frame")
    cli.add_argument("--reference-poses",required=True,type=Path)
    cli.add_argument("--out",required=True,type=Path)
    cli.add_argument("--fps",type=int,default=15)
    cli.add_argument("--seconds",type=int,default=6)
    args=cli.parse_args()
    models=[(name,Path(path)) for name,path in (model.split("=",1) for model in args.model)]
    export(models,args.reference_poses,args.out,args.fps,args.seconds)


if __name__=="__main__":
    main()
