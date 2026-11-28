"""Scene-independent media intake, rotation and motion-adaptive keyframes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps

from reconstruction.workflow import sha256

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
VIDEOS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".mpg", ".mpeg"}
TURNS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def list_sources(inputs):
    sources = []
    for path in inputs:
        path = path.resolve()
        if path.is_dir():
            sources += sorted(p for p in path.iterdir()
                              if p.is_file() and p.suffix.lower() in IMAGES | VIDEOS | {".zip"})
        elif path.is_file():
            sources.append(path)
        else:
            raise FileNotFoundError(path)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Input contains no supported media or contains duplicate paths")
    if any(p.suffix.lower() not in IMAGES | VIDEOS | {".zip"} for p in sources):
        raise ValueError("Unsupported media extension")
    return sources


def recorder_archive(path, dst):
    files = {"arcore_recording.mp4", "integrity_sha256.json", "r4_camera.json", "session.json", "pose.csv", "imu.csv"}
    dst.mkdir()
    with ZipFile(path) as archive:
        members = {n: [x for x in archive.infolist() if Path(x.filename).name == n] for n in files}
        if any(len(members[n]) != 1 for n in ("arcore_recording.mp4", "integrity_sha256.json")):
            raise ValueError("ZIP must contain one Recorder video and one integrity manifest")
        wanted = [a[0] for a in members.values() if len(a) == 1]
        if sum(x.file_size for x in wanted) > shutil.disk_usage(dst).free * .85:
            raise OSError("Insufficient free disk for Recorder files")
        for info in wanted:
            with archive.open(info) as src, (dst / Path(info.filename).name).open("xb") as target:
                shutil.copyfileobj(src, target, 1024**2)
    declared = {Path(x["path"]).name: x for x in json.loads((dst/"integrity_sha256.json").read_text())["files"]}
    for member in dst.iterdir():
        if member.name == "integrity_sha256.json":
            continue
        row = declared.get(member.name)
        if (not row or row["size_bytes"] != member.stat().st_size or
                (row.get("sha256") and sha256(member) != row["sha256"])):
            raise ValueError(f"Recorder integrity failure: {member.name}")
    return dst/"arcore_recording.mp4"


def probe(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
               "-show_entries", "stream=width,height,avg_frame_rate:stream_side_data=rotation:"
                                "stream_tags=rotate:frame=best_effort_timestamp_time", "-of", "json", str(path)]
    doc = json.loads(subprocess.check_output(command, text=True))
    stream = doc["streams"][0]
    pts = np.array([float(x.get("best_effort_timestamp_time", "nan")) for x in doc["frames"]])
    if len(pts)<3 or not np.isfinite(pts).all() or (np.diff(pts)<0).any():
        raise ValueError("Decoded video has missing or discontinuous PTS")
    side = next((x["rotation"] for x in stream.get("side_data_list",[]) if "rotation" in x),None)
    tag = stream.get("tags",{}).get("rotate")
    rotation = (-int(round(float(side))) if side is not None else int(round(float(tag or 0)))) % 360
    if rotation not in (0,90,180,270):
        raise ValueError(f"Unsupported encoded non-right-angle display rotation: {rotation}")
    return pts,rotation,(stream["width"],stream["height"]),("container_rotation" if side is not None or tag is not None else "encoded_axes_no_orientation_tag")


def recorder_metadata(root, size, rotation):
    session = json.loads((root/"session.json").read_text()) if (root/"session.json").is_file() else {}
    doc = json.loads((root/"r4_camera.json").read_text()) if (root/"r4_camera.json").is_file() else {}
    intr = doc.get("first_frame_intrinsics",{})
    w,h = size
    k = None
    if (intr.get("width"),intr.get("height")) == size:
        fx,fy,cx,cy = [float(intr[key]) for key in ("fx","fy","cx","cy")]
        if min(fx,fy)>0 and 0<cx<w and 0<cy<h:
            k = {0:[fx,fy,cx,cy],90:[fy,fx,h-1-cy,cx],180:[fx,fy,w-1-cx,h-1-cy],270:[fy,fx,cy,w-1-cx]}[rotation]
    result = {"calibration": "recorder_integrity_verified_initial_intrinsics" if k else "uncalibrated_sfm_estimate",
              "initial_camera_k":k,"tracking_logged":(root/"pose.csv").is_file(),
              "imu_logged":(root/"imu.csv").is_file(),"distortion":"optimizer_calibration_or_unknown",
              "depth_samples_reported":session.get("depth_capture",{}).get("sample_count"),
              "phone_pose_used":False}
    if (root/"pose.csv").is_file() and (root/"imu.csv").is_file():
        from .timing import audit_imu
        try:
            result["imu_audit"]=audit_imu(root)
        except (KeyError, ValueError, IndexError, OSError) as exc:
            result["imu_audit"]={"status":"insufficient_or_invalid","reason":str(exc)}
    return result


def reduced(image):
    h,w = image.shape[:2]
    scale = min(1,400/max(w,h))
    return cv2.cvtColor(cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)


def inspect(gray, previous):
    blurred = cv2.Laplacian(gray,cv2.CV_32F).var()
    dark=float(np.mean(gray<16));clipped=float(np.mean(gray>248))
    pts=cv2.goodFeaturesToTrack(gray,maxCorners=100,qualityLevel=.015,minDistance=7)
    stat={"sharpness":float(blurred),"dark_fraction":dark,"clipped_fraction":clipped,
          "texture_features":0 if pts is None else len(pts),"flow_overlap":None,
          "rotation_proxy_degrees":None,"translation_residual_px":None,"flow_px":None,
          "moving_or_nonrigid_fraction_proxy":None,"ambiguity_proxy":None,
          "brightness":float(np.mean(gray)),"scene_cut_candidate":False}
    if previous is None or previous.shape!=gray.shape:
        return stat
    a=cv2.goodFeaturesToTrack(previous,maxCorners=120,qualityLevel=.015,minDistance=6)
    if a is None or len(a)<8:
        return stat
    b,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,a,None,winSize=(19,19),maxLevel=3)
    if b is None:
        return stat
    aa=a[status.ravel() > 0].reshape(-1,2)
    bb=b[status.ravel() > 0].reshape(-1,2)
    stat["flow_overlap"]=len(aa)/len(a)
    if len(aa)<8:
        stat["scene_cut_candidate"]=bool(np.mean(cv2.absdiff(gray,previous))>28)
        return stat
    model,inliers=cv2.estimateAffinePartial2D(aa,bb,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=400)
    flow=np.linalg.norm(bb-aa,axis=1)
    stat["flow_px"]=float(np.median(flow))
    if model is not None:
        agreement=float(np.mean(inliers))
        stat["moving_or_nonrigid_fraction_proxy"]=1-agreement
        stat["rotation_proxy_degrees"]=float(abs(np.degrees(np.arctan2(model[1,0],model[0,0]))))
        stat["translation_residual_px"]=float(np.median(np.linalg.norm(aa@model[:,:2].T+model[:,2]-bb,axis=1)))
    # Repeated texture and occlusion cannot be uniquely identified by optical flow.
    stat["ambiguity_proxy"]=int(stat["flow_overlap"]<.55)
    stat["scene_cut_candidate"]=bool(stat["flow_overlap"]<.25 and np.mean(cv2.absdiff(gray,previous))>30)
    return stat


def save_frame(frame, directory, group, index, pts, info, rows, orientation):
    name=f"c{group:03d}/f{index:09d}.jpg"
    path=directory/name
    path.parent.mkdir(exist_ok=True)
    if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,96]):
        raise OSError(f"Could not save frame {name}")
    rows.append(dict(name=name,group=group,source_index=index,pts_s=pts,
                     width=frame.shape[1],height=frame.shape[0],orientation=orientation,**info))


def collect(sources, work, profile, progress):
    raw=work/"frames";raw.mkdir()
    rows=[]; reports=[]; group=0; still_index=0
    images_in_group=False
    for source_number, source in enumerate(sources, 1):
        suffix=source.suffix.lower()
        if suffix in IMAGES:
            with Image.open(source) as img:
                frame=cv2.cvtColor(np.array(ImageOps.exif_transpose(img).convert("RGB")),cv2.COLOR_RGB2BGR)
            save_frame(frame,raw,group,still_index,None,inspect(reduced(frame),None),rows,"exif_normalized")
            still_index+=1;images_in_group=True
            continue
        if images_in_group:
            group+=1;images_in_group=False;still_index=0
        rec=work/f"recorder_{group:03d}"
        video=recorder_archive(source,rec) if suffix==".zip" else source
        pts,rotation,size,orientation=probe(video)
        meta=recorder_metadata(rec,size,rotation) if suffix==".zip" else {}
        report={"group":group,"native_frames":len(pts),"duration_s":float(pts[-1]-pts[0]),
                "raw_size":list(size),"rotation_cw":rotation,"orientation_source":orientation,**meta}
        minimum=profile["min_spacing"]
        selected=0; last=-1; gray_before=None; duplicates=0; cut_count=0; segment=0
        cap=cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
        for i,t in enumerate(pts):
            ok,frame=cap.read()
            if not ok:
                raise OSError(f"Video decode stopped before PTS at frame {i}")
            if last>=0 and t-pts[last]<minimum and i<len(pts)-1:
                continue
            if rotation:frame=cv2.rotate(frame,TURNS[rotation])
            gray=reduced(frame)
            score=inspect(gray,gray_before)
            if score["scene_cut_candidate"]:
                cut_count+=1
            gap=float(t-pts[last]) if last>=0 else 0
            change=float(np.mean(cv2.absdiff(gray,gray_before))) if gray_before is not None else 999.
            if change<1.15 and i<len(pts)-1:
                duplicates+=1
                continue
            flow=score["flow_px"] or 0
            detail=score["texture_features"]
            threshold=profile["flow_delta"]*(.78 if detail<28 else 1.)
            keep=(last<0 or i==len(pts)-1 or gap>=profile["max_spacing"]
                  or flow>=threshold or change>15 or score["scene_cut_candidate"])
            if not keep:
                continue
            # Preserve temporal bridges even under blur when no clearer view
            # exists in the interval. The flag survives to the quality report.
            if gap<profile["max_spacing"]*.85 and score["sharpness"]<13 and not score["scene_cut_candidate"] and last>=0:
                continue
            if score["scene_cut_candidate"]:segment+=1
            score["segment"]=segment
            save_frame(frame,raw,group,i,float(t),score,rows,orientation)
            gray_before=gray;last=i;selected+=1
        cap.release()
        report.update(selected=selected,near_duplicates_skipped=duplicates,scene_cut_candidates=cut_count)
        reports.append(report);group+=1
        # Extracted sidecars and videos are scratch data within the intake stage.
        if suffix==".zip":
            shutil.rmtree(rec)
        progress(f"intake · source {source_number}/{len(sources)} · {selected} keyframes")
    if len(rows)<3:
        raise ValueError("Fewer than three usable views")
    return {"frames": rows,"videos":reports,
            "quality_limits":["Flow residual is a parallax proxy, not measured depth.",
                              "Tracking ambiguity, moving subjects and occlusion are uncertain."]}
