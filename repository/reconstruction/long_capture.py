"""Motion-adaptive original Recorder video to a fresh, globally matched COLMAP model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from time import monotonic

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .workflow import Blocked, emit_json, sha256

NAME = re.compile(r"frame_(\d{8})\.jpg")
AXES = np.diag([1., -1., -1.])


def read_recorder(root: Path, rotation: int, video_hash: str | None = None):
    integrity = json.loads((root / "integrity_sha256.json").read_text())
    by_name = {row["path"]: row for row in integrity["files"]}
    required = ("arcore_recording.mp4", "session.json", "pose.csv", "imu.csv", "r4_camera.json")
    for name in required:
        rec = by_name[name]
        path = root / name
        if path.stat().st_size != rec["size_bytes"] or (rec.get("sha256") and sha256(path) != rec["sha256"]):
            raise Blocked(f"Recorder integrity mismatch: {name}")
    video = root / required[0]
    digest = sha256(video)
    if video_hash and digest != video_hash.lower():
        raise Blocked("Original Recorder video hash mismatch")
    session = json.loads((root / "session.json").read_text())
    camera = json.loads((root / "r4_camera.json").read_text())["first_frame_intrinsics"]
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=width,height:frame=best_effort_timestamp_time", "-of", "json", str(video)
    ], text=True))
    pts = np.asarray([float(frame["best_effort_timestamp_time"]) for frame in stream["frames"]])
    if len(pts) < 3 or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise Blocked("Missing or nonmonotonic native video presentation timestamps")
    sensor = stream["streams"][0]
    w, h = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = (float(camera[key]) for key in ("fx", "fy", "cx", "cy"))
    if (w, h) != (sensor["width"], sensor["height"]) or min(fx, fy) <= 0 or not (0 < cx < w and 0 < cy < h):
        raise Blocked("Recorder intrinsics are invalid for the encoded sensor resolution")
    k = {
        0: (fx, fy, cx, cy, w, h), 90: (fy, fx, h-1-cy, cx, h, w),
        180: (fx, fy, w-1-cx, h-1-cy, w, h), 270: (fy, fx, cy, w-1-cx, h, w)
    }[rotation]
    with (root / "pose.csv").open(newline="") as fp:
        rows = [r for r in csv.DictReader(fp) if r["tracking"] == "TRACKING"]
    start = int(session["first_frame_timestamp_ns"])
    keys = ("tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw")
    # Keep integer nanoseconds before subtracting the session origin.
    unique = {int(r["timestamp_ns"]): [float(r[key]) for key in keys] for r in rows}
    timestamps = sorted(unique)
    t = np.asarray([(v-start)*1e-9 for v in timestamps])
    xyzq = np.asarray([unique[v] for v in timestamps])
    if len(t) < 4 or not np.isfinite(xyzq).all() or np.max(abs(np.linalg.norm(xyzq[:, 3:], axis=1)-1)) > .02:
        raise Blocked("Invalid Recorder pose trace")
    raw = Rotation.from_quat(xyzq[:, 3:])
    with (root / "imu.csv").open(newline="") as fp:
        imu = [(int(r["timestamp_ns"]), [float(r[k]) for k in ("v0", "v1", "v2")])
               for r in csv.DictReader(fp) if r["sensor"] == "GYRO"]
    gyro_t = np.asarray([(stamp-start)*1e-9 for stamp, _ in imu])
    gyro_w = np.linalg.norm([v for _, v in imu], axis=1)
    pose_rate = (raw[1:] * raw[:-1].inv()).magnitude() / np.diff(t)
    stable = (np.diff(t) > .008) & (np.diff(t) < .09) & (t[1:] <= gyro_t[-1]) & (t[1:] >= gyro_t[0])
    corr = float(np.corrcoef(pose_rate[stable], np.interp(t[1:][stable], gyro_t, gyro_w))[0, 1])
    if not np.isfinite(corr):
        corr = None
    return video, pts, k, (t, xyzq[:, :3], raw), {
        "video_sha256": digest, "native_pts_start": float(pts[0]), "native_pts_end": float(pts[-1]),
        "native_pts_count": len(pts), "clock_source": "Recorder relative native video PTS, offset supplied",
        "gyro_pose_rate_pearson": corr, "gyro_pose_samples": int(stable.sum()),
        "raw_to_upright_clockwise": rotation, "upright_intrinsics": list(k),
        "pose_duration_s": float(t[-1]-t[0]), "surveyed": False,
    }


def poses_at(pts, trace, rotation, offset=0.):
    t, positions, rotations = trace
    q = np.asarray(pts) + offset
    nearest = np.searchsorted(t, q).clip(1, len(t)-1)
    gap = np.maximum(t[nearest]-t[nearest-1], 0)
    valid = (q >= t[0]) & (q <= t[-1]) & (gap <= .10)
    clipped = q.clip(t[0], t[-1])
    centers = np.column_stack([np.interp(clipped,t,positions[:, j]) for j in range(3)])
    rotation_world = Slerp(t, rotations)(clipped)
    pixel = Rotation.from_euler("z", rotation, degrees=True).as_matrix()
    w2c = pixel @ AXES @ rotation_world.inv().as_matrix()
    return centers, w2c, valid


def select_frames(video, pts, centers, axes, pose_ok, out, rotation, roster, min_hz, max_hz):
    if not 0 < min_hz <= max_hz:
        raise Blocked("Require positive min_hz <= max_hz")
    if roster:
        names = [f["name"] for f in json.loads(roster.read_text())["frames"]]
        if any(not NAME.fullmatch(name) for name in names):
            raise Blocked("Invalid frame name in historical roster")
        chosen = {int(NAME.fullmatch(n)[1]) for n in names}
        if len(chosen) != len(names) or max(chosen) >= len(pts):
            raise Blocked("Invalid/duplicate frame index in historical roster")
    else:
        chosen = None
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    rows, descriptors = [], []
    prev_gray = None
    for index, stamp in enumerate(pts):
        has_image, image = cap.read()
        if not has_image:
            raise Blocked(f"Decoder ended before presentation timestamp {index}")
        if chosen is not None and index not in chosen:
            continue
        if rotation:
            image = cv2.rotate(image, turn[rotation])
        gray = cv2.cvtColor(cv2.resize(image, (144, 256)), cv2.COLOR_BGR2GRAY)
        if chosen is None and rows and index != len(pts)-1:
            last = rows[-1]["source_frame"]
            dt = stamp-pts[last]
            if dt < 1/max_hz:
                continue
            moved = np.linalg.norm(centers[index]-centers[last])
            degrees = Rotation.from_matrix(axes[index]@axes[last].T).magnitude()*180/np.pi
            visual_change = np.mean(cv2.absdiff(gray,prev_gray))
            fast = pose_ok[index] and pose_ok[last] and (moved >= .055 or degrees >= 4.)
            if dt < 1/min_hz and visual_change < 9. and not fast:
                continue
        filename = f"frame_{index:08d}.jpg"
        if not cv2.imwrite(str(out/filename),image,[cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise IOError(f"Failed writing {filename}")
        descriptor = cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:9,:9].reshape(-1)[1:]
        descriptors.append(descriptor / max(np.linalg.norm(descriptor),1e-9))
        rows.append({"name": filename,"source_frame": index,"native_pts_s": float(stamp),
                     "source_image_sha256": sha256(out/filename)})
        prev_gray = gray
    cap.release()
    return rows, np.asarray(descriptors)


def propose_pairs(rows, desc, centers, axes, pose_ok, span_s, use_prior):
    frames = np.asarray([row["source_frame"] for row in rows])
    time = np.asarray([row["native_pts_s"] for row in rows])
    pairs, loop_candidates = set(), set()
    for i in range(len(frames)):
        score = desc @ desc[i]

        def top(candidates, count):
            if not len(candidates):
                return []
            count = min(count,len(candidates))
            chosen = np.argpartition(score[candidates], -count)[-count:]
            return sorted(candidates[chosen],key=lambda j:(-score[j],int(j)))

        # Dense short-range support plus longer links across consecutive local windows.
        for j in range(i+1, min(i+7,len(frames))):
            pairs.add((i,j))
        ahead = np.flatnonzero((time-time[i] > .45) & (time-time[i] <= span_s))
        for j in top(ahead,4):
            pairs.add((i,int(j)))
        far = np.flatnonzero(abs(time-time[i]) > span_s)
        if use_prior:
            track = frames[far]
            center_gap = np.linalg.norm(centers[track]-centers[frames[i]],axis=1)
            look = axes[track,2,:] @ axes[frames[i],2,:]
            near = far[(center_gap < 1.75) & (look > .55) & pose_ok[track] & pose_ok[frames[i]]]
        else:
            near = np.empty(0,dtype=int)
        best = top(far,3)+top(near,3)
        for j in best:
            if score[j] >= .55:
                loop_candidates.add(tuple(sorted((i,int(j)))))
    return sorted(pairs|loop_candidates), loop_candidates


def reconstruct(root, out, rotation, roster, video_hash, min_hz, max_hz, feature_size, offset, threads, select_only=False):

    t0 = monotonic()
    video, pts, k, trace, info = read_recorder(root,rotation,video_hash)
    centers, axes, pose_ok = poses_at(pts,trace,rotation,offset)
    pose_usable = info["gyro_pose_rate_pearson"] is not None and info["gyro_pose_rate_pearson"] >= .45
    out.mkdir(parents=True,exist_ok=False)
    images = out/"images"
    images.mkdir()
    rows, desc = select_frames(video,pts,centers,axes,pose_ok & pose_usable,images,rotation,roster,min_hz,max_hz)
    names = [row["name"] for row in rows]
    pairs, loops = propose_pairs(rows,desc,centers,axes,pose_ok,8.,pose_usable)
    pair_file = out/"pairs.txt"
    pair_file.write_text("".join(f"{names[i]} {names[j]}\n" for i,j in pairs))
    emit_json(out/"recorder_qa.json",{**info,"prior_used_for_pair_proposals_only": pose_usable,
                 "pose_interpolation_valid": int(pose_ok.sum()),"offset_s":offset,
                 "frame_selection":"historical indices" if roster else "motion adaptive, native PTS",
                 "selected":rows,"loop_candidates":len(loops),"pair_candidates":len(pairs),
                 "selected_per_10s":[sum(bool(start<=f["native_pts_s"]<start+10) for f in rows)
                                     for start in np.arange(0,pts[-1],10)]})
    if select_only:
        return
    import pycolmap as pc

    db = out/"database.db"
    e = pc.FeatureExtractionOptions(max_image_size=feature_size,num_threads=threads,use_gpu=False)
    e.sift.max_num_features = 4500
    reader = pc.ImageReaderOptions(camera_model="OPENCV",camera_params=",".join(map(str,(*k[:4],0,0,0,0))))
    pc.extract_features(db,images,camera_mode=pc.CameraMode.SINGLE,reader_options=reader,extraction_options=e,device=pc.Device.cpu)
    match = pc.FeatureMatchingOptions(num_threads=threads,use_gpu=False,max_num_matches=5000)
    pairing = pc.ImportedPairingOptions(match_list_path=pair_file)
    pc.match_image_pairs(db,matching_options=match,pairing_options=pairing,
                         verification_options=pc.TwoViewGeometryOptions(min_num_inliers=25),device=pc.Device.cpu)
    with sqlite3.connect(db) as connection:
        image_ids = {key:name for key,name in connection.execute("SELECT image_id,name FROM images")}
        verified_pairs = [(image_ids[i],image_ids[j],n,kind) for pair,n,kind in connection.execute(
            "SELECT pair_id,rows,config FROM two_view_geometries WHERE rows>=25")
            for i,j in [pc.pair_id_to_image_pair(pair)]]
    name_index = {name:i for i,name in enumerate(names)}
    loop_rows = [p for p in verified_pairs if tuple(sorted((name_index[p[0]],name_index[p[1]]))) in loops]
    emit_json(out/"verified_pairs.json",{"verified_count":len(verified_pairs),
                 "nonlocal_geometry_verified":loop_rows,"pose_graph_claim":"COLMAP global bundle adjustment"})
    if not verified_pairs:
        raise Blocked("Empty geometrically verified pair table; mapper cannot resume")
    models = out/"sparse"
    models.mkdir()
    options = pc.IncrementalPipelineOptions(num_threads=threads,random_seed=42,ba_refine_focal_length=True,
                                            ba_refine_extra_params=False, min_model_size=8)
    options.mapper.init_min_tri_angle = 6.0
    options.ba_global_max_num_iterations = 40
    options.ba_global_frames_ratio = 1.5
    options.ba_global_points_ratio = 1.5
    recs = pc.incremental_mapping(db,images,models,options)
    if not recs:
        raise Blocked("Global mapper did not find a usable model")
    best = max(recs,key=lambda i:(recs[i].num_reg_images(),recs[i].num_points3D(),-i))
    rec = recs[best]
    rec.export_PLY(out/"sparse.ply")
    text_dir = out/"model_text"
    text_dir.mkdir()
    rec.write_text(text_dir)
    registered = {rec.images[i].name for i in rec.reg_image_ids()}
    facts = [{"start_s":int(start),"selected":sum(bool(start<=f["native_pts_s"]<start+10) for f in rows),
              "registered":sum(bool(start<=f["native_pts_s"]<start+10 and f["name"] in registered) for f in rows)}
             for start in np.arange(0,pts[-1]+1,10)]
    emit_json(out/"sfm_receipt.json",{"engine":f"PyCOLMAP {pc.__version__}",
                 "selected":len(rows),"registered":len(registered),"sparse_points3D":rec.num_points3D(),
                 "coverage":facts,"verification_edges":len(verified_pairs),"nonlocal_edges":len(loop_rows),
                 "components":{str(i):{"registered":m.num_reg_images(),"points":m.num_points3D()} for i,m in recs.items()},
                 "elapsed_s":monotonic()-t0,"is_metric":False,"best_model":best,
                 "mean_internal_reprojection_px":rec.compute_mean_reprojection_error()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder",type=Path,required=True,help="Extracted original Recorder archive")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--rotation",type=int,choices=(0,90,180,270),required=True)
    parser.add_argument("--video-sha256")
    parser.add_argument("--historical-roster",type=Path)
    parser.add_argument("--min-hz",type=float,default=3.)
    parser.add_argument("--max-hz",type=float,default=10.)
    parser.add_argument("--feature-size",type=int,default=1280)
    parser.add_argument("--pose-clock-offset",type=float,default=0.)
    parser.add_argument("--threads",type=int,default=6)
    parser.add_argument("--select-only",action="store_true",help="Write frame roster and candidate pair list without running COLMAP")
    a=parser.parse_args()
    reconstruct(a.recorder,a.out,a.rotation,a.historical_roster,a.video_sha256,a.min_hz,a.max_hz,
                a.feature_size,a.pose_clock_offset,a.threads,a.select_only)


if __name__ == "__main__":
    main()
