"""Fail-closed audit of a historical COLMAP capture; never assigns its poses to new video frames."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .workflow import Blocked,parse_colmap_text,sha256,emit_json,checked_run

def audit(run_dir:Path, model_dir:Path, receipt_file:Path, *,
          expected_images_sha256:str, expected_source_sha256:str,
          sample_dir:Path|None=None)->dict:
    checked_run(run_dir)
    images_text=model_dir/"images.txt"
    cameras_text=model_dir/"cameras.txt"
    if not (images_text.is_file() and cameras_text.is_file()):
        raise Blocked("Missing historical COLMAP TXT camera files")
    if sha256(images_text)!=expected_images_sha256:
        raise Blocked("Original historical COLMAP images SHA changed")
    receipt=json.loads(receipt_file.read_text(encoding="utf-8"))
    cams, imgs=parse_colmap_text(model_dir)
    orig=receipt.get("manifest",{}).get("files",{})
    if len(imgs)!=192 or len(cams)!=1 or set(imgs)!=set(orig):
        raise Blocked("Historical 192-image camera/receipt identity mismatch")
    if receipt.get("capture_video_sha256")!=expected_source_sha256:
        raise Blocked("Historical original capture hash mismatch")
    checked=[]
    if sample_dir:
        for frame in sorted(sample_dir.glob("frame_*.jpg")):
            if frame.name not in orig or sha256(frame)!=orig[frame.name]["sha256"]:
                raise Blocked("Sample image receipt hash mismatch: "+frame.name)
            checked.append(frame.name)
    out=run_dir/"historical_camera_reference_v1.json"
    coverage_path=run_dir/"historical_pose_coverage_v1.json"
    if out.exists() or coverage_path.exists():
        raise Blocked("Historical reference output already exists: cannot overwrite")
    poses=[]; directions=[]; centers=[]
    for name,im in sorted(imgs.items()):
        w2c=np.asarray(im["world_to_camera_4x4"],dtype=float)
        c2w=np.linalg.inv(w2c)
        if not np.allclose(w2c@c2w,np.eye(4),atol=1e-7):
            raise Blocked("Improper historical camera pose")
        cam=cams[im["camera_id"]]
        poses.append({"view_id":name.removesuffix(".jpg"),"image_name":name,
           "image_sha256":orig[name]["sha256"],"camera":cam,
           "world_to_camera_4x4":w2c.tolist(),"camera_to_world_4x4":c2w.tolist(),
           "observations_3d":im["observations_3d"],"source_frame":"historical_colmap_opencv",
           "metric_scale_verified":False})
        directions.append(c2w[:3,:3]@np.array([0.,0.,1.]))
        centers.append(c2w[:3,3])
    direction=np.asarray(directions)
    pair=np.degrees(np.arccos(np.clip(direction@direction.T,-1.,1.)))
    np.fill_diagonal(pair,np.nan)
    nearest=np.nanmin(pair,axis=1)
    from datetime import datetime,timezone
    manifest={"schema_version":"mycon.reconstruction.historical_camera_reference.v1",
       "run_id":run_dir.name,"created_at_utc":datetime.now(timezone.utc).isoformat(),
       "source_video_sha256":expected_source_sha256,"images_txt_sha256":sha256(images_text),
       "cameras_txt_sha256":sha256(cameras_text),"frames_receipt_sha256":sha256(receipt_file),
       "source_frame":"historical_colmap_world_unscaled","units":"arbitrary_reconstruction_units",
       "new_colmap_executed":False,"pose_transfer_to_new_video_allowed":False,
       "metric_scale_verified":False,"registration_verified":False,"poses":poses}
    emit_json(out,manifest)
    report={"schema_version":"mycon.reconstruction.historical_pose_coverage.v1",
       "run_id":run_dir.name,"reference_sha256":sha256(out),
       "verified_sample_count":len(checked),"verified_sample_images":checked,
       "historical_registered_views":len(poses),
       "historical_median_nearest_look_direction_deg":float(np.nanmedian(nearest)),
       "historical_max_look_direction_deg":float(np.nanmax(pair)),
       "source_frame":"historical_colmap_world_unscaled","units":"arbitrary_reconstruction_units",
       "structural_surface_visibility":None,"new_colmap_executed":False,
       "independent_test_split_verified":False,"metric_scale_verified":False}
    emit_json(coverage_path,report)
    return report

def main(argv=None):
    p=argparse.ArgumentParser()
    for option in ("run-dir","model-dir","receipt-file","sample-dir"):
        p.add_argument("--"+option,type=Path,required=option!="sample-dir")
    p.add_argument("--expected-images-sha256",required=True)
    p.add_argument("--expected-source-sha256",required=True)
    a=p.parse_args(argv)
    try:
        r=audit(a.run_dir,a.model_dir,a.receipt_file,
                expected_images_sha256=a.expected_images_sha256,
                expected_source_sha256=a.expected_source_sha256,
                sample_dir=a.sample_dir)
        print(json.dumps({"status":"historical_only","views":r["historical_registered_views"],
                         "verified_samples":r["verified_sample_count"]}))
        return 0
    except (Blocked,OSError,ValueError,KeyError) as exc:
        print(json.dumps({"status":"blocked","reason":str(exc)}))
        return 2
if __name__=="__main__":raise SystemExit(main())
