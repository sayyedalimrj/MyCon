"""Create-only human-review comparison strips and multi-view MP4 from verified renders.

Visual QA is not ground truth of room surfaces or proof of model improvement.
No actual rendering or training is performed by this module.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .workflow import Blocked, checked_run, emit_json, rel, sha256


def export(run_dir: Path, render_manifest: Path, *, fps: int = 4) -> dict:
    import cv2
    if fps < 1 or fps > 30:
        raise Blocked("FPS must be 1..30")
    manifest = checked_run(run_dir)
    poses = run_dir / "camera_poses.json"
    scores = run_dir / "per_view_metrics.csv"
    if not poses.exists() or not scores.exists():
        raise Blocked("Need fresh camera_poses and held-out score table")
    incoming = json.loads(render_manifest.read_text(encoding="utf-8"))
    if incoming.get("schema_version") != "mycon.reconstruction.render_manifest.v1" or incoming.get("run_id") != run_dir.name:
        raise Blocked("Bad renderer manifest schema/run ID")
    if incoming.get("camera_poses_sha256") != sha256(poses):
        raise Blocked("Rendered camera spec SHA mismatches fresh pose file")
    frames = {x["view_id"]:x for x in manifest["frames"]}
    registered = {x["view_id"]:x for x in json.loads(poses.read_text(encoding="utf-8"))["poses"]}
    target = run_dir / "visual_review"
    if target.exists():
        raise Blocked("Refusing overwrite of visual_review")
    # Validate all inputs BEFORE creating the target directory.
    validated=[]
    for pred in incoming.get("renders", []):
        view=pred.get("view_id")
        src=frames.get(view)
        pose=registered.get(view)
        if not src or src["split"] != "test" or not pose or not pose.get("registered"):
            raise Blocked(f"Non-test or unregistered rendered view: {view}")
        if src["image_sha256"] != pred.get("image_sha256") or pose["image_sha256"] != src["image_sha256"]:
            raise Blocked("Render view does not match frozen test image identity")
        pred_path=(render_manifest.parent / pred["render_relpath"]).resolve()
        if not pred_path.is_relative_to(render_manifest.parent.resolve()) or not pred_path.is_file() or sha256(pred_path) != pred.get("render_sha256"):
            raise Blocked(f"Missing/tampered render: {view}")
        left=cv2.imread(str(run_dir/src["image_relpath"]), cv2.IMREAD_COLOR)
        right=cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise Blocked(f"Render shape/source failure: {view}")
        validated.append((view,src,left,right,pred_path))
    if not validated:
        raise Blocked("No verified real prediction images to review")
    target.mkdir(exist_ok=False)
    clips=[]
    writer=None
    mp4=target/"heldout_multiview_comparison.mp4"
    rows=[]
    try:
        for i,(view,src,left,right,pred_path) in enumerate(validated):
            label=f"GT / observed | fresh GS render | {view}"
            canvas=cv2.hconcat([left,right])
            # A status line goes inside image, not on original or prediction files.
            cv2.putText(canvas,label,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1,cv2.LINE_AA)
            out=target/f"comparison_{i:04d}.png"
            if not cv2.imwrite(str(out), canvas):
                raise Blocked("Failed writing comparison strip")
            if writer is None:
                h,w=canvas.shape[:2]
                writer=cv2.VideoWriter(str(mp4),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
                if not writer.isOpened():
                    raise Blocked("MP4 encoder unavailable; no video claimed")
            writer.write(canvas)
            rows.append({"view_id":view,"source_image_sha256":src["image_sha256"],
                         "render_sha256":sha256(pred_path),"comparison_relpath":rel(out,run_dir),
                         "comparison_sha256":sha256(out),"split":"test"})
    finally:
        if writer is not None:
            writer.release()
    if not mp4.is_file() or mp4.stat().st_size==0:
        raise Blocked("MP4 not successfully written")
    result={"schema_version":"mycon.reconstruction.visual_review.v1", "run_id":run_dir.name,
            "source_render_manifest_sha256":sha256(render_manifest),"camera_poses_sha256":sha256(poses),
            "per_view_scores_sha256":sha256(scores),"coordinate_frame":"colmap_opencv_world_to_camera",
            "units":"reconstruction_units_unscaled","metric_scale_verified":False,
            "preview_video_relpath":rel(mp4,run_dir),"preview_video_sha256":sha256(mp4),
            "view_count":len(rows),"frames":rows,
            "claim_limit":"VISUAL_QA_ONLY_not_surface_GT_not_independent_model_superiority"}
    emit_json(target/"visual_review_manifest.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--render-manifest",type=Path,required=True)
    parser.add_argument("--fps",type=int,default=4)
    a=parser.parse_args(argv)
    try:
        r=export(a.run_dir,a.render_manifest,fps=a.fps)
        print(json.dumps({"status":"review_bundle_ready","views":r["view_count"],"video":r["preview_video_relpath"]}))
        return 0
    except (OSError,RuntimeError,KeyError,ValueError) as e:
        print(json.dumps({"status":"blocked","reason":str(e)}))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
