"""Fail-closed, read-only verification of independent MyCON Reconstruction manifests.

This is a small standalone integration adapter. Full round-2 CPU implementation
and 19 tests are provided as a separate downloadable research source bundle;
do not imply that this adapter alone reconstructs the original room.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "mycon.reconstruction.dataset.v1"

class Blocked(RuntimeError):
    pass

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def audit_run(run_dir: Path) -> dict:
    run_dir=run_dir.resolve()
    manifest_path=run_dir/"dataset_manifest.json"
    if not manifest_path.is_file():
        raise Blocked("No dataset_manifest.json")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("run_id") != run_dir.name:
        raise Blocked("Dataset schema/run ID mismatch")
    frames=manifest.get("frames")
    if not isinstance(frames,list) or not frames:
        raise Blocked("No frame records")
    seen=set()
    counts={"train":0,"test":0,"unknown":0}
    sessions={"train":set(),"test":set()}
    for frame in frames:
        view=str(frame.get("view_id",""))
        expected=frame.get("image_sha256")
        split=frame.get("split")
        if not view or split not in ("train","test") or not isinstance(expected,str) or len(expected)!=64:
            raise Blocked("Invalid frame split/view/hash")
        path=(run_dir/str(frame.get("image_relpath",""))).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file() or path.is_symlink():
            raise Blocked("Unsafe/missing frame file")
        actual=sha256(path)
        if actual != expected or actual in seen:
            raise Blocked("Frame modified or duplicate image across splits")
        seen.add(actual)
        counts[split]+=1
        sessions[split].add(str(frame.get("capture_session_id","")))
    if not counts["train"] or not counts["test"]:
        raise Blocked("No independent train/test split")
    if manifest.get("train_test_protocol")=="capture_session_heldout" and sessions["train"].intersection(sessions["test"]):
        raise Blocked("Capture session leaked across split")
    if manifest.get("metric_scale_verified") is not False:
        raise Blocked("Unverified metric scale must be false")
    poses=run_dir/"camera_poses.json"
    if poses.exists():
        data=json.loads(poses.read_text(encoding="utf-8"))
        if data.get("schema_version")!="mycon.reconstruction.camera_poses.v1" or data.get("run_id")!=run_dir.name:
            raise Blocked("Camera manifest version/run mismatch")
        found={}
        for pose in data.get("poses",[]):
            key=pose.get("view_id")
            if key in found:
                raise Blocked("Duplicate view ID in camera poses")
            found[key]=pose
        if set(found)!={f["view_id"] for f in frames}:
            raise Blocked("Missing/extra pose records")
        for frame in frames:
            pose=found[frame["view_id"]]
            if pose.get("image_sha256")!=frame["image_sha256"] or pose.get("split")!=frame["split"]:
                raise Blocked("Frame SHA/split mismatch in camera")
            if pose.get("registered") and (not pose.get("camera") or not pose.get("world_to_camera_4x4")):
                raise Blocked("Registered camera missing fresh calibration/extrinsics")
    return {"schema_version":"mycon.reconstruction.audit.v1","run_id":run_dir.name,
            "source_manifest_sha256":sha256(manifest_path),
            "frame_counts":counts,"poses_present":poses.exists(),
            "metric_scale_verified":False, "field_validated":False}

def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True,type=Path)
    a=parser.parse_args(argv)
    try:
        print(json.dumps(audit_run(a.run_dir),indent=2))
    except (OSError,ValueError,Blocked,KeyError) as exc:
        print(json.dumps({"status":"BLOCKED","reason":str(exc)}))
        return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
