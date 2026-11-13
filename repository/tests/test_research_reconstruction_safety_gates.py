"""Synthetic-only fail-closed contract tests, not a real room evaluation."""
import hashlib, json
from pathlib import Path
import pytest
from reconstruction.safety_gates import Blocked,audit_run

def fixture(tmp_path):
    run=tmp_path/"recon-round2-test";run.mkdir()
    frames=[]
    for split,session in [("train","session_A"),("test","session_B")]:
        path=run/(split+".png")
        path.write_bytes((split+" synthetic bytes").encode())
        frames.append({"view_id":split,"capture_session_id":session,"split":split,
                       "image_relpath":path.name,"image_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    m={"schema_version":"mycon.reconstruction.dataset.v1","run_id":run.name,
       "metric_scale_verified":False,"train_test_protocol":"capture_session_heldout","frames":frames}
    (run/"dataset_manifest.json").write_text(json.dumps(m))
    return run,m

def test_valid_split_and_manifest(tmp_path):
    run,_=fixture(tmp_path)
    assert audit_run(run)["frame_counts"]["train"]==1

def test_source_integrity(tmp_path):
    run,_=fixture(tmp_path);(run/"train.png").write_bytes(b"tampered")
    with pytest.raises(Blocked,match="modified"):audit_run(run)

def test_same_session_leak(tmp_path):
    run,m=fixture(tmp_path)
    m["frames"][1]["capture_session_id"]="session_A"
    (run/"dataset_manifest.json").write_text(json.dumps(m))
    with pytest.raises(Blocked,match="session leaked"):audit_run(run)

def test_false_metric_scale_claim(tmp_path):
    run,m=fixture(tmp_path);m["metric_scale_verified"]=True
    (run/"dataset_manifest.json").write_text(json.dumps(m))
    with pytest.raises(Blocked,match="metric scale"):audit_run(run)

def test_pose_missing_or_wrong_sha(tmp_path):
    run,m=fixture(tmp_path)
    poses=[{"view_id":f["view_id"],"split":f["split"],"image_sha256":f["image_sha256"],
            "registered":False} for f in m["frames"]]
    poses[0]["image_sha256"]="0"*64
    (run/"camera_poses.json").write_text(json.dumps({"schema_version":"mycon.reconstruction.camera_poses.v1","run_id":run.name,"poses":poses}))
    with pytest.raises(Blocked,match="SHA/split"):audit_run(run)

def test_registered_pose_requires_camera(tmp_path):
    run,m=fixture(tmp_path)
    poses=[{"view_id":f["view_id"],"split":f["split"],"image_sha256":f["image_sha256"],
            "registered":True} for f in m["frames"]]
    (run/"camera_poses.json").write_text(json.dumps({"schema_version":"mycon.reconstruction.camera_poses.v1","run_id":run.name,"poses":poses}))
    with pytest.raises(Blocked,match="calibration"):audit_run(run)
