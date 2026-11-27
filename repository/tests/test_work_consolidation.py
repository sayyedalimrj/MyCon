"""Synthetic cross-stream tests. Engineering and field validation are separate."""
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import pytest

from pipeline.stage_07_cleanup.r_frame_geometry import generate
from pipeline.stage_09_progress.enrich_scientific_evidence import enrich_scientific_evidence_rows
from reconstruction.workflow import Blocked, _choose_train_model, report_poses, run_colmap, sha256
from viewer_civil.core import ViewerInputError, load_bundle
from viewer_civil.producer_bridge import convert_r_run
from test_research_geometry_r_frame import fixture


@pytest.fixture
def sources(tmp_path):
    files = fixture(tmp_path)
    run = tmp_path / "recon-work-test"
    model = run / "colmap/train_text"
    model.mkdir(parents=True)
    for key in ("cameras", "images", "points"):
        shutil.copyfile(files[key], model / files[key].name)
        files[key] = model / files[key].name
    heldout = run / "colmap/registered_text"
    heldout.mkdir()
    shutil.copyfile(files["cameras"], heldout / "cameras.txt")
    registered = heldout / "images.txt"
    registered.write_text(files["images"].read_text() + "5 1 0 0 0 0.1 0 0 1 test.jpg\n\n")
    photos = run / "images"
    photos.mkdir()
    frames = []
    for i in range(1, 6):
        name = f"frame_{i:06}.jpg" if i < 5 else "test.jpg"
        path = photos / name
        Image.new("RGB", (640, 480), (i*30, 75, 80)).save(path)
        frames.append({"view_id": name, "image_relpath": f"images/{name}",
                       "image_sha256": sha256(path), "width": 640, "height": 480,
                       "capture_session_id": "synthetic", "split": "train" if i < 5 else "test"})
    dataset = run / "dataset_manifest.json"
    dataset.write_text(json.dumps({"schema_version": "mycon.reconstruction.dataset.v1",
                                   "run_id": run.name, "synthetic": True,
                                   "metric_scale_verified": False, "frames": frames}))
    files.update(photos=sorted(photos.glob("frame_*.jpg"))[:3],
                 r_dataset=dataset, r_poses=run / "camera_poses.json")
    return run, model, heldout, files


def test_frozen_train_to_a_to_viewer_and_b_keeps_unknown(sources):
    run, model, heldout, files = sources
    report_poses(run, model, heldout)
    result = generate(**files, max_planes=2, trials=20, min_points=180, threshold=.025)
    assert result["R_shared_world_frame"] is True
    root = files["root"]
    a = Path(result["output_dir"]) / "civil_r_frame_manifest.json"
    destination = root / "viewer-test"
    convert_r_run(run, destination, project_root=root, a_manifest=a.relative_to(root).as_posix())
    bundle = load_bundle(destination)
    assert bundle["manifest"]["geometry_alignment_status"] == "same_frozen_train_model_internal_nonmetric"
    assert bundle["assets"]["planes"]["planes"][0]["state"] == "unknown"
    assert bundle["deviation_enabled"] is False
    rows, summary = enrich_scientific_evidence_rows(
        [{"global_id": "synthetic-only"}], manifest_path=None, project_root=root,
        a_handoff_manifest=a, expected_a_handoff_sha256=sha256(a))
    assert summary["upstream_receipt_audit"]["A"]["integrity_valid"]
    assert summary["upstream_receipt_audit"]["direct_geometry_gate"] == "blocked"
    assert rows[0]["physical_progress_estimate"] is None
    assert rows[0]["geometry_evidence_state"].startswith("unknown")
    (a.parent / "viewer_overlay.json").write_text("{}")
    with pytest.raises(ViewerInputError, match="receipt SHA256"):
        convert_r_run(run, root / "tampered-viewer", project_root=root,
                      a_manifest=a.relative_to(root).as_posix())


@pytest.mark.parametrize("tamper, expected", [
    ("pose", "world-to-camera pose mismatch"),
    ("points", "train text model or dataset receipt mismatch"),
    ("photo", "frozen R training split"),
])
def test_frozen_train_handoff_rejects_bad_pose_model_or_photo(sources, tamper, expected):
    run, model, heldout, files = sources
    report_poses(run, model, heldout)
    if tamper == "pose":
        pose_path = files["r_poses"]
        document = json.loads(pose_path.read_text())
        document["poses"][0]["world_to_camera_4x4"][0][3] = 42
        pose_path.write_text(json.dumps(document))
    elif tamper == "points":
        with files["points"].open("a") as stream:
            stream.write("# different 3D file\n")
    else:
        files["photos"][0] = run / "images/test.jpg"
    with pytest.raises(ValueError, match=expected):
        generate(**files)
    assert not (files["root"] / "runs").exists()


def test_heldout_registration_cannot_move_train_world(sources):
    run, model, heldout, files = sources
    poses = heldout / "images.txt"
    poses.write_text(poses.read_text().replace("1 1 0 0 0 0.0 ", "1 1 0 0 0 12.0 ", 1))
    with pytest.raises(Blocked, match="changed frozen train camera or pose"):
        report_poses(run, model, heldout)
    assert not files["r_poses"].exists()


def test_tampered_train_list_never_reaches_colmap(tmp_path):
    from test_research_reconstruction_workflow import select_run

    run, dataset = select_run(tmp_path)
    test = next(frame for frame in dataset["frames"] if frame["split"] == "test")
    with (run / "train_image_names.txt").open("a") as stream:
        stream.write(Path(test["image_relpath"]).relative_to("images").as_posix() + "\n")
    with pytest.raises(Blocked, match="Training list differs from the frozen split"):
        run_colmap(run, sys.executable, execute=True)
    assert not (run / "colmap").exists()



def test_sparse_component_selection_prioritizes_camera_registration(tmp_path):
    for label, registered, points in [("0", 17, 1625), ("1", 20, 1522), ("2", 2, 15)]:
        model = tmp_path / label
        model.mkdir()
        (model / "images.bin").write_bytes(registered.to_bytes(8, "little"))
        (model / "points3D.bin").write_bytes(points.to_bytes(8, "little"))
        (model / "cameras.bin").write_bytes(b"fixture")
    selected, counts = _choose_train_model(tmp_path, 49)
    assert selected.name == "1"
    assert len(counts) == 3
