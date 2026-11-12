"""Deterministic lightweight tests; no GPU, pipeline or actual site files."""
from __future__ import annotations

import json

import pytest

from benchmarks.mycon_2026.metrics import (compare_anchors, ifc_element_metrics,
                                           progress_metrics, read_ply_xyz,
                                           reconstruction_metrics)
from benchmarks.mycon_2026.protocol import validate_manifest
from benchmarks.mycon_2026.runner import run

pytestmark = pytest.mark.lightweight


def test_geometry_identity_and_translation():
    gt = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]
    result = reconstruction_metrics(gt, gt, tau_m=0.05)
    assert result["fscore_at_tau"] == 1
    assert result["chamfer_l1_m"] == 0
    moved = [(x + 0.1, y, z) for x, y, z in gt]
    result = reconstruction_metrics(moved, gt, tau_m=0.05)
    assert result["fscore_at_tau"] == 0
    assert result["chamfer_l1_m"] > 0


def test_read_ascii_ply(tmp_path):
    f = tmp_path / "points.ply"
    f.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 2 3\n")
    cloud, n = read_ply_xyz(f)
    assert n == 2 and cloud == [(0, 0, 0), (1, 2, 3)]


def test_anchor_scale_and_tre():
    truth = [{"anchor_id": "1", "x_m": "0", "y_m": "0", "z_m": "0"},
             {"anchor_id": "2", "x_m": "2", "y_m": "0", "z_m": "0"}]
    predicted = [dict(truth[0]), {**truth[1], "x_m": "2.2"}]
    out = compare_anchors(predicted, truth)
    assert out["scale_error_pct"] == pytest.approx(10)
    assert out["tre_mean_m"] == pytest.approx(0.1)


def test_missing_is_unknown_not_unbuilt():
    truth = [{"global_id": "A", "status": "built", "completion_fraction": "1", "weight": "1"},
             {"global_id": "B", "status": "not_built", "completion_fraction": "0", "weight": "1"},
             {"global_id": "C", "status": "partial", "completion_fraction": "0.4", "weight": "1"}]
    pred = [{"global_id": "A", "completion_state": "complete_candidate",
             "physical_progress_estimate": "0.9", "evidence_confidence": "0.8"},
            {"global_id": "B", "status": "not_evidenced", "confidence": "0.9"}]
    m = progress_metrics(pred, truth)
    assert m["n_unknown"] == 2
    assert m["unknown_rate"] == pytest.approx(2 / 3)
    assert m["per_class"]["not_built"]["support_evaluated"] == 0
    assert m["n_progress_pairs"] == 1
    assert m["progress_mae_fraction_known"] == pytest.approx(0.1)
    assert m["project_progress_lower_bound_fraction"] == pytest.approx(0.3)
    assert m["project_progress_upper_bound_fraction"] == pytest.approx(2.9 / 3)


def test_reject_bad_confidence():
    with pytest.raises(ValueError):
        progress_metrics([{"global_id": "A", "status": "built", "confidence": "1.2"}],
                         [{"global_id": "A", "status": "built"}])


def test_ifc_identity_class():
    pred = [{"global_id": "A", "ifc_class": "IfcWall"}, {"global_id": "B", "ifc_class": "IfcBeam"}]
    gt = [{"global_id": "A", "ifc_class": "IfcWall"}, {"global_id": "C", "ifc_class": "IfcSlab"}]
    result = ifc_element_metrics(pred, gt)
    assert result["identity_precision"] == pytest.approx(0.5)
    assert result["identity_recall"] == pytest.approx(0.5)
    assert result["ifc_class_accuracy_matched"] == 1
    assert result["n_invalid_ifc_globalid_format_pred"] == 2


def test_manifest_group_leakage():
    digest1, digest2 = "a" * 64, "b" * 64
    samples = [{"id": "s1", "site_id": "site", "building_id": "b1", "capture_session_id": "c1",
                "split": "train", "source_sha256": digest1, "methods": {"colmap": {}}},
               {"id": "s2", "site_id": "site", "building_id": "b1", "capture_session_id": "c2",
                "split": "test", "source_sha256": digest2, "methods": {"colmap": {}}}]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({"schema_version": "mycon.validation.v1", "samples": samples})


def test_manifest_repeated_hash_leakage():
    s = {"id": "a", "site_id": "s", "building_id": "1", "capture_session_id": "c1",
         "split": "train", "source_sha256": "f" * 64, "methods": {"x": {}}}
    t = {**s, "id": "b", "building_id": "2", "capture_session_id": "c2", "split": "test"}
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({"schema_version": "mycon.validation.v1", "samples": [s, t]})


def test_synthetic_tool_scoring_is_not_thesis_data(tmp_path):
    dataset = tmp_path / "inputs"
    dataset.mkdir()
    points = "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n"
    (dataset / "gt.ply").write_text(points)
    (dataset / "pred.ply").write_text(points)
    manifest = {"schema_version": "mycon.validation.v1", "samples": [
        {"id": "fixture", "site_id": "demo", "building_id": "demo", "capture_session_id": "demo-session",
         "split": "test", "data_kind": "synthetic", "source_sha256": "c" * 64,
         "ground_truth": {"reconstruction_gt_ply": "gt.ply"},
         "methods": {"fixture-method": {"reconstruction_ply": "pred.ply"}}}]}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "runs"
    info = run(dataset / "manifest.json", out, run_id="bench-smoke", include_fixtures=True)
    assert info["scientific_status"] == "not_executed"
    assert json.loads((out / "bench-smoke" / "scientific_results.json").read_text()) == []
    cases = json.loads((out / "bench-smoke" / "cases.json").read_text())
    assert next(x for x in cases if x["task"] == "reconstruction")["metrics"]["fscore_at_tau"] == 1
    with pytest.raises(FileExistsError):
        run(dataset / "manifest.json", out, run_id="bench-smoke", include_fixtures=True)


def test_missing_real_outputs_not_executed(tmp_path):
    dataset = tmp_path / "inputs"
    dataset.mkdir()
    path = dataset / "manifest.json"
    path.write_text(json.dumps({"schema_version": "mycon.validation.v1", "samples": [{
        "id": "case", "site_id": "one", "building_id": "one", "capture_session_id": "c1",
        "split": "test", "source_sha256": "e" * 64,
        "ground_truth": {}, "methods": {"ours": {}}}]}))
    info = run(path, tmp_path / "run_outputs", run_id="bench-no-real")
    assert info["scientific_status"] == "not_executed"


def test_partially_observed_is_not_partial_progress():
    p = [{"global_id": "X", "status": "partially_observed", "confidence": "high"}]
    gt = [{"global_id": "X", "status": "partial", "completion_fraction": "0.5"}]
    m = progress_metrics(p, gt)
    assert m["n_unknown"] == 1
    assert m["micro_accuracy_selective"] is None
    assert m["ece_10bin_selective"] is None


def test_site_split_leakage():
    s = {"id": "a", "site_id": "s", "building_id": "1", "capture_session_id": "c1",
         "split": "train", "source_sha256": "a" * 64, "methods": {"x": {}}}
    t = {**s, "id": "b", "building_id": "2", "capture_session_id": "c2",
         "source_sha256": "b" * 64, "split": "test"}
    with pytest.raises(ValueError, match="Site leakage"):
        validate_manifest({"schema_version": "mycon.validation.v1", "samples": [s, t]})


def test_binary_ply(tmp_path):
    import struct
    f = tmp_path / "binary.ply"
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    f.write_bytes(header + struct.pack("<ffffff", 1, 2, 3, 4, 5, 6))
    points, total = read_ply_xyz(f)
    assert total == 2
    assert points == [(1., 2., 3.), (4., 5., 6.)]


def test_real_requires_independent_gt_review(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "capture.txt").write_text("raw")
    import hashlib
    digest = hashlib.sha256(b"raw").hexdigest()
    manifest = {"schema_version": "mycon.validation.v1", "samples": [{
        "id": "s", "site_id": "s", "building_id": "b", "capture_session_id": "c",
        "split": "test", "source_sha256": digest, "source_path": "capture.txt",
        "data_kind": "real", "ground_truth": {}, "methods": {"ours": {}}}]}
    (base / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "out"
    result = run(base / "manifest.json", out, run_id="bench-reject-unreviewed")
    assert result["scientific_status"] == "not_executed"
    case = json.loads((out / "bench-reject-unreviewed" / "cases.json").read_text())[0]
    assert case["reason"] == "real_sample_missing_independent_gt_provenance"
