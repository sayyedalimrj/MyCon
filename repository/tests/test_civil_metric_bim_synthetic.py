"""Synthetic software QA only: never a surveyed building or real IFC acceptance."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from civil_metric_bim.core import (
    build_transform, deviation, length_scale, run, umeyama, verify_geometry,
)


def approval():
    return {"status": "approved", "reviewer": "synthetic_test_operator", "evidence_id": "fixture-only"}


def span(name, length, role="fit", source=None, actual=True, kind="field_measurement"):
    return {
        "reference_id": name, "source_kind": kind, "physical_source_id": source or name,
        "role": role, "scan_endpoints_xyz": [[0, 0, 0], [length, 0, 0]],
        "nominal_length_m": 2 * length + 0.01,
        "actual_length_m": 2 * length if actual else None,
        "actual_uncertainty_m": 0.002,
        "scan_distance_uncertainty_units": 0.0001,
        "approval": approval(),
        "confidence": {"grade": "high_reviewed", "calibrated_probability": None},
    }


def references(fit=True):
    ss = [span("block_A", 1, kind="block"),
          span("steel_B", 2, kind="steel_section"),
          span("survey_C", 1.5),
          span("survey_D", 3, role="control")]
    if not fit:
        for r in ss[:3]:
            r["approval"] = {"status": "proposed"}
    return {"schema_version": "mycon.length_references.v1",
            "source_frame": "colmap_reconstruction_world",
            "source_units": "reconstruction_units", "references": ss}


def features(with_matches=True):
    samples = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 0.5]]
    matches = []
    if with_matches:
        for i, xyz in enumerate(samples):
            matches.append({
                "match_id": "f"+str(i), "physical_feature_id": "feature_"+str(i),
                "kind": "corner", "role": "control" if i == 4 else "fit",
                "scan_xyz": xyz,
                "bim_xyz_m": (2 * np.array(xyz) + np.array([1, 2, 3])).tolist(),
                "observation_state": "observed_supported", "approval": approval()})
    return {
        "schema_version": "mycon.bim_feature_adapter.v1",
        "source_frame": "colmap_reconstruction_world",
        "target_frame": "bim_world", "target_units": "m",
        "model_provenance": {"kind": "synthetic_fixture_only"},
        "matches": matches,
        "planes": [{
            "plane_id": "floor", "point_xyz_m": [1, 2, 2],
            "normal_xyz": [0, 0, 1],
            "finite_vertices_xyz_m": [[1, 2, 2], [3, 2, 2], [3, 4, 2], [1, 4, 2]],
            "observation_support": "reviewed_finite",
            "ifc_global_id": None,
            "approval": approval(),
        }],
        "samples": [{
            "sample_id": "signed", "scan_xyz": [0.5, 0.5, -0.45],
            "observation_state": "observed", "bim_plane_id": "floor",
        }, {
            "sample_id": "occluded", "scan_xyz": [0.5, 0.5, -0.45],
            "observation_state": "occluded", "bim_plane_id": "floor",
        }, {
            "sample_id": "outside", "scan_xyz": [5.0, 5.0, -0.45],
            "observation_state": "observed", "bim_plane_id": "floor",
        }, {
            "sample_id": "no_mapping", "scan_xyz": [0.5, 0.5, -0.45],
            "observation_state": "observed", "bim_plane_id": None,
        }],
    }


def save(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_length_scale_uses_manual_actual_not_nominal():
    s = length_scale(references())
    assert s["status"] == "scale_candidate_review_required"
    assert s["scale_m_per_reconstruction_unit"] == pytest.approx(2.0)
    assert s["uncertainty_1sigma_m_per_unit"] > 0
    assert s["fit_reference_ids"] == ["block_A", "steel_B", "survey_C"]
    assert s["heldout_reference_ids"] == ["survey_D"]
    assert s["references"][0]["nominal_minus_actual_m"] == pytest.approx(0.01)
    assert s["metric_validation"] == "pending"


def test_nominal_only_never_accepted_for_scale():
    r = references()
    r["references"][0]["actual_length_m"] = None
    s = length_scale(r)
    assert s["scale_m_per_reconstruction_unit"] is None
    assert s["references"][0]["state"] == "nominal_only_not_accepted"


def test_low_confidence_review_rejected():
    r = references()
    r["references"][0]["confidence"]["grade"] = "medium_provisional"
    assert length_scale(r)["scale_m_per_reconstruction_unit"] is None


def test_unapproved_references_rejected():
    s = length_scale(references(fit=False))
    assert s["scale_m_per_reconstruction_unit"] is None


def test_fit_control_group_leakage_is_fatal():
    r = references()
    r["references"][-1]["physical_source_id"] = "steel_B"
    with pytest.raises(ValueError, match="reuse"):
        length_scale(r)


def test_inconsistent_length_sources_never_certified():
    r = references()
    r["references"][2]["actual_length_m"] = 5.0
    s = length_scale(r)
    assert s["scale_m_per_reconstruction_unit"] is None
    assert s["status"] == "inconsistent_fit_references"


def test_sim3_with_length_scale_and_heldout():
    r = references()
    m = features()
    a, tx = build_transform(m, length_scale(r))
    assert tx is not None
    assert a["scale"] == pytest.approx(2.0)
    assert np.array(a["rotation"]) == pytest.approx(np.eye(3), abs=1e-10)
    assert a["translation"] == pytest.approx([1, 2, 3], abs=1e-10)
    assert a["heldout_match_ids"] == ["f4"]
    assert a["metric_scale_verified"] is False
    assert a["can_feed_stage8"] is False
    assert a["metric_validation"] == "pending"


def test_independent_bim_only_path():
    m = features()
    a, tx = build_transform(m, length_scale(None))
    assert tx is not None
    assert a["scale_source"] == "bim_correspondences_unverified"
    assert a["scale"] == pytest.approx(2.0)
    assert a["metric_validation"] == "pending"


def test_rotation_translation_scale():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    y = 1.75 * (x @ R.T) + [3, 1, 4]
    s, foundR, t = umeyama(x, y, fixed_scale=None)
    assert s == pytest.approx(1.75)
    assert foundR == pytest.approx(R)
    assert t == pytest.approx([3, 1, 4])


def test_rank_deficient_match_fail_closed():
    m = features()
    for i, e in enumerate(m["matches"][:4]):
        e["scan_xyz"] = [float(i), 0, 0]
        e["bim_xyz_m"] = [2 * float(i), 0, 0]
    a, tx = build_transform(m, length_scale(None))
    assert tx is None
    assert a["status"] == "blocked_degenerate_correspondences"


def test_reviewed_finite_mask_signed_and_missing():
    _, tx = build_transform(features(), length_scale(references()))
    out = deviation(features(), tx)
    assert out[0]["valid_mask"] is True
    assert out[0]["signed_distance_m"] == pytest.approx(0.1)
    assert out[0]["unsigned_distance_m"] == pytest.approx(0.1)
    assert all(not x["valid_mask"] and x["signed_distance_m"] is None for x in out[1:])
    assert out[1]["reason"] == "not_observed_or_occluded"


def test_unreviewed_plane_masks_every_sample():
    m = features()
    m["planes"][0]["approval"] = {"status": "proposed"}
    _, tx = build_transform(m, length_scale(references()))
    out = deviation(m, tx)
    assert all(not x["valid_mask"] for x in out)


def test_create_only_outputs_and_hashed_manifest(tmp_path):
    r = save(tmp_path / "references.json", references())
    b = save(tmp_path / "features.json", features())
    manifest = run(tmp_path, "metric-bim-synthetic-001",
                   references_json=r, bim_json=b)
    target = tmp_path / "runs/metric-bim-synthetic-001/metric_bim"
    expected = ["scale_estimate.json", "alignment_transform.json",
                "alignment_report.csv", "heldout_control_errors.csv",
                "deviation_map.csv", "deviation_heatmap.json", "viewer_handoff.json"]
    assert sorted(manifest["outputs"]) == sorted(expected)
    for name in expected:
        file = target / name
        assert file.is_file()
        assert manifest["outputs"][name]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    with (target / "heldout_control_errors.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # one held-out length and one held-out corner
    assert max(float(x["unsigned_error_m"]) for x in rows) < 1e-9
    assert manifest["metric_validation"] == "pending"
    assert manifest["physical_progress_state"] == "unknown"
    assert json.loads((target / "alignment_transform.json").read_text())["can_feed_stage8"] is False
    assert r.read_text() == json.dumps(references())
    with pytest.raises(FileExistsError):
        run(tmp_path, "metric-bim-synthetic-001", references_json=r, bim_json=b)


def test_length_only_heldout_evaluated_without_any_bim(tmp_path):
    r = save(tmp_path / "references.json", references())
    manifest = run(tmp_path, "metric-bim-length-only-005", references_json=r)
    root = tmp_path / "runs/metric-bim-length-only-005/metric_bim"
    scale = json.loads((root / "scale_estimate.json").read_text())
    alignment = json.loads((root / "alignment_transform.json").read_text())
    with (root / "heldout_control_errors.csv").open() as stream:
        controls = list(csv.DictReader(stream))
    assert scale["scale_m_per_reconstruction_unit"] == pytest.approx(2.0)
    assert alignment["scale"] is None
    assert len(controls) == 1
    assert controls[0]["state"] == "measured_candidate"
    assert float(controls[0]["unsigned_error_m"]) == pytest.approx(0.0)
    assert alignment["metric_validation"] == "pending"


def test_bim_only_candidate_scale_is_not_field_verified(tmp_path):
    b = save(tmp_path / "bim.json", features())
    run(tmp_path, "metric-bim-bim-only-006", bim_json=b)
    target = tmp_path / "runs/metric-bim-bim-only-006/metric_bim"
    scale = json.loads((target / "scale_estimate.json").read_text())
    assert scale["scale_m_per_reconstruction_unit"] is None
    assert scale["bim_correspondence_scale_candidate_m_per_unit"] == pytest.approx(2.0)
    assert scale["bim_correspondence_uncertainty_1sigma_m_per_unit"] is None
    assert scale["metric_validation"] == "pending"


def test_missing_bim_outputs_are_unknown_not_zero(tmp_path):
    b = save(tmp_path / "features.json", features(with_matches=False))
    result = run(tmp_path, "metric-bim-no-match-002", bim_json=b)
    root = tmp_path / result["outputs"]["deviation_heatmap.json"]["path"]
    heat = json.loads(root.read_text())
    assert heat["valid_sample_count"] == 0
    assert all(x["signed_distance_m"] is None for x in heat["points"])


def test_reject_corrupted_upstream_geometry_hash(tmp_path):
    good = tmp_path / "runs/geom-ready-fake/geometry"
    good.mkdir(parents=True)
    src = good / "clean_cloud.npy"
    src.write_bytes(b"fixture")
    manifest = {
        "schema_version": "mycon.comparison_ready_geometry.v1",
        "coordinate_frame": "colmap_reconstruction_world",
        "metric_scale_verified": False, "source_path": "",
        "source_sha256": None,
        "outputs": {"clean_cloud.npy": {
            "path": "runs/geom-ready-fake/geometry/clean_cloud.npy",
            "sha256": hashlib.sha256(b"fixture").hexdigest()}}}
    p = save(good / "comparison_ready.json", manifest)
    verified = verify_geometry(p, tmp_path, "colmap_reconstruction_world")
    assert verified["source_sha256"] is None
    src.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_geometry(p, tmp_path, "colmap_reconstruction_world")


def test_validation_blocks_mixed_frame_and_bad_units(tmp_path):
    r = references()
    b = features()
    b["source_frame"] = "other"
    rp = save(tmp_path / "references.json", r)
    bp = save(tmp_path / "features.json", b)
    with pytest.raises(ValueError, match="frames disagree"):
        run(tmp_path, "metric-bim-frame-003", references_json=rp, bim_json=bp)
    assert not (tmp_path / "runs/metric-bim-frame-003").exists()


def test_negative_run_id_and_observability(tmp_path):
    b = features()
    b["samples"][0]["observation_state"] = "unknown"
    p = save(tmp_path / "features.json", b)
    with pytest.raises(ValueError, match="Run ID"):
        run(tmp_path, "../bad", bim_json=p)
    result = run(tmp_path, "metric-bim-unknown-004", bim_json=p)
    heat = json.loads((tmp_path / result["outputs"]["deviation_heatmap.json"]["path"]).read_text())
    assert heat["points"][0]["valid_mask"] is False
