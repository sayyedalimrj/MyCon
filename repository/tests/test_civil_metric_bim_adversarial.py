"""Adversarial synthetic D checks; no field data or real BIM dimensions."""
import csv
import json

import numpy as np
import pytest

from civil_metric_bim.core import (
    build_transform, length_scale, run, validate_inputs, robust_similarity, deviation,
)
from test_civil_metric_bim_synthetic import approval, features, references, save, span


def test_20_outlier_measurement_is_not_used_in_length_scale():
    r = references()
    bad = span("bogus_section", 1.0, source="bogus_physical_group")
    bad["actual_length_m"] = 3.0
    r["references"].append(bad)
    result = length_scale(r)
    assert result["scale_m_per_reconstruction_unit"] == pytest.approx(2.0)
    assert "bogus_section" not in result["fit_reference_ids"]
    assert next(x for x in result["references"] if x["reference_id"] == "bogus_section")["state"] == "rejected_inconsistent_span"


def test_21_two_good_spans_plus_wrong_measurement_fails():
    r = references()
    r["references"][2]["actual_length_m"] = 7.0
    result = length_scale(r)
    assert result["scale_m_per_reconstruction_unit"] is None
    assert result["status"] == "inconsistent_fit_references"


def test_22_only_two_independent_fit_lengths_remain_unscaled():
    r = references()
    r["references"] = r["references"][0:2] + r["references"][3:]
    assert length_scale(r)["scale_m_per_reconstruction_unit"] is None


def test_23_no_holdout_keeps_proposal_explicit():
    r = references()
    r["references"] = r["references"][:-1]
    result = length_scale(r)
    assert result["scale_m_per_reconstruction_unit"] == pytest.approx(2)
    assert result["status"] == "scale_candidate_without_approved_independent_control"
    assert result["approved_heldout_reference_count"] == 0
    assert result["metric_validation"] == "pending"


@pytest.mark.parametrize("mutator,match", [
    (lambda r, b: r.update(source_units="centimeters"), "reconstruction_units"),
    (lambda r, b: b.update(source_units="millimeters"), "reconstruction_units"),
    (lambda r, b: b.update(target_units="mm"), "metres"),
    (lambda r, b: b.update(target_frame="colmap_reconstruction_world"), "bim_world"),
])
def test_24_27_unit_or_frame_mismatch_rejected(mutator, match):
    r, b = references(), features()
    mutator(r, b)
    with pytest.raises(ValueError, match=match):
        validate_inputs(r, b)


def test_28_unreviewed_nominal_profile_not_actual():
    r = references()
    for x in r["references"]:
        x["nominal_length_m"] = 3500
        x["actual_length_m"] = None
    result = length_scale(r)
    assert result["scale_m_per_reconstruction_unit"] is None
    assert all(x.get("candidate_scale") is None for x in result["references"])


def test_29_empty_or_missing_finite_surface_rejected():
    b = features()
    b["planes"][0]["finite_vertices_xyz_m"] = [[0, 0, 0]]
    with pytest.raises(ValueError, match="finite polygons"):
        validate_inputs(None, b)


def test_30_inconsistent_plane_normal_point_rejected():
    b = features()
    b["planes"][0]["finite_vertices_xyz_m"][0][2] += 0.1
    with pytest.raises(ValueError, match="polygon and its plane"):
        validate_inputs(None, b)


def test_31_zero_normal_rejected_without_nan():
    b = features()
    b["planes"][0]["normal_xyz"] = [0, 0, 0]
    with pytest.raises(ValueError, match="Zero plane normal"):
        validate_inputs(None, b)


def test_32_no_bim_reference_does_not_create_registration(tmp_path):
    r = save(tmp_path / "lengths.json", references())
    result = run(tmp_path, "metric-bim-missing-bim-033", references_json=r)
    path = tmp_path / result["outputs"]["alignment_transform.json"]["path"]
    a = json.loads(path.read_text())
    assert a["matrix4x4"] is None
    assert a["can_feed_stage8"] is False


def test_33_no_bim_and_no_lengths_rejected_before_artifacts(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        run(tmp_path, "metric-bim-no-inputs-034")
    assert not (tmp_path / "runs").exists()


def test_34_missing_real_ifc_asset_blocks_model_claim(tmp_path):
    bim = features()
    bim["model_provenance"] = {"kind": "IFC", "revision": "synthetic-revision",
                               "asset": {"path": "nonexistent.ifc", "sha256": "0" * 64}}
    p = save(tmp_path / "bim.json", bim)
    with pytest.raises(ValueError, match="BIM source asset"):
        run(tmp_path, "metric-bim-bad-ifc-035", bim_json=p)


def test_35_robust_umeyama_rejects_bad_fit_correspondence_not_heldout():
    b = features()
    b["matches"][4]["role"] = "fit"
    b["matches"].append({
        "match_id": "independent_holdout", "physical_feature_id": "holdout-unique",
        "kind": "corner", "role": "control",
        "scan_xyz": [0.7, 0.3, 0.8], "bim_xyz_m": [2.4, 2.6, 4.6],
        "observation_state": "observed_supported", "approval": approval(),
    })
    b["matches"][0]["bim_xyz_m"] = [30, 20, 10]
    b["registration"] = {"method": "ransac_umeyama", "inlier_threshold_m": .001,
                          "max_trials": 100, "seed": 42}
    a, tx = build_transform(b, length_scale(None))
    assert tx is not None
    assert a["scale"] == pytest.approx(2.0, abs=1e-10)
    assert a["fit_outlier_match_ids"] == ["f0"]
    assert a["fit_inlier_match_ids"] == ["f1", "f2", "f3", "f4"]
    assert a["heldout_match_ids"] == ["independent_holdout"]
    assert a["can_feed_stage8"] is False


def test_36_ransac_requires_operator_threshold():
    b = features()
    b["registration"] = {"method": "ransac_umeyama"}
    a, tx = build_transform(b, length_scale(None))
    assert tx is None
    assert a["status"] == "blocked_robust_registration"
    assert "inlier_threshold_m" in a["reason"]


def test_37_ransac_degeneracy_and_mismatched_inliers_cannot_be_certified():
    src = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
    dst = src * 2
    with pytest.raises(ValueError, match="No robust similarity consensus"):
        robust_similarity(src, dst, fixed_scale=None, threshold_m=.01)


def test_38_shape4_handoff_is_read_only_candidate_with_verified_links(tmp_path):
    b = save(tmp_path / "bim.json", features())
    r = save(tmp_path / "references.json", references())
    result = run(tmp_path, "metric-bim-shape4-039", bim_json=b, references_json=r)
    base = tmp_path / "runs/metric-bim-shape4-039/metric_bim"
    viewer = json.loads((base / "viewer_handoff.json").read_text())
    assert viewer["schema_version"] == "mycon.metric_bim_viewer_handoff.v1"
    assert viewer["metric_coloring_allowed"] is False
    assert viewer["metric_validation"] == "pending"
    assert all(asset["path"].startswith("runs/") and len(asset["sha256"]) == 64
               for asset in viewer["viewer_assets"].values())
    heat = json.loads((base / "deviation_heatmap.json").read_text())
    assert heat["valid_sample_count"] == 1
    assert sum(x["valid_mask"] for x in heat["points"]) == 1
    assert all(x["signed_distance_m"] is None for x in heat["points"] if not x["valid_mask"])
    assert result["can_feed_stage8"] is False


def test_39_plain_umeyama_bad_correspondence_emits_fit_error_not_certification():
    b = features()
    b["matches"][0]["bim_xyz_m"] = [100, 100, 100]
    a, tx = build_transform(b, length_scale(None))
    assert tx is not None
    assert a["fit_residuals"]["max_error_m"] > 1
    assert a["metric_validation"] == "pending"
    assert a["registration_verified"] is False
