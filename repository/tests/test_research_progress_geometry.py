"""Isolated scientific tests; no Open3D/GPU needed."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.stage_09_progress.point_assignment import (
    assign_exclusive_points, nearest_element_distances,
)
from pipeline.stage_09_progress.evidence_metrics import element_scientific_fields


pytestmark = pytest.mark.lightweight


def _box(a, b):
    return {"bounds_min": a, "bounds_max": b}


def test_overlap_cannot_double_count_scan_points():
    pts = np.asarray([[0.5, 0, 0], [1.5, 0, 0], [2.5, 0, 0], [4, 0, 0]], float)
    owner, candidates = assign_exclusive_points(
        pts, [_box([0,-1,-1], [2,1,1]), _box([1,-1,-1], [3,1,1])]
    )
    assert owner.tolist() == [0, -2, 1, -1]
    assert candidates.tolist() == [1, 2, 1, 0]


def test_identical_boxes_are_not_independent_element_evidence():
    owner, counts = assign_exclusive_points(
        np.zeros((8, 3)), [_box([-1,-1,-1], [1,1,1])] * 2
    )
    assert np.all(owner == -2)
    assert np.all(counts == 2)


def test_invalid_boxes_cannot_claim_or_steal_points():
    owner, _ = assign_exclusive_points(
        np.zeros((1, 3)), [_box([1,1,1], [-1,-1,-1]), _box([-1,-1,-1], [1,1,1])]
    )
    assert owner.tolist() == [1]


def test_tiny_ifc_geometry_and_zero_scan_remain_undefined():
    assert nearest_element_distances(np.zeros((1,3)), np.empty((0,3))).size == 0
    assert nearest_element_distances(np.empty((0,3)), np.zeros((1,3))).size == 0


def test_nearest_distance_uses_element_owned_reference_not_other_element():
    targets = np.asarray([[1.0, 0, 0]], float)
    disjoint_scan_for_this_element = np.asarray([[3.0, 0, 0]], float)
    distances = nearest_element_distances(targets, disjoint_scan_for_this_element)
    assert distances.tolist() == pytest.approx([2.0])


def test_unobserved_not_synonym_for_not_built():
    fields = element_scientific_fields({
        "coverage": "0.0", "registration_confidence": "high",
        "assigned_scan_point_count": 0,
        "observability_ratio": 0, "observability_source": "raycast_zbuffer",
        "geometry_adapter_valid": True, "geometry_evidence_state": "occluded",
    })
    assert fields["geometry_evidence_state"] == "occluded"
    assert fields["physical_progress_estimate"] is None
    assert fields["observability_ratio"] == 0.0


def test_coverage_cannot_fabricate_observability_or_evidence_confidence():
    fields = element_scientific_fields({
        "coverage": 1, "point_count_evaluated": 9999,
        "confidence": 0.99, "registration_confidence": "high",
    })
    assert fields["observability_ratio"] is None
    assert fields["evidence_confidence"] is None
    assert fields["physical_progress_estimate"] is None


def test_registration_failure_and_overlap_separate_from_geom_nonconformance():
    common = {
        "assigned_scan_point_count": 0, "ambiguous_scan_point_count": 7,
        "observability_ratio": 1, "observability_source": "raycast_zbuffer",
    }
    assert element_scientific_fields({**common, "registration_confidence":"low"})["geometry_evidence_state"] == "uncertain_registration"
    assert element_scientific_fields({**common, "registration_confidence":"high"})["geometry_evidence_state"] == "unknown_geometry_identity"
    # Coverage alone is not a QA-certified geometric nonconformance.
    geom = {**common, "assigned_scan_point_count": 12, "ambiguous_scan_point_count": 0,
            "coverage":0.40, "in_tolerance_ratio":0.40, "registration_confidence":"high"}
    assert element_scientific_fields(geom)["geometry_evidence_state"] == "unknown_geometry_identity"
    inspected = {**geom, "geometry_adapter_valid": True, "geometry_evidence_state": "geometric_nonconformance"}
    assert element_scientific_fields(inspected)["geometry_evidence_state"] == "geometric_nonconformance"


def test_invalid_quantities_or_unsigned_claims_are_not_progress():
    base = {
        "planned_quantity": 10, "executed_quantity": 9,
        "quantity_unit": "m2", "quantity_source": "signed_site_log",
    }
    assert element_scientific_fields(base)["physical_progress_estimate"] is None
    assert element_scientific_fields({**base, "quantity_verified": True, "executed_quantity": 11})["physical_progress_estimate"] is None
    assert element_scientific_fields({**base, "quantity_verified": True})["physical_progress_estimate"] == pytest.approx(0.9)
    assert element_scientific_fields({**base, "quantity_verified": True, "executed_quantity": 0})["physical_progress_estimate"] == 0


def test_confidence_requires_separate_calibration():
    row = {
        "registration_confidence": "high", "evidence_confidence": 0.8,
        "evidence_confidence_source": "heldout_calibration",
        "observability_ratio": 0.3, "observability_source": "manual_occlusion_audit",
        "geometry_adapter_valid": True, "evidence_calibration_ref": "SYNTH-HELDOUT-CALIBRATION",
    }
    fields = element_scientific_fields(row)
    assert fields["evidence_confidence"] == 0.8
    assert fields["observability_ratio"] == 0.3
    assert element_scientific_fields({**row, "registration_confidence":"low"})["evidence_confidence"] is None
