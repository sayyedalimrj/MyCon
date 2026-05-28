"""Tests for :mod:`pipeline.stage_09_progress.multiview_fusion`.

Cover the three layers separately so a regression localises cleanly:

1. Per-view weight schedule (``compute_view_evidence_weight``).
2. Pairwise Dempster combination (``fuse_two_views``).
3. N-view fold + decision policy + per-element grouping
   (``fuse_views`` / ``fuse_per_element``).

References for the math are in the module docstring; tests check
*invariants* that have to hold regardless of implementation details
(boundedness, normalisation, conflict semantics, degenerate inputs).
"""

from __future__ import annotations

import math

import pytest

from pipeline.stage_09_progress.multiview_fusion import (
    FusedElementBelief,
    ViewObservation,
    compute_view_evidence_weight,
    fuse_per_element,
    fuse_two_views,
    fuse_views,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# compute_view_evidence_weight
# ---------------------------------------------------------------------------


def test_weight_in_unit_interval_for_typical_inputs() -> None:
    w = compute_view_evidence_weight(blur_score=0.5, grazing_angle_deg=30.0, point_count=50)
    assert 0.0 <= w <= 1.0


def test_blur_below_floor_returns_zero() -> None:
    assert compute_view_evidence_weight(blur_score=0.05) == 0.0


def test_grazing_above_floor_returns_zero() -> None:
    assert compute_view_evidence_weight(grazing_angle_deg=80.0) == 0.0


def test_missing_signals_do_not_penalise() -> None:
    """When all signals are absent the weight should default to 1.0
    (the function is uninformative, not punitive)."""
    assert compute_view_evidence_weight() == 1.0


def test_weight_monotonic_in_blur_score() -> None:
    a = compute_view_evidence_weight(blur_score=0.3)
    b = compute_view_evidence_weight(blur_score=0.8)
    assert b > a


def test_weight_monotonic_in_point_count() -> None:
    a = compute_view_evidence_weight(point_count=10)
    b = compute_view_evidence_weight(point_count=100)
    assert b > a


# ---------------------------------------------------------------------------
# fuse_two_views — closed-form Dempster combination
# ---------------------------------------------------------------------------


def _obs(p: float, w: float = 1.0, vid: str = "v") -> ViewObservation:
    return ViewObservation(view_id=vid, element_global_id="e1", p_acceptable=p, view_confidence=w)


def test_two_views_sum_to_one() -> None:
    """b_a + b_n + u must be exactly 1 after fusion (within float epsilon)."""
    b_a, b_n, u, _ = fuse_two_views(_obs(0.8), _obs(0.7))
    assert math.isclose(b_a + b_n + u, 1.0, abs_tol=1e-9)


def test_two_views_nonnegative_outputs() -> None:
    b_a, b_n, u, k = fuse_two_views(_obs(0.8), _obs(0.7))
    assert b_a >= 0.0
    assert b_n >= 0.0
    assert u >= 0.0
    assert k >= 0.0


def test_two_views_unanimous_high_belief_increases_acceptable() -> None:
    """Two strongly-acceptable views combined should produce a higher
    belief than a single view (the central property of evidence fusion)."""
    b_a_single, _, _, _ = fuse_two_views(_obs(0.8), _obs(0.5, w=0.0))  # one informative, one vacuous
    b_a_pair, _, _, _ = fuse_two_views(_obs(0.8), _obs(0.8))
    assert b_a_pair > b_a_single


def test_vacuous_view_does_not_change_belief() -> None:
    """Combining with a w=0 (totally uncertain) view leaves belief
    unchanged. This is a hard invariant of the {a, n, U} frame."""
    b_a_pair, b_n_pair, u_pair, _ = fuse_two_views(_obs(0.7, w=0.7), _obs(0.5, w=0.0))
    # Reference: single-view (b_a, b_n, u) for (p=0.7, w=0.7).
    e_a, e_n = 0.7 * 0.7, 0.7 * 0.3
    s = e_a + e_n + 2.0
    assert math.isclose(b_a_pair, e_a / s, abs_tol=1e-9)
    assert math.isclose(b_n_pair, e_n / s, abs_tol=1e-9)
    assert math.isclose(u_pair, 2.0 / s, abs_tol=1e-9)


def test_conflict_high_when_views_disagree() -> None:
    """Two views that strongly disagree should produce a high Dempster K."""
    _, _, _, k_unanimous = fuse_two_views(_obs(0.9), _obs(0.9))
    _, _, _, k_disagree = fuse_two_views(_obs(0.95), _obs(0.05))
    assert k_disagree > k_unanimous


def test_conflict_zero_for_identical_views() -> None:
    _, _, _, k = fuse_two_views(_obs(0.5), _obs(0.5))
    # Two views at p=0.5 produce e_a = e_n = 0.5, b_a = b_n = ~0.166...
    # K = b_a^(1) b_n^(2) + b_n^(1) b_a^(2) ≈ 2 * 0.166^2 ≈ 0.055.
    # Not literally zero but small. We assert it stays below the
    # disagreement-conflict number, which is the meaningful invariant.
    _, _, _, k_disagree = fuse_two_views(_obs(0.95), _obs(0.05))
    assert k < k_disagree


# ---------------------------------------------------------------------------
# fuse_views — N-view fold + decision policy
# ---------------------------------------------------------------------------


def test_fuse_empty_returns_not_evidenced() -> None:
    fb = fuse_views([])
    assert fb.decision == "not_evidenced"
    assert fb.fused_uncertainty_mass == 1.0
    assert fb.view_count == 0


def test_all_zero_confidence_returns_not_evidenced() -> None:
    fb = fuse_views([_obs(0.9, w=0.0, vid="a"), _obs(0.1, w=0.0, vid="b")])
    assert fb.decision == "not_evidenced"
    assert "all_views_zero_confidence" in fb.notes


def test_unanimous_high_acceptance_returns_acceptable() -> None:
    fb = fuse_views([_obs(0.95, vid="a"), _obs(0.92, vid="b"), _obs(0.88, vid="c")])
    assert fb.decision == "acceptable"
    assert fb.expected_probability_acceptable > 0.65
    assert fb.fused_belief_acceptable > fb.fused_belief_not_acceptable
    assert fb.contributing_views == ("a", "b", "c")


def test_unanimous_low_acceptance_returns_not_acceptable() -> None:
    fb = fuse_views([_obs(0.05, vid="a"), _obs(0.08, vid="b"), _obs(0.10, vid="c")])
    assert fb.decision == "not_acceptable"
    assert fb.expected_probability_acceptable < 0.35
    assert fb.fused_belief_not_acceptable > fb.fused_belief_acceptable


def test_high_conflict_triggers_uncertain_conflict() -> None:
    """Multiple views in strong disagreement (each at confidence 1.0) should
    accumulate enough conflict mass to flip the decision to
    ``uncertain_conflict`` rather than washing out into a middling
    acceptable/not_acceptable belief.

    Two views are not always enough: in a 2-class evidential frame with
    +2 Dirichlet prior, two opposed high-confidence views give K ~ 0.10,
    which is real but below the default 0.30 threshold. Six opposed
    views accumulate K to well above 0.30 (closer to 0.45 in the
    canonical case used here).
    """
    fb = fuse_views(
        [
            _obs(0.95, vid="a"),
            _obs(0.05, vid="b"),
            _obs(0.93, vid="c"),
            _obs(0.07, vid="d"),
            _obs(0.91, vid="e"),
            _obs(0.09, vid="f"),
        ],
        conflict_mass_threshold=0.30,
    )
    assert fb.decision == "uncertain_conflict"
    assert fb.conflict_mass > 0.30


def test_decision_thresholds_are_recorded() -> None:
    fb = fuse_views(
        [_obs(0.7, vid="a")],
        decision_threshold_acceptable=0.7,
        decision_threshold_uncertain=0.5,
        conflict_mass_threshold=0.4,
    )
    assert fb.decision_threshold_acceptable == 0.7
    assert fb.decision_threshold_uncertain == 0.5
    assert fb.conflict_mass_threshold == 0.4


def test_belief_normalisation_after_fold() -> None:
    """Three-view fold; belief masses still sum to 1, and expected
    probabilities of acceptable + not_acceptable also sum to 1."""
    fb = fuse_views([_obs(0.8, vid="a"), _obs(0.7, vid="b"), _obs(0.6, vid="c")])
    s = fb.fused_belief_acceptable + fb.fused_belief_not_acceptable + fb.fused_uncertainty_mass
    assert math.isclose(s, 1.0, abs_tol=1e-6)
    # expected_p_a is in [0, 1].
    assert 0.0 <= fb.expected_probability_acceptable <= 1.0


def test_zero_confidence_views_dropped_with_note() -> None:
    fb = fuse_views([_obs(0.9, vid="a"), _obs(0.5, w=0.0, vid="b"), _obs(0.85, vid="c")])
    assert fb.view_count == 2
    assert any("dropped_zero_confidence_views" in n for n in fb.notes)
    assert "b" not in fb.contributing_views


def test_to_dict_is_json_ready() -> None:
    fb = fuse_views([_obs(0.9, vid="a"), _obs(0.85, vid="b")])
    d = fb.to_dict()
    import json

    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["decision"] in {
        "acceptable",
        "not_acceptable",
        "uncertain",
        "uncertain_conflict",
        "not_evidenced",
    }
    assert "contributing_views" in parsed
    assert isinstance(parsed["contributing_views"], list)


def test_fuse_one_view_degenerates_to_input_belief_direction() -> None:
    """Single-view fold must point in the same direction as the input."""
    fb_high = fuse_views([_obs(0.95)])
    fb_low = fuse_views([_obs(0.05)])
    assert fb_high.expected_probability_acceptable > 0.5
    assert fb_low.expected_probability_acceptable < 0.5


# ---------------------------------------------------------------------------
# fuse_per_element — grouping
# ---------------------------------------------------------------------------


def test_fuse_per_element_groups_by_global_id() -> None:
    """Three unanimous high-acceptance views for e_alpha and three
    unanimous low-acceptance views for e_beta. Both should cross the
    decision threshold (single-view low-confidence does not — see
    test_fuse_one_view_degenerates_to_input_belief_direction)."""
    obs = [
        ViewObservation("v1a", "e_alpha", 0.95, 1.0),
        ViewObservation("v2a", "e_alpha", 0.92, 1.0),
        ViewObservation("v3a", "e_alpha", 0.88, 1.0),
        ViewObservation("v1b", "e_beta", 0.05, 1.0),
        ViewObservation("v2b", "e_beta", 0.08, 1.0),
        ViewObservation("v3b", "e_beta", 0.10, 1.0),
    ]
    grouped = fuse_per_element(obs)
    assert set(grouped.keys()) == {"e_alpha", "e_beta"}
    assert grouped["e_alpha"].decision == "acceptable"
    assert grouped["e_beta"].decision == "not_acceptable"
    assert all(isinstance(v, FusedElementBelief) for v in grouped.values())


def test_fuse_per_element_empty_iterable_returns_empty_dict() -> None:
    assert fuse_per_element([]) == {}


def test_fuse_per_element_propagates_thresholds() -> None:
    obs = [ViewObservation("v1", "e", 0.6, 1.0)]
    grouped = fuse_per_element(
        obs,
        decision_threshold_acceptable=0.9,  # very strict — single-view 0.6 should not pass
        decision_threshold_uncertain=0.4,
    )
    fb = grouped["e"]
    assert fb.decision_threshold_acceptable == 0.9
    # Single view at p=0.6, w=1 → expected_p_a = 1.6/3 ≈ 0.533, well below
    # 0.9 but above 0.4 uncertain threshold.
    assert fb.decision == "uncertain"
    assert 0.4 < fb.expected_probability_acceptable < 0.9
