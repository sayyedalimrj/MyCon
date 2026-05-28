"""Tests for :mod:`pipeline.stage_10_copilot.grounding_guard`.

These tests pin the contract that the VLM grounding guard provides:

- claim extraction is deterministic and covers numeric / named-entity /
  acceptance kinds;
- numeric verification respects the per-quantity tolerance schedule;
- named-entity verification is conservative (substring + reverse
  substring matches);
- acceptance/rejection verification cross-checks the evidence package's
  ``confidence_flags`` and any deterministic per-element ``status``;
- the top-level ``ground_answer`` returns a stable, JSON-shaped
  :class:`GroundingResult`;
- ``attach_grounding_guard`` slots into the existing Stage 10 response
  augmentation chain without overwriting prior validation results.
"""

from __future__ import annotations

import json

import pytest

from pipeline.stage_10_copilot.grounding_guard import (
    DEFAULT_TOLERANCES,
    GROUNDING_SCHEMA_VERSION,
    Claim,
    ClaimKind,
    GroundingResult,
    attach_grounding_guard,
    extract_claims,
    ground_answer,
    verify_claim,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def test_extract_numeric_claim_with_unit_and_quantity() -> None:
    claims = extract_claims("The wall offset is 3.2 cm.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    assert numeric[0].quantity == "offset"
    # 3.2 cm -> 0.032 m canonical
    assert numeric[0].value == pytest.approx(0.032, abs=1e-9)
    assert numeric[0].unit == "m"


def test_extract_numeric_claim_with_percent_unit() -> None:
    claims = extract_claims("Activity completion of 75%.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    # 75% -> 0.75 fraction
    assert numeric[0].value == pytest.approx(0.75, abs=1e-9)
    assert numeric[0].unit == "fraction"


def test_extract_numeric_claim_without_named_quantity() -> None:
    claims = extract_claims("It is 12 mm.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    # 12 mm -> 0.012 m
    assert numeric[0].value == pytest.approx(0.012, abs=1e-9)
    assert numeric[0].quantity is None


def test_extract_global_id_entity() -> None:
    # IFC GlobalIds are exactly 22 base64-like characters.
    answer = "Element 2N3RfMfeDD$AbcDefghijk is acceptable."
    claims = extract_claims(answer)
    entities = [c for c in claims if c.entity_kind == "global_id"]
    assert len(entities) == 1
    assert entities[0].entity_value == "2N3RfMfeDD$AbcDefghijk"


def test_extract_ifc_class_entity() -> None:
    claims = extract_claims("This IfcWall component is fine.")
    ifc = [c for c in claims if c.entity_kind == "ifc_class"]
    assert len(ifc) == 1
    assert ifc[0].entity_value == "IfcWall"


def test_extract_activity_id_entity_via_pattern() -> None:
    claims = extract_claims("Activity A0432 is on schedule.")
    a = [c for c in claims if c.entity_kind == "activity_id"]
    assert len(a) == 1
    assert a[0].entity_value == "A0432"


def test_extract_activity_id_entity_via_known_list() -> None:
    """Caller-supplied known IDs widen the activity-ID match."""
    claims = extract_claims(
        "Activity FOUND-007 is complete.",
        known_activity_ids=["FOUND-007", "FOUND-008"],
    )
    a = [c for c in claims if c.entity_kind == "activity_id"]
    assert any(c.entity_value == "FOUND-007" for c in a)


def test_extract_acceptance_and_rejection() -> None:
    claims = extract_claims("We accept this wall but reject that column.")
    polarities = {c.acceptance_polarity for c in claims if c.kind == ClaimKind.ACCEPTANCE}
    assert polarities == {"accept", "reject"}


def test_extract_handles_empty_answer_gracefully() -> None:
    assert extract_claims("") == []


def test_extract_deduplicates_repeated_global_ids() -> None:
    answer = "2N3RfMfeDD$AbcDefghijk is fine. Again, 2N3RfMfeDD$AbcDefghijk is fine."
    entities = [c for c in extract_claims(answer) if c.entity_kind == "global_id"]
    assert len(entities) == 1


# ---------------------------------------------------------------------------
# Numeric claim verification
# ---------------------------------------------------------------------------


def _evidence(**fields) -> dict:
    """Build a Stage-10-shaped evidence package for the tests."""
    base = {
        "metrics": {},
        "confidence_flags": [],
        "evidence_used": ["runs/r1/reports/element_progress.json"],
    }
    base.update(fields)
    return base


def test_numeric_match_within_tolerance() -> None:
    """Claim of 3.0 cm offset matches an evidence offset of 3.1 cm
    because absolute tolerance is 5 mm."""
    evidence = _evidence(metrics={"offset": 0.031})  # 3.1 cm in metres
    claim = Claim(kind=ClaimKind.NUMERIC, text="3.0 cm", value=0.030, unit="m", quantity="offset")
    v = verify_claim(claim, evidence)
    assert v.status == "matched"
    assert v.evidence_path == "metrics.offset"


def test_numeric_unsupported_when_out_of_tolerance() -> None:
    """A claim of 50 cm offset is *not* matched by an evidence value of
    3 cm — distance 0.47 m vs tol max(0.10*0.03, 0.005) = 5 mm."""
    evidence = _evidence(metrics={"offset": 0.030})
    claim = Claim(kind=ClaimKind.NUMERIC, text="50 cm", value=0.50, unit="m", quantity="offset")
    v = verify_claim(claim, evidence)
    assert v.status == "unsupported"
    # Closest match is reported with its distance.
    assert any("closest_match_distance" in n for n in v.notes)


def test_numeric_unverifiable_when_quantity_field_absent() -> None:
    """Claim of '5% coverage' against evidence with no coverage field
    is unverifiable, not unsupported (Pelican-style: don't over-flag)."""
    evidence = _evidence(metrics={"offset": 0.030})
    claim = Claim(kind=ClaimKind.NUMERIC, text="5%", value=0.05, unit="fraction", quantity="coverage")
    v = verify_claim(claim, evidence)
    assert v.status == "unverifiable"


def test_numeric_no_quantity_falls_back_to_proximity_match() -> None:
    """Claim with no named quantity can still match any near-equal numeric
    leaf (lenient mode for free-form VLM answers)."""
    evidence = _evidence(metrics={"f_score": 0.81})
    claim = Claim(kind=ClaimKind.NUMERIC, text="0.80", value=0.80, unit="fraction", quantity=None)
    v = verify_claim(claim, evidence)
    assert v.status == "matched"


def test_per_quantity_tolerance_schedule_used() -> None:
    """Default tolerance for ``completion`` is ±2 % absolute. A claim of
    75% completion is matched by an evidence value of 76% (diff = 0.01)
    but not 80% (diff = 0.05)."""
    ev_close = _evidence(metrics={"completion": 0.76})
    ev_far = _evidence(metrics={"completion": 0.80})
    claim = Claim(kind=ClaimKind.NUMERIC, text="75%", value=0.75, unit="fraction", quantity="completion")
    assert verify_claim(claim, ev_close).status == "matched"
    assert verify_claim(claim, ev_far).status == "unsupported"


# ---------------------------------------------------------------------------
# Named-entity verification
# ---------------------------------------------------------------------------


def test_named_entity_matched_when_string_appears_in_evidence() -> None:
    evidence = _evidence(selected_context={"element_global_id": "2N3RfMfeDD$AbcDefghijk"})
    claim = Claim(
        kind=ClaimKind.NAMED_ENTITY,
        text="2N3RfMfeDD$AbcDefghijk",
        entity_kind="global_id",
        entity_value="2N3RfMfeDD$AbcDefghijk",
    )
    v = verify_claim(claim, evidence)
    assert v.status == "matched"


def test_named_entity_unsupported_when_string_absent() -> None:
    evidence = _evidence(selected_context={"element_global_id": "differentXXXXXXXXXXXXXX"})
    claim = Claim(
        kind=ClaimKind.NAMED_ENTITY,
        text="2N3RfMfeDD$AbcDefghijk",
        entity_kind="global_id",
        entity_value="2N3RfMfeDD$AbcDefghijk",
    )
    v = verify_claim(claim, evidence)
    assert v.status == "unsupported"


# ---------------------------------------------------------------------------
# Acceptance / rejection verification
# ---------------------------------------------------------------------------


def test_acceptance_blocked_when_evidence_carries_risk_flags() -> None:
    evidence = _evidence(confidence_flags=["registration_low", "metric_artifacts_missing_or_incomplete"])
    claim = Claim(kind=ClaimKind.ACCEPTANCE, text="<accept>", acceptance_polarity="accept")
    v = verify_claim(claim, evidence)
    assert v.status == "unsupported"


def test_acceptance_passes_when_no_risk_flags() -> None:
    evidence = _evidence(confidence_flags=["evidence_package_complete_enough_for_mock_answer"])
    claim = Claim(kind=ClaimKind.ACCEPTANCE, text="<accept>", acceptance_polarity="accept")
    v = verify_claim(claim, evidence)
    assert v.status == "matched"


def test_rejection_always_marked_matched() -> None:
    """Rejection claims are conservative and never flagged unsupported."""
    evidence = _evidence(confidence_flags=[])
    claim = Claim(kind=ClaimKind.ACCEPTANCE, text="<reject>", acceptance_polarity="reject")
    v = verify_claim(claim, evidence)
    assert v.status == "matched"


# ---------------------------------------------------------------------------
# ground_answer — top-level pipeline
# ---------------------------------------------------------------------------


def test_ground_answer_passes_for_well_supported_answer() -> None:
    answer = "Element 2N3RfMfeDD$AbcDefghijk has 76% completion. Accept."
    evidence = {
        "metrics": {"completion": 0.76},
        "confidence_flags": ["evidence_package_complete_enough_for_mock_answer"],
        "selected_context": {"element_global_id": "2N3RfMfeDD$AbcDefghijk"},
    }
    res = ground_answer(answer, evidence)
    assert res.passed is True
    assert res.n_unsupported == 0
    assert res.n_matched >= 2  # numeric + named entity matches


def test_ground_answer_fails_for_hallucinated_numeric_claim() -> None:
    answer = "The offset is 50 cm so we approve."
    evidence = {
        "metrics": {"offset": 0.030},
        "confidence_flags": [],
    }
    res = ground_answer(answer, evidence)
    assert res.passed is False
    assert res.n_unsupported >= 1
    assert any(t.startswith("vlm_unsupported_claims") for t in res.risk_tokens)


def test_ground_answer_returns_stable_schema_version() -> None:
    res = ground_answer("anything", {})
    assert res.schema_version == GROUNDING_SCHEMA_VERSION


def test_ground_answer_passes_when_no_extractable_claims_but_under_threshold() -> None:
    res = ground_answer("Hello there.", {})
    # No claims means n_unsupported = 0, so passed=True under the default
    # threshold of 0; risk_tokens still surfaces the no-claims diagnostic.
    assert res.passed is True
    assert "vlm_answer_carries_no_extractable_claims" in res.risk_tokens


def test_ground_answer_threshold_can_be_lenient() -> None:
    answer = "The offset is 99 cm."
    evidence = {"metrics": {"offset": 0.030}}
    strict = ground_answer(answer, evidence, pass_threshold_unsupported=0)
    lenient = ground_answer(answer, evidence, pass_threshold_unsupported=2)
    assert strict.passed is False
    assert lenient.passed is True


def test_ground_answer_result_is_json_round_trippable() -> None:
    res = ground_answer("Activity A0432 has 75% completion.", {"metrics": {"completion": 0.75}})
    s = json.dumps(res.to_dict())
    parsed = json.loads(s)
    assert parsed["schema_version"] == GROUNDING_SCHEMA_VERSION
    assert "per_claim" in parsed


# ---------------------------------------------------------------------------
# attach_grounding_guard
# ---------------------------------------------------------------------------


def test_attach_grounding_guard_does_not_overwrite_existing() -> None:
    response = {
        "answer": "any",
        "grounding_guard": {"passed": True, "from": "previous"},
    }
    out = attach_grounding_guard(response)
    assert out["grounding_guard"] == {"passed": True, "from": "previous"}


def test_attach_grounding_guard_forces_low_confidence_on_failure() -> None:
    response = {
        "answer": "The offset is 99 cm so we approve.",
        "evidence_package": {
            "metrics": {"offset": 0.030},
            "confidence_flags": [],
        },
        "confidence": "high",
        "risks_or_uncertainty": [],
    }
    out = attach_grounding_guard(response)
    assert out["confidence"] == "low"
    assert any(r.startswith("vlm_unsupported_claims") for r in out["risks_or_uncertainty"])
    assert out["grounding_guard"]["passed"] is False


def test_attach_grounding_guard_preserves_high_confidence_on_pass() -> None:
    response = {
        "answer": "Element 2N3RfMfeDD$AbcDefghijk has 76% completion.",
        "evidence_package": {
            "metrics": {"completion": 0.76},
            "selected_context": {"element_global_id": "2N3RfMfeDD$AbcDefghijk"},
        },
        "confidence": "high",
        "risks_or_uncertainty": [],
    }
    out = attach_grounding_guard(response)
    assert out["confidence"] == "high"
    assert out["grounding_guard"]["passed"] is True


def test_attach_grounding_guard_handles_non_dict_response() -> None:
    """Invalid response types should be returned unchanged."""
    assert attach_grounding_guard("not a dict") == "not a dict"  # type: ignore[arg-type]
    assert attach_grounding_guard(None) is None  # type: ignore[arg-type]


def test_attach_grounding_guard_falls_back_to_response_fields_when_no_package() -> None:
    """If no full evidence_package is present, the guard should use
    ``confidence_flags`` and ``evidence_used`` from the response itself."""
    response = {
        "answer": "Accept.",
        "confidence_flags": ["registration_low"],  # would block acceptance
        "evidence_used": ["runs/r1/element_progress.json"],
    }
    out = attach_grounding_guard(response)
    assert out["grounding_guard"]["passed"] is False



# ---------------------------------------------------------------------------
# Phase 5: extra unit conversions
# ---------------------------------------------------------------------------


def test_extract_numeric_claim_with_inches() -> None:
    """Imperial inches should canonicalise to metres (1 in = 0.0254 m)."""
    claims = extract_claims("Wall thickness is 8 inches.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    assert numeric[0].quantity == "thickness"
    assert numeric[0].unit == "m"
    assert numeric[0].value == pytest.approx(8 * 0.0254, abs=1e-9)


def test_extract_numeric_claim_with_inch_singular() -> None:
    claims = extract_claims("Spacing of 1 inch.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    assert numeric[0].value == pytest.approx(0.0254, abs=1e-9)


def test_extract_numeric_claim_with_in_short_form() -> None:
    claims = extract_claims("12 in offset.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    # 'in' must match because it's followed by a space (not a letter),
    # and our negative lookahead is on the letter class.
    assert len(numeric) == 1
    assert numeric[0].value == pytest.approx(12 * 0.0254, abs=1e-9)


def test_extract_numeric_claim_with_feet() -> None:
    claims = extract_claims("Clearance height of 9 ft.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    assert numeric[0].unit == "m"
    assert numeric[0].value == pytest.approx(9 * 0.3048, abs=1e-9)


def test_extract_numeric_claim_with_radians() -> None:
    """Radians should canonicalise to degrees."""
    claims = extract_claims("Skew angle of 1.5 rad.")
    numeric = [c for c in claims if c.kind == ClaimKind.NUMERIC]
    assert len(numeric) == 1
    assert numeric[0].quantity == "angle"
    assert numeric[0].unit == "deg"
    assert numeric[0].value == pytest.approx(1.5 * 57.295779513082323, abs=1e-6)


def test_unit_pattern_does_not_match_letter_m_inside_word() -> None:
    """'meeting' / 'minimum' / 'morning' must NOT trip the 'm' unit
    matcher; the negative lookahead (?![A-Za-z]) is what guards this."""
    assert extract_claims("There is a meeting in the morning.") == []
    assert extract_claims("The minimum is required.") == []
    # And as a sanity check the standalone 'm' still works:
    inflate = extract_claims("Distance is 4 m.")
    assert any(c.unit == "m" and c.value == 4.0 for c in inflate)


def test_imperial_inches_match_evidence_in_metres_within_tolerance() -> None:
    """End-to-end: a claim of '8 inches thickness' (= 0.2032 m) should
    match an evidence package value of 0.20 m for 'thickness' under the
    default tolerance (rel 10 % or abs 5 mm, whichever is larger)."""
    answer = "Wall thickness is 8 inches."
    evidence = {"metrics": {"thickness": 0.20}}
    result = ground_answer(answer, evidence)
    assert result.passed is True
    assert result.n_unsupported == 0


def test_imperial_inches_unsupported_when_evidence_disagrees() -> None:
    """8 inches = 0.2032 m must NOT match an evidence value of 0.05 m."""
    answer = "Wall thickness is 8 inches."
    evidence = {"metrics": {"thickness": 0.05}}
    result = ground_answer(answer, evidence)
    assert result.passed is False
    assert result.n_unsupported >= 1


# ---------------------------------------------------------------------------
# Phase 5: plug-in claim extractor (Protocol)
# ---------------------------------------------------------------------------


def test_default_claim_extractor_is_a_regex_extractor() -> None:
    from pipeline.stage_10_copilot.grounding_guard import (
        DEFAULT_CLAIM_EXTRACTOR,
        RegexClaimExtractor,
    )

    assert isinstance(DEFAULT_CLAIM_EXTRACTOR, RegexClaimExtractor)


def test_regex_claim_extractor_matches_extract_claims_function() -> None:
    """The default extractor must be byte-for-byte equivalent to
    :func:`extract_claims` so callers swapping in a custom extractor
    can do A/B comparisons against the regex baseline."""
    from pipeline.stage_10_copilot.grounding_guard import RegexClaimExtractor

    answer = "Wall offset 3.2 cm. Activity A0432 has 75% completion."
    a = extract_claims(answer)
    b = RegexClaimExtractor().extract(answer)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_claim_extractor_protocol_runtime_checkable() -> None:
    """ClaimExtractor must be a runtime-checkable Protocol so consumers
    can ``isinstance(x, ClaimExtractor)`` without subclassing it."""
    from pipeline.stage_10_copilot.grounding_guard import (
        ClaimExtractor,
        DEFAULT_CLAIM_EXTRACTOR,
        RegexClaimExtractor,
    )

    assert isinstance(DEFAULT_CLAIM_EXTRACTOR, ClaimExtractor)
    assert isinstance(RegexClaimExtractor(), ClaimExtractor)

    class _NotAnExtractor:
        pass

    assert not isinstance(_NotAnExtractor(), ClaimExtractor)


def test_ground_answer_uses_custom_extractor_when_supplied() -> None:
    """A custom extractor that returns a single named-entity claim must
    be honoured by ground_answer; the regex output is bypassed entirely."""
    from pipeline.stage_10_copilot.grounding_guard import Claim, ClaimKind

    class _SyntheticExtractor:
        def extract(self, answer, *, known_activity_ids=()):
            # Return a single named-entity claim regardless of the answer
            # text -- so we can prove the verifier consumed *our* output.
            return [
                Claim(
                    kind=ClaimKind.NAMED_ENTITY,
                    text="SyntheticElement",
                    entity_kind="ifc_class",
                    entity_value="SyntheticElement",
                )
            ]

    answer = "irrelevant text with no IFC classes"
    evidence = {"selected_context": {"element_global_id": "SyntheticElement"}}
    result = ground_answer(answer, evidence, claim_extractor=_SyntheticExtractor())
    assert result.n_claims == 1
    assert result.per_claim[0].claim.entity_value == "SyntheticElement"
    assert result.per_claim[0].status == "matched"


def test_attach_grounding_guard_threads_custom_extractor() -> None:
    """attach_grounding_guard forwards claim_extractor to ground_answer."""
    from pipeline.stage_10_copilot.grounding_guard import Claim, ClaimKind

    class _ZeroClaimExtractor:
        def extract(self, answer, *, known_activity_ids=()):
            return []

    response = {"answer": "literally anything", "evidence_package": {}}
    out = attach_grounding_guard(response, claim_extractor=_ZeroClaimExtractor())
    # Zero claims -> guard passes (no unsupported), risk_tokens contains
    # the no-claims diagnostic.
    assert out["grounding_guard"]["passed"] is True
    assert out["grounding_guard"]["n_claims"] == 0
    assert "vlm_answer_carries_no_extractable_claims" in out["grounding_guard"]["risk_tokens"]


def test_default_claim_extractor_is_module_level_singleton() -> None:
    """``DEFAULT_CLAIM_EXTRACTOR`` is exported in __all__ so plug-in
    authors can subclass / wrap it."""
    from pipeline.stage_10_copilot import grounding_guard

    assert "DEFAULT_CLAIM_EXTRACTOR" in grounding_guard.__all__
    assert "ClaimExtractor" in grounding_guard.__all__
    assert "RegexClaimExtractor" in grounding_guard.__all__
