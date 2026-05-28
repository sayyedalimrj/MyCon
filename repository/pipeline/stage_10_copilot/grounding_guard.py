"""VLM evidence-grounding guardrails for the Construction Copilot.

Why this module exists
----------------------

The existing :mod:`pipeline.stage_10_copilot.answer_validator` checks
*policy-level* properties of a copilot answer (was evidence supplied?
does the answer claim acceptance even though the run carries low-quality
risks?). It does **not** check whether the **specific numeric or named
claims** inside the answer are actually supported by the deterministic
evidence package produced by the rest of the pipeline.

That gap is exactly the failure mode flagged by the recent VLM-on-
construction-site studies, including Ersoz, *Demystifying the Potential
of ChatGPT-4 Vision for Construction Progress Monitoring*
([arXiv 2412.16108](https://arxiv.org/abs/2412.16108), Dec 2024) and
Wang et al., *Can Vision-Language Models Understand Construction
Workers?* ([arXiv 2601.10835](https://arxiv.org/abs/2601.10835), Jan
2026). Both papers show that current VLMs frequently emit fluent but
unsupported quantitative or named claims when asked about construction
imagery.

This module implements a **claim-level grounding guard**. It is
inspired directly by Pelican (claim decomposition + program-of-thought
verification, [arXiv 2407.02352](https://arxiv.org/abs/2407.02352),
2024), CoRGI (verified chain-of-thought with post-hoc visual grounding,
[arXiv 2508.00378](https://arxiv.org/abs/2508.00378), 2025), and Liu &
Liang's *Multi-Modal Hallucination Control by Visual Information
Grounding* ([arXiv 2403.14003](https://arxiv.org/abs/2403.14003), 2024),
adapted to the deterministic, structured evidence package this codebase
already produces.

What it does
------------

Given a VLM answer string and the structured evidence package returned by
:mod:`pipeline.stage_10_copilot.evidence_builder`, the guard:

1. **Decomposes** the answer into atomic claims of three kinds:
   - *numeric claims*: extracted with units (mm, cm, m, %, deg) along
     with the matched value and an optional named entity ("offset",
     "completion", "angle", "deviation", ...);
   - *named-entity claims*: GlobalId / IFC class / activity ID
     references;
   - *acceptance/rejection claims*: textual "accept", "reject",
     "complete", "incomplete" assertions.
2. For each claim, **verifies** it against the evidence package:
   - numeric claims must match a numeric field in the evidence package
     within a configurable tolerance (default ±10 %% relative or
     ±0.005 absolute, whichever is larger; see :data:`DEFAULT_TOLERANCES`);
   - named-entity claims must literally appear in the package (or in
     a registered alias);
   - acceptance/rejection claims must be consistent with the evidence
     package's ``confidence_flags`` and any deterministic
     ``element_status`` field.
3. Returns a :class:`GroundingResult` summarising matched / unsupported
   claims, the per-claim evidence path, and a boolean ``passed`` flag
   plus a list of risk tokens compatible with the existing
   ``risks_or_uncertainty`` channel of the Stage 10 response.

Design notes
------------

- The guard is **purely deterministic** and dependency-free. It never
  calls a model. This is intentional: it serves as the *post-hoc
  verification* layer.
- The guard is **side-effect free**. It returns a result; the caller
  decides how to attach it. The wrapper :func:`attach_grounding_guard`
  mirrors :func:`pipeline.stage_10_copilot.api.attach_answer_validation`
  so it slots into the existing response-augmentation chain.
- The guard is **conservative**: when the evidence package is missing a
  field that a claim could be matched against, the claim is marked
  ``unverifiable`` (not ``unsupported``), so we never over-flag.
- The default tolerance schedule comes from typical construction
  measurement-survey practice: ±5 mm absolute on linear measurements,
  ±2 %% on percent-complete claims (LOD 350 dimensional tolerance,
  AIA E202).

The guard is intentionally simple: claim decomposition uses regular
expressions, not an LLM. This keeps it auditable and reproducible.
A future revision can replace the regex extractor with a trained NER
model behind the same interface; the contract remains stable.

References
----------

- Sahu et al., *Pelican: Correcting Hallucination in Vision-LLMs via
  Claim Decomposition and Program of Thought Verification*, 2024.
- *CoRGI: Verified Chain-of-Thought Reasoning with Post-hoc Visual
  Grounding*, 2025.
- Liu, Liang et al., *Multi-Modal Hallucination Control by Visual
  Information Grounding*, 2024.
- Ersoz, *Demystifying the Potential of ChatGPT-4 Vision for
  Construction Progress Monitoring*, 2024.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ClaimKind",
    "Claim",
    "ClaimVerification",
    "GroundingResult",
    "DEFAULT_TOLERANCES",
    "DEFAULT_UNIT_FACTORS",
    "extract_claims",
    "verify_claim",
    "ground_answer",
    "attach_grounding_guard",
]


class ClaimKind:
    """String constants for claim kinds. Not an Enum so it serialises trivially."""

    NUMERIC = "numeric"
    NAMED_ENTITY = "named_entity"
    ACCEPTANCE = "acceptance"


@dataclass(frozen=True)
class Claim:
    """One extracted atomic claim from the VLM answer."""

    kind: str
    text: str
    # Numeric claims:
    value: float | None = None
    unit: str | None = None
    quantity: str | None = None  # canonical name: "offset", "completion", ...
    # Named-entity claims:
    entity_kind: str | None = None  # "global_id" / "ifc_class" / "activity_id"
    entity_value: str | None = None
    # Acceptance claims:
    acceptance_polarity: str | None = None  # "accept" / "reject"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimVerification:
    """Outcome of verifying one claim against the evidence package."""

    claim: Claim
    status: str  # "matched" / "unsupported" / "unverifiable"
    evidence_path: str = ""  # dotted path into the evidence package
    expected_value: Any = None
    tolerance_used: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status,
            "evidence_path": self.evidence_path,
            "expected_value": self.expected_value,
            "tolerance_used": self.tolerance_used,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class GroundingResult:
    """Aggregate result for one (answer, evidence) pair."""

    schema_version: str
    n_claims: int
    n_matched: int
    n_unsupported: int
    n_unverifiable: int
    passed: bool
    risk_tokens: tuple[str, ...]
    per_claim: tuple[ClaimVerification, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_claims": self.n_claims,
            "n_matched": self.n_matched,
            "n_unsupported": self.n_unsupported,
            "n_unverifiable": self.n_unverifiable,
            "passed": self.passed,
            "risk_tokens": list(self.risk_tokens),
            "per_claim": [c.to_dict() for c in self.per_claim],
        }


# ---------------------------------------------------------------------------
# Defaults (operating points)
# ---------------------------------------------------------------------------


# Per-quantity tolerance schedule. ``(rel, abs)``: a numeric claim matches
# the evidence value when ``|claim - expected| <= max(rel * |expected|, abs)``.
# Defaults reflect typical construction measurement-survey practice
# (LOD 350 dimensional tolerance, AIA E202).
DEFAULT_TOLERANCES: Mapping[str, tuple[float, float]] = {
    "offset": (0.10, 0.005),         # rel ±10 % or abs ±5 mm in metres
    "deviation": (0.10, 0.005),
    "distance": (0.10, 0.005),
    "thickness": (0.10, 0.005),
    "completion": (0.02, 0.02),      # ±2 % absolute on completion percentage
    "percent_complete": (0.02, 0.02),
    "coverage": (0.05, 0.02),
    "in_tolerance_ratio": (0.05, 0.02),
    "f_score": (0.05, 0.02),
    "angle": (0.10, 1.0),             # ±10 % or ±1 degree
    "default": (0.10, 0.01),
}


# Unit conversion factors to canonical metres / dimensionless. Keys are
# lower-case unit strings.
DEFAULT_UNIT_FACTORS: Mapping[str, float] = {
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
    "%": 0.01,
    "percent": 0.01,
    "deg": 1.0,
    "degrees": 1.0,
    "rad": 57.295779513082323,  # to degrees
}


_NUMERIC_PATTERN = re.compile(
    r"""
    (?P<quantity>            # optional named quantity preceding the number
        offset|deviation|distance|thickness|completion|percent[_\s]complete
        |coverage|in[_\s]tolerance(?:[_\s]ratio)?|f[_\s-]?score|angle
    )?
    \s*
    (?:of|is|=|:)?           # connectors
    \s*
    (?P<value>[-+]?\d+(?:\.\d+)?)
    \s*
    (?P<unit>mm|cm|m|%|percent|deg|degrees|rad)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_GLOBAL_ID_PATTERN = re.compile(r"\b([0-9A-Za-z_$]{22})\b")  # IFC GlobalId is 22 chars
_IFC_CLASS_PATTERN = re.compile(r"\b(Ifc[A-Z][A-Za-z]+)\b")
_ACTIVITY_ID_PATTERN = re.compile(r"\b(A\d{3,6})\b")  # e.g. A0123 — sentinel; the
                                                      # caller can pass extra known
                                                      # activity IDs to widen the
                                                      # match.

_ACCEPT_PATTERN = re.compile(
    r"\b(accept(?:ed|s)?|approve(?:d|s)?|complete[d]?|passes?|on\s+schedule)\b",
    re.IGNORECASE,
)
_REJECT_PATTERN = re.compile(
    r"\b(reject(?:ed|s)?|fail(?:ed|s)?|incomplete|behind\s+schedule|not\s+acceptable)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def _normalise_quantity(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "percent_complete": "percent_complete",
        "in_tolerance": "in_tolerance_ratio",
        "in_tolerance_ratio": "in_tolerance_ratio",
        "f_score": "f_score",
        "fscore": "f_score",
    }
    return aliases.get(s, s)


def _to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Convert ``(value, unit)`` to canonical (metres / fraction / degrees)."""
    u = unit.strip().lower()
    if u in DEFAULT_UNIT_FACTORS:
        if u in {"%", "percent"}:
            return value * DEFAULT_UNIT_FACTORS[u], "fraction"
        if u in {"deg", "degrees", "rad"}:
            return value * DEFAULT_UNIT_FACTORS[u], "deg"
        return value * DEFAULT_UNIT_FACTORS[u], "m"
    return value, u


def extract_claims(
    answer: str,
    *,
    known_activity_ids: Iterable[str] = (),
) -> list[Claim]:
    """Split the answer into atomic numeric, entity, and acceptance claims.

    The extractor is deliberately simple and fully deterministic. Coverage
    is validated by the unit tests, not by an LLM. ``known_activity_ids``
    lets the caller widen the activity-ID pattern beyond the ``A####``
    sentinel.
    """
    if not answer:
        return []
    claims: list[Claim] = []

    # --- numeric claims ----------------------------------------------
    for m in _NUMERIC_PATTERN.finditer(answer):
        try:
            v = float(m.group("value"))
        except (TypeError, ValueError):
            continue
        unit = m.group("unit")
        quantity = _normalise_quantity(m.group("quantity"))
        canonical_value, canonical_unit = _to_canonical(v, unit)
        claims.append(
            Claim(
                kind=ClaimKind.NUMERIC,
                text=m.group(0).strip(),
                value=canonical_value,
                unit=canonical_unit,
                quantity=quantity,
            )
        )

    # --- named-entity claims -----------------------------------------
    seen_entities: set[tuple[str, str]] = set()
    for m in _GLOBAL_ID_PATTERN.finditer(answer):
        ent = m.group(1)
        key = ("global_id", ent)
        if key in seen_entities:
            continue
        seen_entities.add(key)
        claims.append(
            Claim(
                kind=ClaimKind.NAMED_ENTITY,
                text=ent,
                entity_kind="global_id",
                entity_value=ent,
            )
        )
    for m in _IFC_CLASS_PATTERN.finditer(answer):
        ent = m.group(1)
        key = ("ifc_class", ent)
        if key in seen_entities:
            continue
        seen_entities.add(key)
        claims.append(
            Claim(
                kind=ClaimKind.NAMED_ENTITY,
                text=ent,
                entity_kind="ifc_class",
                entity_value=ent,
            )
        )
    for known in set(known_activity_ids):
        if not known:
            continue
        if known in answer:
            key = ("activity_id", known)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            claims.append(
                Claim(
                    kind=ClaimKind.NAMED_ENTITY,
                    text=known,
                    entity_kind="activity_id",
                    entity_value=known,
                )
            )
    for m in _ACTIVITY_ID_PATTERN.finditer(answer):
        ent = m.group(1)
        key = ("activity_id", ent)
        if key in seen_entities:
            continue
        seen_entities.add(key)
        claims.append(
            Claim(
                kind=ClaimKind.NAMED_ENTITY,
                text=ent,
                entity_kind="activity_id",
                entity_value=ent,
            )
        )

    # --- acceptance / rejection claims -------------------------------
    has_accept = bool(_ACCEPT_PATTERN.search(answer))
    has_reject = bool(_REJECT_PATTERN.search(answer))
    if has_accept:
        claims.append(
            Claim(
                kind=ClaimKind.ACCEPTANCE,
                text="<accept>",
                acceptance_polarity="accept",
            )
        )
    if has_reject:
        claims.append(
            Claim(
                kind=ClaimKind.ACCEPTANCE,
                text="<reject>",
                acceptance_polarity="reject",
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Evidence-package walking
# ---------------------------------------------------------------------------


def _walk_numeric_fields(
    obj: Any,
    *,
    path: str = "",
) -> Iterable[tuple[str, str, float]]:
    """Yield ``(dotted_path, leaf_name, numeric_value)`` for every numeric
    leaf in a JSON-shaped object. Leaf name is the last path segment,
    used for quantity matching."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else str(k)
            yield from _walk_numeric_fields(v, path=new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            yield from _walk_numeric_fields(v, path=new_path)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)) and not _is_nan(float(obj)):
        leaf = path.rsplit(".", 1)[-1] if "." in path else path
        leaf = leaf.split("[")[0] or leaf
        yield path, leaf.lower(), float(obj)


def _walk_string_fields(obj: Any, *, path: str = "") -> Iterable[tuple[str, str]]:
    """Yield ``(dotted_path, string_value)`` for every string leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else str(k)
            yield from _walk_string_fields(v, path=new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_string_fields(v, path=f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _is_nan(x: float) -> bool:
    return x != x


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _tolerance_for_quantity(
    quantity: str | None,
    tolerances: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    if quantity is None:
        return tolerances.get("default", (0.10, 0.01))
    return tolerances.get(quantity.lower(), tolerances.get("default", (0.10, 0.01)))


def verify_claim(
    claim: Claim,
    evidence: Mapping[str, Any],
    *,
    tolerances: Mapping[str, tuple[float, float]] = DEFAULT_TOLERANCES,
) -> ClaimVerification:
    """Verify one claim against the evidence package."""

    if claim.kind == ClaimKind.NUMERIC:
        return _verify_numeric(claim, evidence, tolerances)
    if claim.kind == ClaimKind.NAMED_ENTITY:
        return _verify_named_entity(claim, evidence)
    if claim.kind == ClaimKind.ACCEPTANCE:
        return _verify_acceptance(claim, evidence)
    return ClaimVerification(
        claim=claim,
        status="unverifiable",
        notes=(f"unknown_claim_kind:{claim.kind}",),
    )


def _verify_numeric(
    claim: Claim,
    evidence: Mapping[str, Any],
    tolerances: Mapping[str, tuple[float, float]],
) -> ClaimVerification:
    if claim.value is None:
        return ClaimVerification(claim=claim, status="unverifiable", notes=("numeric_value_missing",))

    rel, abs_ = _tolerance_for_quantity(claim.quantity, tolerances)
    best_match: tuple[str, float, float] | None = None  # (path, value, distance)

    quantity_l = (claim.quantity or "").lower()

    for path, leaf, value in _walk_numeric_fields(evidence):
        # If the claim names a quantity, prefer matches whose leaf name
        # contains that quantity; otherwise fall back to numeric proximity.
        if quantity_l and quantity_l not in leaf:
            continue
        diff = abs(value - claim.value)
        tol = max(rel * abs(value), abs_)
        if best_match is None or diff < best_match[2]:
            best_match = (path, value, diff)
        if diff <= tol:
            return ClaimVerification(
                claim=claim,
                status="matched",
                evidence_path=path,
                expected_value=value,
                tolerance_used=tol,
            )

    if best_match is None and quantity_l:
        # Quantity-named field never appeared in the evidence -> unverifiable.
        return ClaimVerification(
            claim=claim,
            status="unverifiable",
            notes=(f"no_field_named:{quantity_l}",),
        )

    if best_match is not None:
        return ClaimVerification(
            claim=claim,
            status="unsupported",
            evidence_path=best_match[0],
            expected_value=best_match[1],
            tolerance_used=max(rel * abs(best_match[1]), abs_),
            notes=(f"closest_match_distance:{best_match[2]:.6g}",),
        )

    return ClaimVerification(
        claim=claim,
        status="unverifiable",
        notes=("no_numeric_evidence_present",),
    )


def _verify_named_entity(
    claim: Claim,
    evidence: Mapping[str, Any],
) -> ClaimVerification:
    if not claim.entity_value:
        return ClaimVerification(claim=claim, status="unverifiable", notes=("entity_value_missing",))
    needle = claim.entity_value
    for path, value in _walk_string_fields(evidence):
        if needle == value or needle in value:
            return ClaimVerification(
                claim=claim,
                status="matched",
                evidence_path=path,
                expected_value=value,
            )
        # Also match when an evidence value is a substring of the claim
        # (e.g. claim says "IfcWall" and evidence has "ifcwall_segment").
        if value and value in needle:
            return ClaimVerification(
                claim=claim,
                status="matched",
                evidence_path=path,
                expected_value=value,
            )
    return ClaimVerification(
        claim=claim,
        status="unsupported",
        notes=(f"no_evidence_string_matches:{needle}",),
    )


def _verify_acceptance(
    claim: Claim,
    evidence: Mapping[str, Any],
) -> ClaimVerification:
    """Acceptance / rejection claims are checked against ``confidence_flags``
    and any deterministic ``element_status`` field in the evidence."""

    flags = evidence.get("confidence_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags_text = " ".join(str(f) for f in flags).lower()

    metrics = evidence.get("metrics") or {}
    statuses: list[str] = []
    if isinstance(metrics, dict):
        for v in metrics.values():
            if isinstance(v, dict):
                data = v.get("data") or {}
                if isinstance(data, dict):
                    s = data.get("status") or data.get("element_status")
                    if isinstance(s, str):
                        statuses.append(s.lower())
    statuses_text = " ".join(statuses)

    risk_indicators = ("low_quality", "registration_low", "incomplete", "behind", "not_acceptable")
    has_risk = any(tok in flags_text or tok in statuses_text for tok in risk_indicators)

    if claim.acceptance_polarity == "accept":
        if has_risk:
            return ClaimVerification(
                claim=claim,
                status="unsupported",
                notes=("acceptance_claim_with_risk_in_evidence",),
            )
        return ClaimVerification(
            claim=claim,
            status="matched",
            evidence_path="confidence_flags",
            expected_value=flags,
        )
    if claim.acceptance_polarity == "reject":
        # Rejection claims are always at least *consistent* — they err on
        # the safe side. We mark them ``matched`` so they do not pollute
        # the unsupported counter.
        return ClaimVerification(
            claim=claim,
            status="matched",
            evidence_path="confidence_flags",
            expected_value=flags,
        )
    return ClaimVerification(claim=claim, status="unverifiable", notes=("unknown_acceptance_polarity",))


# ---------------------------------------------------------------------------
# Top-level guard
# ---------------------------------------------------------------------------


GROUNDING_SCHEMA_VERSION = "grounding_guard.v1"


def ground_answer(
    answer: str,
    evidence: Mapping[str, Any],
    *,
    tolerances: Mapping[str, tuple[float, float]] = DEFAULT_TOLERANCES,
    known_activity_ids: Iterable[str] = (),
    pass_threshold_unsupported: int = 0,
) -> GroundingResult:
    """Run the full grounding-guard pipeline on one (answer, evidence) pair.

    ``pass_threshold_unsupported`` is the maximum number of unsupported
    claims the answer may have and still ``passed = True``. Default 0
    (any unsupported claim fails the guard); set to 1 to be lenient.
    """
    claims = extract_claims(answer, known_activity_ids=known_activity_ids)
    verifications = [verify_claim(c, evidence, tolerances=tolerances) for c in claims]

    n_matched = sum(1 for v in verifications if v.status == "matched")
    n_unsupported = sum(1 for v in verifications if v.status == "unsupported")
    n_unverifiable = sum(1 for v in verifications if v.status == "unverifiable")
    n_claims = len(verifications)

    risks: list[str] = []
    if n_unsupported > 0:
        risks.append(f"vlm_unsupported_claims:{n_unsupported}")
    if n_unverifiable > 0:
        risks.append(f"vlm_unverifiable_claims:{n_unverifiable}")
    if n_claims == 0:
        risks.append("vlm_answer_carries_no_extractable_claims")

    passed = n_unsupported <= pass_threshold_unsupported

    return GroundingResult(
        schema_version=GROUNDING_SCHEMA_VERSION,
        n_claims=n_claims,
        n_matched=n_matched,
        n_unsupported=n_unsupported,
        n_unverifiable=n_unverifiable,
        passed=passed,
        risk_tokens=tuple(risks),
        per_claim=tuple(verifications),
    )


def attach_grounding_guard(
    response: dict[str, Any],
    *,
    tolerances: Mapping[str, tuple[float, float]] = DEFAULT_TOLERANCES,
    known_activity_ids: Iterable[str] = (),
    pass_threshold_unsupported: int = 0,
) -> dict[str, Any]:
    """Run :func:`ground_answer` on a Stage 10 response dict and attach
    the result.

    Mirrors :func:`pipeline.stage_10_copilot.api.attach_answer_validation`
    so the existing response-augmentation chain absorbs the guard
    cleanly.

    On guard failure:

    - the response's ``confidence`` is forced to ``"low"`` (the same
      escalation policy used by ``answer_validator``);
    - the guard's ``risk_tokens`` are appended to
      ``risks_or_uncertainty``.
    """
    if not isinstance(response, dict):
        return response

    if "grounding_guard" in response:
        return response

    answer_text = str(response.get("answer", ""))
    evidence_pkg = response.get("evidence_package") or {}
    if not isinstance(evidence_pkg, dict):
        evidence_pkg = {}

    # Fallback: if no full package is on the response, look at
    # generated_view_paths / evidence_used / evidence_package_path.
    if not evidence_pkg:
        evidence_pkg = {
            "evidence_used": response.get("evidence_used") or [],
            "confidence_flags": response.get("confidence_flags") or [],
            "metrics": response.get("metrics") or {},
        }

    result = ground_answer(
        answer_text,
        evidence_pkg,
        tolerances=tolerances,
        known_activity_ids=known_activity_ids,
        pass_threshold_unsupported=pass_threshold_unsupported,
    )
    response["grounding_guard"] = result.to_dict()

    if not result.passed:
        response["confidence"] = "low"
        risks = response.get("risks_or_uncertainty")
        if risks is None:
            risks = []
        elif isinstance(risks, str):
            risks = [risks]
        elif not isinstance(risks, list):
            risks = [str(risks)]
        risks.extend(result.risk_tokens)
        response["risks_or_uncertainty"] = risks

    return response
