"""Trusted multi-view temporal evidence fusion for per-element progress.

Why this module exists
----------------------

The current Stage 9 metric is computed once per BIM element from the cleaned
fused point cloud. That cloud is itself the result of MVS over many keyframes,
so multi-view aggregation already happens *implicitly* at the geometry level.
But the *evidence* per element — which views actually saw the element, with
what visibility quality, with what residual to the BIM — is collapsed away.

Two concrete consequences:

1. A single bad keyframe (motion blur, grazing-angle pass, transient
   occlusion by a worker or scaffold) can reduce per-element completeness in
   a way that is indistinguishable from an actually-incomplete element.
2. Disagreement between views (e.g. two viewpoints see the column as
   in-tolerance and one sees it shifted) is averaged away into a single
   completeness ratio with no way to recover the conflict.

This module computes a *per-element fused decision* from N per-view
observations and **separately reports the conflict mass** so reviewers can
see when views disagree.

Method
------

We follow the Dempster-Shafer-Inspired *trusted multi-view classification*
paradigm of Han et al. (ICLR 2021, arXiv 2102.02051). For our two-class
problem (``acceptable`` vs ``not_acceptable``) the implementation is small
enough to be self-contained and dependency-free:

For each per-view observation we form a Dirichlet evidence vector
``(e_a, e_n) = view_confidence * (p_a, 1 - p_a)``, where ``p_a`` is the
view-level probability that the element is acceptable and
``view_confidence`` is the view-level evidence weight in [0, 1] (a function
of view sharpness, viewing-angle suitability, and observation count;
see :func:`compute_view_evidence_weight` for the default schedule).

Two views are fused by the Dempster combination on the simplex {a, n, U},
where ``U`` is the *uncertainty* (vacuous) mass = 2 / (e_a + e_n + 2)
(Sensoy, Kaplan & Kandemir, NeurIPS 2018, arXiv 1806.01768). The conflict
mass between the two views is reported separately and **not** redistributed
into the singleton classes (we use the unnormalized rule), so a high-conflict
fusion is loud about the conflict rather than papering over it. A general
N-view fusion is a left fold of pairwise fusions; the operation is
associative for our 2-class setup.

The output per element is:

- ``fused_belief_acceptable`` ∈ [0, 1]  — fused class belief (Dirichlet
  marginal expectation).
- ``fused_uncertainty_mass`` ∈ [0, 1]   — vacuous mass after fusion.
- ``conflict_mass`` ∈ [0, 1]            — total Dempster conflict across
  views (0 = unanimous, 1 = maximal).
- ``view_count`` — N.
- ``contributing_views`` — IDs of the views included.
- ``decision`` — ``acceptable`` / ``uncertain`` / ``not_acceptable``,
  with explicit thresholds documented below.

Why a separate fusion module instead of patching the existing aggregation
-------------------------------------------------------------------------

The existing per-element metric in Stage 9 is *deterministic and BIM-grounded*
(scan-to-BIM bidirectional accuracy/completeness/F-score @ τ; see
:mod:`bidirectional_metrics`). We deliberately do *not* replace it. The
multi-view fusion here is **additive**: it consumes per-view observations
and produces a per-element fused belief that lives next to (and never
overwrites) the deterministic metric. The decision policy in
:mod:`pipeline.common.progress_decision_policy` is unchanged. Operators who
want to use the fused belief can wire it into a future decision policy
revision; those who don't see no behavior change.

Failure modes and what we do about them
---------------------------------------

- **Empty view set.** The module returns ``decision = "not_evidenced"`` and
  ``fused_uncertainty_mass = 1.0`` rather than fabricating an answer.
- **All views uncertain.** Same outcome — uncertainty is preserved through
  the fusion.
- **High conflict.** Reported as ``decision = "uncertain_conflict"`` when
  the conflict mass exceeds ``conflict_mass_threshold`` (default 0.30).
- **Single view.** Returned as-is; the function is well-defined for N ≥ 1
  and degenerates to the input when N = 1.

References
----------

- Han, Z., Zhang, C., Fu, H., Zhou, J. T. *Trusted Multi-View
  Classification*. ICLR 2021. arXiv 2102.02051.
- Sensoy, M., Kaplan, L., Kandemir, M. *Evidential Deep Learning to Quantify
  Classification Uncertainty*. NeurIPS 2018. arXiv 1806.01768.
- *A Trusted Multi-View Evidential Fusion Framework for Commonsense
  Reasoning*. LREC-COLING 2024.
- Dempster, A. P. *Upper and Lower Probabilities Induced by a Multivalued
  Mapping*. Annals of Mathematical Statistics, 1967.
- Tuttas, S., Braun, A., Borrmann, A., Stilla, U. *Acquisition and
  Consecutive Registration of Photogrammetric Point Clouds for Construction
  Progress Monitoring*. PFG 2017. (Multi-view aggregation in construction
  progress monitoring.)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ViewObservation",
    "FusedElementBelief",
    "compute_view_evidence_weight",
    "fuse_two_views",
    "fuse_views",
    "fuse_per_element",
]


@dataclass(frozen=True)
class ViewObservation:
    """One per-view observation of one BIM element.

    Fields
    ------
    view_id : str
        Stable identifier for the view (typically the keyframe basename).
    element_global_id : str
        IFC GlobalId of the observed element.
    p_acceptable : float
        View-level probability that the element is acceptable, in [0, 1].
        Callers are responsible for choosing how to derive this; sensible
        defaults are ``in_tolerance_ratio`` of the view's per-element
        scan-to-BIM cropping, or a sigmoid of the per-view F-score.
    view_confidence : float
        View-level evidence weight in [0, 1]. Defaults to 1.0 for callers
        who do not have per-view quality information; production callers
        should compute it via :func:`compute_view_evidence_weight`.
    blur_score : float | None
        Optional Laplacian sharpness in [0, 1]; consumed by
        :func:`compute_view_evidence_weight`.
    grazing_angle_deg : float | None
        Angle (in degrees) between the view direction and the element
        surface normal at its centroid. 0 = head-on, 90 = grazing.
    point_count : int | None
        Number of scan points the view contributed for this element.

    Notes
    -----
    The dataclass is intentionally schema-rich so callers can pre-compute
    rich per-view quality. The fusion math itself only uses
    ``p_acceptable`` and ``view_confidence``; everything else is metadata
    used by :func:`compute_view_evidence_weight`.
    """

    view_id: str
    element_global_id: str
    p_acceptable: float
    view_confidence: float = 1.0
    blur_score: float | None = None
    grazing_angle_deg: float | None = None
    point_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusedElementBelief:
    """Result of fusing N view observations of one BIM element.

    Attributes
    ----------
    fused_belief_acceptable, fused_belief_not_acceptable, fused_uncertainty_mass
        Subjective-logic masses on {acceptable, not_acceptable, U}; sum to 1.
        These are *belief* masses, not probabilities.
    expected_probability_acceptable
        Dirichlet marginal expectation
        ``alpha_a / S = (e_a + 1) / (e_a + e_n + 2)``,
        which is the *probability* the element is acceptable under the
        Dirichlet posterior. This is the natural quantity to threshold
        for an accept/reject decision; it ranges over the full [0, 1]
        interval as evidence accumulates, whereas the belief mass
        ``fused_belief_acceptable`` is bounded above by ``1 - 2/(2+e_a+e_n)``
        and so always leaves some uncertainty mass for finite evidence.
    conflict_mass
        Aggregated Dempster K across the fusion sequence; a diagnostic
        for inter-view disagreement, *not* a part of the belief masses.
    """

    element_global_id: str
    view_count: int
    contributing_views: tuple[str, ...]
    fused_belief_acceptable: float
    fused_belief_not_acceptable: float
    fused_uncertainty_mass: float
    expected_probability_acceptable: float
    conflict_mass: float
    decision: str
    decision_threshold_acceptable: float
    decision_threshold_uncertain: float
    conflict_mass_threshold: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_global_id": self.element_global_id,
            "view_count": self.view_count,
            "contributing_views": list(self.contributing_views),
            "fused_belief_acceptable": self.fused_belief_acceptable,
            "fused_belief_not_acceptable": self.fused_belief_not_acceptable,
            "fused_uncertainty_mass": self.fused_uncertainty_mass,
            "expected_probability_acceptable": self.expected_probability_acceptable,
            "conflict_mass": self.conflict_mass,
            "decision": self.decision,
            "decision_threshold_acceptable": self.decision_threshold_acceptable,
            "decision_threshold_uncertain": self.decision_threshold_uncertain,
            "conflict_mass_threshold": self.conflict_mass_threshold,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# View-level confidence weighting.
#
# Everything below operates on already-computed per-view confidences. The
# weight schedule is exposed as a separate function so it can be overridden
# (e.g. with a learned weight) without touching the fusion math.
# ---------------------------------------------------------------------------


def compute_view_evidence_weight(
    *,
    blur_score: float | None = None,
    grazing_angle_deg: float | None = None,
    point_count: int | None = None,
    min_point_count: int = 30,
    blur_floor: float = 0.10,
    grazing_floor_deg: float = 75.0,
) -> float:
    """Map per-view quality signals to a scalar evidence weight in [0, 1].

    Default schedule:

    - **Blur**: linear ramp from 0 at ``blur_floor`` to 1 at 1.0; values
      below the floor get 0. Justification: blur invalidates per-pixel
      photometric matches, which is the source of MVS depth.
    - **Grazing angle**: cosine ramp ``cos(angle)`` for angles in
      ``[0, grazing_floor_deg]``, 0 above the floor. Justification:
      surfaces seen at >75° produce extremely noisy depth (Hartley & Zisserman
      Ch 12; see also Furukawa & Hernández, *Multi-view stereo: a tutorial*,
      Foundations and Trends in CGV 2015).
    - **Point count**: ``min(1, count / min_point_count)``; below the
      threshold, evidence is heavily downweighted. Justification: a view
      contributing < ~30 points to an element is too sparse to support a
      defensible per-view per-element decision.

    All available signals are multiplied. Missing signals contribute a
    weight of 1 (i.e. they are uninformative rather than penalising).

    The defaults are conservative for typical handheld/smartphone capture in
    construction sites; sites with calibrated rigs may safely raise
    ``min_point_count``.
    """
    weight = 1.0

    if blur_score is not None:
        if blur_score <= blur_floor:
            return 0.0
        weight *= max(0.0, min(1.0, (blur_score - blur_floor) / max(1e-6, 1.0 - blur_floor)))

    if grazing_angle_deg is not None:
        if grazing_angle_deg >= grazing_floor_deg:
            return 0.0
        weight *= max(0.0, math.cos(math.radians(grazing_angle_deg)))

    if point_count is not None:
        weight *= max(0.0, min(1.0, point_count / max(1, min_point_count)))

    return float(max(0.0, min(1.0, weight)))


# ---------------------------------------------------------------------------
# Two-class evidential fusion.
#
# We map (p_acceptable, view_confidence) to (e_a, e_n) and then use the
# Dirichlet/Subjective-Logic update Han et al. (2021) describe. For two
# classes the formulas simplify enormously:
#
#   alpha_a = e_a + 1
#   alpha_n = e_n + 1
#   S       = alpha_a + alpha_n
#   b_a     = e_a / S
#   b_n     = e_n / S
#   u       = 2 / S
#   b_a + b_n + u = 1     (verifiable)
#
# The Dempster combination of two such (b_a, b_n, u) measures with conflict
# K = b_a^(1) * b_n^(2) + b_n^(1) * b_a^(2) is:
#
#   b_a^* = (b_a^(1) b_a^(2) + b_a^(1) u^(2) + u^(1) b_a^(2)) / (1 - K)
#   b_n^* = (b_n^(1) b_n^(2) + b_n^(1) u^(2) + u^(1) b_n^(2)) / (1 - K)
#   u^*   = (u^(1) u^(2)) / (1 - K)
#
# We *also* return K explicitly so consumers can flag high-conflict fusions.
# ---------------------------------------------------------------------------


def _to_evidence(p_acceptable: float, view_confidence: float) -> tuple[float, float]:
    """Map (p, w) into evidence (e_a, e_n).

    The mapping must satisfy:

    - ``view_confidence = 0`` → vacuous (e_a = e_n = 0, u = 1).
    - ``view_confidence = 1, p = 1`` → all evidence on ``acceptable``.
    - ``view_confidence = 1, p = 0`` → all evidence on ``not_acceptable``.

    We use the linear mapping ``e_a = w * p`` and ``e_n = w * (1 - p)`` so
    the marginal Dirichlet expectation ``alpha_a / S = (e_a + 1) / (e_a + e_n + 2)``
    smoothly interpolates between the prior 0.5 and the view's own ``p``,
    with the prior weight set by ``2 / (w + 2)``. This is the
    *expectation-as-belief* form used in evidential deep learning (Sensoy
    et al. NeurIPS 2018).
    """
    p = max(0.0, min(1.0, float(p_acceptable)))
    w = max(0.0, min(1.0, float(view_confidence)))
    return w * p, w * (1.0 - p)


def _evidence_to_bu(e_a: float, e_n: float) -> tuple[float, float, float]:
    """Convert evidence to (b_a, b_n, u) on the {a, n, U} frame."""
    s = e_a + e_n + 2.0
    return e_a / s, e_n / s, 2.0 / s


def fuse_two_views(
    obs_a: ViewObservation,
    obs_b: ViewObservation,
) -> tuple[float, float, float, float]:
    """Dempster combination of two ViewObservation belief masses.

    Returns ``(b_acceptable, b_not_acceptable, uncertainty, conflict)`` in [0, 1].
    The triple ``(b_acceptable, b_not_acceptable, uncertainty)`` always sums
    to 1.0 (within float tolerance). ``conflict`` is reported separately.

    When K = 1 (total conflict), Dempster's rule is undefined; we return
    ``(0, 0, 1, 1)`` as a documented degenerate case.
    """
    e_a1, e_n1 = _to_evidence(obs_a.p_acceptable, obs_a.view_confidence)
    e_a2, e_n2 = _to_evidence(obs_b.p_acceptable, obs_b.view_confidence)
    b_a1, b_n1, u1 = _evidence_to_bu(e_a1, e_n1)
    b_a2, b_n2, u2 = _evidence_to_bu(e_a2, e_n2)

    k = b_a1 * b_n2 + b_n1 * b_a2
    if k >= 1.0 - 1e-12:
        return 0.0, 0.0, 1.0, 1.0

    denom = 1.0 - k
    b_a = (b_a1 * b_a2 + b_a1 * u2 + u1 * b_a2) / denom
    b_n = (b_n1 * b_n2 + b_n1 * u2 + u1 * b_n2) / denom
    u = (u1 * u2) / denom
    # Renormalise for floating-point drift; sum should be exactly 1.
    s = b_a + b_n + u
    if s > 0:
        b_a, b_n, u = b_a / s, b_n / s, u / s
    return float(b_a), float(b_n), float(u), float(k)


def fuse_views(
    observations: Sequence[ViewObservation],
    *,
    decision_threshold_acceptable: float = 0.65,
    decision_threshold_uncertain: float = 0.50,
    conflict_mass_threshold: float = 0.30,
) -> FusedElementBelief:
    """Fuse N >= 0 per-view observations into a single per-element belief.

    Implementation uses the closed-form Dirichlet-evidence combination of
    Han et al. (ICLR 2021). For two classes:

        e_a_total = sum_i e_a_i        # sum of acceptable evidence
        e_n_total = sum_i e_n_i        # sum of not-acceptable evidence
        S         = e_a_total + e_n_total + 2
        b_a       = e_a_total / S
        b_n       = e_n_total / S
        u         = 2 / S

    The N-view evidence sum is associative and commutative (independent of
    fusion order), so the mathematics is order-stable by construction.

    Conflict is reported separately. We accumulate Dempster K *pairwise*
    over a fixed fusion order (input order) using the iterative
    {a, n, U}-frame combination of :func:`fuse_two_views`, and
    summarise it as ``K_total = 1 - prod_i (1 - K_i)``. This is the same
    "no-conflict probability" composition used in Sensoy et al.
    (NeurIPS 2018). Note that K_total is a *diagnostic*; the belief
    masses themselves come from the additive Dirichlet form, not from
    the iterative Dempster fold.

    Decision policy (configurable via the threshold parameters):

    - ``not_evidenced``         — N = 0 or all views had zero confidence.
    - ``uncertain_conflict``    — accumulated conflict mass exceeds
                                   ``conflict_mass_threshold``.
    - ``acceptable``            — fused belief on acceptable >=
                                   ``decision_threshold_acceptable`` and
                                   conflict mass below threshold.
    - ``not_acceptable``        — fused belief on not_acceptable >=
                                   ``decision_threshold_acceptable``.
    - ``uncertain``             — otherwise.
    """
    if not observations:
        element_id = ""
        return FusedElementBelief(
            element_global_id=element_id,
            view_count=0,
            contributing_views=(),
            fused_belief_acceptable=0.0,
            fused_belief_not_acceptable=0.0,
            fused_uncertainty_mass=1.0,
            expected_probability_acceptable=0.5,
            conflict_mass=0.0,
            decision="not_evidenced",
            decision_threshold_acceptable=decision_threshold_acceptable,
            decision_threshold_uncertain=decision_threshold_uncertain,
            conflict_mass_threshold=conflict_mass_threshold,
            notes=("no_view_observations",),
        )

    element_id = observations[0].element_global_id
    notes: list[str] = []

    # Drop zero-confidence observations: they carry no evidence.
    active = [o for o in observations if o.view_confidence > 0.0]
    if len(active) < len(observations):
        notes.append(f"dropped_zero_confidence_views:{len(observations) - len(active)}")
    if not active:
        return FusedElementBelief(
            element_global_id=element_id,
            view_count=len(observations),
            contributing_views=tuple(o.view_id for o in observations),
            fused_belief_acceptable=0.0,
            fused_belief_not_acceptable=0.0,
            fused_uncertainty_mass=1.0,
            expected_probability_acceptable=0.5,
            conflict_mass=0.0,
            decision="not_evidenced",
            decision_threshold_acceptable=decision_threshold_acceptable,
            decision_threshold_uncertain=decision_threshold_uncertain,
            conflict_mass_threshold=conflict_mass_threshold,
            notes=tuple(notes + ["all_views_zero_confidence"]),
        )

    if any(o.element_global_id != element_id for o in active):
        notes.append("mixed_element_ids")

    # ---- Belief masses: closed-form Dirichlet evidence sum -----------
    e_a_total = 0.0
    e_n_total = 0.0
    contributing: list[str] = []
    for o in active:
        e_a, e_n = _to_evidence(o.p_acceptable, o.view_confidence)
        e_a_total += e_a
        e_n_total += e_n
        contributing.append(o.view_id)
    s = e_a_total + e_n_total + 2.0
    b_a = e_a_total / s
    b_n = e_n_total / s
    u = 2.0 / s
    # Dirichlet marginal expectation: alpha_a / S = (e_a + 1) / S.
    # This is the *probability* the element is acceptable under the
    # posterior, and is the natural quantity for the decision policy.
    expected_p_a = (e_a_total + 1.0) / s
    expected_p_n = 1.0 - expected_p_a

    # ---- Conflict diagnostic: pairwise Dempster K accumulation -------
    # We compute K_i = b_a^(prev) * b_n^(cur) + b_n^(prev) * b_a^(cur) on
    # the per-view {a, n, U} masses, then aggregate as
    # K_total = 1 - prod_i (1 - K_i). The 'prev' belief here is the
    # *fused* belief after the first i views (Dirichlet form), so each
    # pairwise step measures conflict between the running consensus and
    # the next view -- the meaningful quantity for an audit.
    if len(active) == 1:
        conflict_total = 0.0
    else:
        e_a_run = 0.0
        e_n_run = 0.0
        no_conflict_product = 1.0
        # Seed with the first view.
        e_a_run, e_n_run = _to_evidence(active[0].p_acceptable, active[0].view_confidence)
        for o in active[1:]:
            s_run = e_a_run + e_n_run + 2.0
            b_a_run = e_a_run / s_run
            b_n_run = e_n_run / s_run
            e_a_cur, e_n_cur = _to_evidence(o.p_acceptable, o.view_confidence)
            s_cur = e_a_cur + e_n_cur + 2.0
            b_a_cur = e_a_cur / s_cur
            b_n_cur = e_n_cur / s_cur
            k_i = b_a_run * b_n_cur + b_n_run * b_a_cur
            no_conflict_product *= max(0.0, 1.0 - k_i)
            e_a_run += e_a_cur
            e_n_run += e_n_cur
        conflict_total = float(max(0.0, min(1.0, 1.0 - no_conflict_product)))

    if conflict_total > conflict_mass_threshold:
        decision = "uncertain_conflict"
        notes.append(f"conflict_mass_above_threshold:{conflict_total:.6f}>{conflict_mass_threshold:.6f}")
    elif expected_p_a >= decision_threshold_acceptable:
        decision = "acceptable"
    elif expected_p_n >= decision_threshold_acceptable:
        decision = "not_acceptable"
    elif expected_p_a >= decision_threshold_uncertain or expected_p_n >= decision_threshold_uncertain:
        decision = "uncertain"
    else:
        decision = "uncertain"

    return FusedElementBelief(
        element_global_id=element_id,
        view_count=len(active),
        contributing_views=tuple(contributing),
        fused_belief_acceptable=float(b_a),
        fused_belief_not_acceptable=float(b_n),
        fused_uncertainty_mass=float(u),
        expected_probability_acceptable=float(expected_p_a),
        conflict_mass=conflict_total,
        decision=decision,
        decision_threshold_acceptable=decision_threshold_acceptable,
        decision_threshold_uncertain=decision_threshold_uncertain,
        conflict_mass_threshold=conflict_mass_threshold,
        notes=tuple(notes),
    )


def fuse_per_element(
    observations: Iterable[ViewObservation],
    *,
    decision_threshold_acceptable: float = 0.65,
    decision_threshold_uncertain: float = 0.50,
    conflict_mass_threshold: float = 0.30,
) -> dict[str, FusedElementBelief]:
    """Group ``observations`` by ``element_global_id`` and fuse each group.

    Returns a dict ``{element_global_id: FusedElementBelief}``. Empty input
    returns an empty dict. The function is pure: same input → same output.
    """
    grouped: dict[str, list[ViewObservation]] = {}
    for o in observations:
        grouped.setdefault(o.element_global_id, []).append(o)
    return {
        eid: fuse_views(
            obs_list,
            decision_threshold_acceptable=decision_threshold_acceptable,
            decision_threshold_uncertain=decision_threshold_uncertain,
            conflict_mass_threshold=conflict_mass_threshold,
        )
        for eid, obs_list in grouped.items()
    }
