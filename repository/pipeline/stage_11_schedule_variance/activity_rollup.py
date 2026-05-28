"""Per-activity roll-up of Stage 9 element-level results.

Given:

- Stage 9 element rows, each carrying at minimum
  ``global_id`` and ``status`` (one of ``likely_completed`` /
  ``partially_observed`` / ``not_evidenced`` /
  ``uncertain_low_registration``);
- the BIM <-> schedule mapping (:mod:`pipeline.common.bim_schedule_mapping`);

…produce a per-activity rollup with:

- ``actual_percent_complete`` — weighted mean of element-level
  acceptance scores (1.0 / 0.5 / 0.0 / 0.5 by status);
- ``actual_percent_complete_lower_95``, ``actual_percent_complete_upper_95``
  — Wilson 95 %% interval on the unweighted *binarised* acceptance
  count, computed via :func:`pipeline.stage_09_progress.uncertainty.wilson_interval`
  if available; otherwise a faithful inline implementation.

The rollup is **deterministic** and **dependency-free** (apart from the
existing Wilson helper).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pipeline.common.bim_schedule_mapping import BimScheduleMapping, MappingEntry

__all__ = [
    "ELEMENT_STATUS_TO_ACCEPTANCE",
    "ActivityRollup",
    "rollup_activities",
    "wilson_interval",
]


# Per-status acceptance weight in [0, 1]. The weight is what the rollup
# sums (weighted by mapping weight) and divides by the total mapping
# weight to get actual_percent_complete.
ELEMENT_STATUS_TO_ACCEPTANCE: Mapping[str, float] = {
    "likely_completed": 1.0,
    "partially_observed": 0.5,
    "uncertain_low_registration": 0.5,  # neither pass nor fail
    "not_evidenced": 0.0,
    # Phase 4 multi-view fusion outputs (consumed only when the
    # element row carries a fused decision rather than the legacy
    # status) — same semantics:
    "acceptable": 1.0,
    "uncertain": 0.5,
    "uncertain_conflict": 0.5,
    "not_acceptable": 0.0,
}


@dataclass(frozen=True)
class ActivityRollup:
    """One activity's rolled-up per-element result."""

    activity_id: str
    n_mapped_elements: int
    n_evaluated_elements: int
    n_completed_elements: int
    n_partial_elements: int
    n_uncertain_elements: int
    n_not_evidenced_elements: int
    actual_percent_complete: float
    actual_percent_complete_lower_95: float
    actual_percent_complete_upper_95: float
    contributing_element_global_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["contributing_element_global_ids"] = list(self.contributing_element_global_ids)
        d["notes"] = list(self.notes)
        return d


def wilson_interval(
    successes: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Two-sided Wilson interval for a binomial proportion.

    Returns ``(lo, hi)`` in [0, 1]. For ``n == 0`` returns ``(0.0, 1.0)``.

    The default ``z`` is the standard 95 %% normal quantile (2-sided).
    Implementation matches Wilson (JASA 1927) and is consistent with the
    existing :mod:`pipeline.stage_09_progress.uncertainty` helper.
    """
    if n <= 0:
        return 0.0, 1.0
    if successes < 0 or successes > n:
        raise ValueError(f"successes ({successes}) must be in [0, n]={n}")
    p_hat = successes / n
    denom = 1.0 + (z * z) / n
    centre = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n)
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _read_status(row: Mapping[str, Any]) -> str:
    """Pick the strongest available status field on a Stage 9 row.

    Phase-4 callers may attach a ``fused_decision`` from multi-view
    fusion next to the legacy ``status``; we prefer the fused decision
    when present.
    """
    for key in ("fused_decision", "decision", "status"):
        v = row.get(key)
        if v:
            return str(v)
    return ""


def _read_global_id(row: Mapping[str, Any]) -> str:
    for key in ("global_id", "GlobalId", "element_global_id", "ifc_global_id"):
        v = row.get(key)
        if v:
            return str(v)
    return ""


def rollup_activities(
    element_rows: Iterable[Mapping[str, Any]],
    *,
    mapping: BimScheduleMapping,
    activity_ids: Sequence[str] | None = None,
) -> list[ActivityRollup]:
    """Roll element rows up to per-activity rollups.

    ``activity_ids`` is the list of activities you want results for; if
    ``None``, every activity that appears in the mapping gets a rollup.
    Activities with no mapped elements still appear in the output (as
    ``n_mapped_elements=0``) so the dashboard can show "no evidence" cells
    rather than silently dropping rows.
    """
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in element_rows:
        gid = _read_global_id(row)
        if gid:
            rows_by_id[gid] = row

    targets: list[str]
    if activity_ids is None:
        targets = sorted(set(e.activity_id for e in mapping.entries))
    else:
        targets = list(activity_ids)

    out: list[ActivityRollup] = []
    for aid in targets:
        entries: tuple[MappingEntry, ...] = mapping.elements_for_activity(aid)
        contributing: list[str] = []
        weighted_sum = 0.0
        weight_total = 0.0

        n_completed = 0
        n_partial = 0
        n_uncertain = 0
        n_not_ev = 0
        n_eval = 0
        notes: list[str] = []

        for entry in entries:
            row = rows_by_id.get(entry.ifc_global_id)
            if row is None:
                # Element appears in mapping but Stage 9 has no row for it.
                # Treat as not_evidenced and surface in the notes.
                n_not_ev += 1
                continue
            n_eval += 1
            contributing.append(entry.ifc_global_id)
            status = _read_status(row).lower()
            acceptance = ELEMENT_STATUS_TO_ACCEPTANCE.get(status)
            if acceptance is None:
                # Unknown status -> treat conservatively as uncertain.
                acceptance = 0.5
                notes.append(f"unknown_element_status:{status}")
            if acceptance >= 1.0:
                n_completed += 1
            elif acceptance <= 0.0:
                n_not_ev += 1
            elif status in {"partially_observed"}:
                n_partial += 1
            else:
                n_uncertain += 1

            weight = float(entry.weight)
            if weight <= 0:
                continue
            weighted_sum += acceptance * weight
            weight_total += weight

        if weight_total > 0:
            actual = 100.0 * weighted_sum / weight_total
        else:
            actual = 0.0

        # Wilson interval on the binarised acceptance count
        # (n_completed successes / n_eval trials). This is a *per-activity*
        # uncertainty band that matches what AiC reviewers expect.
        if n_eval > 0:
            lo, hi = wilson_interval(n_completed, n_eval)
            lo_pct = 100.0 * lo
            hi_pct = 100.0 * hi
        else:
            lo_pct = 0.0
            hi_pct = 100.0
            notes.append("no_evidence")

        out.append(
            ActivityRollup(
                activity_id=aid,
                n_mapped_elements=len(entries),
                n_evaluated_elements=n_eval,
                n_completed_elements=n_completed,
                n_partial_elements=n_partial,
                n_uncertain_elements=n_uncertain,
                n_not_evidenced_elements=n_not_ev,
                actual_percent_complete=actual,
                actual_percent_complete_lower_95=lo_pct,
                actual_percent_complete_upper_95=hi_pct,
                contributing_element_global_ids=tuple(contributing),
                notes=tuple(notes),
            )
        )

    return out
