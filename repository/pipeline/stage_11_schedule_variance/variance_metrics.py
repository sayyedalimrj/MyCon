"""Variance metrics + dashboard summary for Stage 11.

Given:

- a :class:`pipeline.common.schedule_io.Schedule`,
- a list of :class:`ActivityRollup` from :mod:`activity_rollup`,
- a "data date" (the date at which planned-vs-actual is being compared,
  default UTC ``now()``),

…this module produces:

- :class:`ActivityVariance` per activity,
- :class:`ScheduleVarianceReport` aggregating the run, and
- :class:`DashboardSummary` — exactly the JSON the dashboard renders.

Schemas are stable and versioned. The variance_status vocabulary is:

- ``on_schedule`` — ``|actual - planned| <= on_schedule_band_pct``
- ``ahead``      — ``actual - planned > on_schedule_band_pct``
- ``behind``     — ``planned - actual > on_schedule_band_pct``
- ``unknown_evidence`` — when the rollup has no evaluated elements

The default ``on_schedule_band_pct`` is 5.0 percentage points. This is a
deliberately wide band that reflects practitioner tolerance in
construction project controls (cf. Bosché-style scan-vs-BIM dimensional
control: dimensional checks are routinely accepted at +-5 mm absolute,
and percent-complete carries similar measurement noise).

Confidence is mapped from the per-rollup interval width:

- ``high``   — width <= 10 percentage points
- ``medium`` — width <= 25 percentage points
- ``low``    — wider, or no evaluated elements
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pipeline.common.schedule_io import Activity, Schedule, planned_percent_complete_at

from pipeline.stage_11_schedule_variance.activity_rollup import ActivityRollup


__all__ = [
    "SCHEDULE_VARIANCE_SCHEMA_VERSION",
    "DASHBOARD_SUMMARY_SCHEMA_VERSION",
    "DEFAULT_ON_SCHEDULE_BAND_PCT",
    "ActivityVariance",
    "ScheduleVarianceReport",
    "DashboardSummary",
    "classify_status",
    "build_variance_report",
    "build_dashboard_summary",
]


SCHEDULE_VARIANCE_SCHEMA_VERSION = "schedule_variance.v1"
DASHBOARD_SUMMARY_SCHEMA_VERSION = "dashboard_summary.v1"

DEFAULT_ON_SCHEDULE_BAND_PCT = 5.0


@dataclass(frozen=True)
class ActivityVariance:
    """One activity row with planned, actual, variance, status, confidence."""

    activity_id: str
    activity_name: str
    planned_percent_complete: float
    actual_percent_complete: float
    actual_percent_complete_lower_95: float
    actual_percent_complete_upper_95: float
    schedule_variance_percent: float  # actual - planned, in pct points
    status: str
    confidence: str
    n_evaluated_elements: int
    n_mapped_elements: int
    risks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risks"] = list(self.risks)
        return d


@dataclass(frozen=True)
class ScheduleVarianceReport:
    """Run-wide variance summary."""

    schema_version: str
    data_date_utc: str
    on_schedule_band_pct: float
    n_activities: int
    n_on_schedule: int
    n_ahead: int
    n_behind: int
    n_unknown_evidence: int
    overall_planned_percent_complete: float
    overall_actual_percent_complete: float
    overall_actual_lower_95: float
    overall_actual_upper_95: float
    overall_schedule_variance_percent: float
    activities: tuple[ActivityVariance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_date_utc": self.data_date_utc,
            "on_schedule_band_pct": self.on_schedule_band_pct,
            "n_activities": self.n_activities,
            "n_on_schedule": self.n_on_schedule,
            "n_ahead": self.n_ahead,
            "n_behind": self.n_behind,
            "n_unknown_evidence": self.n_unknown_evidence,
            "overall_planned_percent_complete": self.overall_planned_percent_complete,
            "overall_actual_percent_complete": self.overall_actual_percent_complete,
            "overall_actual_lower_95": self.overall_actual_lower_95,
            "overall_actual_upper_95": self.overall_actual_upper_95,
            "overall_schedule_variance_percent": self.overall_schedule_variance_percent,
            "activities": [a.to_dict() for a in self.activities],
        }


@dataclass(frozen=True)
class DashboardSummary:
    """Exactly the JSON the dashboard ``ScheduleCompare`` page consumes."""

    schema_version: str
    data_date_utc: str
    kpi_planned_percent: float
    kpi_actual_percent: float
    kpi_actual_lower_95: float
    kpi_actual_upper_95: float
    kpi_variance_percent: float
    kpi_n_activities: int
    kpi_n_on_schedule: int
    kpi_n_behind: int
    kpi_n_ahead: int
    kpi_n_unknown_evidence: int
    activities: tuple[ActivityVariance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_date_utc": self.data_date_utc,
            "kpi": {
                "planned_percent": self.kpi_planned_percent,
                "actual_percent": self.kpi_actual_percent,
                "actual_lower_95": self.kpi_actual_lower_95,
                "actual_upper_95": self.kpi_actual_upper_95,
                "variance_percent": self.kpi_variance_percent,
                "n_activities": self.kpi_n_activities,
                "n_on_schedule": self.kpi_n_on_schedule,
                "n_behind": self.kpi_n_behind,
                "n_ahead": self.kpi_n_ahead,
                "n_unknown_evidence": self.kpi_n_unknown_evidence,
            },
            "activities": [a.to_dict() for a in self.activities],
        }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_status(
    planned_pct: float,
    actual_pct: float,
    n_evaluated_elements: int,
    *,
    on_schedule_band_pct: float = DEFAULT_ON_SCHEDULE_BAND_PCT,
) -> str:
    """Map planned + actual + evidence count to one of
    ``on_schedule`` / ``ahead`` / ``behind`` / ``unknown_evidence``.
    """
    if n_evaluated_elements <= 0:
        return "unknown_evidence"
    delta = actual_pct - planned_pct
    if delta > on_schedule_band_pct:
        return "ahead"
    if -delta > on_schedule_band_pct:
        return "behind"
    return "on_schedule"


def _confidence_from_interval_width(width_pct: float, n_eval: int) -> str:
    if n_eval <= 0:
        return "low"
    if width_pct <= 10.0:
        return "high"
    if width_pct <= 25.0:
        return "medium"
    return "low"


def _build_risk_tokens(rollup: ActivityRollup, status: str, confidence: str) -> tuple[str, ...]:
    risks: list[str] = []
    if status == "behind":
        risks.append("schedule_behind")
    if status == "unknown_evidence":
        risks.append("no_evidence_for_activity")
    if confidence == "low":
        risks.append("low_confidence_actual_percent")
    if rollup.n_mapped_elements > 0 and rollup.n_evaluated_elements == 0:
        risks.append("mapped_but_unobserved")
    if rollup.n_uncertain_elements > 0:
        risks.append(f"uncertain_elements:{rollup.n_uncertain_elements}")
    return tuple(risks)


def _utc_iso(dt: _dt.datetime | None = None) -> str:
    if dt is None:
        dt = _dt.datetime.now(_dt.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    else:
        dt = dt.astimezone(_dt.timezone.utc)
    return dt.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Build report + dashboard
# ---------------------------------------------------------------------------


def build_variance_report(
    *,
    schedule: Schedule,
    rollups: Sequence[ActivityRollup],
    data_date: _dt.datetime | None = None,
    on_schedule_band_pct: float = DEFAULT_ON_SCHEDULE_BAND_PCT,
) -> ScheduleVarianceReport:
    """Compute the per-activity + run-wide variance report.

    ``rollups`` is matched to ``schedule`` by ``activity_id``. Rollups
    whose ``activity_id`` is not in ``schedule`` are silently skipped
    (the upstream :func:`validate_mapping` is the right place to surface
    those mismatches).
    """
    when = data_date if data_date is not None else _dt.datetime.now(_dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    else:
        when = when.astimezone(_dt.timezone.utc)

    rollups_by_id: Mapping[str, ActivityRollup] = {r.activity_id: r for r in rollups}

    activities_out: list[ActivityVariance] = []
    n_on, n_ahead, n_behind, n_unknown = 0, 0, 0, 0

    sum_weighted_planned = 0.0
    sum_weighted_actual = 0.0
    weight_total = 0.0
    sum_weighted_lo = 0.0
    sum_weighted_hi = 0.0

    for activity in schedule.activities:
        rollup = rollups_by_id.get(activity.activity_id)
        if rollup is None:
            rollup = ActivityRollup(
                activity_id=activity.activity_id,
                n_mapped_elements=0,
                n_evaluated_elements=0,
                n_completed_elements=0,
                n_partial_elements=0,
                n_uncertain_elements=0,
                n_not_evidenced_elements=0,
                actual_percent_complete=0.0,
                actual_percent_complete_lower_95=0.0,
                actual_percent_complete_upper_95=100.0,
                notes=("no_mapping_for_activity",),
            )

        planned_pct = activity.planned_percent_complete_at(when)
        # If the schedule explicitly carried a planner-asserted percent,
        # prefer it. (This lets the schedule retain authority on
        # heterogeneous-rate activities.)
        if activity.percent_complete is not None:
            planned_pct = float(activity.percent_complete)

        actual = rollup.actual_percent_complete
        lo = rollup.actual_percent_complete_lower_95
        hi = rollup.actual_percent_complete_upper_95
        variance = actual - planned_pct

        status = classify_status(
            planned_pct, actual, rollup.n_evaluated_elements,
            on_schedule_band_pct=on_schedule_band_pct,
        )
        confidence = _confidence_from_interval_width(hi - lo, rollup.n_evaluated_elements)
        risks = _build_risk_tokens(rollup, status, confidence)

        if status == "on_schedule":
            n_on += 1
        elif status == "ahead":
            n_ahead += 1
        elif status == "behind":
            n_behind += 1
        else:
            n_unknown += 1

        # Weight the run-wide aggregate by the number of mapped elements;
        # activities with more elements contribute more to the overall %%.
        # An activity with no mapped elements contributes nothing.
        weight = float(rollup.n_mapped_elements)
        sum_weighted_planned += planned_pct * weight
        sum_weighted_actual += actual * weight
        sum_weighted_lo += lo * weight
        sum_weighted_hi += hi * weight
        weight_total += weight

        activities_out.append(
            ActivityVariance(
                activity_id=activity.activity_id,
                activity_name=activity.activity_name,
                planned_percent_complete=planned_pct,
                actual_percent_complete=actual,
                actual_percent_complete_lower_95=lo,
                actual_percent_complete_upper_95=hi,
                schedule_variance_percent=variance,
                status=status,
                confidence=confidence,
                n_evaluated_elements=rollup.n_evaluated_elements,
                n_mapped_elements=rollup.n_mapped_elements,
                risks=risks,
            )
        )

    if weight_total > 0:
        overall_planned = sum_weighted_planned / weight_total
        overall_actual = sum_weighted_actual / weight_total
        overall_lo = sum_weighted_lo / weight_total
        overall_hi = sum_weighted_hi / weight_total
    else:
        overall_planned = 0.0
        overall_actual = 0.0
        overall_lo = 0.0
        overall_hi = 100.0

    return ScheduleVarianceReport(
        schema_version=SCHEDULE_VARIANCE_SCHEMA_VERSION,
        data_date_utc=_utc_iso(when),
        on_schedule_band_pct=on_schedule_band_pct,
        n_activities=len(activities_out),
        n_on_schedule=n_on,
        n_ahead=n_ahead,
        n_behind=n_behind,
        n_unknown_evidence=n_unknown,
        overall_planned_percent_complete=overall_planned,
        overall_actual_percent_complete=overall_actual,
        overall_actual_lower_95=overall_lo,
        overall_actual_upper_95=overall_hi,
        overall_schedule_variance_percent=overall_actual - overall_planned,
        activities=tuple(activities_out),
    )


def build_dashboard_summary(report: ScheduleVarianceReport) -> DashboardSummary:
    """Project the variance report into the dashboard-shaped JSON."""
    return DashboardSummary(
        schema_version=DASHBOARD_SUMMARY_SCHEMA_VERSION,
        data_date_utc=report.data_date_utc,
        kpi_planned_percent=report.overall_planned_percent_complete,
        kpi_actual_percent=report.overall_actual_percent_complete,
        kpi_actual_lower_95=report.overall_actual_lower_95,
        kpi_actual_upper_95=report.overall_actual_upper_95,
        kpi_variance_percent=report.overall_schedule_variance_percent,
        kpi_n_activities=report.n_activities,
        kpi_n_on_schedule=report.n_on_schedule,
        kpi_n_behind=report.n_behind,
        kpi_n_ahead=report.n_ahead,
        kpi_n_unknown_evidence=report.n_unknown_evidence,
        activities=report.activities,
    )
