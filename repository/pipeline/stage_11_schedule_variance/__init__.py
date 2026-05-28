"""Stage 11: schedule variance.

Closes the end-to-end finishing loop by joining Stage 9 per-element
results with the canonical schedule (Phase 4
:mod:`pipeline.common.schedule_io`) and the BIM<->schedule mapping
(:mod:`pipeline.common.bim_schedule_mapping`), and produces:

- ``activity_progress.json`` — per-activity planned vs actual %% with
  Wilson 95 %% confidence intervals and explicit risk tokens.
- ``schedule_variance.json`` — run-wide variance summary suitable for
  the dashboard ``ScheduleCompare`` page (see
  ``docs/end_to_end_finishing_plan.md``).
- ``dashboard_summary.json`` — exactly the JSON the GUI renders in one
  shot.

This stage is intentionally thin: it consumes structured outputs from
upstream and produces structured outputs downstream; it does no
geometry. That's why it lives outside the heavy-deps stages.
"""

from __future__ import annotations

from pipeline.stage_11_schedule_variance.activity_rollup import (  # noqa: F401
    ActivityRollup,
    rollup_activities,
)
from pipeline.stage_11_schedule_variance.variance_metrics import (  # noqa: F401
    DASHBOARD_SUMMARY_SCHEMA_VERSION,
    DEFAULT_ON_SCHEDULE_BAND_PCT,
    SCHEDULE_VARIANCE_SCHEMA_VERSION,
    ActivityVariance,
    DashboardSummary,
    ScheduleVarianceReport,
    build_dashboard_summary,
    build_variance_report,
    classify_status,
)
