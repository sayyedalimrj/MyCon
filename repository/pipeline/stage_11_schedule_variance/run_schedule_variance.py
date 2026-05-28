"""Stage 11 CLI runner.

Reads:

  - ``--schedule-csv``           canonical schedule CSV (see
                                  :mod:`pipeline.common.schedule_io`).
  - ``--mapping-csv``            BIM <-> schedule mapping CSV (see
                                  :mod:`pipeline.common.bim_schedule_mapping`).
  - ``--element-metrics-csv``    Stage 9 ``element_metrics_csv`` output.

Writes:

  - ``--activity-progress-json`` per-activity rollup with Wilson 95 %% intervals.
  - ``--schedule-variance-json`` run-wide schedule-variance report.
  - ``--dashboard-summary-json`` dashboard-shaped JSON summary.

Optionally accepts ``--data-date-utc YYYY-MM-DD`` to control the date at
which planned vs actual is compared. Defaults to ``now()`` UTC.

Provenance
----------

Every output JSON carries a ``provenance`` block recording the input
file paths, sha-256 hashes, and timestamp. This makes the artefacts
auditable end-to-end without depending on external metadata.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.bim_schedule_mapping import load_mapping_csv  # noqa: E402
from pipeline.common.schedule_io import (  # noqa: E402
    SCHEDULE_SCHEMA_VERSION,
    load_schedule_csv,
    parse_iso_datetime,
)
from pipeline.stage_11_schedule_variance.activity_rollup import (  # noqa: E402
    rollup_activities,
)
from pipeline.stage_11_schedule_variance.variance_metrics import (  # noqa: E402
    DEFAULT_ON_SCHEDULE_BAND_PCT,
    build_dashboard_summary,
    build_variance_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _build_provenance(
    *,
    schedule_csv: Path,
    mapping_csv: Path,
    element_metrics_csv: Path,
    data_date_utc: str,
    on_schedule_band_pct: float,
) -> dict[str, Any]:
    return {
        "stage": "stage_11_schedule_variance",
        "schema_version": "stage_11_provenance.v1",
        "generated_at_utc": _utc_iso_now(),
        "data_date_utc": data_date_utc,
        "on_schedule_band_pct": on_schedule_band_pct,
        "inputs": {
            "schedule_csv": {
                "path": str(schedule_csv.resolve()),
                "sha256": _sha256(schedule_csv),
                "bytes": schedule_csv.stat().st_size,
            },
            "mapping_csv": {
                "path": str(mapping_csv.resolve()),
                "sha256": _sha256(mapping_csv),
                "bytes": mapping_csv.stat().st_size,
            },
            "element_metrics_csv": {
                "path": str(element_metrics_csv.resolve()),
                "sha256": _sha256(element_metrics_csv),
                "bytes": element_metrics_csv.stat().st_size,
            },
        },
    }


def _read_element_metrics_csv(path: Path) -> list[dict[str, Any]]:
    """Read Stage 9's element_metrics_csv into a list of dicts.

    The function is tolerant of extra columns (forward-compat: Phase 4
    multi-view-fusion outputs may add columns alongside the legacy
    Stage 9 ones) and drops rows missing ``global_id``.
    """
    if not path.exists():
        raise FileNotFoundError(f"element metrics CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gid = (r.get("global_id") or r.get("GlobalId") or "").strip()
            if not gid:
                continue
            rows.append(dict(r))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule-csv", required=True, type=Path)
    p.add_argument("--mapping-csv", required=True, type=Path)
    p.add_argument("--element-metrics-csv", required=True, type=Path)
    p.add_argument("--activity-progress-json", required=True, type=Path)
    p.add_argument("--schedule-variance-json", required=True, type=Path)
    p.add_argument("--dashboard-summary-json", required=True, type=Path)
    p.add_argument(
        "--data-date-utc",
        default=None,
        help="ISO date or datetime (UTC). Defaults to now() UTC.",
    )
    p.add_argument(
        "--on-schedule-band-pct",
        type=float,
        default=DEFAULT_ON_SCHEDULE_BAND_PCT,
        help="Tolerance (in percentage points) inside which an activity is 'on_schedule'.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.data_date_utc:
        try:
            data_date = parse_iso_datetime(args.data_date_utc)
        except ValueError as exc:
            print(f"STAGE_11_FAILED: bad --data-date-utc: {exc}", file=sys.stderr)
            return 1
    else:
        data_date = _dt.datetime.now(_dt.timezone.utc)

    try:
        schedule = load_schedule_csv(args.schedule_csv)
        mapping = load_mapping_csv(args.mapping_csv)
        element_rows = _read_element_metrics_csv(args.element_metrics_csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"STAGE_11_FAILED: input error: {exc}", file=sys.stderr)
        return 1

    rollups = rollup_activities(element_rows, mapping=mapping)
    report = build_variance_report(
        schedule=schedule,
        rollups=rollups,
        data_date=data_date,
        on_schedule_band_pct=args.on_schedule_band_pct,
    )
    dashboard = build_dashboard_summary(report)

    provenance = _build_provenance(
        schedule_csv=args.schedule_csv,
        mapping_csv=args.mapping_csv,
        element_metrics_csv=args.element_metrics_csv,
        data_date_utc=report.data_date_utc,
        on_schedule_band_pct=args.on_schedule_band_pct,
    )

    activity_progress = {
        "schema_version": "activity_progress.v1",
        "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
        "n_activities": len(rollups),
        "rollups": [r.to_dict() for r in rollups],
        "provenance": provenance,
    }

    variance_payload = report.to_dict()
    variance_payload["provenance"] = provenance

    dashboard_payload = dashboard.to_dict()
    dashboard_payload["provenance"] = provenance

    for path, payload in (
        (args.activity_progress_json, activity_progress),
        (args.schedule_variance_json, variance_payload),
        (args.dashboard_summary_json, dashboard_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"STAGE_11_OK n_activities={report.n_activities} "
        f"on={report.n_on_schedule} ahead={report.n_ahead} "
        f"behind={report.n_behind} unknown={report.n_unknown_evidence} "
        f"overall_actual={report.overall_actual_percent_complete:.2f}%% "
        f"overall_planned={report.overall_planned_percent_complete:.2f}%% "
        f"variance={report.overall_schedule_variance_percent:+.2f}pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
