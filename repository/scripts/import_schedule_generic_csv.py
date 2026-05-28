#!/usr/bin/env python3
"""Map an arbitrary vendor CSV export into the canonical schedule CSV.

Usage
-----

    python3 scripts/import_schedule_generic_csv.py \
        --input  schedules/vendor.csv \
        --output configs/schedules/site01.csv \
        --activity-id-column "Task ID" \
        --activity-name-column "Task Name" \
        --planned-start-column "Start Date" \
        --planned-finish-column "Finish Date" \
        [--wbs-column "WBS"] \
        [--percent-complete-column "% Complete"] \
        [--predecessors-column "Predecessors"] \
        [--trade-column "Trade"] \
        [--location-column "Location"] \
        [--encoding utf-8] \
        [--summary-json runs/imports/site01.json]

Date column values are passed through unchanged; if the vendor uses
locale-specific formats (e.g. ``31/04/2026``) it is the operator's
responsibility to convert them to ISO-8601 *before* running this
importer. We deliberately do not attempt locale guessing — silently
swapping day and month is a class of error not worth risking on a
construction project schedule.

Empty rows and rows missing the required activity_id / start / finish
are skipped with per-reason counters surfaced in the summary JSON.
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


def _normalise_iso_date_or_datetime(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    # Accept space separator: "2026-04-01 08:00" -> "2026-04-01T08:00".
    if " " in s and "T" not in s:
        return s.replace(" ", "T", 1)
    return s


def convert_rows(
    reader: csv.DictReader,
    *,
    activity_id_column: str,
    activity_name_column: str,
    planned_start_column: str,
    planned_finish_column: str,
    wbs_column: str | None = None,
    percent_complete_column: str | None = None,
    predecessors_column: str | None = None,
    trade_column: str | None = None,
    location_column: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    skip: dict[str, int] = {}
    seen_ids: set[str] = set()
    if reader.fieldnames is None:
        return [], {"empty_csv_no_header": 1}
    needed = {activity_id_column, activity_name_column, planned_start_column, planned_finish_column}
    missing = [c for c in needed if c not in reader.fieldnames]
    if missing:
        return [], {f"missing_required_column:{','.join(missing)}": 1}

    for row in reader:
        aid = (row.get(activity_id_column) or "").strip()
        if not aid:
            skip["missing_activity_id"] = skip.get("missing_activity_id", 0) + 1
            continue
        if aid in seen_ids:
            skip["duplicate_activity_id"] = skip.get("duplicate_activity_id", 0) + 1
            continue
        start = (row.get(planned_start_column) or "").strip()
        finish = (row.get(planned_finish_column) or "").strip()
        if not start or not finish:
            skip["missing_start_or_finish"] = skip.get("missing_start_or_finish", 0) + 1
            continue
        seen_ids.add(aid)
        rows.append(
            {
                "activity_id": aid,
                "activity_name": (row.get(activity_name_column) or "").strip(),
                "wbs_code": (row.get(wbs_column) or "").strip() if wbs_column else "",
                "planned_start_iso": _normalise_iso_date_or_datetime(start),
                "planned_finish_iso": _normalise_iso_date_or_datetime(finish),
                "percent_complete": (row.get(percent_complete_column) or "").strip()
                if percent_complete_column
                else "",
                "predecessors": (row.get(predecessors_column) or "").strip() if predecessors_column else "",
                "trade": (row.get(trade_column) or "").strip() if trade_column else "",
                "location": (row.get(location_column) or "").strip() if location_column else "",
            }
        )
    return rows, skip


def write_canonical_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "activity_id",
        "activity_name",
        "wbs_code",
        "planned_start_iso",
        "planned_finish_iso",
        "percent_complete",
        "predecessors",
        "trade",
        "location",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--activity-id-column", required=True)
    p.add_argument("--activity-name-column", required=True)
    p.add_argument("--planned-start-column", required=True)
    p.add_argument("--planned-finish-column", required=True)
    p.add_argument("--wbs-column", default=None)
    p.add_argument("--percent-complete-column", default=None)
    p.add_argument("--predecessors-column", default=None)
    p.add_argument("--trade-column", default=None)
    p.add_argument("--location-column", default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.input.exists():
        print(f"IMPORT_FAILED: input does not exist: {args.input}", file=sys.stderr)
        return 1

    raw = args.input.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode(args.encoding)
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    import io

    reader = csv.DictReader(io.StringIO(text))
    rows, skip = convert_rows(
        reader,
        activity_id_column=args.activity_id_column,
        activity_name_column=args.activity_name_column,
        planned_start_column=args.planned_start_column,
        planned_finish_column=args.planned_finish_column,
        wbs_column=args.wbs_column,
        percent_complete_column=args.percent_complete_column,
        predecessors_column=args.predecessors_column,
        trade_column=args.trade_column,
        location_column=args.location_column,
    )
    write_canonical_csv(rows, args.output)

    summary = {
        "schema_version": "schedule_import_summary.v1",
        "importer": "generic_csv",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(args.input.resolve()),
        "input_sha256": sha,
        "input_bytes": len(raw),
        "output_path": str(args.output.resolve()),
        "n_tasks_kept": len(rows),
        "n_tasks_skipped": sum(skip.values()),
        "skip_reasons": [list(p) for p in sorted(skip.items())],
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "IMPORT_OK importer=generic_csv "
        f"kept={summary['n_tasks_kept']} skipped={summary['n_tasks_skipped']} "
        f"out={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
