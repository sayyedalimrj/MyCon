#!/usr/bin/env python3
"""Convert a Primavera P6 ``.xer`` export into the canonical schedule CSV.

Why XER and not the P6 XML
--------------------------

P6's XML export is heavyweight (thousands of fields per project) and
its XSD changes between versions. The flat-text ``.xer`` format is
remarkably stable: it has been the preferred interchange format for over
a decade, and *every* P6 install can produce it. We parse that.

XER format (1-line summary)
---------------------------

XER files are tab-separated tables prefixed with control records:

    %T <TableName>
    %F <field1> <field2> ...        # column header for the table
    %R <value1> <value2> ...        # rows
    %T <NextTable>
    ...

We only need the ``TASK`` and ``TASKPRED`` tables; everything else is
ignored. Field names are stable across P6 versions:

- ``task_code``        -> ``activity_id``
- ``task_name``        -> ``activity_name``
- ``wbs_id``           -> ``wbs_code``  (left as the WBS code; if you
                                          want hierarchical WBS, run the
                                          XER through P6's own export
                                          and use the XML importer)
- ``early_start_date`` -> ``planned_start_iso``
- ``early_end_date``   -> ``planned_finish_iso``
- ``phys_complete_pct`` (preferred) or ``act_complete_pct``
                       -> ``percent_complete``

``TASKPRED`` rows give predecessor links keyed by ``task_id`` and
``pred_task_id``; we resolve both back to ``task_code`` so the
predecessors column references the human-readable IDs.

Trade and location are not first-class P6 concepts; we leave both
columns blank.

Usage
-----

    python3 scripts/import_schedule_p6_xer.py \
        --input  schedules/site01.xer \
        --output configs/schedules/site01.csv \
        [--summary-json runs/imports/site01.json]
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_xer_tables(text: str) -> dict[str, list[dict[str, str]]]:
    """Walk the .xer text and return ``{table_name: [{col: val, ...}, ...]}``."""
    tables: dict[str, list[dict[str, str]]] = {}
    current_table: str | None = None
    current_fields: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        if line.startswith("%T"):
            parts = line.split("\t", 1)
            current_table = parts[1].strip() if len(parts) > 1 else None
            current_fields = None
            if current_table:
                tables.setdefault(current_table, [])
        elif line.startswith("%F"):
            if current_table is None:
                continue
            current_fields = [f.strip() for f in line.split("\t")[1:]]
        elif line.startswith("%R"):
            if current_table is None or current_fields is None:
                continue
            values = line.split("\t")[1:]
            row = {f: (v if i < len(values) else "") for i, (f, v) in enumerate(zip(current_fields, values))}
            tables[current_table].append(row)
        # Other %... lines (header, end, etc.) are intentionally ignored.
    return tables


def _format_iso(raw: str) -> str:
    """P6 timestamps are ``YYYY-MM-DD HH:MM`` or just ``YYYY-MM-DD``.
    Convert to ISO-8601 with a 'T' separator; preserve naive form."""
    if not raw:
        return ""
    s = raw.strip()
    if " " in s:
        return s.replace(" ", "T")
    return s


def convert_xer_to_canonical_rows(
    xer_text: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return ``(rows, skip_counts)`` from a parsed XER text."""
    tables = _parse_xer_tables(xer_text)
    skip: dict[str, int] = {}
    if "TASK" not in tables:
        skip["no_task_table"] = 1
        return [], skip

    # Build task_id -> task_code map for predecessor resolution.
    by_internal_id: dict[str, str] = {}
    for task in tables["TASK"]:
        tid = task.get("task_id", "").strip()
        code = task.get("task_code", "").strip()
        if tid and code:
            by_internal_id[tid] = code

    # predecessors[task_code] = [pred_code, ...]
    predecessors: dict[str, list[str]] = {}
    for link in tables.get("TASKPRED", []):
        succ_id = link.get("task_id", "").strip()
        pred_id = link.get("pred_task_id", "").strip()
        succ_code = by_internal_id.get(succ_id)
        pred_code = by_internal_id.get(pred_id)
        if not succ_code or not pred_code:
            continue
        bucket = predecessors.setdefault(succ_code, [])
        if pred_code not in bucket:
            bucket.append(pred_code)

    rows: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for task in tables["TASK"]:
        code = task.get("task_code", "").strip()
        if not code:
            skip["missing_task_code"] = skip.get("missing_task_code", 0) + 1
            continue
        if code in seen_codes:
            skip["duplicate_task_code"] = skip.get("duplicate_task_code", 0) + 1
            continue
        start = task.get("early_start_date") or task.get("target_start_date") or ""
        finish = task.get("early_end_date") or task.get("target_end_date") or ""
        if not start.strip() or not finish.strip():
            skip["missing_start_or_finish"] = skip.get("missing_start_or_finish", 0) + 1
            continue
        seen_codes.add(code)
        pct = task.get("phys_complete_pct") or task.get("act_complete_pct") or ""
        rows.append(
            {
                "activity_id": code,
                "activity_name": task.get("task_name", "").strip(),
                "wbs_code": task.get("wbs_id", "").strip(),
                "planned_start_iso": _format_iso(start),
                "planned_finish_iso": _format_iso(finish),
                "percent_complete": pct.strip(),
                "predecessors": ",".join(predecessors.get(code, [])),
                "trade": "",
                "location": "",
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
    p.add_argument("--summary-json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.exists():
        print(f"IMPORT_FAILED: input does not exist: {in_path}", file=sys.stderr)
        return 1

    raw = in_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    rows, skip = convert_xer_to_canonical_rows(text)
    write_canonical_csv(rows, out_path)

    summary = {
        "schema_version": "schedule_import_summary.v1",
        "importer": "p6_xer",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(in_path.resolve()),
        "input_sha256": sha,
        "input_bytes": len(raw),
        "output_path": str(out_path.resolve()),
        "n_tasks_kept": len(rows),
        "n_tasks_skipped": sum(skip.values()),
        "skip_reasons": [list(p) for p in sorted(skip.items())],
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "IMPORT_OK importer=p6_xer "
        f"kept={summary['n_tasks_kept']} skipped={summary['n_tasks_skipped']} "
        f"out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
