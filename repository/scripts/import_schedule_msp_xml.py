#!/usr/bin/env python3
"""Convert a Microsoft Project XML export into the canonical schedule CSV.

Why MS Project XML and not ``.mpp``
-----------------------------------

``.mpp`` is a closed binary format whose support is fragile across
Microsoft Project versions. Every modern MSP version exposes a stable,
documented XML serialisation through *File → Save As → XML*. We parse
that.

This script is intentionally a **side-car**: the pipeline never parses
``.mpp`` itself; it only reads the canonical CSV defined in
:mod:`pipeline.common.schedule_io`. Operators run this importer on the
side, commit the resulting CSV, and the pipeline picks it up.

What we extract
---------------

For each ``<Task>`` element we read:

- ``UID`` or ``ID`` -> ``activity_id``
- ``Name`` -> ``activity_name``
- ``OutlineNumber`` or ``WBS`` -> ``wbs_code``
- ``Start`` -> ``planned_start_iso``
- ``Finish`` -> ``planned_finish_iso``
- ``PercentComplete`` -> ``percent_complete``
- ``PredecessorLink``/``PredecessorUID`` -> comma-joined ``predecessors``

Trade and location are not first-class MSP concepts; we leave both
columns blank (the operator can hand-fill them in the CSV).

Provenance
----------

The importer prints a one-line summary to stdout and, when invoked with
``--summary-json``, also writes a machine-readable summary recording
``input_path``, ``input_sha256``, ``n_tasks_total``, ``n_tasks_kept``,
and per-skip-reason counts. This pairs naturally with the existing
provenance discipline of the pipeline.

Usage
-----

    python3 scripts/import_schedule_msp_xml.py \
        --input  schedules/site01.xml \
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
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# MSP XML uses the http://schemas.microsoft.com/project namespace.
MSP_NS = "http://schemas.microsoft.com/project"


def _strip_ns(tag: str) -> str:
    """Drop ``{namespace}`` prefixes from an element tag name."""
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _child_text(elem: ET.Element, name: str) -> str:
    """Find a direct child whose local name is ``name`` and return its
    stripped text. Returns the empty string if the child is missing or
    empty. Namespace-agnostic so it works on both default-namespaced and
    bare exports."""
    for child in elem:
        if _strip_ns(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _format_iso_datetime(raw: str) -> str:
    """Normalise an MSP datetime string to ISO-8601 UTC.

    MSP exports timestamps in local time without an offset
    (``2026-04-01T08:00:00``). We **do not** attempt to guess the project
    time zone — instead we keep the wall-clock and emit it without an
    offset. :func:`pipeline.common.schedule_io.parse_iso_datetime` then
    treats it as UTC, which is the documented behaviour for naive inputs.
    """
    if not raw:
        return ""
    s = raw.strip()
    # Some exports use ``YYYY-MM-DD`` for date-only fields; pass through.
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    # Strip a trailing timezone if present; keep the wall-clock.
    if s.endswith("Z"):
        return s[:-1]
    if "+" in s[10:]:
        return s.split("+", 1)[0]
    return s


def _extract_predecessors(task: ET.Element) -> list[str]:
    """Collect predecessor UIDs from one task element.

    MSP exports predecessor links as nested ``<PredecessorLink>``
    elements with a ``<PredecessorUID>`` child. We deduplicate and keep
    insertion order.
    """
    found: list[str] = []
    seen: set[str] = set()
    for child in task:
        if _strip_ns(child.tag) != "PredecessorLink":
            continue
        for sub in child:
            if _strip_ns(sub.tag) == "PredecessorUID":
                uid = (sub.text or "").strip()
                if uid and uid not in seen:
                    seen.add(uid)
                    found.append(uid)
    return found


def _iter_tasks(root: ET.Element):
    """Yield every Task element regardless of namespace placement.

    Some exports nest tasks under ``<Tasks>``, some under
    ``<Project>/<Tasks>``; we walk depth-first and yield any element
    whose local tag is ``Task``.
    """
    for elem in root.iter():
        if _strip_ns(elem.tag) == "Task":
            yield elem


def convert_xml_to_canonical_rows(
    xml_text: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Parse the XML text and return ``(rows, skip_counts)``.

    Skip counts are returned as a side-channel rather than logged so the
    caller (CLI, tests) can decide what to do with them.
    """
    rows: list[dict[str, str]] = []
    skip_counts: dict[str, int] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        skip_counts["xml_parse_error"] = 1
        # Re-raise so the CLI can produce a clear non-zero exit.
        raise ValueError(f"could not parse MSP XML: {exc}") from exc

    seen_ids: set[str] = set()
    for task in _iter_tasks(root):
        # MSP marks the project summary task with UID=0 and no Start /
        # Finish; skip it so it doesn't pollute the CSV.
        uid = _child_text(task, "UID") or _child_text(task, "ID")
        if not uid or uid == "0":
            skip_counts["summary_or_blank_uid"] = skip_counts.get("summary_or_blank_uid", 0) + 1
            continue

        if uid in seen_ids:
            skip_counts["duplicate_uid"] = skip_counts.get("duplicate_uid", 0) + 1
            continue

        start_raw = _child_text(task, "Start")
        finish_raw = _child_text(task, "Finish")
        if not start_raw or not finish_raw:
            skip_counts["missing_start_or_finish"] = skip_counts.get("missing_start_or_finish", 0) + 1
            continue

        seen_ids.add(uid)
        rows.append(
            {
                "activity_id": uid,
                "activity_name": _child_text(task, "Name"),
                "wbs_code": _child_text(task, "OutlineNumber") or _child_text(task, "WBS"),
                "planned_start_iso": _format_iso_datetime(start_raw),
                "planned_finish_iso": _format_iso_datetime(finish_raw),
                "percent_complete": _child_text(task, "PercentComplete"),
                "predecessors": ",".join(_extract_predecessors(task)),
                "trade": "",
                "location": "",
            }
        )
    return rows, skip_counts


def write_canonical_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    """Write the canonical schedule CSV with a stable column order."""
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="MS Project XML export")
    p.add_argument("--output", required=True, type=Path, help="Canonical schedule CSV")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional importer summary JSON")
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

    try:
        rows, skip_counts = convert_xml_to_canonical_rows(text)
    except ValueError as exc:
        print(f"IMPORT_FAILED: {exc}", file=sys.stderr)
        return 2

    write_canonical_csv(rows, out_path)

    summary = {
        "schema_version": "schedule_import_summary.v1",
        "importer": "msp_xml",
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(in_path.resolve()),
        "input_sha256": sha,
        "input_bytes": len(raw),
        "output_path": str(out_path.resolve()),
        "n_tasks_kept": len(rows),
        "n_tasks_skipped": sum(skip_counts.values()),
        "skip_reasons": [list(p) for p in sorted(skip_counts.items())],
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "IMPORT_OK importer=msp_xml "
        f"kept={summary['n_tasks_kept']} skipped={summary['n_tasks_skipped']} "
        f"out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
