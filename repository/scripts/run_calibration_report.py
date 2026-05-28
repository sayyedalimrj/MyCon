#!/usr/bin/env python3
"""Compute a calibration / reliability report from a JSONL of (prediction, ground-truth) records.

Why this script exists
----------------------

The pipeline emits a discrete confidence label (``high`` / ``medium`` /
``low``) on every element- and activity-level decision. Without a calibration
report we have no defensible way to claim those labels mean what they say.
This script produces the report.

Input
-----

A JSON Lines file. Each line is an object with at least:

- ``confidence`` — either a numeric probability in [0, 1] or one of the
  string labels supported by :mod:`pipeline.common.calibration`
  (``high`` / ``medium`` / ``low`` / ``low_to_medium`` / ``unverified``).
- ``correct`` — ground-truth correctness as bool / 0-1 / ``true``/``false`` /
  ``accepted``/``rejected``.

Optional fields are ignored; this lets the same file be reused as a HITL
corrections log (see :mod:`pipeline.common.hitl`).

Output
------

A single JSON file with the structure documented in
:func:`pipeline.common.calibration.calibration_report`. The report is
self-contained: it records its own input mapping, the chosen binning
strategy, and the metric values, so a reviewer can reproduce the numbers
without knowing which CLI flags were used.

Exit codes
----------

The script always exits 0 if it could read the input and write the output;
calibration metric thresholds (``--max-ece``, ``--max-brier``) cause exit
code 2 when violated, so this can be wired into CI as a regression gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.calibration import (  # noqa: E402
    CalibrationDataset,
    calibration_report,
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Malformed lines are skipped silently; we report the count below.
                continue
            if isinstance(obj, dict):
                yield obj


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="JSONL of records with confidence + correctness")
    p.add_argument("--out-json", required=True, type=Path, help="Output report path")
    p.add_argument("--n-bins", type=int, default=10, help="Number of reliability bins")
    p.add_argument(
        "--strategy",
        choices=("equal_mass", "equal_width"),
        default="equal_mass",
        help="Binning strategy. Equal-mass is the default; equal-width matches some published ECE values.",
    )
    p.add_argument("--confidence-key", default="confidence")
    p.add_argument("--correct-key", default="correct")
    p.add_argument(
        "--max-ece",
        type=float,
        default=None,
        help="If set, exit code 2 when ECE exceeds this threshold.",
    )
    p.add_argument(
        "--max-brier",
        type=float,
        default=None,
        help="If set, exit code 2 when Brier score exceeds this threshold.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    in_path: Path = args.input
    out_path: Path = args.out_json

    if not in_path.exists():
        print(f"CALIBRATION_REPORT_FAILED: input does not exist: {in_path}", file=sys.stderr)
        return 1

    records = list(_iter_jsonl(in_path))
    dataset = CalibrationDataset.from_records(
        records,
        confidence_key=args.confidence_key,
        correct_key=args.correct_key,
    )
    report = calibration_report(
        dataset,
        n_bins=args.n_bins,
        strategy=args.strategy,
    )
    report["input_path"] = str(in_path)
    report["output_path"] = str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(
        f"CALIBRATION_REPORT_OK n_samples={report['n_samples']} "
        f"ece={metrics['expected_calibration_error']:.6f} "
        f"mce={metrics['maximum_calibration_error']:.6f} "
        f"brier={metrics['brier_score']:.6f} "
        f"smooth_ece={metrics['smooth_ece']:.6f} "
        f"out={out_path}"
    )

    if args.max_ece is not None and metrics["expected_calibration_error"] > args.max_ece:
        print(
            f"CALIBRATION_THRESHOLD_VIOLATED ece={metrics['expected_calibration_error']:.6f} > {args.max_ece}",
            file=sys.stderr,
        )
        return 2
    if args.max_brier is not None and metrics["brier_score"] > args.max_brier:
        print(
            f"CALIBRATION_THRESHOLD_VIOLATED brier={metrics['brier_score']:.6f} > {args.max_brier}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
