from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.stage_08_bim_eval.metric_alignment_quality import enrich_metric_alignment_report_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate/enrich Stage 8 metric alignment quality report.")
    parser.add_argument(
        "--report",
        default="runs/2026-04-30_site01_baseline/reports/metric_alignment_report.json",
        help="Metric alignment report JSON path.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if quality gate fails.")
    args = parser.parse_args()

    report = Path(args.report)
    if not report.exists():
        print(f"METRIC_ALIGNMENT_QUALITY_MISSING report={report}")
        return 1 if args.strict else 0

    enriched = enrich_metric_alignment_report_file(report)
    gate = enriched.get("quality_gate", {})
    passed = bool(gate.get("passed", False))
    failures = gate.get("failures", []) or []

    print(
        "METRIC_ALIGNMENT_QUALITY_OK "
        f"passed={str(passed).lower()} "
        f"confidence={enriched.get('confidence')} "
        f"report={report}"
    )

    if failures:
        print("failures=" + ",".join(str(x) for x in failures))

    return 0 if passed or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
