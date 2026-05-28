from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_09_progress.progress_interpretation import upgrade_progress_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Add conservative scientific interpretation fields to Stage 9 outputs.")
    parser.add_argument("--element-metrics", default="data/bim/metrics/site01/element_metrics.csv")
    parser.add_argument("--activity-progress", default="data/bim/metrics/site01/activity_progress.csv")
    parser.add_argument("--output-element", default="data/bim/metrics/site01/element_metrics_interpreted.csv")
    parser.add_argument("--output-activity", default="data/bim/metrics/site01/activity_progress_interpreted.csv")
    parser.add_argument("--output-summary", default="runs/2026-04-30_site01_baseline/reports/progress_interpretation_summary.json")
    args = parser.parse_args()

    summary = upgrade_progress_outputs(
        element_metrics_csv=Path(args.element_metrics),
        activity_progress_csv=Path(args.activity_progress),
        output_element_csv=Path(args.output_element),
        output_activity_csv=Path(args.output_activity),
        output_summary_json=Path(args.output_summary),
    )

    print(
        "STAGE_09_PROGRESS_INTERPRETATION_OK "
        f"elements={summary['element_count']} "
        f"activities={summary['activity_count']} "
        f"summary={args.output_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
