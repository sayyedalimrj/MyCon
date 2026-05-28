from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_08_bim_eval.metric_anchor_validation import validate_metric_anchor_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate metric benchmark/control-point anchor files.")
    parser.add_argument("--anchors", default="data/bim/design/metric_anchors.csv")
    parser.add_argument("--known-distances", default="data/bim/design/known_distances.csv")
    parser.add_argument("--output", default="runs/2026-04-30_site01_baseline/reports/metric_anchor_validation.json")
    parser.add_argument("--min-registration-anchors", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = validate_metric_anchor_files(
        Path(args.anchors),
        Path(args.known_distances),
        output_json=Path(args.output),
        min_registration_anchors=args.min_registration_anchors,
    )

    print(
        "METRIC_ANCHOR_VALIDATION_OK "
        f"status={result.status} "
        f"ready={str(result.ready_for_metric_alignment).lower()} "
        f"registration_anchors={result.complete_registration_anchor_count} "
        f"scale_anchors={result.complete_scale_anchor_count} "
        f"usable_known_distances={result.usable_known_distance_count} "
        f"output={args.output}"
    )

    if result.failures:
        print("failures=" + ",".join(result.failures))
    if result.warnings:
        print("warnings=" + ",".join(result.warnings))

    return 1 if args.strict and not result.ready_for_metric_alignment else 0


if __name__ == "__main__":
    raise SystemExit(main())
