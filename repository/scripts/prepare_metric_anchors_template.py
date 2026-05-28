from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_08_bim_eval.metric_anchor_validation import prepare_metric_anchor_template


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a working metric anchor CSV template.")
    parser.add_argument("--source", default="data/bim/design/metric_anchors.csv")
    parser.add_argument("--output", default="data/bim/design/metric_anchors_working.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = prepare_metric_anchor_template(Path(args.source), Path(args.output), force=args.force)

    print("METRIC_ANCHOR_TEMPLATE_OK")
    print(f"source={args.source}")
    print(f"output={out}")
    print("Fill scan_x, scan_y, scan_z for at least three anchors before running metric alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
