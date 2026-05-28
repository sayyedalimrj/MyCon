
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_08_bim_eval.visual_anchor_triangulation import (
    merge_picked_scan_anchors,
    triangulate_visual_anchors,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triangulate visual anchor observations from COLMAP poses and merge them into metric anchors."
    )
    parser.add_argument("--cameras-txt", default="data/sparse_refined/site01/0/cameras.txt")
    parser.add_argument("--images-txt", default="data/sparse_refined/site01/0/images.txt")
    parser.add_argument("--observations-csv", default="data/bim/design/visual_anchor_observations.csv")
    parser.add_argument("--metric-anchors-template", default="data/bim/design/metric_anchors.csv")
    parser.add_argument("--picked-output", default="data/bim/design/picked_scan_anchors.csv")
    parser.add_argument("--merged-output", default="data/bim/design/metric_anchors_working.csv")
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--max-mean-reprojection-error-px", type=float, default=12.0)
    args = parser.parse_args()

    results = triangulate_visual_anchors(
        cameras_txt=PROJECT_ROOT / args.cameras_txt,
        images_txt=PROJECT_ROOT / args.images_txt,
        observations_csv=PROJECT_ROOT / args.observations_csv,
        output_picked_csv=PROJECT_ROOT / args.picked_output,
        min_views=args.min_views,
        max_mean_reprojection_error_px=args.max_mean_reprojection_error_px,
    )

    merged = merge_picked_scan_anchors(
        metric_anchors_template_csv=PROJECT_ROOT / args.metric_anchors_template,
        picked_scan_anchors_csv=PROJECT_ROOT / args.picked_output,
        output_metric_anchors_csv=PROJECT_ROOT / args.merged_output,
    )

    ok_count = sum(1 for item in results if item.status == "ok")
    print(
        "STAGE_08_VISUAL_ANCHORS_OK "
        f"triangulated={len(results)} ok={ok_count} merged={merged} "
        f"picked={args.picked_output} merged_csv={args.merged_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
