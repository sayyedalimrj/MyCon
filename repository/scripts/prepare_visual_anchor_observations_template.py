from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FIELDS = [
    "anchor_id",
    "observation_index",
    "image_name",
    "u",
    "v",
    "confidence",
    "source",
    "source_candidate_id",
    "notes",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _candidate_count(path: Path) -> int:
    return len(_read_rows(path))


def prepare_template(
    metric_anchors_csv: Path,
    output_csv: Path,
    *,
    observations_per_anchor: int,
    force: bool,
) -> Path:
    if not metric_anchors_csv.exists():
        raise FileNotFoundError(f"Missing metric anchors CSV: {metric_anchors_csv}")

    if output_csv.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {output_csv}")

    anchors = _read_rows(metric_anchors_csv)
    anchor_ids = []
    for row in anchors:
        anchor_id = str(row.get("anchor_id", "")).strip()
        if anchor_id and anchor_id not in anchor_ids:
            anchor_ids.append(anchor_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for anchor_id in anchor_ids:
        for idx in range(1, observations_per_anchor + 1):
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "observation_index": str(idx),
                    "image_name": "",
                    "u": "",
                    "v": "",
                    "confidence": "manual",
                    "source": "manual_or_edge_candidate",
                    "source_candidate_id": "",
                    "notes": "Fill image_name,u,v for the same physical BIM anchor observed in this frame.",
                }
            )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare visual anchor observation CSV template.")
    parser.add_argument("--metric-anchors", default="data/bim/design/metric_anchors.csv")
    parser.add_argument("--edge-candidates", default="data/bim/design/edge_anchor_candidates.csv")
    parser.add_argument("--output", default="data/bim/design/visual_anchor_observations_template.csv")
    parser.add_argument("--observations-per-anchor", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = prepare_template(
        Path(args.metric_anchors),
        Path(args.output),
        observations_per_anchor=args.observations_per_anchor,
        force=args.force,
    )

    print("VISUAL_ANCHOR_OBSERVATION_TEMPLATE_OK")
    print(f"metric_anchors={args.metric_anchors}")
    print(f"edge_candidates={args.edge_candidates}")
    print(f"edge_candidate_count={_candidate_count(Path(args.edge_candidates))}")
    print(f"output={out}")
    print("Fill at least 2 observations for at least 3 anchors before triangulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
