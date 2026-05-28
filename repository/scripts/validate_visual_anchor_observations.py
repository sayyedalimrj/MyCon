from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



REQUIRED_FIELDS = ["anchor_id", "image_name", "u", "v"]


def _read_colmap_image_names(images_txt: Path) -> set[str]:
    """Read image names directly from COLMAP images.txt.

    COLMAP image lines have:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME

    The following line contains POINTS2D and is ignored.
    """
    names: set[str] = set()
    if not images_txt.exists():
        return names

    lines = images_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        parts = raw.split()
        if len(parts) < 10:
            continue

        try:
            int(parts[0])
            [float(x) for x in parts[1:8]]
            int(parts[8])
        except ValueError:
            continue

        names.add(" ".join(parts[9:]))

    return names




def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_ok(value: Any) -> bool:
    try:
        if value is None or str(value).strip() == "":
            return False
        float(value)
        return True
    except ValueError:
        return False


def validate_visual_anchor_observations(
    observations_csv: Path,
    images_txt: Path,
    output_json: Path | None = None,
    *,
    min_anchors: int = 3,
    min_observations_per_anchor: int = 2,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    if not observations_csv.exists():
        failures.append(f"missing_observations_csv:{observations_csv}")
        rows: list[dict[str, str]] = []
    else:
        rows = _read_rows(observations_csv)

    if not images_txt.exists():
        failures.append(f"missing_colmap_images_txt:{images_txt}")
        image_names: set[str] = set()
    else:
        image_names = _read_colmap_image_names(images_txt)

    complete_by_anchor: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown_images: list[str] = []
    incomplete_rows = 0

    for row in rows:
        anchor_id = str(row.get("anchor_id", "")).strip()
        image_name = str(row.get("image_name", "")).strip()
        u = row.get("u")
        v = row.get("v")

        complete = bool(anchor_id and image_name and _float_ok(u) and _float_ok(v))
        if not complete:
            incomplete_rows += 1
            continue

        if image_names and image_name not in image_names:
            unknown_images.append(image_name)
            continue

        complete_by_anchor[anchor_id].append(row)

    valid_anchor_ids = sorted(
        anchor_id
        for anchor_id, obs in complete_by_anchor.items()
        if len(obs) >= min_observations_per_anchor
    )

    if len(valid_anchor_ids) < min_anchors:
        failures.append(f"insufficient_visual_anchors:{len(valid_anchor_ids)}<{min_anchors}")

    if unknown_images:
        failures.append(f"unknown_images:{len(sorted(set(unknown_images)))}")

    status = "ready" if not failures else "incomplete"

    payload = {
        "status": status,
        "ready_for_triangulation": status == "ready",
        "observation_row_count": len(rows),
        "incomplete_row_count": incomplete_rows,
        "valid_anchor_count": len(valid_anchor_ids),
        "valid_anchor_ids": valid_anchor_ids,
        "observations_per_anchor": {k: len(v) for k, v in sorted(complete_by_anchor.items())},
        "unknown_images": sorted(set(unknown_images)),
        "failures": failures,
        "warnings": warnings,
        "thresholds": {
            "min_anchors": min_anchors,
            "min_observations_per_anchor": min_observations_per_anchor,
        },
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate visual anchor observations before triangulation.")
    parser.add_argument("--observations", default="data/bim/design/visual_anchor_observations.csv")
    parser.add_argument("--images-txt", default="data/sparse_refined/site01/_work/stats/stage_04_refined_txt/images.txt")
    parser.add_argument("--output", default="runs/2026-04-30_site01_baseline/reports/visual_anchor_observation_validation.json")
    parser.add_argument("--min-anchors", type=int, default=3)
    parser.add_argument("--min-observations-per-anchor", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    result = validate_visual_anchor_observations(
        Path(args.observations),
        Path(args.images_txt),
        Path(args.output),
        min_anchors=args.min_anchors,
        min_observations_per_anchor=args.min_observations_per_anchor,
    )

    print(
        "VISUAL_ANCHOR_OBSERVATION_VALIDATION_OK "
        f"status={result['status']} "
        f"ready={str(result['ready_for_triangulation']).lower()} "
        f"valid_anchors={result['valid_anchor_count']} "
        f"rows={result['observation_row_count']} "
        f"output={args.output}"
    )

    if result["failures"]:
        print("failures=" + ",".join(result["failures"]))

    return 1 if args.strict and not result["ready_for_triangulation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
