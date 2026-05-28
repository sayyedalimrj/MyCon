from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_colmap_image_names(images_txt: Path, limit: int) -> list[str]:
    if not images_txt.exists():
        return []

    names: list[str] = []
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
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

        names.append(" ".join(parts[9:]))
        if len(names) >= limit:
            break

    return names


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a lightweight packet for manual visual anchor picking.")
    parser.add_argument("--metric-anchors", default="data/bim/design/metric_anchors.csv")
    parser.add_argument("--edge-candidates", default="data/bim/design/edge_anchor_candidates.csv")
    parser.add_argument("--images-txt", default="data/sparse_refined/site01/_work/stats/stage_04_refined_txt/images.txt")
    parser.add_argument("--output-dir", default="data/bim/design/visual_anchor_picking_packet")
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors = _read_csv(Path(args.metric_anchors))
    candidates = _read_csv(Path(args.edge_candidates))
    images = _read_colmap_image_names(Path(args.images_txt), limit=args.max_images)

    anchor_rows = []
    for row in anchors:
        anchor_rows.append({
            "anchor_id": row.get("anchor_id", ""),
            "description": row.get("description", row.get("name", "")),
            "bim_x_m": row.get("bim_x_m", row.get("bim_x", "")),
            "bim_y_m": row.get("bim_y_m", row.get("bim_y", "")),
            "bim_z_m": row.get("bim_z_m", row.get("bim_z", "")),
            "recommended_pick_priority": "",
            "notes": "Pick stable visible corners: column corner, wall-slab corner, opening corner, slab edge intersection.",
        })

    candidate_rows = candidates[: args.max_candidates]

    image_rows = [{"image_name": name, "pick_priority": "", "notes": ""} for name in images]

    _write_csv(
        out_dir / "anchor_pick_list.csv",
        anchor_rows,
        ["anchor_id", "description", "bim_x_m", "bim_y_m", "bim_z_m", "recommended_pick_priority", "notes"],
    )

    if candidate_rows:
        _write_csv(out_dir / "edge_candidates_preview.csv", candidate_rows, list(candidate_rows[0].keys()))
    else:
        _write_csv(out_dir / "edge_candidates_preview.csv", [], ["candidate_id", "image_name", "u1", "v1", "u2", "v2"])

    _write_csv(out_dir / "registered_images_preview.csv", image_rows, ["image_name", "pick_priority", "notes"])

    instructions = {
        "status": "prepared",
        "purpose": "Use this packet to manually select visual anchors before triangulation.",
        "next_file_to_create": "data/bim/design/visual_anchor_observations.csv",
        "minimum_requirement": "At least 3 anchors, each observed in at least 2 registered COLMAP images. Prefer 3 to 5 images per anchor.",
        "files": {
            "anchor_pick_list": str(out_dir / "anchor_pick_list.csv"),
            "edge_candidates_preview": str(out_dir / "edge_candidates_preview.csv"),
            "registered_images_preview": str(out_dir / "registered_images_preview.csv"),
        },
        "counts": {
            "anchors": len(anchor_rows),
            "edge_candidates_total": len(candidates),
            "edge_candidates_preview": len(candidate_rows),
            "registered_images_preview": len(image_rows),
        },
    }

    (out_dir / "README_visual_anchor_picking.json").write_text(
        json.dumps(instructions, indent=2),
        encoding="utf-8",
    )

    print("VISUAL_ANCHOR_PICKING_PACKET_OK")
    print(f"output_dir={out_dir}")
    print(f"anchors={len(anchor_rows)}")
    print(f"registered_images_preview={len(image_rows)}")
    print(f"edge_candidates_total={len(candidates)}")
    print(f"edge_candidates_preview={len(candidate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
