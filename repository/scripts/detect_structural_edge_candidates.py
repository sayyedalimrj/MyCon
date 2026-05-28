
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def image_files(images_dir: Path, max_images: int) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files: list[Path] = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    files = sorted(set(files))
    return files[:max_images]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect structural line/edge candidates from images for manual anchor labeling."
    )
    parser.add_argument("--images-dir", default="data/sfm/site01/images")
    parser.add_argument("--output-csv", default="data/bim/design/edge_anchor_candidates.csv")
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--max-lines-per-image", type=int, default=80)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=80)
    parser.add_argument("--min-line-length", type=int, default=80)
    parser.add_argument("--max-line-gap", type=int, default=12)
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"OpenCV/cv2 is required for edge candidate detection: {exc}")

    root = Path(__file__).resolve().parents[1]
    images_dir = root / args.images_dir
    output_csv = root / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for img_path in image_files(images_dir, args.max_images):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        edges = cv2.Canny(gray, args.canny_low, args.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180.0,
            threshold=args.hough_threshold,
            minLineLength=args.min_line_length,
            maxLineGap=args.max_line_gap,
        )
        if lines is None:
            continue

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in line]
            length = math.hypot(x2 - x1, y2 - y1)
            candidates.append((length, (x1, y1, x2, y2)))

        for idx, (length, (x1, y1, x2, y2)) in enumerate(
            sorted(candidates, reverse=True)[: args.max_lines_per_image]
        ):
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            rows.append(
                {
                    "image_name": img_path.name,
                    "candidate_id": f"{img_path.stem}_line_{idx:03d}",
                    "u1_px": x1,
                    "v1_px": y1,
                    "u2_px": x2,
                    "v2_px": y2,
                    "mid_u_px": f"{(x1 + x2) / 2.0:.3f}",
                    "mid_v_px": f"{(y1 + y2) / 2.0:.3f}",
                    "length_px": f"{length:.3f}",
                    "angle_deg": f"{angle:.3f}",
                    "method": "houghlinesp_edge_candidate",
                    "anchor_id": "",
                    "notes": "Fill anchor_id manually if this line/corner corresponds to a BIM benchmark edge.",
                }
            )

    fields = [
        "image_name",
        "candidate_id",
        "u1_px",
        "v1_px",
        "u2_px",
        "v2_px",
        "mid_u_px",
        "mid_v_px",
        "length_px",
        "angle_deg",
        "method",
        "anchor_id",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"STAGE_08_EDGE_CANDIDATES_OK candidates={len(rows)} output={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
