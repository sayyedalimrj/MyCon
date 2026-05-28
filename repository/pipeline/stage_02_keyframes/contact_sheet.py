"""Contact sheet generation for selected keyframes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from pipeline.common.paths import atomic_output_path, ensure_parent_dir


def create_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    *,
    thumb_width: int,
    max_images: int,
    label: bool = True,
) -> dict[str, object]:
    """Create a JPEG contact sheet from keyframe images.

    The implementation uses a two-pass layout to avoid holding every thumbnail
    in memory. Only the final bounded sheet and one thumbnail are resident at a
    time. This matters for long first-run selections while keeping the Stage 2
    dependency footprint minimal.
    """
    ensure_parent_dir(output_path)
    selected_paths = list(image_paths[: max(1, max_images)])
    if not selected_paths:
        raise ValueError("No image paths were provided for the contact sheet.")
    specs: list[tuple[Path, int]] = []
    for path in selected_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            continue
        thumb_height = max(1, int(round(height * (thumb_width / float(width)))))
        specs.append((path, thumb_height))
    if not specs:
        raise ValueError("Could not read any keyframe images for the contact sheet.")

    cell_h = max(height for _, height in specs)
    cols = int(math.ceil(math.sqrt(len(specs))))
    rows = int(math.ceil(len(specs) / cols))
    sheet = np.zeros((rows * cell_h, cols * thumb_width, 3), dtype=np.uint8)

    rendered = 0
    for idx, (path, _thumb_height) in enumerate(specs):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            continue
        thumb_height = max(1, int(round(height * (thumb_width / float(width)))))
        thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        if label:
            cv2.putText(
                thumb,
                f"{idx + 1:03d}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        row = idx // cols
        col = idx % cols
        y0 = row * cell_h
        x0 = col * thumb_width
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb
        rendered += 1

    if rendered == 0:
        raise ValueError("Could not render any keyframe thumbnails for the contact sheet.")

    with atomic_output_path(output_path) as tmp_path:
        ok = cv2.imwrite(str(tmp_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"Failed to write contact sheet: {tmp_path}")
    return {
        "contact_sheet": str(output_path),
        "image_count": rendered,
        "rows": rows,
        "cols": cols,
        "thumb_width": thumb_width,
        "memory_strategy": "two_pass_bounded",
    }
