
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pipeline.stage_08_bim_eval.visual_anchor_triangulation import (
    merge_picked_scan_anchors,
    triangulate_visual_anchors,
)


def _write_colmap_text_model(root: Path) -> tuple[Path, Path]:
    cameras = root / "cameras.txt"
    images = root / "images.txt"

    cameras.write_text(
        "\n".join(
            [
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]",
                "1 PINHOLE 1000 1000 100 100 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    images.write_text(
        "\n".join(
            [
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
                "1 1 0 0 0 0 0 0 1 img1.jpg",
                "",
                "2 1 0 0 0 -1 0 0 1 img2.jpg",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return cameras, images


def test_visual_anchor_triangulation_from_two_colmap_views(tmp_path: Path) -> None:
    cameras, images = _write_colmap_text_model(tmp_path)
    obs = tmp_path / "visual_anchor_observations.csv"
    picked = tmp_path / "picked_scan_anchors.csv"

    obs.write_text(
        "\n".join(
            [
                "anchor_id,image_name,u_px,v_px,confidence,method",
                "A,img1.jpg,0,0,1.0,manual_corner",
                "A,img2.jpg,-20,0,1.0,manual_corner",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results = triangulate_visual_anchors(cameras, images, obs, picked, min_views=2)
    assert len(results) == 1
    assert results[0].anchor_id == "A"
    assert results[0].status == "ok"
    assert np.allclose(results[0].scan_xyz, np.array([0.0, 0.0, 5.0]), atol=1e-6)

    rows = list(csv.DictReader(picked.open("r", encoding="utf-8", newline="")))
    assert rows[0]["anchor_id"] == "A"
    assert float(rows[0]["scan_z_m"]) == 5.0


def test_merge_picked_scan_anchors_into_metric_template(tmp_path: Path) -> None:
    template = tmp_path / "metric_anchors.csv"
    picked = tmp_path / "picked_scan_anchors.csv"
    output = tmp_path / "metric_anchors_working.csv"

    template.write_text(
        "anchor_id,label,bim_x_m,bim_y_m,bim_z_m\nA,column_edge,0,0,0\nB,wall_corner,1,0,0\n",
        encoding="utf-8",
    )
    picked.write_text(
        "anchor_id,scan_x_m,scan_y_m,scan_z_m,view_count,mean_reprojection_error_px,status,source\n"
        "A,10,20,30,3,2.5,ok,visual_anchor_triangulation\n",
        encoding="utf-8",
    )

    merged = merge_picked_scan_anchors(template, picked, output)
    assert merged == 1

    rows = list(csv.DictReader(output.open("r", encoding="utf-8", newline="")))
    assert rows[0]["scan_x_m"] == "10"
    assert rows[0]["scan_anchor_status"] == "ok"
    assert rows[1]["scan_x_m"] == ""
