from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_visual_anchor_observations_template import prepare_template
from scripts.validate_visual_anchor_observations import validate_visual_anchor_observations


def _write_metric_anchors(path: Path) -> None:
    rows = [
        {"anchor_id": "A", "description": "A", "bim_x_m": "0", "bim_y_m": "0", "bim_z_m": "0", "scan_x": "", "scan_y": "", "scan_z": "", "use_for_scale": "true", "use_for_registration": "true"},
        {"anchor_id": "B", "description": "B", "bim_x_m": "1", "bim_y_m": "0", "bim_z_m": "0", "scan_x": "", "scan_y": "", "scan_z": "", "use_for_scale": "true", "use_for_registration": "true"},
        {"anchor_id": "C", "description": "C", "bim_x_m": "0", "bim_y_m": "1", "bim_z_m": "0", "scan_x": "", "scan_y": "", "scan_z": "", "use_for_scale": "true", "use_for_registration": "true"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_images_txt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 0 0 0 1 frame_001.jpg",
                "",
                "2 1 0 0 0 1 0 0 1 frame_002.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_observations(path: Path, complete: bool) -> None:
    rows = []
    for anchor_id in ["A", "B", "C"]:
        rows.append({"anchor_id": anchor_id, "image_name": "frame_001.jpg", "u": "10", "v": "20"})
        rows.append({"anchor_id": anchor_id, "image_name": "frame_002.jpg", "u": "11", "v": "21"})
    if not complete:
        rows = rows[:4]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor_id", "image_name", "u", "v"])
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_visual_anchor_observation_template(tmp_path: Path) -> None:
    metric = tmp_path / "metric_anchors.csv"
    output = tmp_path / "visual_anchor_observations_template.csv"
    _write_metric_anchors(metric)

    prepare_template(metric, output, observations_per_anchor=3, force=True)

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 9
    assert rows[0]["anchor_id"] == "A"
    assert rows[0]["image_name"] == ""


def test_validate_visual_anchor_observations_ready(tmp_path: Path) -> None:
    observations = tmp_path / "visual_anchor_observations.csv"
    images_txt = tmp_path / "images.txt"
    output = tmp_path / "report.json"
    _write_observations(observations, complete=True)
    _write_images_txt(images_txt)

    result = validate_visual_anchor_observations(observations, images_txt, output)

    assert result["status"] == "ready"
    assert result["ready_for_triangulation"] is True
    assert result["valid_anchor_count"] == 3
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready"


def test_validate_visual_anchor_observations_incomplete(tmp_path: Path) -> None:
    observations = tmp_path / "visual_anchor_observations.csv"
    images_txt = tmp_path / "images.txt"
    output = tmp_path / "report.json"
    _write_observations(observations, complete=False)
    _write_images_txt(images_txt)

    result = validate_visual_anchor_observations(observations, images_txt, output)

    assert result["status"] == "incomplete"
    assert result["ready_for_triangulation"] is False
    assert result["valid_anchor_count"] == 2
