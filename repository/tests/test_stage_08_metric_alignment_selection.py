from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.stage_08_bim_eval.metric_alignment_selection import (
    attach_metric_alignment_selection,
    estimate_full_sim3,
    extract_anchor_correspondences,
)


@dataclass
class Anchor:
    anchor_id: str
    scan_xyz: list[float]
    bim_xyz: list[float]


def _make_anchors() -> list[Anchor]:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    scale = 2.0
    translation = np.asarray([10.0, -3.0, 0.5], dtype=float)
    target = scale * source + translation

    anchors = [
        Anchor(f"A{i}", scan_xyz=source[i].tolist(), bim_xyz=target[i].tolist())
        for i in range(len(source))
    ]

    # Add one wrong/outlier anchor.
    anchors.append(
        Anchor(
            "BAD",
            scan_xyz=[3.0, 3.0, 3.0],
            bim_xyz=[100.0, 100.0, 100.0],
        )
    )

    return anchors


def test_extract_anchor_correspondences_from_dataclasses() -> None:
    ids, scan, bim = extract_anchor_correspondences(_make_anchors())

    assert ids[0] == "A0"
    assert scan.shape == (6, 3)
    assert bim.shape == (6, 3)


def test_full_sim3_recovers_known_transform_without_outlier() -> None:
    anchors = _make_anchors()[:5]
    _, scan, bim = extract_anchor_correspondences(anchors)

    transform = estimate_full_sim3(scan, bim)

    assert abs(transform.scale - 2.0) < 1e-8
    assert transform.residual_max_m < 1e-8


def test_attach_metric_alignment_selection_prefers_ransac_with_outlier() -> None:
    report = {
        "status": "ok",
        "quality_gate": {
            "thresholds": {
                "residual_fail_m": 0.15,
            }
        },
    }

    updated = attach_metric_alignment_selection(report, _make_anchors())
    selection = updated["metric_alignment_selection"]

    assert selection["status"] == "ok"
    assert selection["selected_solution"] == "ransac_sim3"
    assert selection["ransac_sim3"]["inlier_count"] >= 5
    assert "BAD" in selection["recommended_outlier_anchor_ids"]
    assert updated["selected_metric_solution"] == "ransac_sim3"
    assert updated["recommended_transform_scan_to_bim"]["residual_max_m"] < 1e-8
    assert len(updated["matrix4x4"]) == 4
