from __future__ import annotations

import numpy as np

from pipeline.stage_08_bim_eval.visual_anchor_quality import (
    VisualAnchorQualityThresholds,
    angle_between_rays_deg,
    camera_center_from_world_to_camera,
    evaluate_visual_anchor_quality,
    min_ray_angle_from_camera_centers,
    summarize_visual_anchor_quality,
)


def test_ray_angle_computation() -> None:
    angle = angle_between_rays_deg(
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    )

    assert abs(angle - 90.0) < 1e-8


def test_camera_center_from_colmap_world_to_camera_pose() -> None:
    rotation = np.eye(3)
    translation = np.asarray([-2.0, 0.0, 0.0])

    center = camera_center_from_world_to_camera(rotation, translation)

    assert np.allclose(center, [2.0, 0.0, 0.0])


def test_min_ray_angle_from_camera_centers() -> None:
    point = np.asarray([0.0, 0.0, 5.0])
    centers = [
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ]

    angle = min_ray_angle_from_camera_centers(point, centers)

    assert angle is not None
    assert angle > 1.0


def test_quality_accepts_good_anchor() -> None:
    result = evaluate_visual_anchor_quality(
        {"anchor_id": "A", "x": 1.0, "y": 2.0, "z": 3.0},
        observation_count=3,
        min_ray_angle_deg=8.0,
        reprojection_errors=[0.5, 1.0, 1.5],
    )

    assert result.accepted is True
    assert result.status == "accepted"
    assert result.failures == []


def test_quality_rejects_bad_anchor() -> None:
    result = evaluate_visual_anchor_quality(
        {"anchor_id": "B", "x": 1.0, "y": 2.0, "z": 3.0},
        observation_count=1,
        min_ray_angle_deg=0.2,
        reprojection_errors=[2.0, 15.0],
        thresholds=VisualAnchorQualityThresholds(min_observations=2),
    )

    assert result.accepted is False
    assert any(item.startswith("insufficient_observations") for item in result.failures)
    assert any(item.startswith("ray_angle_too_small") for item in result.failures)
    assert any(item.startswith("reprojection_error_too_high") for item in result.failures)


def test_quality_summary() -> None:
    good = evaluate_visual_anchor_quality(
        {"anchor_id": "A", "x": 1.0, "y": 2.0, "z": 3.0},
        observation_count=3,
        min_ray_angle_deg=8.0,
        reprojection_errors=[1.0],
    )
    bad = evaluate_visual_anchor_quality(
        {"anchor_id": "B", "x": 1.0, "y": 2.0, "z": 3.0},
        observation_count=1,
        min_ray_angle_deg=0.1,
        reprojection_errors=[20.0],
    )

    summary = summarize_visual_anchor_quality([good, bad])

    assert summary["status"] == "has_rejections"
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["accepted_anchor_ids"] == ["A"]
    assert summary["rejected_anchor_ids"] == ["B"]
