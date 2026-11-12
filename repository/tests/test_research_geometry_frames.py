"""Synthetic unit tests only: mathematical contracts, no real-site claim."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.stage_08_bim_eval.research_frames import (
    gravity_to_bim_rotation, transform_points, validate_scan_to_bim_transform,
)


def test_gravity_rotation_is_rigid_and_aligns_up() -> None:
    scan = np.array([0.0, 1.0, 0.0])
    bim = np.array([0.0, 0.0, 1.0])
    r = gravity_to_bim_rotation(scan, bim)
    assert np.allclose(r @ scan, bim, atol=1e-8)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-8)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_opposite_gravity_is_handled_without_reflection() -> None:
    source = [0.0, 0.0, -1.0]
    target = [0.0, 0.0, 1.0]
    r = gravity_to_bim_rotation(source, target)
    assert np.allclose(r @ source, target, atol=1e-8)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_meter_transform_preserves_uniform_scale_and_translation() -> None:
    m = np.eye(4)
    m[:3, :3] *= 2.0
    m[:3, 3] = [1.0, -2.0, 3.0]
    detail = validate_scan_to_bim_transform(m)
    assert detail["scale_scan_to_bim"] == pytest.approx(2.0)
    assert detail["metric_scale_independently_verified"] is False
    points = transform_points([[1, 0, 0], [0, 0, 1]], m)
    assert np.allclose(points, [[3, -2, 3], [1, -2, 5]])
    with pytest.raises(ValueError, match="Unexpected metric scale"):
        validate_scan_to_bim_transform(m, allow_scale=False)


def test_reject_shear_reflection_and_invalid_row() -> None:
    bad = np.eye(4)
    bad[0, 1] = 0.2
    with pytest.raises(ValueError, match="Nonuniform"):
        validate_scan_to_bim_transform(bad)
    bad = np.eye(4)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="reflecting"):
        validate_scan_to_bim_transform(bad)
    bad = np.eye(4)
    bad[3, 0] = 0.1
    with pytest.raises(ValueError, match="last row"):
        validate_scan_to_bim_transform(bad)


pytestmark = pytest.mark.lightweight
