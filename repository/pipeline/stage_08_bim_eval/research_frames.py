"""Pure, side-effect-free research frame/scale diagnostics for Stage 8.

Does not rewrite Stage 8's canonical scan_aligned.ply or transformation JSON.
All coordinates passed to these functions must be meters.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _unit_vector(vector: Any) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("Expected finite xyz vector")
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Zero vector cannot define gravity direction")
    return v / norm


def gravity_to_bim_rotation(scan_up: Any, bim_up: Any) -> np.ndarray:
    """Proper minimal rotation mapping scan up to BIM up, without reflection."""
    a, b = _unit_vector(scan_up), _unit_vector(bim_up)
    cross = np.cross(a, b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    length = float(np.linalg.norm(cross))
    if length < 1e-12:
        if dot > 0:
            return np.eye(3, dtype=np.float64)
        # Deterministic perpendicular axis for exactly opposite gravity.
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = _unit_vector(np.cross(a, basis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / length
    skew = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]], dtype=np.float64)
    theta = math.atan2(length, dot)
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * skew @ skew


def validate_scan_to_bim_transform(matrix: Any, *, allow_scale: bool = True,
                                   scale_tolerance: float = 1e-3,
                                   rigid_tolerance: float = 1e-5) -> dict:
    """Check a 4x4 similarity transform and report scale, reflection, and orthogonality.

    Matrix maps scan-metric to design IFC/BIM-world-metric coordinates. A
    determinant-positive similarity is necessary, not sufficient, evidence of
    correct georeferencing or dimensions.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("Expected finite 4x4 scan-to-BIM matrix")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=rigid_tolerance):
        raise ValueError("Invalid homogeneous transform last row")
    r_scaled = a[:3, :3]
    determinant = float(np.linalg.det(r_scaled))
    if determinant <= 0:
        raise ValueError("Degenerate or reflecting transform is not valid for BIM registration")
    scale = float(np.cbrt(determinant))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid uniform scale")
    r = r_scaled / scale
    orth_error = float(np.linalg.norm(r.T @ r - np.eye(3), ord="fro"))
    if orth_error > rigid_tolerance:
        raise ValueError(f"Nonuniform or sheared matrix: orthonormality_error={orth_error:.9g}")
    if not allow_scale and abs(scale - 1.0) > scale_tolerance:
        raise ValueError(f"Unexpected metric scale={scale:.9g}; needs measured anchors")
    return {
        "scale_scan_to_bim": scale,
        "rotation_orthogonality_error": orth_error,
        "has_reflection": False,
        "translation_m": a[:3, 3].tolist(),
        "matrix_valid_sim3": True,
        "metric_scale_independently_verified": False,
        "frame": "scan_to_design_ifc_world",
        "units": "meters_only_if_upstream_declared_and_verified",
    }


def transform_points(points: Any, matrix: Any) -> np.ndarray:
    """Apply a validated Sim3; the caller must document metric units."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite Nx3 points")
    validate_scan_to_bim_transform(matrix)
    transform = np.asarray(matrix, dtype=np.float64)
    return p @ transform[:3, :3].T + transform[:3, 3]


def colmap_camera_to_bim_w2c(colmap_world_to_camera: Any, scan_to_bim: Any) -> np.ndarray:
    """Convert a rigid COLMAP (unscaled SfM) W2C into metric BIM-world W2C.

    If BIM = s*R*SfM+t, the BIM camera center is s*R*C_sfm+t, while
    R_cam_bim = R_cam_sfm * R_bim_sfm.T. Do not compose raw homogeneous
    transforms directly: their product would make camera rotation nonrigid.
    """
    colmap = np.asarray(colmap_world_to_camera, dtype=np.float64)
    if colmap.shape != (4, 4) or not np.isfinite(colmap).all():
        raise ValueError("COLMAP W2C must be finite 4x4")
    validate_scan_to_bim_transform(colmap, allow_scale=False)
    transform_qa = validate_scan_to_bim_transform(scan_to_bim)
    sim = np.asarray(scan_to_bim, dtype=np.float64)
    scale = transform_qa["scale_scan_to_bim"]
    R_bim_sfm = sim[:3, :3] / scale
    R_cam_sfm = colmap[:3, :3]
    center_sfm = -R_cam_sfm.T @ colmap[:3, 3]
    center_bim = scale * (R_bim_sfm @ center_sfm) + sim[:3, 3]
    R_cam_bim = R_cam_sfm @ R_bim_sfm.T
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = R_cam_bim
    output[:3, 3] = -R_cam_bim @ center_bim
    validate_scan_to_bim_transform(output, allow_scale=False)
    return output
