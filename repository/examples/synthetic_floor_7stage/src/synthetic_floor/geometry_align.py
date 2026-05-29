"""Coordinate-frame alignment for meshes imported into Blender.

Why this exists
---------------
The synthetic floor geometry is authored **Z-up** (height along +Z, the
room laid out in the XY plane: x in [0, length], y in [0, width],
z in [0, height]). ``mesh_builder`` exports it as a GLB via trimesh,
which writes the vertices **verbatim** (no axis rotation node).

Blender's glTF importer, however, assumes glTF's **Y-up** convention and
applies a fixed +90 degree rotation about X to bring the data into
Blender's Z-up world. The net effect is that our Z-up room is rotated so
its *height* lands on Blender's Y axis and its *width* lands on Blender's
Z axis. The room ends up "standing up", far outside the hard-coded
camera path / lights / window portals (which are all expressed in the
authored Z-up frame). The camera then sees almost nothing but the sky —
producing the infamous "bright scene, nothing but light" renders.

This module computes the rigid transform that maps the imported geometry
**back** into the authored Z-up frame, so the camera, the window light
portals, and the ceiling lights all line up with real geometry again.

The maths here are pure NumPy and Blender-free so they can be unit
tested on a laptop. ``blender_gpu_renderer`` builds a ``mathutils.Matrix``
from :func:`compute_alignment`'s 4x4 result and applies it to the
imported root objects.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

Vec3 = Sequence[float]


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    xs = (bmin[0], bmax[0])
    ys = (bmin[1], bmax[1])
    zs = (bmin[2], bmax[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def _rotated_extents(bmin: np.ndarray, bmax: np.ndarray, R: np.ndarray) -> np.ndarray:
    rc = _corners(bmin, bmax) @ R.T
    return rc.max(axis=0) - rc.min(axis=0)


def author_bbox_from_elements(elements_payload: dict) -> Optional[tuple[list[float], list[float]]]:
    """Compute the overall (min, max) bbox from an elements sidecar payload."""
    elements = (elements_payload or {}).get("elements") or []
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
    for e in elements:
        bmin = e.get("box_min")
        bmax = e.get("box_max")
        if bmin is None or bmax is None or len(bmin) != 3 or len(bmax) != 3:
            continue
        mins.append([float(v) for v in bmin])
        maxs.append([float(v) for v in bmax])
    if not mins:
        return None
    arr_min = np.array(mins).min(axis=0)
    arr_max = np.array(maxs).max(axis=0)
    return arr_min.tolist(), arr_max.tolist()


def compute_alignment(
    blender_min: Vec3,
    blender_max: Vec3,
    *,
    author_min: Optional[Vec3] = None,
    author_max: Optional[Vec3] = None,
    tol_ratio: float = 0.15,
) -> dict:
    """Return the rigid transform that maps imported geometry to the authored frame.

    Parameters
    ----------
    blender_min, blender_max:
        World-space bounding box of the imported geometry as seen by
        Blender (after its glTF Y-up to Z-up conversion).
    author_min, author_max:
        The authored Z-up bounding box (from the elements sidecar). When
        available we pick the rotation whose resulting extents best match
        the authored extents, and we re-anchor the min corner exactly.
        When unavailable we fall back to a "smallest extent is up"
        heuristic suitable for a thin floor plate.

    Returns
    -------
    dict with keys:
      ``mode``        : which correction was chosen
      ``matrix``      : 4x4 homogeneous transform (row-major list of lists)
      ``rotation``    : 3x3 rotation (list of lists)
      ``translation`` : 3-vector (list)
      ``needs_change``: bool, False when the geometry was already aligned
      ``author_extents`` / ``blender_extents`` / ``result_extents``
    """
    bmin = np.array(blender_min, dtype=float)
    bmax = np.array(blender_max, dtype=float)
    be = bmax - bmin

    candidates: dict[str, np.ndarray] = {
        "identity": np.eye(3),
        "rot_x_-90": _rot_x(-90.0),
        "rot_x_+90": _rot_x(90.0),
        "rot_y_-90": _rot_y(-90.0),
        "rot_y_+90": _rot_y(90.0),
    }

    if author_min is not None and author_max is not None:
        amin = np.array(author_min, dtype=float)
        amax = np.array(author_max, dtype=float)
        ae = amax - amin

        def score(R: np.ndarray) -> float:
            return float(np.abs(_rotated_extents(bmin, bmax, R) - ae).sum())

        # Prefer identity when it already matches (stable, no needless flips),
        # then the glTF Y-up fix (rot_x_-90), then the rest.
        order = ["identity", "rot_x_-90", "rot_x_+90", "rot_y_-90", "rot_y_+90"]
        best_name = min(order, key=lambda k: (round(score(candidates[k]), 6), order.index(k)))
        R = candidates[best_name]
        result_ext = _rotated_extents(bmin, bmax, R)
        rc = _corners(bmin, bmax) @ R.T
        rmin = rc.min(axis=0)
        T = amin - rmin
        author_ext = ae.tolist()
    else:
        # Heuristic: a floor plate is thin in its up direction, so the axis
        # with the smallest extent should become +Z.
        up = int(np.argmin(be))
        if up == 2:
            best_name, R = "identity", candidates["identity"]
        elif up == 1:
            best_name, R = "rot_x_-90", candidates["rot_x_-90"]
        else:  # up == 0  -> rotate about Y to bring X up onto Z
            best_name, R = "rot_y_-90", candidates["rot_y_-90"]
        result_ext = _rotated_extents(bmin, bmax, R)
        rc = _corners(bmin, bmax) @ R.T
        rmin = rc.min(axis=0)
        T = -rmin  # anchor min corner to the origin
        author_ext = None

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T

    needs_change = best_name != "identity" or float(np.abs(T).max()) > 1e-6

    return {
        "mode": best_name,
        "matrix": M.tolist(),
        "rotation": R.tolist(),
        "translation": T.tolist(),
        "needs_change": bool(needs_change),
        "author_extents": author_ext,
        "blender_extents": be.tolist(),
        "result_extents": result_ext.tolist(),
    }
