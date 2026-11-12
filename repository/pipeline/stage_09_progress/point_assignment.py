"""Exclusive AABB attribution for Stage 9. No geometry is changed upstream.

An AABB intersection is a candidate region, never an IFC ownership proof.
When several expanded boxes contain a point it is *unassigned*, not copied
to every element. Production-quality resolution of overlaps requires Stage 8
to export per-element surface samples/triangle IDs and a visibility model.
"""
from __future__ import annotations

import numpy as np


def assign_exclusive_points(
    points: np.ndarray,
    elements: list[dict],
    margin_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (owner, candidate_count), shape (N,).

    owner = -1 (outside), -2 (ambiguous overlap), or zero-based element index.
    Invalid or missing bounds cannot claim points. O(N * elements), O(N) RAM.
    """
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and nonnegative")
    owners = np.full(len(xyz), -1, dtype=np.int64)
    counts = np.zeros(len(xyz), dtype=np.int32)
    finite = np.all(np.isfinite(xyz), axis=1)
    for index, element in enumerate(elements):
        try:
            lower = np.asarray(element["bounds_min"], dtype=float).reshape(3)
            upper = np.asarray(element["bounds_max"], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(lower <= upper)):
            continue
        inside = finite & np.all((xyz >= lower - margin_m) & (xyz <= upper + margin_m), axis=1)
        first = inside & (counts == 0)
        owners[first] = index
        owners[inside & ~first] = -2
        counts[inside] += 1
    return owners, counts


def nearest_element_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Element-owned query -> element-owned reference NN distances in metres.

    Returns an empty array if either set is empty (no invented failures).
    SciPy KD-tree if available; bounded-memory NumPy fallback for small tests.
    """
    q = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(q) == 0 or len(r) == 0:
        return np.empty(0, dtype=np.float64)
    if not (np.isfinite(q).all() and np.isfinite(r).all()):
        raise ValueError("non-finite element points")
    try:
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(r).query(q, k=1, workers=-1)
        return np.asarray(distances, dtype=np.float64)
    except ImportError:
        out = np.empty(len(q), dtype=np.float64)
        # 128 by M, not N by M, to avoid quadratic memory allocation.
        for start in range(0, len(q), 128):
            chunk = q[start : start + 128]
            out[start : start + len(chunk)] = np.sqrt(
                np.min(np.sum((chunk[:, None, :] - r[None, :, :]) ** 2, axis=2), axis=1)
            )
        return out
