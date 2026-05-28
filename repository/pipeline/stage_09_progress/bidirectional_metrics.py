"""Bidirectional scan-to-BIM completeness/accuracy metrics for Stage 9.

The historical Stage 9 metric was a *one-sided* proximity ratio: for each BIM
element bbox-cropped target, compute nearest-neighbor distances from the BIM
to the nearest scan point and report ``mean(distance ≤ τ)``. That metric
cannot distinguish "not built" from "not observed", and cannot distinguish
"built incorrectly" from "out of view".

This module computes the principled bidirectional pair plus their F-score:

- **accuracy** = fraction of *scan* points (cropped to the element bbox plus
  margin) whose nearest BIM-element point is within ``tau_m`` — i.e. of the
  scan we observed for this element, what fraction agrees with the design.
- **completeness** = fraction of *BIM-element* points whose nearest scan point
  is within ``tau_m`` — i.e. of the BIM surface we expected to see, what
  fraction was observed.
- **f_score** = harmonic mean of the two (Knapitsch et al., SIGGRAPH 2017).

The function is intentionally pure (no I/O, no Open3D imports). Callers
provide already-cropped point arrays; this lets the function stand alone and
makes it trivial to unit-test deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .uncertainty import f_score, wilson_interval


@dataclass(frozen=True)
class BidirectionalResult:
    """Per-element bidirectional metric record."""

    tau_m: float
    accuracy: float
    completeness: float
    f_score: float
    accuracy_count_evaluated: int
    accuracy_count_in_tolerance: int
    completeness_count_evaluated: int
    completeness_count_in_tolerance: int
    accuracy_wilson: tuple[float, float, float]
    completeness_wilson: tuple[float, float, float]
    f_score_wilson: tuple[float, float, float]
    notes: list[str]

    def to_row(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for CSV concatenation.

        Keys are additive — none of them collide with the legacy Stage 9
        columns. The ``notes`` field is joined with ``;`` so rows remain
        single-line.
        """
        ac_lo, ac_pt, ac_hi = self.accuracy_wilson
        cm_lo, cm_pt, cm_hi = self.completeness_wilson
        f_lo, f_pt, f_hi = self.f_score_wilson
        return {
            "bidirectional_tau_m": f"{self.tau_m:.6f}",
            "accuracy": f"{self.accuracy:.6f}",
            "completeness": f"{self.completeness:.6f}",
            "f_score": f"{self.f_score:.6f}",
            "accuracy_n_evaluated": str(self.accuracy_count_evaluated),
            "accuracy_n_in_tolerance": str(self.accuracy_count_in_tolerance),
            "completeness_n_evaluated": str(self.completeness_count_evaluated),
            "completeness_n_in_tolerance": str(self.completeness_count_in_tolerance),
            "accuracy_wilson_lo": f"{ac_lo:.6f}",
            "accuracy_wilson_hi": f"{ac_hi:.6f}",
            "completeness_wilson_lo": f"{cm_lo:.6f}",
            "completeness_wilson_hi": f"{cm_hi:.6f}",
            "f_score_wilson_lo": f"{f_lo:.6f}",
            "f_score_wilson_hi": f"{f_hi:.6f}",
            "bidirectional_notes": ";".join(self.notes),
        }


def _nearest_distances_kdtree(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the nearest-neighbor distance for each row of ``query`` in ``target``.

    Uses scipy.spatial.cKDTree when available (fastest path); falls back to a
    pure NumPy O(n*m) computation when SciPy is missing. The fallback is only
    intended for unit tests with small fixtures, not for production runs.
    """
    if query.size == 0 or target.size == 0:
        return np.zeros((0,), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree  # noqa: WPS433

        distances, _ = cKDTree(target).query(query, k=1, workers=-1)
        return np.asarray(distances, dtype=np.float64)
    except Exception:
        # NumPy fallback: ||q - t|| for each pair, take min over t per q.
        diffs = query[:, None, :] - target[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        return np.sqrt(np.min(d2, axis=1))


def compute_bidirectional(
    scan_points: np.ndarray,
    bim_points: np.ndarray,
    tau_m: float,
) -> BidirectionalResult:
    """Compute accuracy, completeness, F-score @ τ, and Wilson CIs for one element.

    Both arguments must be ``(N, 3)`` float arrays already cropped to the
    element bounding-box-plus-margin. ``scan_points`` are the scan points to
    evaluate against the BIM element; ``bim_points`` are the BIM-element
    surface samples (or the bbox-cropped BIM cloud).

    When either side is empty, the corresponding metric is reported as 0 with
    ``count_evaluated=0`` and a note. The F-score is then 0; this matches the
    Tanks-and-Temples convention that "no observations" should not give credit
    rather than divide by zero.
    """
    notes: list[str] = []

    scan_pts = np.asarray(scan_points, dtype=np.float64).reshape(-1, 3) if len(scan_points) else np.zeros((0, 3), dtype=np.float64)
    bim_pts = np.asarray(bim_points, dtype=np.float64).reshape(-1, 3) if len(bim_points) else np.zeros((0, 3), dtype=np.float64)

    if scan_pts.size == 0:
        notes.append("no_scan_points_for_element")
        accuracy = 0.0
        accuracy_n = 0
        accuracy_k = 0
    else:
        # accuracy: scan -> nearest BIM-element point within tau
        if bim_pts.size == 0:
            notes.append("no_bim_points_for_element")
            accuracy = 0.0
            accuracy_n = int(scan_pts.shape[0])
            accuracy_k = 0
        else:
            d_scan_to_bim = _nearest_distances_kdtree(scan_pts, bim_pts)
            accuracy_n = int(d_scan_to_bim.size)
            accuracy_k = int(np.sum(d_scan_to_bim <= tau_m))
            accuracy = (accuracy_k / accuracy_n) if accuracy_n > 0 else 0.0

    if bim_pts.size == 0:
        if "no_bim_points_for_element" not in notes:
            notes.append("no_bim_points_for_element")
        completeness = 0.0
        completeness_n = 0
        completeness_k = 0
    else:
        if scan_pts.size == 0:
            completeness = 0.0
            completeness_n = int(bim_pts.shape[0])
            completeness_k = 0
        else:
            d_bim_to_scan = _nearest_distances_kdtree(bim_pts, scan_pts)
            completeness_n = int(d_bim_to_scan.size)
            completeness_k = int(np.sum(d_bim_to_scan <= tau_m))
            completeness = (completeness_k / completeness_n) if completeness_n > 0 else 0.0

    f = f_score(accuracy, completeness)

    accuracy_wilson = wilson_interval(accuracy_k, accuracy_n)
    completeness_wilson = wilson_interval(completeness_k, completeness_n)
    # F-score CI: harmonic-mean propagation is non-trivial; use the harmonic
    # mean of the two Wilson endpoints as a conservative envelope. This is a
    # documented convention rather than a derivation; we record it as such.
    f_lo = f_score(accuracy_wilson[0], completeness_wilson[0])
    f_hi = f_score(accuracy_wilson[2], completeness_wilson[2])
    f_score_wilson = (float(min(f_lo, f_hi)), float(f), float(max(f_lo, f_hi)))

    return BidirectionalResult(
        tau_m=float(tau_m),
        accuracy=float(accuracy),
        completeness=float(completeness),
        f_score=float(f),
        accuracy_count_evaluated=accuracy_n,
        accuracy_count_in_tolerance=accuracy_k,
        completeness_count_evaluated=completeness_n,
        completeness_count_in_tolerance=completeness_k,
        accuracy_wilson=accuracy_wilson,
        completeness_wilson=completeness_wilson,
        f_score_wilson=f_score_wilson,
        notes=notes,
    )


__all__ = ["BidirectionalResult", "compute_bidirectional"]
