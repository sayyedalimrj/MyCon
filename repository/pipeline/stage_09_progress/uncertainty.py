"""Uncertainty-quantification helpers for Stage 9 progress metrics.

Exposes two principled tools that are missing from the historical Stage 9
implementation:

- :func:`wilson_interval` — Wilson score interval for proportions. Preferred
  over the normal-approximation interval for small samples and for proportions
  near 0 or 1, both of which are common in element-level scan-to-BIM coverage
  (a tiny element may have only a handful of evaluated NN distances).
  Reference: Wilson, *Probable Inference, the Law of Succession, and
  Statistical Inference*, JASA 22(158), 1927.

- :func:`bootstrap_ci` — percentile bootstrap CI for a real-valued statistic
  (mean, median, p95). Reference: Efron, *Bootstrap Methods: Another Look at
  the Jackknife*, Annals of Statistics 7(1), 1979.

Both functions are pure NumPy. The bootstrap takes a seedable
``numpy.random.Generator`` so callers can wire reproducibility through
:mod:`pipeline.common.determinism`.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

# Two-sided z-quantile for 95 % normal CI (used in Wilson formula).
_Z_AT_95: float = 1.959963984540054


def wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = _Z_AT_95,
) -> tuple[float, float, float]:
    """Return ``(lower, point_estimate, upper)`` for a binomial proportion.

    Successes are clamped to ``[0, n]``. When ``n == 0`` the interval is
    ``(0.0, 0.0, 0.0)`` and the caller should treat the metric as undefined,
    not as zero.

    The default ``z=1.96`` corresponds to a 95 % CI. The interval is
    *symmetric in normalized success counts*, not in raw counts; that is its
    correctness improvement over the textbook normal-approximation CI for
    small ``n``.
    """
    if n is None or n <= 0:
        return 0.0, 0.0, 0.0
    k = max(0, min(int(successes), int(n)))
    n_f = float(n)
    p_hat = k / n_f
    z2 = z * z
    denom = 1.0 + z2 / n_f
    centre = (p_hat + z2 / (2.0 * n_f)) / denom
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) / n_f) + (z2 / (4.0 * n_f * n_f)))) / denom
    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    # Floating-point can produce upper = 0.9999999999999999 when p_hat = 1.0
    # (or symmetrically lower ≈ 1e-17 when p_hat = 0.0). Snap to the exact
    # boundary in those cases so the public contract "clamps to [0, 1]" holds.
    if k == 0:
        lower = 0.0
    if k == int(n):
        upper = 1.0
    return float(lower), float(p_hat), float(upper)


def f_score(accuracy: float, completeness: float) -> float:
    """Harmonic mean of accuracy (precision-style) and completeness (recall-style).

    Returns 0 when either component is zero or non-finite. F-score @ τ with
    τ matching ``progress.deviation_threshold_m`` is the standard
    Tanks-and-Temples-style aggregation (Knapitsch et al., SIGGRAPH 2017).
    """
    if not (math.isfinite(accuracy) and math.isfinite(completeness)):
        return 0.0
    if accuracy <= 0.0 or completeness <= 0.0:
        return 0.0
    return float(2.0 * accuracy * completeness / (accuracy + completeness))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a real-valued summary statistic.

    Returns ``(lower, point_estimate, upper)`` at confidence level ``1 - alpha``.
    When fewer than 2 finite samples are present the function returns
    ``(nan, nan, nan)`` rather than fabricating a degenerate interval.

    The percentile bootstrap is preferred to the normal-approximation bootstrap
    here because the distribution of element-level NN distances on a
    construction scan is heavy-tailed (occlusion + scaffolding outliers) and
    skewed; the percentile method does not assume symmetry.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if rng is None:
        rng = np.random.default_rng()
    boot = np.empty(int(n_boot), dtype=np.float64)
    n = arr.size
    for i in range(int(n_boot)):
        sample = arr[rng.integers(0, n, size=n)]
        boot[i] = float(statistic(sample))
    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return point, point, point
    lo = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return lo, point, hi


__all__ = ["wilson_interval", "f_score", "bootstrap_ci"]
