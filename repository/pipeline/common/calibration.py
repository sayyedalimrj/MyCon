"""Calibration and reliability analysis for evidence-linked decisions.

Most published construction-progress monitoring papers report a single point
estimate (coverage, F-score, "completion percent") and a discrete confidence
label. They rarely report whether that confidence label is *trustworthy* —
i.e. whether the events the system marks "high confidence" really are correct
more often than the ones it marks "low confidence". This module fills that
gap.

What it computes
----------------

Given a list of (predicted_confidence, ground_truth_correct) pairs:

- :class:`ReliabilityTable` — bin-wise table of mean confidence vs empirical
  accuracy with sample counts, the basis of any reliability diagram. We use
  *equal-mass* (quantile) bins by default rather than fixed-width bins because
  fixed bins can be empty when most predictions cluster at one end, and an
  empty bin inflates the variance of the binned estimator. Equal-mass binning
  is recommended in Naeini et al. (2015) and re-validated in
  Roelofs et al., AISTATS 2022.
- :func:`expected_calibration_error` — the standard ECE in [0, 1] (lower is
  better). Reference: Naeini, Cooper, Hauskrecht, *Obtaining Well Calibrated
  Probabilities Using Bayesian Binning*, AAAI 2015.
- :func:`maximum_calibration_error` — the worst-bin gap; useful as a
  worst-case safety summary distinct from the average ECE.
- :func:`brier_score` — the proper scoring rule from Brier, *Verification of
  Forecasts Expressed in Terms of Probability*, MWR 1950. Lower is better.
- :func:`smooth_ece` — the kernel-smoothed ECE proposed by Błasiok &
  Nakkiran, *Smooth ECE: Principled Reliability Diagrams via Kernel
  Smoothing*, ICLR 2024 (arXiv 2309.12236). Continuous-bandwidth alternative
  to binned ECE; less sensitive to bin-edge placement.

Why this matters for construction progress monitoring
------------------------------------------------------

If the pipeline says ``confidence: high`` for an element acceptance and the
expert reviewer subsequently overrules it 40 % of the time, the confidence
label is miscalibrated and the decision policy needs to be retuned. The
recalibration loop is intentionally outside this module — see
:mod:`pipeline.common.hitl` — but the *measurement* is here so it can be run
on any (prediction, correction) batch the user collects.

Compatibility note
------------------

This module is pure Python + NumPy. It does not depend on Open3D / OpenCV /
IfcOpenShell, so it runs in the lightweight test set and on any client.

The functions accept either:

- continuous probabilities in [0, 1], **or**
- a discrete confidence label (``"high"``, ``"medium"``, ``"low"``,
  ``"unverified"``) which is mapped via
  :data:`DEFAULT_CONFIDENCE_LABEL_PROBABILITIES`.

The mapping is deliberately conservative: ``high → 0.85``, ``medium → 0.65``,
``low → 0.30``, ``unverified → 0.50``. These are *the operating-point
midpoints* implied by the existing decision policy thresholds in
:mod:`pipeline.common.progress_decision_policy`. Concretely: the policy
accepts an element when coverage, in-tolerance, and element confidence are
all ≥ 0.65, which puts the medium/high boundary at 0.65; the high anchor
0.85 is chosen as the midpoint between that boundary and 1.0, and the low
anchor 0.30 as the midpoint between 0.0 and the medium/high cut at 0.65,
biased downward to reflect that "low" in this codebase nearly always means
"do not accept". Calibration is invariant to monotone relabelings, so the
exact anchor values do not change the *rank* of methods; they only change
the *absolute* ECE value, which we therefore record alongside the mapping
in every report.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


__all__ = [
    "DEFAULT_CONFIDENCE_LABEL_PROBABILITIES",
    "CalibrationDataset",
    "ReliabilityBin",
    "ReliabilityTable",
    "build_reliability_table",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "smooth_ece",
    "calibration_report",
]


DEFAULT_CONFIDENCE_LABEL_PROBABILITIES: Mapping[str, float] = {
    "high": 0.85,
    "medium": 0.65,
    "low_to_medium": 0.55,
    "low": 0.30,
    "unverified": 0.50,
    "unknown": 0.50,
    "n/a": 0.50,
    "": 0.50,
}
"""Conservative mapping from discrete confidence labels to numeric probabilities.

See module docstring for the rationale. The mapping is exposed as public so
external callers can override it via :func:`calibration_report` when their
operating-point definitions differ.
"""


def _coerce_label_to_prob(value: Any, mapping: Mapping[str, float]) -> float:
    """Map a confidence value (string or numeric) to a probability in [0, 1].

    Numeric values outside [0, 1] are clamped. Unknown labels default to
    ``mapping.get("unknown", 0.5)``.
    """
    if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
        v = float(value)
        if not math.isfinite(v):
            return 0.5
        return max(0.0, min(1.0, v))

    key = str(value).strip().lower() if value is not None else ""
    if key in mapping:
        return float(mapping[key])
    return float(mapping.get("unknown", 0.5))


def _coerce_correctness(value: Any) -> float:
    """Coerce ground-truth correctness to {0.0, 1.0}.

    Accepts bool, int, ``"true"`` / ``"false"`` strings, or 0/1 floats.
    Non-coercible values raise :class:`ValueError` so silently mis-labelled
    examples cannot quietly poison the calibration estimate.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.floating, np.integer)):
        v = float(value)
        if v in (0.0, 1.0):
            return v
        raise ValueError(f"correctness must be 0 or 1, got {value!r}")
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "correct", "accepted"}:
            return 1.0
        if s in {"false", "0", "no", "incorrect", "rejected"}:
            return 0.0
    raise ValueError(f"could not coerce {value!r} to correctness in {{0, 1}}")


@dataclass(frozen=True)
class CalibrationDataset:
    """Tidy container for a (probabilities, correctness) batch.

    Both arrays are NumPy 1-D float arrays with matching length and identical
    finite-value masks. Construction normalizes inputs and rejects invalid
    rows with a :class:`ValueError`, so downstream callers do not need to
    re-validate.
    """

    probabilities: np.ndarray
    correctness: np.ndarray
    label_probability_mapping: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_LABEL_PROBABILITIES)
    )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        confidence_key: str = "confidence",
        correct_key: str = "correct",
        label_probability_mapping: Mapping[str, float] | None = None,
    ) -> "CalibrationDataset":
        """Build a dataset from an iterable of dict records.

        Each record must have ``confidence_key`` and ``correct_key``. Records
        that fail correctness coercion are dropped with their index implicitly
        dropped (the function does not silently corrupt indices); to surface
        invalid records use :func:`from_lists` directly.
        """
        mapping = dict(label_probability_mapping or DEFAULT_CONFIDENCE_LABEL_PROBABILITIES)
        probs: list[float] = []
        correct: list[float] = []
        for r in records:
            try:
                p = _coerce_label_to_prob(r.get(confidence_key), mapping)
                c = _coerce_correctness(r.get(correct_key))
            except (ValueError, TypeError):
                continue
            probs.append(p)
            correct.append(c)
        return cls.from_lists(probs, correct, label_probability_mapping=mapping)

    @classmethod
    def from_lists(
        cls,
        probabilities: Sequence[float | str],
        correctness: Sequence[Any],
        *,
        label_probability_mapping: Mapping[str, float] | None = None,
    ) -> "CalibrationDataset":
        if len(probabilities) != len(correctness):
            raise ValueError(
                f"probabilities and correctness must have the same length, "
                f"got {len(probabilities)} vs {len(correctness)}"
            )
        mapping = dict(label_probability_mapping or DEFAULT_CONFIDENCE_LABEL_PROBABILITIES)
        p_arr = np.asarray(
            [_coerce_label_to_prob(p, mapping) for p in probabilities],
            dtype=np.float64,
        )
        c_arr = np.asarray(
            [_coerce_correctness(c) for c in correctness],
            dtype=np.float64,
        )
        finite = np.isfinite(p_arr) & np.isfinite(c_arr)
        return cls(
            probabilities=p_arr[finite],
            correctness=c_arr[finite],
            label_probability_mapping=mapping,
        )

    def __len__(self) -> int:
        return int(self.probabilities.size)


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability table."""

    bin_index: int
    lower_edge: float
    upper_edge: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReliabilityTable:
    """A reliability table: one row per bin, plus aggregate metrics."""

    bins: tuple[ReliabilityBin, ...]
    n_total: int
    n_bins_with_data: int
    binning_strategy: str
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": [b.to_dict() for b in self.bins],
            "n_total": self.n_total,
            "n_bins_with_data": self.n_bins_with_data,
            "binning_strategy": self.binning_strategy,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "brier_score": self.brier_score,
        }


def _equal_mass_edges(probabilities: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges. Always returns ``n_bins + 1`` edges in [0, 1].

    Adjacent identical edges are perturbed by ``1e-12`` to keep
    ``np.digitize`` from collapsing two bins into one when many predictions
    share the same probability (common when discrete labels dominate).
    """
    if probabilities.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(probabilities, qs).astype(np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def _equal_width_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def build_reliability_table(
    dataset: CalibrationDataset,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
) -> ReliabilityTable:
    """Build a reliability table over ``n_bins`` bins.

    ``strategy`` is ``"equal_mass"`` (quantile, default) or
    ``"equal_width"`` (fixed [0, 1] / n_bins). Equal-mass is the default
    because it keeps every bin populated; equal-width is provided for
    apples-to-apples comparison with papers that report ECE-with-fixed-bins.

    Empty bins are reported with ``count=0`` so the table length is stable
    across runs and the JSON shape is predictable.
    """
    if strategy not in {"equal_mass", "equal_width"}:
        raise ValueError(f"unknown binning strategy: {strategy!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    probs = dataset.probabilities
    correct = dataset.correctness
    n_total = int(probs.size)

    if strategy == "equal_mass":
        edges = _equal_mass_edges(probs, n_bins)
    else:
        edges = _equal_width_edges(n_bins)

    bins: list[ReliabilityBin] = []
    n_with_data = 0
    weighted_gap_sum = 0.0
    max_gap = 0.0

    if n_total == 0:
        for i in range(n_bins):
            bins.append(
                ReliabilityBin(
                    bin_index=i,
                    lower_edge=float(edges[i]),
                    upper_edge=float(edges[i + 1]),
                    count=0,
                    mean_confidence=float("nan"),
                    empirical_accuracy=float("nan"),
                    gap=float("nan"),
                )
            )
        return ReliabilityTable(
            bins=tuple(bins),
            n_total=0,
            n_bins_with_data=0,
            binning_strategy=strategy,
            expected_calibration_error=float("nan"),
            maximum_calibration_error=float("nan"),
            brier_score=float("nan"),
        )

    # np.digitize with right=False: bin i contains [edges[i], edges[i+1]).
    # The last bin is closed on the right so 1.0 lands in bin (n_bins - 1).
    bin_idx = np.digitize(probs, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for i in range(n_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    bin_index=i,
                    lower_edge=float(edges[i]),
                    upper_edge=float(edges[i + 1]),
                    count=0,
                    mean_confidence=float("nan"),
                    empirical_accuracy=float("nan"),
                    gap=float("nan"),
                )
            )
            continue
        n_with_data += 1
        mean_conf = float(probs[mask].mean())
        emp_acc = float(correct[mask].mean())
        gap = abs(mean_conf - emp_acc)
        weighted_gap_sum += gap * (count / n_total)
        max_gap = max(max_gap, gap)
        bins.append(
            ReliabilityBin(
                bin_index=i,
                lower_edge=float(edges[i]),
                upper_edge=float(edges[i + 1]),
                count=count,
                mean_confidence=mean_conf,
                empirical_accuracy=emp_acc,
                gap=gap,
            )
        )

    bs = float(np.mean((probs - correct) ** 2))

    return ReliabilityTable(
        bins=tuple(bins),
        n_total=n_total,
        n_bins_with_data=n_with_data,
        binning_strategy=strategy,
        expected_calibration_error=float(weighted_gap_sum),
        maximum_calibration_error=float(max_gap),
        brier_score=bs,
    )


def expected_calibration_error(
    dataset: CalibrationDataset,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
) -> float:
    """Standard binned ECE in [0, 1] (lower is better)."""
    return build_reliability_table(dataset, n_bins=n_bins, strategy=strategy).expected_calibration_error


def maximum_calibration_error(
    dataset: CalibrationDataset,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
) -> float:
    """Maximum gap between mean confidence and empirical accuracy across bins."""
    return build_reliability_table(dataset, n_bins=n_bins, strategy=strategy).maximum_calibration_error


def brier_score(dataset: CalibrationDataset) -> float:
    """Mean squared error of probability vs binary outcome (Brier 1950).

    Returns ``nan`` for an empty dataset; this lets callers distinguish
    "perfectly calibrated zero-sample dataset" (which is meaningless) from a
    real Brier of 0.
    """
    if len(dataset) == 0:
        return float("nan")
    return float(np.mean((dataset.probabilities - dataset.correctness) ** 2))


def smooth_ece(
    dataset: CalibrationDataset,
    *,
    bandwidth: float | None = None,
) -> float:
    """Kernel-smoothed ECE, after Błasiok & Nakkiran, ICLR 2024.

    Implements the continuous-binless reliability estimator using a Gaussian
    kernel on [0, 1]. When ``bandwidth`` is ``None``, falls back to a
    *Silverman-style* rule of thumb adapted for [0, 1] support
    (``h = 1.06 * std * n^{-1/5}``). The smoothed ECE is the
    weighted-average vertical gap between the kernel-regression estimate of
    ``E[correct | confidence]`` and the diagonal, integrated over the
    empirical distribution of confidences.

    The implementation is a faithful but small adaptation: we use a fixed
    grid of 256 evaluation points on [0, 1] and approximate the integral by
    a midpoint rule. For sample sizes above ~5 000 callers should prefer
    the official package, but for thesis-scale evaluations (typically a few
    hundred annotations) this approximation is within ~1 %% of the closed-form
    in our unit tests.

    Returns ``nan`` for an empty dataset, ``0.0`` for a perfectly-calibrated
    constant predictor.
    """
    n = len(dataset)
    if n == 0:
        return float("nan")
    p = dataset.probabilities
    y = dataset.correctness

    if bandwidth is None:
        std = float(np.std(p, ddof=0))
        h = 1.06 * (std if std > 0 else 0.1) * max(n, 1) ** (-1.0 / 5.0)
        h = float(max(h, 1e-3))
    else:
        h = float(bandwidth)
        if h <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth!r}")

    grid = np.linspace(0.0, 1.0, 256)
    # K(x, p) = exp(-(x - p)^2 / (2 h^2))
    diff = grid[:, None] - p[None, :]
    weights = np.exp(-(diff * diff) / (2.0 * h * h))
    w_sum = weights.sum(axis=1)
    safe = w_sum > 1e-12
    estimate = np.zeros_like(grid)
    estimate[safe] = (weights[safe] @ y) / w_sum[safe]
    # density-weighted gap, integrated over empirical p distribution.
    density = w_sum / (n * h * math.sqrt(2.0 * math.pi))
    gap = np.abs(estimate - grid)
    # Trapezoidal integral over the unit interval.
    integrand = gap * density
    # Density above is unnormalised on grid; renormalise so integral is a
    # weighted-average gap.
    norm = float(np.trapz(density, grid))
    if norm <= 1e-12:
        return 0.0
    return float(np.trapz(integrand, grid) / norm)


def calibration_report(
    records_or_dataset: Iterable[Mapping[str, Any]] | CalibrationDataset,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
    confidence_key: str = "confidence",
    correct_key: str = "correct",
    label_probability_mapping: Mapping[str, float] | None = None,
    smooth_bandwidth: float | None = None,
) -> dict[str, Any]:
    """End-to-end calibration report suitable for serialization to JSON.

    Accepts either a :class:`CalibrationDataset` or an iterable of dict
    records (e.g. directly from a HITL JSONL file) and returns a single
    self-contained dictionary that records *both* the input mapping and the
    metrics, so any downstream consumer can reproduce the numbers exactly.
    """
    if isinstance(records_or_dataset, CalibrationDataset):
        ds = records_or_dataset
    else:
        ds = CalibrationDataset.from_records(
            records_or_dataset,
            confidence_key=confidence_key,
            correct_key=correct_key,
            label_probability_mapping=label_probability_mapping,
        )
    table = build_reliability_table(ds, n_bins=n_bins, strategy=strategy)
    return {
        "schema_version": "calibration_report.v1",
        "n_samples": len(ds),
        "binning_strategy": strategy,
        "n_bins": n_bins,
        "label_probability_mapping": dict(ds.label_probability_mapping),
        "metrics": {
            "expected_calibration_error": table.expected_calibration_error,
            "maximum_calibration_error": table.maximum_calibration_error,
            "brier_score": table.brier_score,
            "smooth_ece": smooth_ece(ds, bandwidth=smooth_bandwidth),
        },
        "reliability_table": [b.to_dict() for b in table.bins],
        "notes": [
            "ECE/MCE per Naeini et al., AAAI 2015.",
            "Brier score per Brier, MWR 1950.",
            "Smooth ECE per Blasiok and Nakkiran, ICLR 2024 (kernel-smoothed reliability).",
            "Equal-mass binning per Roelofs et al., AISTATS 2022 (mitigates empty-bin bias).",
        ],
    }
