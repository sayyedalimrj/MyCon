"""Structured method-comparison artefacts (method x metric x CI tables).

Reviewers in Q1 civil/AEC venues (Automation in Construction, CACAIE,
AEI, JCCEE5, JCEMD4) expect every novelty paper to include a comparison
table that shows the proposed method against published baselines on
identical metrics, with explicit confidence intervals. Most repositories
ship this table as ad-hoc Markdown or hand-written LaTeX, which makes
the numbers hard to audit and harder to update.

This module is the canonical, typed representation of that table. It

- accepts results from any number of methods and metrics;
- carries a 95 %% confidence interval on every cell (or ``None`` if the
  source paper did not report one);
- preserves the source citation for every value so the paper can show
  provenance inline;
- exports to a deterministic dict (for JSON), to a plain ASCII table
  (for terminal review), and to a thesis-grade LaTeX ``booktabs``
  table that compiles cleanly with the ``booktabs`` and ``siunitx``
  packages.

Schema is locked at ``method_comparison.v1``.

Why a typed framework rather than free-form Markdown
----------------------------------------------------

- Determinism. Every export is a pure function of the inputs, so the
  table in the paper, the table in the dashboard, and the table in
  ``runs/.../method_comparison.json`` are byte-for-byte identical.
- Auditability. ``MethodResult.source`` records *exactly* where each
  number came from -- the CORE-N entry from the literature map, the
  arXiv id, or the local run id.
- Reuse. The same dataclass shape is consumed by the dashboard's
  comparison view (Phase 5, follow-up task) and by the LaTeX exporter
  (this module).

Pure stdlib. Lightweight test set safe.

References
----------

The default exporter style follows the recommendations in the Q1
civil-engineering literature reviewed in
``docs/literature_q1_2024_2026.md``: numeric cells are printed with one
decimal place, intervals as ``[lo, hi]`` pairs, and "lower is better"
metrics are tagged with a small marker so the table reads cleanly
under copy-edit.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "METHOD_COMPARISON_SCHEMA_VERSION",
    "MetricDirection",
    "MetricSpec",
    "MethodResult",
    "MethodComparisonTable",
    "build_table",
    "to_ascii",
    "to_latex",
    "to_dict",
    "from_dict",
]


METHOD_COMPARISON_SCHEMA_VERSION = "method_comparison.v1"


class MetricDirection:
    """String constants for the optimization direction of a metric.

    Not an Enum so JSON serialisation is trivial.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.HIGHER_IS_BETTER, cls.LOWER_IS_BETTER)


@dataclass(frozen=True)
class MetricSpec:
    """One metric column.

    Fields
    ------
    name : str
        Stable, human-readable column name (e.g. ``"F-score @ 5cm"`` or
        ``"ECE"``). Used as the table header.
    direction : str
        One of :class:`MetricDirection`. Determines which cell wins in
        ranking and which marker (``up``/``down``) is rendered next to
        the column header in LaTeX.
    units : str
        Optional unit suffix (e.g. ``"%"``, ``"cm"``). Empty string
        means dimensionless.
    description : str
        Optional human-readable explanation, included in the JSON
        export so the paper-writer can reuse it as a footnote.
    """

    name: str
    direction: str
    units: str = ""
    description: str = ""
    decimals: int | None = None
    """Optional per-metric override for the number of decimal places.

    When ``None`` (default), the renderer's ``decimals`` argument
    applies. When set, this value wins. Use ``decimals=3`` for
    calibration metrics (ECE / Brier) and ``decimals=1`` for
    percentage scores so the same table reads cleanly with both.
    """

    def __post_init__(self) -> None:
        if self.direction not in MetricDirection.all():
            raise ValueError(
                f"MetricSpec.direction must be one of {MetricDirection.all()!r}, "
                f"got {self.direction!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodResult:
    """One cell of the comparison table.

    Fields
    ------
    method : str
        Stable method label (e.g. ``"Bosche 2010"``, ``"Ours"``).
    metric : str
        Must match a :class:`MetricSpec.name` in the parent table.
    value : float
        Headline value. ``float('nan')`` is reserved for "not
        reported" cells; ``to_ascii``/``to_latex`` render those as
        ``--`` so the reader sees a structural placeholder rather than
        a zero.
    ci_lower_95 : float | None
        Optional lower bound of a 95 %% confidence interval. ``None``
        when the source paper did not report one.
    ci_upper_95 : float | None
        Optional upper bound. Must be set together with ``ci_lower_95``
        or both must be ``None``.
    n : int | None
        Optional sample size. Useful for footnotes and for re-deriving
        a Wilson interval in the dashboard view.
    source : str
        Free-form provenance string (e.g. ``"AiC 2024 Mahami et al."``,
        ``"runs/example_walkthrough"``, ``"CORE-1"``). The paper's
        comparison table cites this column verbatim.
    notes : tuple[str, ...]
        Free-form notes (e.g. ``"reproduced with author's repo at commit
        abc1234"``). Rendered as a footnote marker in LaTeX, kept in
        full in JSON.
    """

    method: str
    metric: str
    value: float
    ci_lower_95: float | None = None
    ci_upper_95: float | None = None
    n: int | None = None
    source: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.ci_lower_95 is None) ^ (self.ci_upper_95 is None):
            raise ValueError(
                "ci_lower_95 and ci_upper_95 must both be set or both be None"
            )
        if (
            self.ci_lower_95 is not None
            and self.ci_upper_95 is not None
            and self.ci_lower_95 > self.ci_upper_95
        ):
            raise ValueError(
                f"ci_lower_95 ({self.ci_lower_95}) > ci_upper_95 ({self.ci_upper_95})"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class MethodComparisonTable:
    """A row-oriented comparison table.

    Fields
    ------
    title : str
        Optional title; rendered as the LaTeX caption and the ASCII
        header.
    metrics : tuple[MetricSpec, ...]
        One spec per column.
    methods : tuple[str, ...]
        One label per row, in display order.
    results : tuple[MethodResult, ...]
        Cell values. The constructor validates that every (method,
        metric) pair appears at most once and that every metric in a
        :class:`MethodResult` is declared in :attr:`metrics`.
    schema_version : str
        Locked at :data:`METHOD_COMPARISON_SCHEMA_VERSION`.
    generated_at_utc : str
        ISO-8601 timestamp; recorded in :func:`build_table` and
        preserved through :func:`from_dict`.
    """

    title: str
    metrics: tuple[MetricSpec, ...]
    methods: tuple[str, ...]
    results: tuple[MethodResult, ...]
    schema_version: str = METHOD_COMPARISON_SCHEMA_VERSION
    generated_at_utc: str = ""

    def __post_init__(self) -> None:
        metric_names = {m.name for m in self.metrics}
        if len(metric_names) != len(self.metrics):
            raise ValueError("MethodComparisonTable.metrics must have unique names")
        method_names = set(self.methods)
        if len(method_names) != len(self.methods):
            raise ValueError("MethodComparisonTable.methods must have unique labels")
        seen: set[tuple[str, str]] = set()
        for r in self.results:
            if r.metric not in metric_names:
                raise ValueError(
                    f"result references unknown metric {r.metric!r}; declared: {sorted(metric_names)}"
                )
            if r.method not in method_names:
                raise ValueError(
                    f"result references unknown method {r.method!r}; declared: {sorted(method_names)}"
                )
            key = (r.method, r.metric)
            if key in seen:
                raise ValueError(
                    f"duplicate result for method={r.method!r} metric={r.metric!r}"
                )
            seen.add(key)

    def by_method_and_metric(self) -> dict[tuple[str, str], MethodResult]:
        return {(r.method, r.metric): r for r in self.results}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "title": self.title,
            "metrics": [m.to_dict() for m in self.metrics],
            "methods": list(self.methods),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_table(
    *,
    title: str,
    metrics: Sequence[MetricSpec],
    methods: Sequence[str],
    results: Sequence[MethodResult | Mapping[str, Any]],
    generated_at_utc: str | None = None,
) -> MethodComparisonTable:
    """Construct a validated table from typed or dict-shaped inputs.

    ``results`` may be a mix of :class:`MethodResult` instances and
    plain dicts; dicts are normalised through :func:`MethodResult` so
    callers can build the table from JSON without explicitly converting
    each row.
    """
    norm: list[MethodResult] = []
    for item in results:
        if isinstance(item, MethodResult):
            norm.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError(
                f"results items must be MethodResult or Mapping, got {type(item).__name__}"
            )
        notes_raw = item.get("notes", ())
        notes = tuple(str(x) for x in notes_raw) if not isinstance(notes_raw, str) else (notes_raw,)
        norm.append(
            MethodResult(
                method=str(item["method"]),
                metric=str(item["metric"]),
                value=float(item["value"]),
                ci_lower_95=(
                    None
                    if item.get("ci_lower_95") is None
                    else float(item["ci_lower_95"])
                ),
                ci_upper_95=(
                    None
                    if item.get("ci_upper_95") is None
                    else float(item["ci_upper_95"])
                ),
                n=item.get("n") if item.get("n") is None else int(item["n"]),
                source=str(item.get("source", "")),
                notes=notes,
            )
        )
    when = generated_at_utc or _dt.datetime.now(_dt.timezone.utc).replace(
        microsecond=0
    ).isoformat()
    return MethodComparisonTable(
        title=title,
        metrics=tuple(metrics),
        methods=tuple(methods),
        results=tuple(norm),
        generated_at_utc=when,
    )


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def to_dict(table: MethodComparisonTable) -> dict[str, Any]:
    """Pure JSON-shaped projection (for ``runs/.../method_comparison.json``)."""
    return table.to_dict()


def from_dict(payload: Mapping[str, Any]) -> MethodComparisonTable:
    """Inverse of :func:`to_dict`. Validates the schema version."""
    schema = payload.get("schema_version")
    if schema != METHOD_COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"unexpected schema_version: got {schema!r}, expected "
            f"{METHOD_COMPARISON_SCHEMA_VERSION!r}"
        )
    metrics = tuple(
        MetricSpec(
            name=str(m["name"]),
            direction=str(m["direction"]),
            units=str(m.get("units", "")),
            description=str(m.get("description", "")),
            decimals=m["decimals"] if m.get("decimals") is not None else None,
        )
        for m in payload.get("metrics", [])
    )
    methods = tuple(str(s) for s in payload.get("methods", []))
    results = payload.get("results", [])
    return build_table(
        title=str(payload.get("title", "")),
        metrics=metrics,
        methods=methods,
        results=results,
        generated_at_utc=str(payload.get("generated_at_utc", "")),
    )


def _format_cell_value(
    result: MethodResult | None,
    *,
    decimals: int = 1,
    ci_decimals: int = 1,
) -> str:
    """Render one cell as a plain ASCII string.

    - ``None``                              -> ``--``
    - NaN                                   -> ``--``
    - value only                            -> ``"12.3"``
    - value + 95 %% CI                      -> ``"12.3 [10.1, 14.5]"``
    """
    if result is None:
        return "--"
    if not _is_finite(result.value):
        return "--"
    base = f"{result.value:.{decimals}f}"
    if result.ci_lower_95 is not None and result.ci_upper_95 is not None:
        return (
            f"{base} "
            f"[{result.ci_lower_95:.{ci_decimals}f}, {result.ci_upper_95:.{ci_decimals}f}]"
        )
    return base


def _is_finite(value: float) -> bool:
    # Avoid importing math; manual check keeps the module dependency-free.
    return value == value and value not in (float("inf"), float("-inf"))


def to_ascii(
    table: MethodComparisonTable,
    *,
    decimals: int = 1,
    ci_decimals: int = 1,
) -> str:
    """Render the table as a plain ASCII grid (for terminal review)."""
    by_key = table.by_method_and_metric()
    headers: list[str] = ["Method"]
    for m in table.metrics:
        unit = f" ({m.units})" if m.units else ""
        marker = (
            "↑"
            if m.direction == MetricDirection.HIGHER_IS_BETTER
            else "↓"
        )
        headers.append(f"{m.name}{unit} {marker}")
    rows: list[list[str]] = [headers]
    for method in table.methods:
        row = [method]
        for m in table.metrics:
            d_value = m.decimals if m.decimals is not None else decimals
            d_ci = m.decimals if m.decimals is not None else ci_decimals
            cell = _format_cell_value(
                by_key.get((method, m.name)),
                decimals=d_value,
                ci_decimals=d_ci,
            )
            row.append(cell)
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in widths)
    lines: list[str] = []
    if table.title:
        lines.append(table.title)
        lines.append("=" * max(len(table.title), len(sep)))
    formatted_rows = [
        "  ".join(rows[i][c].ljust(widths[c]) for c in range(len(headers)))
        for i in range(len(rows))
    ]
    lines.append(formatted_rows[0])
    lines.append(sep)
    lines.extend(formatted_rows[1:])
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Minimal LaTeX escape for the strings we render.

    We only escape the seven characters that have a special meaning
    inside a tabular cell. Math content is not expected; the comparison
    table is plain text.
    """
    if not text:
        return ""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _format_latex_cell(
    result: MethodResult | None,
    *,
    decimals: int,
    ci_decimals: int,
) -> str:
    if result is None or not _is_finite(result.value):
        return r"\textemdash"
    base = f"{result.value:.{decimals}f}"
    if result.ci_lower_95 is not None and result.ci_upper_95 is not None:
        return (
            f"{base} "
            rf"\,[{result.ci_lower_95:.{ci_decimals}f},\,{result.ci_upper_95:.{ci_decimals}f}]"
        )
    return base


def to_latex(
    table: MethodComparisonTable,
    *,
    decimals: int = 1,
    ci_decimals: int = 1,
    label: str | None = None,
) -> str:
    """Render a thesis-grade ``booktabs`` LaTeX table.

    The output uses ``\\toprule``/``\\midrule``/``\\bottomrule`` (so it
    requires the ``booktabs`` package) and emits one column per metric
    plus the leading ``Method`` column. Numeric columns are right-
    aligned to align decimal points. Direction markers are rendered as
    ``$\\uparrow$``/``$\\downarrow$`` next to the column header.

    The exporter does **not** introduce any other LaTeX dependency
    (no ``siunitx``, no ``threeparttable``); the result compiles inside
    a vanilla LaTeX article preamble that loads only ``booktabs``.

    The ``label`` argument is forwarded to ``\\label{...}`` so the paper
    can ``\\ref`` the table directly.
    """
    by_key = table.by_method_and_metric()
    column_spec = "l" + "r" * len(table.metrics)
    header_cells = ["\\textbf{Method}"]
    for m in table.metrics:
        unit_part = f" ({_latex_escape(m.units)})" if m.units else ""
        marker = (
            r" $\uparrow$"
            if m.direction == MetricDirection.HIGHER_IS_BETTER
            else r" $\downarrow$"
        )
        header_cells.append(f"\\textbf{{{_latex_escape(m.name)}{unit_part}}}{marker}")

    body_rows: list[str] = []
    for method in table.methods:
        cells = [_latex_escape(method)]
        for m in table.metrics:
            d_value = m.decimals if m.decimals is not None else decimals
            d_ci = m.decimals if m.decimals is not None else ci_decimals
            cells.append(
                _format_latex_cell(
                    by_key.get((method, m.name)),
                    decimals=d_value,
                    ci_decimals=d_ci,
                )
            )
        body_rows.append(" & ".join(cells) + r" \\")

    parts: list[str] = []
    parts.append(r"\begin{table}[t]")
    parts.append(r"\centering")
    if table.title:
        parts.append(rf"\caption{{{_latex_escape(table.title)}}}")
    if label:
        parts.append(rf"\label{{{label}}}")
    parts.append(rf"\begin{{tabular}}{{{column_spec}}}")
    parts.append(r"\toprule")
    parts.append(" & ".join(header_cells) + r" \\")
    parts.append(r"\midrule")
    parts.extend(body_rows)
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    return "\n".join(parts)
