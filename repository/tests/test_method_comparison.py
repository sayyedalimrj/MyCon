"""Tests for :mod:`pipeline.common.method_comparison`.

Lock the contract of:

- ``MetricSpec`` / ``MethodResult`` / ``MethodComparisonTable``
  validation (unique names, declared metric/method, no duplicates,
  monotone CI bounds);
- the dict round trip (``to_dict`` -> ``from_dict``);
- the ASCII and LaTeX renderers (per-metric decimals override,
  NaN cells render as placeholder, direction markers honoured,
  LaTeX special characters escaped).
"""

from __future__ import annotations

import json
import math

import pytest

from pipeline.common.method_comparison import (
    METHOD_COMPARISON_SCHEMA_VERSION,
    MethodComparisonTable,
    MethodResult,
    MetricDirection,
    MetricSpec,
    build_table,
    from_dict,
    to_ascii,
    to_dict,
    to_latex,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _toy_table(*, ours_ece: float | None = 0.07) -> MethodComparisonTable:
    return build_table(
        title="Bidirectional accuracy on Walls (synthetic)",
        metrics=[
            MetricSpec(
                name="F-score @ 5cm",
                direction=MetricDirection.HIGHER_IS_BETTER,
                units="%",
                decimals=1,
            ),
            MetricSpec(
                name="ECE",
                direction=MetricDirection.LOWER_IS_BETTER,
                decimals=3,
            ),
        ],
        methods=["Bosche 2010", "Mahami AiC 2024", "Ours"],
        results=[
            MethodResult(
                method="Bosche 2010",
                metric="F-score @ 5cm",
                value=72.4,
                source="AESM 2010",
            ),
            MethodResult(
                method="Mahami AiC 2024",
                metric="F-score @ 5cm",
                value=88.1,
                source="CORE-4",
            ),
            MethodResult(
                method="Ours",
                metric="F-score @ 5cm",
                value=89.3,
                ci_lower_95=85.0,
                ci_upper_95=92.6,
                n=120,
                source="runs/example_walkthrough",
            ),
            MethodResult(
                method="Mahami AiC 2024",
                metric="ECE",
                value=float("nan"),
                source="not reported",
            ),
            MethodResult(
                method="Ours",
                metric="ECE",
                value=ours_ece if ours_ece is not None else float("nan"),
                ci_lower_95=ours_ece - 0.02 if ours_ece is not None else None,
                ci_upper_95=ours_ece + 0.02 if ours_ece is not None else None,
                n=120,
                source="runs/example_walkthrough",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# MetricSpec validation
# ---------------------------------------------------------------------------


def test_metric_spec_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError):
        MetricSpec(name="X", direction="sideways")


def test_metric_spec_accepts_documented_directions() -> None:
    MetricSpec(name="A", direction=MetricDirection.HIGHER_IS_BETTER)
    MetricSpec(name="B", direction=MetricDirection.LOWER_IS_BETTER)


def test_metric_direction_all_lists_both_options() -> None:
    assert set(MetricDirection.all()) == {
        MetricDirection.HIGHER_IS_BETTER,
        MetricDirection.LOWER_IS_BETTER,
    }


# ---------------------------------------------------------------------------
# MethodResult validation
# ---------------------------------------------------------------------------


def test_method_result_rejects_partial_ci() -> None:
    with pytest.raises(ValueError):
        MethodResult(method="Ours", metric="F", value=1.0, ci_lower_95=0.9)
    with pytest.raises(ValueError):
        MethodResult(method="Ours", metric="F", value=1.0, ci_upper_95=1.1)


def test_method_result_rejects_inverted_ci() -> None:
    with pytest.raises(ValueError):
        MethodResult(
            method="Ours", metric="F", value=1.0, ci_lower_95=2.0, ci_upper_95=1.0
        )


def test_method_result_accepts_nan_value_with_no_ci() -> None:
    r = MethodResult(method="Ours", metric="F", value=float("nan"))
    assert math.isnan(r.value)


# ---------------------------------------------------------------------------
# Table validation
# ---------------------------------------------------------------------------


def test_table_rejects_duplicate_metric_names() -> None:
    with pytest.raises(ValueError):
        MethodComparisonTable(
            title="x",
            metrics=(
                MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER),
                MetricSpec(name="F", direction=MetricDirection.LOWER_IS_BETTER),
            ),
            methods=("Ours",),
            results=(),
        )


def test_table_rejects_duplicate_method_labels() -> None:
    with pytest.raises(ValueError):
        MethodComparisonTable(
            title="x",
            metrics=(MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER),),
            methods=("Ours", "Ours"),
            results=(),
        )


def test_table_rejects_result_with_unknown_metric() -> None:
    with pytest.raises(ValueError):
        MethodComparisonTable(
            title="x",
            metrics=(MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER),),
            methods=("Ours",),
            results=(MethodResult(method="Ours", metric="ECE", value=0.0),),
        )


def test_table_rejects_result_with_unknown_method() -> None:
    with pytest.raises(ValueError):
        MethodComparisonTable(
            title="x",
            metrics=(MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER),),
            methods=("Ours",),
            results=(MethodResult(method="Ghost", metric="F", value=0.0),),
        )


def test_table_rejects_duplicate_method_metric_pair() -> None:
    with pytest.raises(ValueError):
        MethodComparisonTable(
            title="x",
            metrics=(MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER),),
            methods=("Ours",),
            results=(
                MethodResult(method="Ours", metric="F", value=1.0),
                MethodResult(method="Ours", metric="F", value=2.0),
            ),
        )


# ---------------------------------------------------------------------------
# build_table accepts plain dicts in the results list
# ---------------------------------------------------------------------------


def test_build_table_accepts_dict_rows() -> None:
    table = build_table(
        title="x",
        metrics=[MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER)],
        methods=["Ours"],
        results=[
            {
                "method": "Ours",
                "metric": "F",
                "value": 0.9,
                "ci_lower_95": 0.85,
                "ci_upper_95": 0.95,
                "n": 50,
                "source": "runs/x",
                "notes": ["one", "two"],
            }
        ],
    )
    assert len(table.results) == 1
    r = table.results[0]
    assert r.value == pytest.approx(0.9)
    assert r.ci_lower_95 == pytest.approx(0.85)
    assert r.ci_upper_95 == pytest.approx(0.95)
    assert r.n == 50
    assert r.source == "runs/x"
    assert r.notes == ("one", "two")


def test_build_table_records_generated_at_when_not_supplied() -> None:
    table = build_table(
        title="x",
        metrics=[MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER)],
        methods=["Ours"],
        results=[],
    )
    assert table.generated_at_utc != ""


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_every_field() -> None:
    table = _toy_table()
    payload = to_dict(table)
    restored = from_dict(payload)
    # Validate we can re-serialize without any change.
    assert to_dict(restored) == payload
    assert restored.schema_version == METHOD_COMPARISON_SCHEMA_VERSION


def test_round_trip_preserves_per_metric_decimals() -> None:
    table = _toy_table()
    restored = from_dict(json.loads(json.dumps(to_dict(table))))
    assert restored.metrics[0].decimals == 1
    assert restored.metrics[1].decimals == 3


def test_from_dict_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError):
        from_dict(
            {
                "schema_version": "method_comparison.vBOGUS",
                "title": "x",
                "metrics": [],
                "methods": [],
                "results": [],
            }
        )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_ascii_renders_per_metric_decimals_correctly() -> None:
    table = _toy_table(ours_ece=0.072)
    out = to_ascii(table)
    assert "89.3 [85.0, 92.6]" in out  # F-score row with 1-decimal rendering
    assert "0.072 [0.052, 0.092]" in out  # ECE row with 3-decimal rendering


def test_ascii_renders_nan_cells_as_placeholder() -> None:
    table = _toy_table()
    out = to_ascii(table)
    # Mahami's ECE is float('nan'); placeholder must be '--'.
    assert "Mahami AiC 2024" in out
    line_with_mahami = [ln for ln in out.splitlines() if ln.startswith("Mahami AiC 2024")][0]
    assert "--" in line_with_mahami


def test_ascii_includes_direction_markers() -> None:
    out = to_ascii(_toy_table())
    assert "↑" in out  # F-score is higher_is_better
    assert "↓" in out  # ECE is lower_is_better


def test_latex_uses_booktabs_rules() -> None:
    out = to_latex(_toy_table(), label="tab:cmp")
    assert r"\toprule" in out
    assert r"\midrule" in out
    assert r"\bottomrule" in out
    assert r"\label{tab:cmp}" in out
    assert r"$\uparrow$" in out
    assert r"$\downarrow$" in out


def test_latex_escapes_percent_sign() -> None:
    out = to_latex(_toy_table())
    # Units '%' must be LaTeX-escaped.
    assert r"\%" in out
    # Bare '%' would only ever appear as a comment marker; it must not
    # appear unescaped on a content line.
    for line in out.splitlines():
        if "%" in line and r"\%" not in line:
            pytest.fail(f"unescaped percent in LaTeX line: {line!r}")


def test_latex_renders_nan_cells_as_textemdash() -> None:
    out = to_latex(_toy_table())
    assert r"\textemdash" in out


def test_latex_omits_label_when_not_supplied() -> None:
    out = to_latex(_toy_table(), label=None)
    assert r"\label{" not in out


def test_ascii_returns_just_one_method_when_only_one_row_present() -> None:
    table = build_table(
        title="single-method",
        metrics=[MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER)],
        methods=["Ours"],
        results=[MethodResult(method="Ours", metric="F", value=0.9)],
    )
    out = to_ascii(table)
    assert "Ours" in out
    assert "single-method" in out


def test_ascii_handles_empty_table_gracefully() -> None:
    table = build_table(
        title="empty",
        metrics=[MetricSpec(name="F", direction=MetricDirection.HIGHER_IS_BETTER)],
        methods=[],
        results=[],
    )
    out = to_ascii(table)
    # Header line must still be present.
    assert "F" in out


# ---------------------------------------------------------------------------
# LaTeX render via the side-car script
# ---------------------------------------------------------------------------


def test_render_method_comparison_latex_script_writes_file(tmp_path) -> None:
    from scripts.render_method_comparison_latex import main as render_main

    table = _toy_table()
    in_path = tmp_path / "table.json"
    in_path.write_text(json.dumps(to_dict(table)), encoding="utf-8")
    out_tex = tmp_path / "table.tex"
    rc = render_main(
        [
            "--input",
            str(in_path),
            "--output",
            str(out_tex),
            "--label",
            "tab:cmp",
        ]
    )
    assert rc == 0
    body = out_tex.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in body
    assert r"\label{tab:cmp}" in body


def test_render_script_rejects_bad_schema_version(tmp_path) -> None:
    from scripts.render_method_comparison_latex import main as render_main

    in_path = tmp_path / "wrong.json"
    in_path.write_text(json.dumps({"schema_version": "method_comparison.vBOGUS"}), encoding="utf-8")
    rc = render_main(["--input", str(in_path), "--output", str(tmp_path / "out.tex")])
    assert rc == 2


def test_render_script_returns_non_zero_when_input_missing(tmp_path) -> None:
    from scripts.render_method_comparison_latex import main as render_main

    rc = render_main(
        ["--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "out.tex")]
    )
    assert rc == 1
