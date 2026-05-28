"""Tests for the end-to-end walkthrough fixture and runner.

These tests pin the contract of the synthetic walkthrough at
:file:`examples/end_to_end/`:

- the runner produces the documented set of artefacts,
- each artefact carries its documented ``schema_version``,
- Stage 11 still classifies activity A0001 as ``on_schedule`` and
  activity A0432 as ``behind`` at the canonical data date,
- the calibration report consumes the HITL log (3 confirms + 3
  overrules) and produces a numeric ECE in [0, 1],
- the grounding-guard demo covers all three failure modes
  (well_grounded passes, hallucinated_numeric fails on a numeric
  claim, unsupported_named_entity fails on a named-entity claim).

The walkthrough is the canonical reproducibility fixture for thesis
defence; locking these invariants here means a regression in any
Phase 4 module that affects the headline result will break this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_end_to_end_walkthrough import main as walkthrough_main


pytestmark = pytest.mark.lightweight


# Canonical inputs ship in the repo under examples/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = PROJECT_ROOT / "examples" / "end_to_end" / "inputs"


def _run_walkthrough(out_dir: Path, *, data_date: str = "2026-04-16") -> int:
    return walkthrough_main(
        [
            "--output-dir", str(out_dir),
            "--inputs-dir", str(INPUTS_DIR),
            "--data-date-utc", data_date,
        ]
    )


# ---------------------------------------------------------------------------
# Run + artefact existence + schema versions
# ---------------------------------------------------------------------------


def test_walkthrough_runs_and_writes_documented_artefacts(tmp_path: Path) -> None:
    rc = _run_walkthrough(tmp_path / "out")
    assert rc == 0
    expected = {
        "activity_progress.json": "activity_progress.v1",
        "schedule_variance.json": "schedule_variance.v1",
        "dashboard_summary.json": "dashboard_summary.v1",
        "calibration_report.json": "calibration_report.v1",
        "grounding_guard_demo.json": "grounding_guard_demo.v1",
        "walkthrough_summary.json": "walkthrough_summary.v1",
    }
    out = tmp_path / "out"
    for fname, schema_version in expected.items():
        path = out / fname
        assert path.exists(), f"walkthrough did not produce {fname}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc.get("schema_version") == schema_version, (
            f"{fname} carries {doc.get('schema_version')!r}, expected {schema_version!r}"
        )


def test_walkthrough_summary_links_every_output_with_sha256(tmp_path: Path) -> None:
    rc = _run_walkthrough(tmp_path / "out")
    assert rc == 0
    summary = json.loads(
        (tmp_path / "out" / "walkthrough_summary.json").read_text(encoding="utf-8")
    )
    files = summary["files"]
    # Every linked output must exist and the recorded sha-256 must match
    # the file on disk byte-for-byte.
    import hashlib

    for entry in files.values():
        path = Path(entry["path"])
        assert path.exists(), entry
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"]


# ---------------------------------------------------------------------------
# Stage 11 invariants on the canonical data date
# ---------------------------------------------------------------------------


def test_stage_11_classifies_a0001_on_schedule_and_a0432_behind(tmp_path: Path) -> None:
    rc = _run_walkthrough(tmp_path / "out")
    assert rc == 0
    variance = json.loads(
        (tmp_path / "out" / "schedule_variance.json").read_text(encoding="utf-8")
    )
    by_id = {a["activity_id"]: a for a in variance["activities"]}
    assert by_id["A0001"]["status"] == "on_schedule"
    assert by_id["A0432"]["status"] == "behind"
    # The canonical fixture intentionally has more behind than ahead
    # activities so the dashboard renders a behind row.
    assert variance["n_behind"] >= 1
    assert variance["n_unknown_evidence"] == 0


# ---------------------------------------------------------------------------
# Calibration invariants
# ---------------------------------------------------------------------------


def test_calibration_report_has_finite_metrics_in_unit_interval(tmp_path: Path) -> None:
    rc = _run_walkthrough(tmp_path / "out")
    assert rc == 0
    report = json.loads(
        (tmp_path / "out" / "calibration_report.json").read_text(encoding="utf-8")
    )
    assert report["n_samples"] == 6  # 6 effective corrections in the fixture
    metrics = report["metrics"]
    for key in (
        "expected_calibration_error",
        "maximum_calibration_error",
        "brier_score",
        "smooth_ece",
    ):
        v = metrics[key]
        assert isinstance(v, (int, float))
        assert 0.0 <= v <= 1.0, f"{key}={v} not in [0, 1]"
    # The walkthrough records its own provenance against the HITL log.
    assert report["walkthrough_provenance"]["n_replayed_records"] >= 6


# ---------------------------------------------------------------------------
# Grounding-guard demo invariants
# ---------------------------------------------------------------------------


def test_grounding_guard_demo_covers_three_failure_modes(tmp_path: Path) -> None:
    rc = _run_walkthrough(tmp_path / "out")
    assert rc == 0
    demo = json.loads(
        (tmp_path / "out" / "grounding_guard_demo.json").read_text(encoding="utf-8")
    )
    by_label = {r["label"]: r for r in demo["results"]}
    # Well-grounded must pass.
    assert by_label["well_grounded"]["passed"] is True
    # Hallucinated numeric must fail with at least one unsupported numeric claim.
    assert by_label["hallucinated_numeric"]["passed"] is False
    assert by_label["hallucinated_numeric"]["n_unsupported"] >= 1
    # Unsupported named entity must fail because the activity_id is not
    # in the local evidence package.
    assert by_label["unsupported_named_entity"]["passed"] is False
    assert by_label["unsupported_named_entity"]["n_unsupported"] >= 1


# ---------------------------------------------------------------------------
# Robustness: bad inputs path returns a non-zero exit code with a useful
# print contract.
# ---------------------------------------------------------------------------


def test_walkthrough_fails_loudly_on_missing_inputs_dir(tmp_path: Path, capsys) -> None:
    rc = walkthrough_main(
        [
            "--output-dir", str(tmp_path / "out"),
            "--inputs-dir", str(tmp_path / "does_not_exist"),
            "--data-date-utc", "2026-04-16",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "WALKTHROUGH_FAILED" in captured.err
