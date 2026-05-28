"""Tests for ``pipeline.service.artifacts``.

The artifact-discovery layer is read-only; tests build a synthetic
``runs/<rid>/reports`` tree and verify each contract:

- One :class:`ArtifactSummary` is returned per registry entry that
  declares a ``report_basename``.
- Present / absent files are correctly distinguished.
- The ``provenance`` block is parsed when present.
- Malformed JSON yields a structured ``parse_error`` rather than raising.
- Top-level keys are previewed up to the documented limit, with long
  string values truncated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.common.registry import STAGE_REGISTRY
from pipeline.service.artifacts import discover_run_artifacts


def test_returns_one_summary_per_registry_stage_with_report_basename(tmp_path: Path) -> None:
    summaries = discover_run_artifacts("does-not-exist", project_root=tmp_path)
    expected = sum(1 for d in STAGE_REGISTRY if d.report_basename)
    assert len(summaries) == expected


def test_marks_missing_files_as_not_present(tmp_path: Path) -> None:
    summaries = discover_run_artifacts("missing-run", project_root=tmp_path)
    assert summaries
    assert all(s.exists is False for s in summaries)
    assert all(s.size_bytes == 0 for s in summaries)
    assert all(s.parse_error is None for s in summaries)


def test_parses_present_report_with_provenance(tmp_path: Path) -> None:
    reports = tmp_path / "runs" / "r-1" / "reports"
    reports.mkdir(parents=True)
    (reports / "registration_report.json").write_text(json.dumps({
        "stage": "stage_08_bim_eval",
        "status": "complete",
        "elapsed_sec": 14.0,
        "icp": {"fitness": 0.62, "inlier_rmse": 0.034},
        "provenance": {
            "schema_version": "1.0",
            "stage": "stage_08_bim_registration",
            "config_hash": "deadbeef",
            "git_sha": "abc123",
        },
    }))
    summaries = discover_run_artifacts("r-1", project_root=tmp_path)
    by_stage = {s.stage: s for s in summaries}
    bim = by_stage["stage_08_bim_registration"]
    assert bim.exists is True
    assert bim.status == "complete"
    assert bim.provenance is not None
    assert bim.provenance["config_hash"] == "deadbeef"
    assert bim.parse_error is None


def test_handles_malformed_json_gracefully(tmp_path: Path) -> None:
    reports = tmp_path / "runs" / "r-2" / "reports"
    reports.mkdir(parents=True)
    (reports / "progress_summary.json").write_text("not-json{[")
    summaries = discover_run_artifacts("r-2", project_root=tmp_path)
    by_stage = {s.stage: s for s in summaries}
    progress = by_stage["stage_09_progress"]
    assert progress.exists is True
    assert progress.status is None
    assert progress.parse_error is not None
    assert "json decode" in progress.parse_error


def test_preview_excludes_provenance_and_caps_strings(tmp_path: Path) -> None:
    reports = tmp_path / "runs" / "r-3" / "reports"
    reports.mkdir(parents=True)
    long_string = "x" * 500
    (reports / "progress_summary.json").write_text(json.dumps({
        "stage": "stage_09_progress",
        "status": "complete",
        "long_field": long_string,
        "provenance": {"config_hash": "deadbeef"},
    }))
    summaries = discover_run_artifacts("r-3", project_root=tmp_path)
    by_stage = {s.stage: s for s in summaries}
    progress = by_stage["stage_09_progress"]
    assert "provenance" not in progress.preview
    assert "long_field" in progress.preview
    # Truncated; original was 500 chars, the preview adds an ellipsis.
    assert len(progress.preview["long_field"]) < 250
    assert progress.preview["long_field"].endswith("…")


def test_to_dict_is_json_round_trippable(tmp_path: Path) -> None:
    reports = tmp_path / "runs" / "r-4" / "reports"
    reports.mkdir(parents=True)
    (reports / "registration_report.json").write_text(json.dumps({
        "stage": "stage_08_bim_eval",
        "status": "complete",
    }))
    summaries = discover_run_artifacts("r-4", project_root=tmp_path)
    encoded = json.dumps([s.to_dict() for s in summaries])
    decoded = json.loads(encoded)
    assert isinstance(decoded, list)
    assert len(decoded) == len(summaries)


def test_handles_non_object_root_in_report(tmp_path: Path) -> None:
    """A report whose top-level JSON is a list, not an object, must surface
    as ``parse_error`` rather than crash."""
    reports = tmp_path / "runs" / "r-5" / "reports"
    reports.mkdir(parents=True)
    (reports / "progress_summary.json").write_text(json.dumps([1, 2, 3]))
    summaries = discover_run_artifacts("r-5", project_root=tmp_path)
    by_stage = {s.stage: s for s in summaries}
    progress = by_stage["stage_09_progress"]
    assert progress.parse_error is not None
    assert "JSON object" in progress.parse_error
