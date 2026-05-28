"""Tests for ``scripts/aggregate_run_metrics.py``.

The aggregator is read-only. We exercise it against a synthetic reports
directory rather than by running the pipeline, so the test is fast and
deterministic and does not need Open3D / COLMAP.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "aggregate_run_metrics.py"


def _load_aggregator_module():
    """Import scripts/aggregate_run_metrics.py without invoking its CLI."""
    spec = importlib.util.spec_from_file_location("aggregate_run_metrics", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_run_metrics"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_reports_dir(tmp_path: Path) -> Path:
    reports = tmp_path / "runs" / "tcase" / "reports"
    reports.mkdir(parents=True)

    (reports / "progress_summary.json").write_text(
        json.dumps(
            {
                "stage": "stage_09_progress",
                "status": "complete",
                "elapsed_sec": 3.45,
                "bidirectional_summary": {
                    "accuracy": 0.82,
                    "completeness": 0.71,
                    "f_score": 0.76,
                    "tau_m": 0.05,
                },
                "registration_quality": {
                    "confidence_label": "medium",
                    "confidence_score": 0.65,
                    "fitness": 0.42,
                },
            }
        )
    )

    (reports / "registration_report.json").write_text(
        json.dumps(
            {
                "stage": "stage_08_bim_eval",
                "status": "complete",
                "elapsed_sec": 14.0,
                "icp": {
                    "method": "point_to_point+point_to_plane",
                    "fitness": 0.62,
                    "inlier_rmse": 0.034,
                    "robust_loss": {
                        "requested": "tukey",
                        "applied": "tukey",
                        "k_m": 0.05,
                        "binding_supports_kernel": True,
                        "fallback_reason": None,
                    },
                },
                "quality_gate": {"passed": True, "failures": []},
            }
        )
    )

    (reports / "dense_summary.json").write_text(
        json.dumps(
            {
                "stage": "stage_05_dense_stereo",
                "status": "ok",
                "elapsed_sec": 120.0,
                "dense_stats": {"fused_vertex_count": 1234567},
            }
        )
    )

    return reports


def test_aggregate_emits_one_record_per_stage(fake_reports_dir: Path) -> None:
    mod = _load_aggregator_module()
    records = mod.aggregate(fake_reports_dir)
    # All 14 known stages produce a record (with status "missing_report" for absent ones).
    assert len(records) == 14
    by_stage = {r["stage"]: r for r in records}
    assert by_stage["stage_05_dense"]["status"] == "ok"
    assert by_stage["stage_08_bim_registration"]["status"] == "complete"
    assert by_stage["stage_09_progress"]["status"] == "complete"
    # Stages without a report are honestly flagged, not silently dropped.
    assert by_stage["stage_03_colmap"]["status"] == "missing_report"


def test_aggregate_surfaces_robust_loss_decision(fake_reports_dir: Path) -> None:
    mod = _load_aggregator_module()
    records = mod.aggregate(fake_reports_dir)
    by_stage = {r["stage"]: r for r in records}
    record = by_stage["stage_08_bim_registration"]
    assert record["icp_robust_loss_requested"] == "tukey"
    assert record["icp_robust_loss_applied"] == "tukey"
    assert record["icp_robust_loss_k_m"] == 0.05
    assert record["icp_method"] == "point_to_point+point_to_plane"


def test_aggregate_surfaces_bidirectional_metrics(fake_reports_dir: Path) -> None:
    mod = _load_aggregator_module()
    records = mod.aggregate(fake_reports_dir)
    by_stage = {r["stage"]: r for r in records}
    record = by_stage["stage_09_progress"]
    assert record["bidirectional_accuracy"] == 0.82
    assert record["bidirectional_completeness"] == 0.71
    assert record["bidirectional_f_score"] == 0.76
    assert record["registration_confidence_label"] == "medium"


def test_write_outputs_produces_json_and_csv(fake_reports_dir: Path, tmp_path: Path) -> None:
    mod = _load_aggregator_module()
    records = mod.aggregate(fake_reports_dir)
    out_json = tmp_path / "run_metrics.json"
    out_csv = tmp_path / "run_metrics.csv"
    mod.write_outputs(records, out_json, out_csv)
    assert out_json.exists() and out_json.stat().st_size > 0
    assert out_csv.exists() and out_csv.stat().st_size > 0

    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(parsed, list) and len(parsed) == 14
    # CSV first line is header; second-line records should be present.
    csv_lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) >= 15  # header + at least 14 stage rows


def test_aggregate_missing_dir_is_handled_by_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_aggregator_module()
    rc = mod.main(["--reports-dir", str(tmp_path / "does_not_exist")])
    assert rc == 1
    assert "AGGREGATE_RUN_METRICS_FAILED" in capsys.readouterr().err
