from __future__ import annotations

import json
from pathlib import Path

from pipeline.stage_07_7_cams_gs_evidence.input_selection import select_inputs
from pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence import run_cams_gs_evidence


def _cfg(tmp_path: Path) -> dict:
    return {
        "project": {"root": str(tmp_path), "name": "site01", "run_id": "test"},
        "cams_gs_evidence": {
            "output_dir": "data/cams_gs/site01/evidence",
            "evidence_json": "data/cams_gs/site01/evidence/cams_gs_evidence.json",
            "summary_json": "runs/test/reports/cams_gs_evidence_summary.json",
            "viewer_html": "exports/cams_gs/site01/index.html",
            "viewer_manifest_json": "exports/cams_gs/site01/cams_gs_viewer_manifest.json",
            "stage45_manifest_json": "data/cams_gs/site01/train_manifest.json",
            "stage45_training_status_json": "data/cams_gs/site01/training_status.json",
            "stage45_dataset_dir": "data/cams_gs/site01/nerfstudio_dataset",
            "viewer_export_manifest_json": "exports/viewer/site01/viewer_manifest.json",
        },
    }


def _write_inputs(tmp_path: Path, *, trained: bool = False) -> None:
    root = tmp_path / "data/cams_gs/site01"
    dataset = root / "nerfstudio_dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "prepared",
        "inputs": {"image_count": 4, "selected_image_count": 4, "dataset_dir": dataset.as_posix()},
        "training": {"training_executed": trained, "nerfstudio_method": "splatfacto"},
    }
    (root / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "training_status.json").write_text(json.dumps({"status": "prepared", "training_executed": trained}), encoding="utf-8")
    viewer = tmp_path / "exports/viewer/site01"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "viewer_manifest.json").write_text(json.dumps({"artifacts": [{"key": "cleaned_cloud"}]}), encoding="utf-8")


def test_select_inputs_reads_stage45_manifest(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    selected = select_inputs(_cfg(tmp_path))
    assert selected.stage45_manifest["status"] == "prepared"
    assert selected.training_status["training_executed"] is False
    assert selected.warnings == []


def test_run_evidence_prepared_no_training(tmp_path: Path) -> None:
    _write_inputs(tmp_path, trained=False)
    evidence = run_cams_gs_evidence(_cfg(tmp_path), force=True, log_level="ERROR")
    assert evidence["status"] == "prepared_no_training"
    assert evidence["readiness"] == "prepared_stub_only"
    assert evidence["is_metric_truth"] is False
    assert Path(evidence["outputs"]["evidence_json"]).exists()
    assert Path(evidence["outputs"]["viewer_html"]).exists()


def test_run_evidence_missing_stage45_is_skip_safe(tmp_path: Path) -> None:
    evidence = run_cams_gs_evidence(_cfg(tmp_path), force=True, log_level="ERROR")
    assert evidence["status"] == "skipped_missing_stage45_manifest"
    assert any("missing_stage45_manifest" in str(w) for w in evidence["warnings"])
    assert Path(evidence["outputs"]["evidence_json"]).exists()
