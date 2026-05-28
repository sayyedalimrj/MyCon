from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_07_7_cams_gs_evidence.run_cams_gs_evidence import run_cams_gs_evidence


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="stage077_smoke_"))

    stage45 = root / "data/cams_gs/site01"
    dataset = stage45 / "nerfstudio_dataset"
    dataset.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": "stage_04_5_cams_gs_prepare",
        "status": "prepared",
        "inputs": {"image_count": 10, "selected_image_count": 8, "dataset_dir": dataset.as_posix()},
        "training": {"training_executed": False, "nerfstudio_method": "splatfacto"},
    }
    (stage45 / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (stage45 / "training_status.json").write_text(json.dumps({"status": "prepared", "training_executed": False}), encoding="utf-8")

    viewer_dir = root / "exports/viewer/site01"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    (viewer_dir / "viewer_manifest.json").write_text(json.dumps({"artifacts": [{"key": "a"}, {"key": "b"}]}), encoding="utf-8")

    cfg = {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
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

    evidence = run_cams_gs_evidence(cfg, force=True, log_level="ERROR")

    assert evidence["status"] == "prepared_no_training"
    assert evidence["is_metric_truth"] is False
    assert Path(evidence["outputs"]["evidence_json"]).exists()
    assert Path(evidence["outputs"]["viewer_html"]).exists()

    print(
        "STAGE_07_7_SMOKE_OK "
        f"status={evidence['status']} "
        f"readiness={evidence['readiness']} "
        f"viewer={evidence['outputs']['viewer_html']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
