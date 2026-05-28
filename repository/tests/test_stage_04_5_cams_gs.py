from __future__ import annotations

from pathlib import Path

from pipeline.stage_04_5_cams_gs.input_selection import select_inputs
from pipeline.stage_04_5_cams_gs.run_cams_gs_prepare import run_cams_gs_prepare


def _cfg(tmp_path: Path) -> dict:
    return {
        "project": {"root": str(tmp_path), "name": "site01", "run_id": "test"},
        "cams_gs": {
            "output_dir": "data/cams_gs/site01",
            "nerfstudio_dataset_dir": "data/cams_gs/site01/nerfstudio_dataset",
            "manifest_json": "data/cams_gs/site01/train_manifest.json",
            "training_status_json": "data/cams_gs/site01/training_status.json",
            "report_json": "runs/test/reports/cams_gs_prepare_summary.json",
            "source_images_dir": "data/sfm/site01/images",
            "source_sparse_model_dir": "data/sparse_refined/site01/0",
            "execute_training": False,
            "max_training_images": 2,
        },
    }


def test_select_inputs_counts_and_caps_images(tmp_path: Path) -> None:
    images = tmp_path / "data/sfm/site01/images"
    images.mkdir(parents=True)
    for idx in range(4):
        (images / f"img_{idx}.jpg").write_bytes(b"jpg")

    sparse = tmp_path / "data/sparse_refined/site01/0"
    sparse.mkdir(parents=True)

    selected = select_inputs(_cfg(tmp_path))

    assert selected.image_count == 4
    assert len(selected.selected_images) == 2
    assert any(str(w).startswith("image_limit_applied") for w in selected.warnings)


def test_run_prepare_writes_manifest(tmp_path: Path) -> None:
    images = tmp_path / "data/sfm/site01/images"
    images.mkdir(parents=True)
    (images / "img.jpg").write_bytes(b"jpg")
    (tmp_path / "data/sparse_refined/site01/0").mkdir(parents=True)

    manifest = run_cams_gs_prepare(_cfg(tmp_path), force=True, log_level="ERROR")

    assert manifest["status"] == "prepared"
    assert Path(manifest["paths"]["manifest_json"]).exists()
    assert Path(manifest["paths"]["training_status_json"]).exists()
    assert manifest["training"]["training_executed"] is False
    assert manifest["is_metric_truth"] is False


def test_run_prepare_skip_safe_without_images(tmp_path: Path) -> None:
    manifest = run_cams_gs_prepare(_cfg(tmp_path), force=True, log_level="ERROR")

    assert manifest["status"] == "skipped_missing_images"
    assert Path(manifest["paths"]["manifest_json"]).exists()
    assert Path(manifest["paths"]["training_status_json"]).exists()
    assert any("missing_or_empty_images_dir" in str(w) for w in manifest["warnings"])
