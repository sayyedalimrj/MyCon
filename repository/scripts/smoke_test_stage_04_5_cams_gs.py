from __future__ import annotations

import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_04_5_cams_gs.run_cams_gs_prepare import run_cams_gs_prepare


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="stage045_smoke_"))

    images_dir = root / "data/sfm/site01/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(5):
        (images_dir / f"frame_{idx:03d}.jpg").write_bytes(b"fake-jpg")

    sparse = root / "data/sparse_refined/site01/0"
    sparse.mkdir(parents=True, exist_ok=True)

    cfg = {
        "project": {"root": str(root), "name": "site01", "run_id": "test"},
        "cams_gs": {
            "output_dir": "data/cams_gs/site01",
            "nerfstudio_dataset_dir": "data/cams_gs/site01/nerfstudio_dataset",
            "manifest_json": "data/cams_gs/site01/train_manifest.json",
            "training_status_json": "data/cams_gs/site01/training_status.json",
            "report_json": "runs/test/reports/cams_gs_prepare_summary.json",
            "source_images_dir": "data/sfm/site01/images",
            "source_sparse_model_dir": "data/sparse_refined/site01/0",
            "execute_training": False,
            "max_training_images": 3,
        },
    }

    manifest = run_cams_gs_prepare(cfg, force=True, log_level="ERROR")

    assert manifest["status"] == "prepared"
    assert manifest["inputs"]["image_count"] == 5
    assert manifest["inputs"]["selected_image_count"] == 3
    assert Path(manifest["paths"]["manifest_json"]).exists()
    assert Path(manifest["paths"]["training_status_json"]).exists()

    print(
        "STAGE_04_5_SMOKE_OK "
        f"status={manifest['status']} "
        f"images={manifest['inputs']['image_count']} "
        f"selected={manifest['inputs']['selected_image_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
