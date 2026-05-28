#!/usr/bin/env python3
"""Fast Stage 3 smoke test using a fake COLMAP executable.

The real COLMAP run is intentionally not used here; the smoke test verifies file
contracts, CLI plumbing, command construction, stats parsing, and reports in a
temporary folder.
"""
from __future__ import annotations

import csv
import stat
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from pipeline.common.config import load_config
from pipeline.stage_03_colmap.run_sparse import run_sparse


def _write_fake_colmap(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def arg_value(name: str, default: str | None = None) -> str | None:
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

if cmd in {"help", "-h", "--help"}:
    print("COLMAP fake 4.0.3")
    print("feature_extractor")
    print("sequential_matcher")
    print("mapper")
    print("model_converter")
    print("model_analyzer")
    raise SystemExit(0)

if cmd == "feature_extractor":
    db = Path(arg_value("--database_path", "/tmp/fake.db"))
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"fake database after feature extraction\n")
    print("Feature extraction complete")
    raise SystemExit(0)

if cmd == "sequential_matcher":
    db = Path(arg_value("--database_path", "/tmp/fake.db"))
    with db.open("ab") as handle:
        handle.write(b"fake matches\n")
    print("Sequential matching complete")
    raise SystemExit(0)

if cmd == "mapper":
    out = Path(arg_value("--output_path", "/tmp/fake_sparse")) / "0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras.bin").write_bytes(b"camera-binary")
    (out / "images.bin").write_bytes(b"image-binary" * 8)
    (out / "points3D.bin").write_bytes(b"points-binary" * 16)
    print("Mapper complete")
    raise SystemExit(0)

if cmd == "model_converter":
    out = Path(arg_value("--output_path", "/tmp/fake_text"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 SIMPLE_RADIAL 60 40 50 30 20 0.01\n",
        encoding="utf-8",
    )
    (out / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "1 1 0 0 0 0 0 0 1 site01_kf_00001_f000001.jpg\n\n"
        "2 1 0 0 0 1 0 0 1 site01_kf_00002_f000002.jpg\n\n"
        "3 1 0 0 0 2 0 0 1 site01_kf_00003_f000003.jpg\n\n"
        "4 1 0 0 0 3 0 0 1 site01_kf_00004_f000004.jpg\n\n",
        encoding="utf-8",
    )
    (out / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "1 0 0 0 255 255 255 0.1 1 0 2 0\n"
        "2 1 0 0 255 255 255 0.1 3 0 4 0\n",
        encoding="utf-8",
    )
    print("Model conversion complete")
    raise SystemExit(0)

if cmd == "model_analyzer":
    print("Registered images: 4")
    print("Points: 2")
    raise SystemExit(0)

print(f"Unsupported fake COLMAP command: {cmd}", file=sys.stderr)
raise SystemExit(2)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_images_and_manifest(root: Path, count: int = 4) -> None:
    key_dir = root / "data" / "frames" / "key" / "site01"
    key_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for idx in range(count):
        image = np.full((40, 60, 3), 30 + idx * 40, dtype=np.uint8)
        image_path = key_dir / f"site01_kf_{idx + 1:05d}_f{idx + 1:06d}.jpg"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Failed to write smoke image: {image_path}")

        rows.append(
            {
                "keyframe_id": f"kf_{idx + 1:05d}",
                "source_frame_index": str(idx + 1),
                "timestamp_sec": f"{idx / 30.0:.6f}",
                "image_path": str(image_path.relative_to(root)),
                "segment_id": "0",
                "sharpness_laplacian": "100.0",
                "exposure_mean": "0.5",
                "motion_score": "0.1",
                "novelty_score": "0.2",
                "quality_score": "0.9",
                "keep_sparse": "true",
                "keep_dense": "true",
                "selection_reason": "smoke",
            }
        )

    manifest = root / "data" / "frames" / "key" / "site01_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_config(root: Path, fake_colmap: Path) -> Path:
    template_path = PROJECT_ROOT / "configs" / "site01.yaml"
    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    cfg["project"]["name"] = "site01"
    cfg["project"]["run_id"] = "smoke_stage_03"
    cfg["project"]["root"] = str(root)
    cfg["project"]["random_seed"] = 42

    cfg.setdefault("inputs", {})
    cfg["inputs"].update(
        {
            "video": "data/raw/site01.mp4",
            "ifc": "data/bim/design/model.ifc",
            "schedule": "data/bim/design/schedule.csv",
        }
    )

    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "keyframes_dir": "data/frames/key/site01",
            "manifest_csv": "data/frames/key/site01_manifest.csv",
            "sfm_dir": "data/sfm/site01",
            "colmap_db": "data/sfm/site01/database.db",
            "sparse_dir": "data/sparse/site01",
            "sparse_report_json": "runs/smoke_stage_03/reports/sparse_stats.json",
            "sparse_mask_dir": "data/masks/site01",
        }
    )

    cfg.setdefault("colmap", {})
    cfg["colmap"].update(
        {
            "executable": str(fake_colmap),
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "feature_type": "ALIKED_N16ROT",
            "matcher_type": "ALIKED_LIGHTGLUE",
            "fallback_feature_type": "SIFT",
            "fallback_matcher_type": "SIFT_LIGHTGLUE",
            "enable_fallback": True,
            "allow_sift_bruteforce_emergency": False,
            "emergency_matcher_type": "SIFT_BRUTEFORCE",
            "matching_strategy": "sequential",
            "sequential_overlap": 3,
            "sequential_quadratic_overlap": False,
            "sequential_loop_detection": False,
            "sequential_vocab_tree_path": None,
            "stage_images_mode": "copy",
            "min_input_images": 2,
            "aliked_max_num_features": 128,
            "sift_max_num_features": 128,
            "mapper_min_num_matches": 3,
            "mapper_multiple_models": True,
            "mapper_extract_colors": False,
            "qt_qpa_platform": "offscreen",
            "download_models": False,
            "model_cache_dir": "data/sfm/model_cache",
            "model_download_timeout_sec": 60,
            "use_existing_masks": False,
            "require_masks": False,
            "mask_path": "data/masks/site01",
            "keep_failed_attempts": True,
            "fail_if_no_sparse_model": True,
        }
    )

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage03_smoke_") as tmp:
        root = Path(tmp)
        fake_colmap = root / "fake_colmap.py"
        _write_fake_colmap(fake_colmap)
        _write_images_and_manifest(root)

        cfg_path = _write_config(root, fake_colmap)
        cfg = load_config(cfg_path)
        report = run_sparse(cfg, force=True, log_level="ERROR")

        if not Path(report["database_path"]).exists():
            raise SystemExit("STAGE_03_SMOKE_FAILED missing database")
        if not Path(report["sparse_model_dir"]).exists():
            raise SystemExit("STAGE_03_SMOKE_FAILED missing sparse model")

        registered = report["sparse_stats"].get("registered_image_count")
        if registered != 4:
            raise SystemExit(f"STAGE_03_SMOKE_FAILED expected 4 registered images, got {registered!r}")

        if report["sparse_stats"].get("pycolmap_in_process_used") is not False:
            raise SystemExit("STAGE_03_SMOKE_FAILED in-process pycolmap stats should be disabled")

        print(f"STAGE_03_SMOKE_OK registered={registered} database={report['database_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
