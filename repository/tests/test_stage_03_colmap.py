from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.stage_03_colmap.extract_features import build_feature_extractor_args
from pipeline.stage_03_colmap.match_features import build_sequential_matcher_args
from pipeline.stage_03_colmap.prepare_images import prepare_sparse_inputs
from pipeline.stage_03_colmap.reconstruct_sparse import find_best_sparse_model
from pipeline.stage_03_colmap.sparse_stats import validate_sparse_binary_model


class DummyConfig:
    def __init__(self, root: Path) -> None:
        self.data = {
            "project": {"name": "site01", "run_id": "test_run", "root": str(root)},
            "paths": {
                "keyframes_dir": "data/frames/key/site01",
                "manifest_csv": "data/frames/key/site01_manifest.csv",
                "sfm_dir": "data/sfm/site01",
                "colmap_db": "data/sfm/site01/database.db",
                "sparse_dir": "data/sparse/site01",
                "sparse_mask_dir": "data/masks/site01",
            },
            "colmap": {
                "camera_model": "SIMPLE_RADIAL",
                "single_camera": True,
                "aliked_max_num_features": 1024,
                "sift_max_num_features": 2048,
                "sequential_overlap": 5,
                "sequential_quadratic_overlap": False,
                "sequential_loop_detection": False,
                "stage_images_mode": "copy",
                "min_input_images": 2,
                "use_existing_masks": False,
                "require_masks": False,
            },
        }

    def get(self, dotted: str, default=None):
        current = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, dotted: str):
        value = self.get(dotted, None)
        if value is None:
            raise KeyError(dotted)
        return value


def _write_jpg(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((40, 60, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), arr)


def _write_manifest(root: Path, count: int = 3) -> None:
    key_dir = root / "data" / "frames" / "key" / "site01"
    rows = []
    for i in range(count):
        img = key_dir / f"site01_kf_{i+1:05d}_f{i+1:06d}.jpg"
        _write_jpg(img, 30 + i * 20)
        rows.append(
            {
                "keyframe_id": f"kf_{i+1:05d}",
                "source_frame_index": str(i + 1),
                "timestamp_sec": str(i / 30),
                "image_path": str(img.relative_to(root)),
                "segment_id": "0",
                "sharpness_laplacian": "100.0",
                "exposure_mean": "0.5",
                "motion_score": "0.1",
                "novelty_score": "0.2",
                "quality_score": "0.9",
                "keep_sparse": "true",
                "keep_dense": "true",
                "selection_reason": "test",
            }
        )
    manifest = root / "data" / "frames" / "key" / "site01_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_sparse_inputs_copies_keep_sparse_images(tmp_path: Path) -> None:
    _write_manifest(tmp_path, count=3)
    prepared = prepare_sparse_inputs(DummyConfig(tmp_path), force=True)
    assert len(prepared.rows) == 3
    assert prepared.active_manifest_csv.exists()
    assert prepared.image_list_txt.exists()
    assert len(list(prepared.stage_images_dir.glob("*.jpg"))) == 3


def test_prepare_sparse_inputs_ignores_symlink_mode_for_docker_safety(tmp_path: Path) -> None:
    _write_manifest(tmp_path, count=2)
    cfg = DummyConfig(tmp_path)
    cfg.data["colmap"]["stage_images_mode"] = "symlink"
    prepared = prepare_sparse_inputs(cfg, force=True)
    staged = sorted(prepared.stage_images_dir.glob("*.jpg"))
    assert staged
    assert all(not path.is_symlink() for path in staged)


def test_feature_extractor_command_contains_aliked_options(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    args = build_feature_extractor_args(cfg, tmp_path / "db.db", tmp_path / "images", "ALIKED_N16ROT")
    assert "--FeatureExtraction.type" in args
    assert "ALIKED_N16ROT" in args
    assert "--AlikedExtraction.max_num_features" in args
    assert "1024" in args


def test_feature_extractor_command_can_use_existing_masks(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    mask_dir = tmp_path / "data" / "masks" / "site01"
    mask_dir.mkdir(parents=True)
    (mask_dir / "dummy.png").write_bytes(b"mask")
    cfg.data["colmap"]["use_existing_masks"] = True
    cfg.data["colmap"]["mask_path"] = "data/masks/site01"
    args = build_feature_extractor_args(cfg, tmp_path / "db.db", tmp_path / "images", "ALIKED_N16ROT")
    assert "--ImageReader.mask_path" in args
    assert str(mask_dir) in args


def test_sequential_matcher_command_enforces_lightglue_and_overlap(tmp_path: Path) -> None:
    cfg = DummyConfig(tmp_path)
    args = build_sequential_matcher_args(cfg, tmp_path / "db.db", "ALIKED_LIGHTGLUE")
    assert "--FeatureMatching.type" in args
    assert "ALIKED_LIGHTGLUE" in args
    assert "--SequentialMatching.overlap" in args
    assert "5" in args


def test_find_best_sparse_model_requires_colmap_bins(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    model0 = sparse / "0"
    model0.mkdir(parents=True)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        (model0 / name).write_bytes(b"fake")
    assert find_best_sparse_model(sparse) == model0


def test_find_best_sparse_model_fails_when_no_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_best_sparse_model(tmp_path / "missing")


def test_sparse_binary_validation_rejects_empty_files(tmp_path: Path) -> None:
    model = tmp_path / "0"
    model.mkdir()
    (model / "cameras.bin").write_bytes(b"not-empty")
    (model / "images.bin").write_bytes(b"")
    (model / "points3D.bin").write_bytes(b"not-empty")
    validation = validate_sparse_binary_model(model)
    assert validation["valid_binary_contract"] is False
    assert validation["files"]["images.bin"]["nonempty"] is False
