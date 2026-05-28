"""Input/output contract checks for Stage 5 dense stereo."""
from __future__ import annotations

from pathlib import Path


class DenseContractError(RuntimeError):
    """Raised when a Stage 5 input or output file contract is invalid."""


SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def validate_sparse_model(path: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for name in SPARSE_FILES:
        item = path / name
        files[name] = {
            "path": str(item),
            "exists": item.exists(),
            "size_bytes": item.stat().st_size if item.exists() else 0,
            "nonempty": item.exists() and item.stat().st_size > 0,
        }
    return {"path": str(path), "valid_binary_contract": all(info["nonempty"] for info in files.values()), "files": files}


def require_sparse_model(path: Path, label: str) -> None:
    validation = validate_sparse_model(path)
    if not validation["valid_binary_contract"]:
        missing = [name for name, info in validation["files"].items() if not info["nonempty"]]
        raise DenseContractError(f"Invalid {label} sparse model at {path}; missing/empty files: {missing}")


def require_image_dir(path: Path, min_images: int) -> list[Path]:
    if not path.exists() or not path.is_dir():
        raise DenseContractError(f"Stage 5 image directory does not exist: {path}")
    images = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}])
    if len(images) < min_images:
        raise DenseContractError(f"Stage 5 needs at least {min_images} input images in {path}; found {len(images)}")
    return images


def assert_fused_ply(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise DenseContractError(f"Stage 5 fused point cloud missing or empty: {path}")
