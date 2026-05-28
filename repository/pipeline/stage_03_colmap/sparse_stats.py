"""Sparse reconstruction statistics for Stage 3 reports.

This module intentionally avoids opening COLMAP binary models with in-process
PyCOLMAP. If the binary files are incomplete or corrupt, the C++ parser can
segfault the Python interpreter before a Python exception is raised. Statistics
are therefore collected through COLMAP subprocesses and text conversion.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from .colmap_cli import ColmapRunner
from .reconstruct_sparse import SPARSE_BIN_NAMES

_IMAGE_COUNT_RE = re.compile(r"Registered images:\s*(\d+)", re.IGNORECASE)
_POINT_COUNT_RE = re.compile(r"Points:\s*(\d+)", re.IGNORECASE)


def validate_sparse_binary_model(model_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    ok = model_dir.exists() and model_dir.is_dir()
    for name in SPARSE_BIN_NAMES:
        path = model_dir / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        files[name] = {"exists": exists, "size_bytes": size, "nonempty": size > 0}
        ok = ok and exists and size > 0
    return {"model_dir": str(model_dir), "valid_binary_contract": bool(ok), "files": files}


def _count_non_comment_lines(path: Path) -> int:
    if not path.exists() or path.stat().st_size <= 0:
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def _parse_text_model(text_dir: Path) -> dict[str, Any]:
    cameras = _count_non_comment_lines(text_dir / "cameras.txt")
    image_data_lines = _count_non_comment_lines(text_dir / "images.txt")
    # COLMAP images.txt has two data lines per image: metadata and points2D.
    images = image_data_lines // 2 if image_data_lines else 0
    points = _count_non_comment_lines(text_dir / "points3D.txt")
    return {
        "camera_count_from_text": cameras,
        "registered_image_count_from_text": images,
        "sparse_point_count_from_text": points,
        "text_files_present": {
            "cameras.txt": (text_dir / "cameras.txt").exists(),
            "images.txt": (text_dir / "images.txt").exists(),
            "points3D.txt": (text_dir / "points3D.txt").exists(),
        },
    }


def run_model_converter_stats(runner: ColmapRunner, model_dir: Path) -> dict[str, Any]:
    validation = validate_sparse_binary_model(model_dir)
    result: dict[str, Any] = {"validation": validation, "returncode": None, "output_tail": []}
    if not validation["valid_binary_contract"]:
        result["error"] = "Sparse binary model files are missing or empty; skipped model_converter."
        return result
    with tempfile.TemporaryDirectory(prefix="stage3_colmap_txt_") as tmp:
        text_dir = Path(tmp)
        record = runner.run(
            ["model_converter", "--input_path", str(model_dir), "--output_path", str(text_dir), "--output_type", "TXT"],
            name="model_converter:stats_txt",
            check=False,
        )
        result.update({"returncode": record.returncode, "output_tail": record.stdout_tail})
        if record.returncode == 0:
            result.update(_parse_text_model(text_dir))
    return result


def run_model_analyzer(runner: ColmapRunner, model_dir: Path) -> dict[str, Any]:
    validation = validate_sparse_binary_model(model_dir)
    parsed: dict[str, Any] = {"validation": validation, "returncode": None, "output_tail": []}
    if not validation["valid_binary_contract"]:
        parsed["error"] = "Sparse binary model files are missing or empty; skipped model_analyzer."
        return parsed
    record = runner.run(["model_analyzer", "--path", str(model_dir)], name="model_analyzer", check=False)
    text = "\n".join(record.stdout_tail)
    parsed.update({"returncode": record.returncode, "output_tail": record.stdout_tail})
    image_match = _IMAGE_COUNT_RE.search(text)
    point_match = _POINT_COUNT_RE.search(text)
    if image_match:
        parsed["registered_image_count_from_analyzer"] = int(image_match.group(1))
    if point_match:
        parsed["sparse_point_count_from_analyzer"] = int(point_match.group(1))
    return parsed


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def build_sparse_stats(
    runner: ColmapRunner,
    model_dir: Path,
    input_image_count: int,
    attempt_name: str,
    fallback_used: bool,
) -> dict[str, Any]:
    validation = validate_sparse_binary_model(model_dir)
    converter = run_model_converter_stats(runner, model_dir)
    analyzer = run_model_analyzer(runner, model_dir)
    registered = _first_int(
        converter.get("registered_image_count_from_text"),
        analyzer.get("registered_image_count_from_analyzer"),
    )
    sparse_points = _first_int(
        converter.get("sparse_point_count_from_text"),
        analyzer.get("sparse_point_count_from_analyzer"),
    )
    cameras = _first_int(converter.get("camera_count_from_text"))
    ratio = None if not registered else float(registered) / float(input_image_count)
    return {
        "attempt_name": attempt_name,
        "fallback_used": fallback_used,
        "input_image_count": input_image_count,
        "registered_image_count": registered,
        "registered_ratio": ratio,
        "sparse_point_count": sparse_points,
        "camera_count": cameras,
        "model_dir": str(model_dir),
        "model_files_present": {name: validation["files"][name]["exists"] for name in SPARSE_BIN_NAMES},
        "model_file_sizes": {name: validation["files"][name]["size_bytes"] for name in SPARSE_BIN_NAMES},
        "binary_contract": validation,
        "pycolmap_in_process_used": False,
        "model_converter": converter,
        "model_analyzer": analyzer,
    }
