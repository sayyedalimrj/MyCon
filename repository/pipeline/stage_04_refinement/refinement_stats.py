"""Stage 4 refinement report helpers.

This module intentionally avoids in-process PyCOLMAP binary parsing. Sparse
model statistics are collected through COLMAP subprocess commands and text files
so corrupt binary models cannot segfault the Python process.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from pipeline.stage_03_colmap.colmap_cli import ColmapRunner


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_INT_RE = re.compile(r"[-+]?\d+")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_first_int(text: str, labels: tuple[str, ...]) -> int | None:
    for line in text.splitlines():
        lower = line.strip().lower()
        if any(label in lower for label in labels):
            match = _INT_RE.search(line)
            if match:
                return int(match.group(0))
    return None


def _parse_first_float(text: str, labels: tuple[str, ...]) -> float | None:
    for line in text.splitlines():
        lower = line.strip().lower()
        if any(label in lower for label in labels):
            values = _FLOAT_RE.findall(line)
            if values:
                return float(values[-1])
    return None


def _is_image_header_line(line: str) -> bool:
    parts = line.split()
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        for idx in range(1, 8):
            float(parts[idx])
        int(parts[8])
    except ValueError:
        return False
    name = parts[9]
    suffix = Path(name).suffix.lower()
    return bool(suffix in _IMAGE_EXTENSIONS or "/" in name or "\\" in name)


def _parse_images_txt(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _is_image_header_line(line):
            count += 1
    return count


def _parse_points3d_txt(path: Path) -> tuple[int, float | None, float | None]:
    if not path.exists():
        return 0, None, None
    count = 0
    errors: list[float] = []
    track_lengths: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            int(parts[0])
            float(parts[1])
            float(parts[2])
            float(parts[3])
            int(parts[4])
            int(parts[5])
            int(parts[6])
            errors.append(float(parts[7]))
        except ValueError:
            continue
        count += 1
        if len(parts) > 8:
            track_lengths.append(max(0, (len(parts) - 8) // 2))
    mean_error = mean(errors) if errors else None
    mean_track_length = mean(track_lengths) if track_lengths else None
    return count, mean_error, mean_track_length


def _parse_cameras_txt(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                int(parts[0])
                int(parts[2])
                int(parts[3])
            except ValueError:
                continue
            count += 1
    return count


def _convert_model_to_text(runner: ColmapRunner, model_dir: Path, text_dir: Path, attempt_name: str) -> None:
    if text_dir.exists():
        shutil.rmtree(text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(text_dir),
        "--output_type",
        "TXT",
    ]
    runner.run(args, name=f"model_converter:{attempt_name}")


def _record_output_text(record: Any) -> str:
    """Return command output without assuming a specific CommandRecord schema."""
    stdout_tail = list(getattr(record, "stdout_tail", []) or [])
    stderr_tail = list(getattr(record, "stderr_tail", []) or [])
    stdout = getattr(record, "stdout", "") or ""
    stderr = getattr(record, "stderr", "") or ""
    parts: list[str] = []
    parts.extend(str(item) for item in stdout_tail)
    parts.extend(str(item) for item in stderr_tail)
    if not parts and stdout:
        parts.append(str(stdout))
    if stderr:
        parts.append(str(stderr))
    return "\n".join(parts)


def _analyze_model(runner: ColmapRunner, model_dir: Path, attempt_name: str) -> str:
    record = runner.run(["model_analyzer", "--path", str(model_dir)], name=f"model_analyzer:{attempt_name}", check=False)
    return _record_output_text(record)


def build_sparse_stats_via_colmap(
    runner: ColmapRunner,
    model_dir: Path,
    input_image_count: int,
    attempt_name: str,
    work_dir: Path,
) -> dict[str, Any]:
    """Build sparse model stats using only COLMAP subprocesses and text parsing."""
    text_dir = work_dir / f"{attempt_name}_txt"
    _convert_model_to_text(runner, model_dir, text_dir, attempt_name)
    analyzer_text = _analyze_model(runner, model_dir, attempt_name)

    registered_from_txt = _parse_images_txt(text_dir / "images.txt")
    points_from_txt, mean_reproj_from_txt, mean_track_length = _parse_points3d_txt(text_dir / "points3D.txt")
    cameras_from_txt = _parse_cameras_txt(text_dir / "cameras.txt")

    registered_from_analyzer = _parse_first_int(analyzer_text, ("registered images", "num_reg_images", "images"))
    points_from_analyzer = _parse_first_int(analyzer_text, ("points", "points3d", "num_points"))
    mean_reproj_from_analyzer = _parse_first_float(
        analyzer_text,
        ("mean reprojection error", "mean track reprojection error", "reprojection error"),
    )

    registered = registered_from_txt or registered_from_analyzer or 0
    points = points_from_txt or points_from_analyzer or 0
    mean_reproj = mean_reproj_from_txt if mean_reproj_from_txt is not None else mean_reproj_from_analyzer
    denominator = max(1, int(input_image_count))

    return {
        "attempt_name": attempt_name,
        "model_dir": str(model_dir),
        "stats_source": "colmap_subprocess_text",
        "pycolmap_in_process_used": False,
        "input_image_count": int(input_image_count),
        "camera_count": int(cameras_from_txt),
        "registered_image_count": int(registered),
        "registered_ratio": round(float(registered) / float(denominator), 6),
        "sparse_point_count": int(points),
        "mean_reprojection_error_px": None if mean_reproj is None else float(mean_reproj),
        "mean_track_length": None if mean_track_length is None else float(mean_track_length),
        "model_analyzer_text_tail": analyzer_text.splitlines()[-50:],
    }


def compute_refinement_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    b_images = _to_int(before.get("registered_image_count"))
    a_images = _to_int(after.get("registered_image_count"))
    b_points = _to_int(before.get("sparse_point_count"))
    a_points = _to_int(after.get("sparse_point_count"))
    b_ratio = _to_float(before.get("registered_ratio"))
    a_ratio = _to_float(after.get("registered_ratio"))
    b_error = _to_float(before.get("mean_reprojection_error_px"))
    a_error = _to_float(after.get("mean_reprojection_error_px"))
    point_loss_ratio = None
    if b_points and a_points is not None:
        point_loss_ratio = max(0.0, float(b_points - a_points) / float(b_points))
    reproj_delta = None if b_error is None or a_error is None else a_error - b_error
    return {
        "registered_image_delta": None if b_images is None or a_images is None else a_images - b_images,
        "sparse_point_delta": None if b_points is None or a_points is None else a_points - b_points,
        "registered_ratio_delta": None if b_ratio is None or a_ratio is None else a_ratio - b_ratio,
        "mean_reprojection_error_delta_px": reproj_delta,
        "point_loss_ratio": point_loss_ratio,
        "before_registered_image_count": b_images,
        "after_registered_image_count": a_images,
        "before_sparse_point_count": b_points,
        "after_sparse_point_count": a_points,
        "before_mean_reprojection_error_px": b_error,
        "after_mean_reprojection_error_px": a_error,
    }


def evaluate_quality_gate(cfg: Any, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Stage 4 quality gates.

    A moderate point-count reduction after BA is treated as a warning, not an
    automatic failure: BA and subsequent filtering may remove outliers. The hard
    gate focuses on minimum usable reconstruction and reprojection-error
    degradation when the error is available.
    """
    from .config_access import cfg_bool, cfg_float, cfg_int

    delta = compute_refinement_delta(before, after)
    min_registered = cfg_int(cfg, "refinement.quality_gate_min_registered_images", 2)
    min_points = cfg_int(cfg, "refinement.quality_gate_min_points", 1)
    max_point_loss_ratio = cfg_float(cfg, "refinement.quality_gate_max_point_loss_ratio", 0.40)
    fail_on_point_loss = cfg_bool(cfg, "refinement.quality_gate_fail_on_point_loss", True)
    max_reproj_increase_ratio = cfg_float(cfg, "refinement.quality_gate_max_reprojection_error_increase_ratio", 0.10)
    max_reproj_increase_abs = cfg_float(cfg, "refinement.quality_gate_max_reprojection_error_increase_abs_px", 0.25)
    fail_on_reproj_increase = cfg_bool(cfg, "refinement.quality_gate_fail_on_reprojection_error_increase", True)

    after_registered = delta["after_registered_image_count"]
    after_points = delta["after_sparse_point_count"]
    before_error = delta["before_mean_reprojection_error_px"]
    after_error = delta["after_mean_reprojection_error_px"]
    point_loss_ratio = delta["point_loss_ratio"]

    failures: list[str] = []
    warnings: list[str] = []
    if after_registered is None or after_registered < min_registered:
        failures.append(f"registered images below threshold: {after_registered} < {min_registered}")
    if after_points is None or after_points < min_points:
        failures.append(f"sparse points below threshold: {after_points} < {min_points}")
    if point_loss_ratio is not None and point_loss_ratio > max_point_loss_ratio:
        message = f"sparse point loss high: {point_loss_ratio:.3f} > {max_point_loss_ratio:.3f}"
        if fail_on_point_loss:
            failures.append(message)
        else:
            warnings.append(message)
    if before_error is not None and after_error is not None:
        allowed_increase = max(max_reproj_increase_abs, before_error * max_reproj_increase_ratio)
        observed_increase = after_error - before_error
        if observed_increase > allowed_increase:
            message = (
                f"mean reprojection error increased too much: {observed_increase:.6f}px > "
                f"{allowed_increase:.6f}px"
            )
            if fail_on_reproj_increase:
                failures.append(message)
            else:
                warnings.append(message)
    else:
        warnings.append("mean reprojection error unavailable; reprojection quality gate skipped")

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "delta": delta,
        "min_registered_images": min_registered,
        "min_points": min_points,
        "max_point_loss_ratio": max_point_loss_ratio,
        "observed_point_loss_ratio": point_loss_ratio,
        "max_reprojection_error_increase_ratio": max_reproj_increase_ratio,
        "max_reprojection_error_increase_abs_px": max_reproj_increase_abs,
    }
