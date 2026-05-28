from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ANCHOR_FIELDS = [
    "anchor_id",
    "description",
    "bim_x_m",
    "bim_y_m",
    "bim_z_m",
    "scan_x",
    "scan_y",
    "scan_z",
    "use_for_scale",
    "use_for_registration",
]

DISTANCE_FIELDS = [
    "distance_id",
    "anchor_a",
    "anchor_b",
    "distance_m",
]


@dataclass(frozen=True)
class MetricAnchorValidation:
    status: str
    ready_for_metric_alignment: bool
    anchor_count: int
    complete_registration_anchor_count: int
    complete_scale_anchor_count: int
    known_distance_count: int
    usable_known_distance_count: int
    estimated_scale_from_known_distances: float | None
    failures: list[str]
    warnings: list[str]
    anchors_missing_scan_coordinates: list[str]
    anchors_missing_bim_coordinates: list[str]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _coord(row: dict[str, Any], prefix: str, axis: str) -> float | None:
    for key in (f"{prefix}_{axis}", f"{prefix}_{axis}_m"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _xyz(row: dict[str, Any], prefix: str) -> tuple[float, float, float] | None:
    vals = tuple(_coord(row, prefix, axis) for axis in ("x", "y", "z"))
    if any(v is None for v in vals):
        return None
    return vals  # type: ignore[return-value]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def prepare_metric_anchor_template(
    source_csv: Path,
    output_csv: Path,
    *,
    force: bool = False,
) -> Path:
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source metric anchors CSV: {source_csv}")

    if output_csv.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing template without force: {output_csv}")

    rows = _read_rows(source_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ANCHOR_FIELDS)
            writer.writeheader()
        return output_csv

    normalized: list[dict[str, Any]] = []
    for row in rows:
        out = {field: row.get(field, "") for field in ANCHOR_FIELDS}
        for axis in ("x", "y", "z"):
            out[f"bim_{axis}_m"] = row.get(f"bim_{axis}_m") or row.get(f"bim_{axis}") or ""
            out[f"scan_{axis}"] = row.get(f"scan_{axis}") or row.get(f"scan_{axis}_m") or ""
        out["use_for_scale"] = row.get("use_for_scale", "true") or "true"
        out["use_for_registration"] = row.get("use_for_registration", "true") or "true"
        normalized.append(out)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANCHOR_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)

    return output_csv


def validate_metric_anchor_files(
    anchors_csv: Path,
    known_distances_csv: Path | None,
    *,
    output_json: Path | None = None,
    min_registration_anchors: int = 3,
) -> MetricAnchorValidation:
    anchors = _read_rows(anchors_csv)
    failures: list[str] = []
    warnings: list[str] = []

    if not anchors_csv.exists():
        failures.append(f"missing_anchors_csv:{anchors_csv}")

    anchor_by_id = {str(row.get("anchor_id", "")).strip(): row for row in anchors if str(row.get("anchor_id", "")).strip()}

    missing_scan: list[str] = []
    missing_bim: list[str] = []
    complete_registration = 0
    complete_scale = 0

    for anchor_id, row in anchor_by_id.items():
        bim_xyz = _xyz(row, "bim")
        scan_xyz = _xyz(row, "scan")

        if bim_xyz is None:
            missing_bim.append(anchor_id)
        if scan_xyz is None:
            missing_scan.append(anchor_id)

        if bim_xyz is not None and scan_xyz is not None and _truthy(row.get("use_for_registration"), True):
            complete_registration += 1

        if bim_xyz is not None and scan_xyz is not None and _truthy(row.get("use_for_scale"), True):
            complete_scale += 1

    if complete_registration < min_registration_anchors:
        failures.append(f"insufficient_registration_anchors:{complete_registration}<{min_registration_anchors}")

    known_rows = _read_rows(known_distances_csv) if known_distances_csv is not None else []
    scale_estimates: list[float] = []

    for row in known_rows:
        a_id = str(row.get("anchor_a", "")).strip()
        b_id = str(row.get("anchor_b", "")).strip()
        known_distance = _float_or_none(row.get("distance_m"))

        a = anchor_by_id.get(a_id)
        b = anchor_by_id.get(b_id)

        if not a or not b or known_distance is None or known_distance <= 0:
            continue

        a_scan = _xyz(a, "scan")
        b_scan = _xyz(b, "scan")
        if a_scan is None or b_scan is None:
            continue

        scan_distance = _distance(a_scan, b_scan)
        if scan_distance <= 1.0e-12:
            warnings.append(f"known_distance_zero_scan_distance:{row.get('distance_id', a_id + '_' + b_id)}")
            continue

        scale_estimates.append(known_distance / scan_distance)

    scale = None
    if scale_estimates:
        ordered = sorted(scale_estimates)
        mid = len(ordered) // 2
        scale = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])

    status = "ready" if not failures else "incomplete"
    ready = status == "ready"

    result = MetricAnchorValidation(
        status=status,
        ready_for_metric_alignment=ready,
        anchor_count=len(anchor_by_id),
        complete_registration_anchor_count=complete_registration,
        complete_scale_anchor_count=complete_scale,
        known_distance_count=len(known_rows),
        usable_known_distance_count=len(scale_estimates),
        estimated_scale_from_known_distances=scale,
        failures=failures,
        warnings=warnings,
        anchors_missing_scan_coordinates=missing_scan,
        anchors_missing_bim_coordinates=missing_bim,
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    return result


def copy_if_missing(source: Path, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
