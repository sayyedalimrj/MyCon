from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from .metric_alignment_quality import enrich_metric_alignment_report


@dataclass(frozen=True)
class MetricAnchor:
    anchor_id: str
    description: str
    bim_xyz: np.ndarray
    scan_xyz: np.ndarray | None
    use_for_scale: bool
    use_for_registration: bool


@dataclass(frozen=True)
class KnownDistance:
    distance_id: str
    anchor_a: str
    anchor_b: str
    distance_m: float


@dataclass(frozen=True)
class Sim3Transform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    source: str = "scan_to_bim"

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        return self.scale * (pts @ self.rotation.T) + self.translation

    def matrix4x4(self) -> np.ndarray:
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = self.scale * self.rotation
        mat[:3, 3] = self.translation
        return mat

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix4x4": self.matrix4x4().tolist(),
        }


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return float(s)


def _coordinate_value(row: dict[str, Any], prefix: str, axis: str) -> float | None:
    """Read flexible coordinate column names.

    Supported forms:
    - bim_x, bim_y, bim_z
    - bim_x_m, bim_y_m, bim_z_m
    - scan_x, scan_y, scan_z
    - scan_x_m, scan_y_m, scan_z_m
    """
    for key in (f"{prefix}_{axis}", f"{prefix}_{axis}_m"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _xyz_from_row(row: dict[str, Any], prefix: str, required: bool) -> np.ndarray | None:
    vals = [_coordinate_value(row, prefix, axis) for axis in ["x", "y", "z"]]
    if any(v is None for v in vals):
        if required:
            raise ValueError(
                f"Missing required coordinate columns for prefix={prefix}. "
                f"Expected {prefix}_x/{prefix}_y/{prefix}_z or "
                f"{prefix}_x_m/{prefix}_y_m/{prefix}_z_m for anchor {row.get('anchor_id')}"
            )
        return None
    return np.asarray(vals, dtype=float)



def read_metric_anchors_csv(path: str | Path) -> dict[str, MetricAnchor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    anchors: dict[str, MetricAnchor] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            anchor_id = str(row.get("anchor_id", "")).strip()
            if not anchor_id:
                continue
            anchors[anchor_id] = MetricAnchor(
                anchor_id=anchor_id,
                description=str(row.get("description", "")).strip(),
                bim_xyz=_xyz_from_row(row, "bim", required=True),
                scan_xyz=_xyz_from_row(row, "scan", required=False),
                use_for_scale=_parse_bool(row.get("use_for_scale"), True),
                use_for_registration=_parse_bool(row.get("use_for_registration"), True),
            )
    return anchors


def read_known_distances_csv(path: str | Path) -> list[KnownDistance]:
    p = Path(path)
    if not p.exists():
        return []

    out: list[KnownDistance] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            distance_id = str(row.get("distance_id", "")).strip()
            anchor_a = str(row.get("anchor_a", "")).strip()
            anchor_b = str(row.get("anchor_b", "")).strip()
            distance_m = _float_or_none(row.get("distance_m"))
            if distance_id and anchor_a and anchor_b and distance_m is not None:
                out.append(KnownDistance(distance_id, anchor_a, anchor_b, float(distance_m)))
    return out


def estimate_scale_from_known_distances(
    anchors: dict[str, MetricAnchor],
    distances: list[KnownDistance],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ratios: list[float] = []

    for item in distances:
        a = anchors.get(item.anchor_a)
        b = anchors.get(item.anchor_b)
        if not a or not b or a.scan_xyz is None or b.scan_xyz is None:
            records.append({
                "distance_id": item.distance_id,
                "status": "missing_scan_anchor",
                "anchor_a": item.anchor_a,
                "anchor_b": item.anchor_b,
                "distance_m": item.distance_m,
            })
            continue

        scan_distance = float(np.linalg.norm(a.scan_xyz - b.scan_xyz))
        if scan_distance <= 1.0e-12:
            records.append({
                "distance_id": item.distance_id,
                "status": "zero_scan_distance",
                "anchor_a": item.anchor_a,
                "anchor_b": item.anchor_b,
                "distance_m": item.distance_m,
                "scan_distance": scan_distance,
            })
            continue

        scale = float(item.distance_m / scan_distance)
        ratios.append(scale)
        records.append({
            "distance_id": item.distance_id,
            "status": "ok",
            "anchor_a": item.anchor_a,
            "anchor_b": item.anchor_b,
            "distance_m": item.distance_m,
            "scan_distance": scan_distance,
            "scale": scale,
        })

    if not ratios:
        return {
            "status": "not_available",
            "scale": None,
            "record_count": len(records),
            "records": records,
        }

    arr = np.asarray(ratios, dtype=float)
    return {
        "status": "ok",
        "scale": float(np.median(arr)),
        "mean_scale": float(np.mean(arr)),
        "std_scale": float(np.std(arr)),
        "min_scale": float(np.min(arr)),
        "max_scale": float(np.max(arr)),
        "record_count": len(records),
        "valid_record_count": int(len(ratios)),
        "records": records,
    }


def _sim3_degeneracy_threshold(src: np.ndarray, dst: np.ndarray) -> float:
    """Return a scale-aware source variance floor for Sim3 fitting.

    Absolute epsilon alone lets near-coincident anchors pass. Use the larger
    scan/BIM bounding-box diagonal as scene scale and require the registration
    anchors to span at least about 0.5 percent of that scale.
    """
    src_diag = float(np.linalg.norm(np.ptp(src, axis=0))) if len(src) else 0.0
    dst_diag = float(np.linalg.norm(np.ptp(dst, axis=0))) if len(dst) else 0.0
    scene_scale = max(src_diag, dst_diag, 1.0)
    return max(1.0e-12, (scene_scale * 0.005) ** 2)


def estimate_sim3_umeyama(scan_points: np.ndarray, bim_points: np.ndarray, estimate_scale: bool = True) -> Sim3Transform:
    src = np.asarray(scan_points, dtype=float)
    dst = np.asarray(bim_points, dtype=float)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 paired points, got scan={src.shape} bim={dst.shape}")
    if len(src) < 3:
        raise ValueError("At least 3 paired anchors are required for Sim3 alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    u, singular_values, vt = np.linalg.svd(cov)

    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[-1, -1] = -1.0

    rotation = u @ s_fix @ vt

    if estimate_scale:
        src_var = float(np.mean(np.sum(src_c * src_c, axis=1)))
        min_src_var = _sim3_degeneracy_threshold(src, dst)
        if src_var <= min_src_var:
            raise ValueError(
                f"Cannot estimate scale from degenerate anchor geometry: "
                f"source_variance={src_var:.12g} <= min_required_variance={min_src_var:.12g}"
            )
        scale = float(np.sum(singular_values * np.diag(s_fix)) / src_var)
    else:
        scale = 1.0

    translation = mu_dst - scale * (rotation @ mu_src)
    return Sim3Transform(scale=scale, rotation=rotation, translation=translation)


def evaluate_anchor_residuals(
    transform: Sim3Transform,
    anchors: list[MetricAnchor],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    residuals: list[float] = []

    for anchor in anchors:
        if anchor.scan_xyz is None:
            continue
        predicted = transform.apply(anchor.scan_xyz.reshape(1, 3))[0]
        residual = float(np.linalg.norm(predicted - anchor.bim_xyz))
        residuals.append(residual)
        rows.append({
            "anchor_id": anchor.anchor_id,
            "description": anchor.description,
            "residual_m": residual,
            "predicted_bim_xyz": predicted.tolist(),
            "target_bim_xyz": anchor.bim_xyz.tolist(),
            "scan_xyz": anchor.scan_xyz.tolist(),
        })

    if not residuals:
        return {
            "count": 0,
            "rmse_m": None,
            "max_error_m": None,
            "mean_error_m": None,
            "records": rows,
        }

    arr = np.asarray(residuals, dtype=float)
    return {
        "count": int(len(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr * arr))),
        "max_error_m": float(np.max(arr)),
        "mean_error_m": float(np.mean(arr)),
        "records": rows,
    }


def _build_metric_alignment_report_unvalidated(
    anchors_csv: str | Path,
    known_distances_csv: str | Path | None = None,
    output_json: str | Path | None = None,
    min_registration_anchors: int = 3,
    residual_warn_m: float = 0.05,
    residual_fail_m: float = 0.15,
) -> dict[str, Any]:
    anchors = read_metric_anchors_csv(anchors_csv)
    distances = read_known_distances_csv(known_distances_csv) if known_distances_csv else []

    usable = [
        a for a in anchors.values()
        if a.use_for_registration and a.scan_xyz is not None
    ]

    known_scale = estimate_scale_from_known_distances(anchors, distances)

    warnings: list[str] = []
    failures: list[str] = []

    if len(usable) < min_registration_anchors:
        failures.append(f"insufficient_registration_anchors:{len(usable)}<{min_registration_anchors}")
        report = {
            "stage": "stage_08_metric_alignment",
            "status": "skipped_insufficient_anchors",
            "confidence": "low",
            "can_feed_stage8": False,
            "anchors_csv": str(anchors_csv),
            "known_distances_csv": str(known_distances_csv) if known_distances_csv else None,
            "anchor_count": len(anchors),
            "usable_registration_anchor_count": len(usable),
            "known_distance_scale": known_scale,
            "transform": None,
            "residuals": None,
            "quality_gate": {
                "passed": False,
                "failures": failures,
                "warnings": warnings,
                "thresholds": {
                    "min_registration_anchors": min_registration_anchors,
                    "residual_warn_m": residual_warn_m,
                    "residual_fail_m": residual_fail_m,
                },
            },
        }
    else:
        scan = np.vstack([a.scan_xyz for a in usable if a.scan_xyz is not None])
        bim = np.vstack([a.bim_xyz for a in usable])
        transform = estimate_sim3_umeyama(scan, bim, estimate_scale=True)
        residuals = evaluate_anchor_residuals(transform, usable)

        rmse = residuals["rmse_m"] or 0.0
        max_error = residuals["max_error_m"] or 0.0

        if rmse > residual_fail_m or max_error > residual_fail_m * 2.0:
            failures.append(f"anchor_residual_too_high:rmse={rmse:.6f},max={max_error:.6f}")
            status = "alignment_failed"
            confidence = "low"
            can_feed = False
        elif rmse > residual_warn_m:
            warnings.append(f"anchor_residual_warning:rmse={rmse:.6f}>{residual_warn_m:.6f}")
            status = "alignment_warning"
            confidence = "medium"
            can_feed = True
        else:
            status = "ok"
            confidence = "high"
            can_feed = True

        ks = known_scale.get("scale")
        if ks is not None:
            ratio = abs(float(ks) - transform.scale) / max(abs(transform.scale), 1.0e-12)
            if ratio > 0.05:
                warnings.append(f"known_distance_scale_differs_from_anchor_sim3:{ratio:.6f}>0.050000")

        report = {
            "stage": "stage_08_metric_alignment",
            "status": status,
            "confidence": confidence,
            "can_feed_stage8": can_feed,
            "anchors_csv": str(anchors_csv),
            "known_distances_csv": str(known_distances_csv) if known_distances_csv else None,
            "anchor_count": len(anchors),
            "usable_registration_anchor_count": len(usable),
            "known_distance_scale": known_scale,
            "transform": transform.to_dict(),
            "residuals": residuals,
            "quality_gate": {
                "passed": not failures,
                "failures": failures,
                "warnings": warnings,
                "thresholds": {
                    "min_registration_anchors": min_registration_anchors,
                    "residual_warn_m": residual_warn_m,
                    "residual_fail_m": residual_fail_m,
                },
            },
        }

    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _rewrite_metric_alignment_report_output(report: dict[str, Any], output_json: Any) -> None:
    if output_json is None:
        return
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _rewrite_metric_alignment_report_output(report: dict[str, Any], output_json: Any) -> None:
    if output_json is None:
        return
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_metric_alignment_report_if_requested(report: dict[str, Any], output_json: Any) -> None:
    if output_json is None:
        return
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_metric_alignment_report(*args, **kwargs):
    """Build Stage 8 metric alignment report and attach robust selection.

    The wrapper must not use wrapper-local anchor lookup because anchors are
    local to _build_metric_alignment_report_unvalidated(). Re-read the same
    anchors CSV here so metric_alignment_selection runs in production too.
    """
    import inspect

    bound = inspect.signature(_build_metric_alignment_report_unvalidated).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    anchors_csv = bound.arguments.get("anchors_csv")
    output_json = bound.arguments.get("output_json")

    report = _build_metric_alignment_report_unvalidated(*args, **kwargs)
    if not isinstance(report, dict):
        return report

    try:
        enriched = enrich_metric_alignment_report(report)
        report.clear()
        report.update(enriched)
    except Exception as exc:  # pragma: no cover
        report.setdefault("metric_alignment_quality_warning", str(exc))

    try:
        from .metric_alignment_selection import attach_metric_alignment_selection

        anchors_for_selection = []
        if anchors_csv is not None:
            anchors_for_selection = list(read_metric_anchors_csv(anchors_csv).values())
        report = attach_metric_alignment_selection(report, anchors_for_selection)
    except Exception as exc:
        warnings = report.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append(f"metric_alignment_selection_failed:{exc}")

    _write_metric_alignment_report_if_requested(report, output_json)
    return report

