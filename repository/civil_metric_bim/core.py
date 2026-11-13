"""Isolated, fail-closed civil scale + BIM feature-registration adapter.

No Stage 8 mutations, no source-cloud rewriting and no construction-status inference.
Only NumPy is required; IFC/mesh conversion stays with a separately reviewed producer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "mycon.metric_bim_alignment.v1"
RUN_PATTERN = re.compile(r"metric-bim-[A-Za-z0-9][A-Za-z0-9_-]{1,80}$")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}$")
ROLES = {"fit", "control"}
FIT_STATES = {"observed_supported", "field_verified"}
SOURCE_KINDS = {"block", "steel_section", "field_measurement"}
CSV_FIELDS = ["record_id", "kind", "role", "state", "signed_error_m",
              "unsigned_error_m", "normal_error_deg", "reason"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, version: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != version:
        raise ValueError("Invalid schema version at " + str(path))
    return value


def point(value: Any, label: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(label + " requires finite xyz")
    return a


def positive(value: Any, label: str, zero_ok: bool = False) -> float:
    v = float(value)
    if not math.isfinite(v) or (v < 0 if zero_ok else v <= 0):
        raise ValueError(label + " requires finite positive value")
    return v


def role(record: dict) -> str:
    value = str(record.get("role", ""))
    if value not in ROLES:
        raise ValueError("Each record must have explicit role fit or control")
    return value


def approved(record: dict) -> bool:
    approval = record.get("approval") or {}
    return (isinstance(approval, dict)
            and approval.get("status") == "approved"
            and bool(str(approval.get("reviewer", "")).strip())
            and bool(str(approval.get("evidence_id", "")).strip()))


def check_unique(records: list[dict], key: str) -> None:
    keys = [str(r.get(key, "")).strip() for r in records]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("Missing/duplicate " + key)


def validate_inputs(refs: dict | None, bim: dict | None) -> str:
    frame = None
    if refs is not None:
        if not isinstance(refs.get("references"), list):
            raise ValueError("Length references must contain references[]")
        if refs.get("source_units") != "reconstruction_units":
            raise ValueError("Length input must explicitly use reconstruction_units")
        frame = str(refs.get("source_frame") or "").strip()
        check_unique(refs["references"], "reference_id")
        for r in refs["references"]:
            role(r)
            conf = r.get("confidence")
            if not isinstance(conf, dict) or conf.get("grade") not in {
                "high_reviewed", "medium_provisional", "low_unverified"}:
                raise ValueError("Each length reference requires a qualitative confidence.grade")
            if conf.get("calibrated_probability") is not None:
                raise ValueError("Calibration of confidence probabilities not supported")
            if r.get("source_kind") not in SOURCE_KINDS:
                raise ValueError("Invalid source_kind")
            if not str(r.get("physical_source_id", "")).strip():
                raise ValueError("physical_source_id required to prevent pseudo-replication")
            endpoints = r.get("scan_endpoints_xyz")
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise ValueError("Length span requires exactly two endpoints")
            a, b = point(endpoints[0], "span A"), point(endpoints[1], "span B")
            if np.linalg.norm(a - b) < 1e-9:
                raise ValueError("Length endpoints degenerate")
            if "nominal_length_m" in r:
                positive(r["nominal_length_m"], "nominal_length_m")
            if r.get("actual_length_m") is not None:
                positive(r["actual_length_m"], "actual_length_m")
                positive(r["actual_uncertainty_m"], "actual_uncertainty_m")
                positive(r["scan_distance_uncertainty_units"],
                         "scan_distance_uncertainty_units", zero_ok=True)
    if bim is not None:
        if bim.get("target_frame") != "bim_world" or bim.get("target_units") != "m":
            raise ValueError("BIM feature adapter requires bim_world and metres")
        if not isinstance(bim.get("matches"), list) or not isinstance(bim.get("planes"), list):
            raise ValueError("BIM adapter requires matches[] and planes[]")
        if not isinstance(bim.get("samples", []), list):
            raise ValueError("BIM samples must be a list")
        check_unique(bim["matches"], "match_id")
        check_unique(bim["planes"], "plane_id")
        provenance = bim.get("model_provenance")
        if not isinstance(provenance, dict) or not provenance.get("kind"):
            raise ValueError("Model provenance.kind is mandatory")
        check_unique(bim.get("samples", []), "sample_id")
        f2 = str(bim.get("source_frame") or "").strip()
        if frame is not None and frame != f2:
            raise ValueError("Reference and BIM source frames disagree")
        frame = f2
        for m in bim["matches"]:
            role(m)
            if m.get("kind") not in {"corner", "plane"}:
                raise ValueError("Feature kind must be corner or plane")
            point(m["scan_xyz"], "match scan"), point(m["bim_xyz_m"], "match BIM")
            if m.get("kind") == "plane" and m.get("scan_normal_xyz") is not None:
                normal = point(m["scan_normal_xyz"], "scan_normal_xyz")
                refnormal = point(m["bim_normal_xyz"], "bim_normal_xyz")
                if min(np.linalg.norm(normal), np.linalg.norm(refnormal)) < 1e-9:
                    raise ValueError("Zero feature normal")
        for p in bim["planes"]:
            point(p["point_xyz_m"], "BIM plane point")
            if np.linalg.norm(point(p["normal_xyz"], "plane normal")) < 1e-9:
                raise ValueError("Zero plane normal")
            poly = np.asarray(p["finite_vertices_xyz_m"], dtype=float)
            if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3 or not np.isfinite(poly).all():
                raise ValueError("BIM planes need finite polygons")
            if np.linalg.matrix_rank(poly - poly[0]) < 2:
                raise ValueError("BIM plane finite polygon is degenerate")
        for sm in bim.get("samples", []):
            point(sm["scan_xyz"], "sample")
    if not frame or frame == "bim_world":
        raise ValueError("Explicit non-BIM reconstruction source_frame required")
    return frame


def length_scale(refs: dict | None) -> dict:
    records = [] if refs is None else refs["references"]
    fit_ids = {str(r["physical_source_id"]) for r in records if role(r) == "fit"}
    control_ids = {str(r["physical_source_id"]) for r in records if role(r) == "control"}
    if fit_ids & control_ids:
        raise ValueError("Scale fit/control reuse the same physical_source_id")
    ratios, sigmas, used = [], [], []
    decisions = []
    for r in records:
        item = {"reference_id": r["reference_id"], "role": r["role"],
                "source_kind": r["source_kind"], "physical_source_id": r["physical_source_id"],
                "nominal_length_m": r.get("nominal_length_m"),
                "actual_length_m": r.get("actual_length_m")}
        item["nominal_minus_actual_m"] = (
            float(r["nominal_length_m"]) - float(r["actual_length_m"])
            if r.get("nominal_length_m") is not None and r.get("actual_length_m") is not None else None)
        item["confidence"] = r.get("confidence")
        if not approved(r) or r.get("confidence", {}).get("grade") != "high_reviewed":
            item["state"] = "unreviewed_or_low_confidence"
        elif r.get("actual_length_m") is None:
            item["state"] = "nominal_only_not_accepted"
        elif r["role"] == "control":
            item["state"] = "reserved_holdout"
        else:
            d = np.linalg.norm(point(r["scan_endpoints_xyz"][0], "a") -
                               point(r["scan_endpoints_xyz"][1], "b"))
            actual = float(r["actual_length_m"])
            s = actual / d
            u = math.hypot(float(r["actual_uncertainty_m"]),
                           s * float(r["scan_distance_uncertainty_units"])) / d
            item.update(state="fit_candidate", candidate_scale=s,
                        scale_uncertainty_1sigma=u)
            ratios.append(s)
            sigmas.append(u)
            used.append(r)
        decisions.append(item)
    status = "insufficient_independent_length_references"
    estimate, uncertainty, inliers = None, None, []
    if len(used) >= 3 and len({r["physical_source_id"] for r in used}) >= 3:
        a = np.asarray(ratios)
        u = np.asarray(sigmas)
        med = float(np.median(a))
        # Fixed preregistered guard: reject inconsistent measured spans rather
        # than shrinking acceptance tolerance to fit contaminated data.
        good = np.abs(a - med) <= np.maximum(0.03 * abs(med), 3 * u)
        if int(good.sum()) >= 3 and len({r["physical_source_id"] for r, ok in zip(used, good) if ok}) >= 3:
            wa = 1.0 / u[good] ** 2
            estimate = float(np.sum(wa * a[good]) / wa.sum())
            formal = float(np.sqrt(1.0 / wa.sum()))
            dispersion = float(np.sqrt(np.mean((a[good] - estimate) ** 2)))
            uncertainty = max(formal, dispersion)
            inliers = [r["reference_id"] for r, ok in zip(used, good) if ok]
            status = "scale_candidate_review_required"
            for row in decisions:
                if row["reference_id"] in inliers:
                    row["state"] = "fit_inlier"
                elif row.get("state") == "fit_candidate":
                    row["state"] = "rejected_inconsistent_span"
        else:
            status = "inconsistent_fit_references"
    return {"schema_version": "mycon.scale_estimate.v1", "status": status,
            "scale_m_per_reconstruction_unit": estimate,
            "uncertainty_1sigma_m_per_unit": uncertainty,
            "uncertainty_interpretation": "approximate propagated measurement uncertainty plus between-span dispersion; not calibrated CI",
            "fit_reference_ids": inliers, "fit_candidate_count": len(used),
            "heldout_reference_ids": [r["reference_id"] for r in records if r["role"] == "control"],
            "references": decisions, "metric_validation": "pending"}


def umeyama(src: np.ndarray, dst: np.ndarray, fixed_scale: float | None) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Expected equal Nx3 feature arrays")
    if len(src) < (4 if fixed_scale is None else 3):
        raise ValueError("Too few independent correspondence points")
    x = src - src.mean(axis=0)
    y = dst - dst.mean(axis=0)
    singular = np.linalg.svd(x, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-6)) if singular[0] > 0 else 0
    if rank < (3 if fixed_scale is None else 2):
        raise ValueError("Geometrically degenerate feature correspondence")
    u, sig, vt = np.linalg.svd((y.T @ x) / len(src))
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    R = u @ d @ vt
    s = (float(np.sum(sig * np.diag(d))) / float(np.mean(np.sum(x ** 2, axis=1)))
         if fixed_scale is None else fixed_scale)
    if not math.isfinite(s) or s <= 0 or abs(np.linalg.det(R) - 1) > 1e-5:
        raise ValueError("Nonphysical similarity transform")
    t = dst.mean(axis=0) - s * (R @ src.mean(axis=0))
    return s, R, t


def match_filter(bim: dict | None, for_fit: bool) -> list[dict]:
    matches = [] if bim is None else bim["matches"]
    selected = []
    for m in matches:
        if m["role"] != ("fit" if for_fit else "control"):
            continue
        if approved(m) and m.get("observation_state") in FIT_STATES:
            selected.append(m)
    return selected


def build_transform(bim: dict | None, scale: dict) -> tuple[dict, tuple | None]:
    matches = match_filter(bim, True)
    hold = match_filter(bim, False)
    fit_sources = {str(m.get("physical_feature_id") or m["match_id"]) for m in matches}
    hold_sources = {str(m.get("physical_feature_id") or m["match_id"]) for m in hold}
    if len(fit_sources) != len(matches):
        raise ValueError("Multiple BIM fit observations of the same physical feature are not independent")
    if fit_sources & hold_sources:
        raise ValueError("BIM fit/control source leakage")
    output = {"schema_version": "mycon.alignment_transform.v1",
              "status": "no_verified_bim_matches",
              "scale_source": None, "scale": None, "rotation": None,
              "translation": None, "matrix4x4": None,
              "fit_match_ids": [m["match_id"] for m in matches],
              "heldout_match_ids": [m["match_id"] for m in hold],
              "metric_validation": "pending", "metric_scale_verified": False,
              "registration_verified": False, "can_feed_stage8": False,
              "reason": "Pending survey/BIM revision and independent reviewed held-out control acceptance"}
    fixed = scale["scale_m_per_reconstruction_unit"]
    if not matches or (fixed is None and len(matches) < 4):
        return output, None
    x = np.asarray([m["scan_xyz"] for m in matches], float)
    y = np.asarray([m["bim_xyz_m"] for m in matches], float)
    try:
        s, R, t = umeyama(x, y, fixed)
    except ValueError as exc:
        output.update(status="blocked_degenerate_correspondences", reason=str(exc))
        return output, None
    matrix = np.eye(4)
    matrix[:3, :3] = s * R
    matrix[:3, 3] = t
    output.update(status="candidate_requires_external_validation",
                  scale_source="length_measurements" if fixed is not None else "bim_correspondences_unverified",
                  scale=s, rotation=R.tolist(), translation=t.tolist(),
                  matrix4x4=matrix.tolist())
    return output, (s, R, t)


def transformed(p: np.ndarray, transform: tuple) -> np.ndarray:
    s, R, t = transform
    return s * (p @ R.T) + t


def evaluate_controls(refs: dict | None, bim: dict | None, transform: tuple | None) -> tuple[list[dict], list[dict]]:
    alignment = []
    controls = []
    if bim is not None:
        for m in bim["matches"]:
            row = {"record_id": m["match_id"], "kind": m["kind"], "role": m["role"],
                   "state": "no_transform" if transform is None else "unapproved_or_unsupported",
                   "signed_error_m": None, "unsigned_error_m": None,
                   "normal_error_deg": None, "reason": "not_reviewed"}
            if transform is not None and approved(m) and m.get("observation_state") in FIT_STATES:
                observed = transformed(point(m["scan_xyz"], "scan"), transform)
                target = point(m["bim_xyz_m"], "bim")
                delta = observed - target
                row["unsigned_error_m"] = float(np.linalg.norm(delta))
                row["state"], row["reason"] = "measured_candidate", "matched_reviewed_feature"
                if m["kind"] == "plane" and m.get("bim_normal_xyz") is not None:
                    n = point(m["bim_normal_xyz"], "normal")
                    n /= np.linalg.norm(n)
                    row["signed_error_m"] = float(delta @ n)
                    if m.get("scan_normal_xyz") is not None:
                        a = point(m["scan_normal_xyz"], "normal")
                        a = transform[1] @ (a / np.linalg.norm(a))
                        row["normal_error_deg"] = float(np.degrees(np.arccos(
                            np.clip(abs(a @ n), -1, 1))))
            alignment.append(row)
            if m["role"] == "control":
                controls.append(dict(row))
    if refs is not None:
        for r in refs["references"]:
            if r["role"] != "control":
                continue
            row = {"record_id": r["reference_id"], "kind": "length",
                   "role": "control", "state": "no_scale_or_unapproved",
                   "signed_error_m": None, "unsigned_error_m": None,
                   "normal_error_deg": None, "reason": "no_approved_holdout_or_scale"}
            if transform is not None and approved(r) and r.get("actual_length_m") is not None:
                distance = np.linalg.norm(point(r["scan_endpoints_xyz"][0], "span") -
                                          point(r["scan_endpoints_xyz"][1], "span")) * transform[0]
                diff = float(distance - float(r["actual_length_m"]))
                row.update(state="measured_candidate", signed_error_m=diff,
                           unsigned_error_m=abs(diff), reason="independent_length_holdout")
            controls.append(row)
    return alignment, controls


def _within_polygon(sample: np.ndarray, poly: np.ndarray, n: np.ndarray) -> bool:
    # Orthogonal projection to plane then a 2D even/odd polygon test.
    axis = int(np.argmax(np.abs(n)))
    pp = np.delete(poly, axis, axis=1)
    x, y = np.delete(sample, axis)
    inside = False
    j = len(pp) - 1
    for i in range(len(pp)):
        xi, yi = pp[i]
        xj, yj = pp[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def deviation(bim: dict | None, transform: tuple | None) -> list[dict]:
    if bim is None:
        return []
    planes = {p["plane_id"]: p for p in bim["planes"]}
    out = []
    for sample in bim.get("samples", []):
        result = {"sample_id": sample["sample_id"],
                  "bim_plane_id": sample.get("bim_plane_id"), "ifc_global_id": None,
                  "valid_mask": False, "signed_distance_m": None,
                  "unsigned_distance_m": None, "reason": "no_data",
                  "source_state": sample.get("observation_state", "unknown"),
                  "xyz_bim_m": (transformed(point(sample["scan_xyz"], "sample"), transform).tolist()
                                if transform is not None else None)}
        pid = sample.get("bim_plane_id")
        p = planes.get(pid)
        if transform is None:
            result["reason"] = "scale_or_alignment_missing"
        elif sample.get("observation_state") != "observed":
            result["reason"] = "not_observed_or_occluded"
        elif p is None or not approved(p):
            result["reason"] = "no_reviewed_plane_correspondence"
        else:
            xyz = transformed(point(sample["scan_xyz"], "xyz"), transform)
            n = point(p["normal_xyz"], "normal")
            n /= np.linalg.norm(n)
            anchor = point(p["point_xyz_m"], "anchor")
            signed = float((xyz - anchor) @ n)
            proj = xyz - signed * n
            poly = np.asarray(p["finite_vertices_xyz_m"], dtype=float)
            if not _within_polygon(proj, poly, n):
                result["reason"] = "outside_finite_surface_extent"
            elif p.get("observation_support") != "reviewed_finite":
                result["reason"] = "finite_extent_not_reviewed"
            else:
                result.update(valid_mask=True, signed_distance_m=signed,
                              unsigned_distance_m=abs(signed), reason="measured_candidate",
                              ifc_global_id=p.get("ifc_global_id"))
        out.append(result)
    return out


def verify_geometry(manifest_file: Path, project_root: Path, frame: str) -> dict:
    manifest = read_json(manifest_file, "mycon.comparison_ready_geometry.v1")
    if manifest.get("coordinate_frame") != frame:
        raise ValueError("Geometry manifest and anchors differ in coordinate frame")
    if manifest.get("metric_scale_verified") is True:
        raise ValueError("Unexpected upstream metric scale assertion")
    for name, asset in (manifest.get("outputs") or {}).items():
        rel = str(asset.get("path", ""))
        p = (project_root / rel).resolve()
        if not p.is_relative_to(project_root.resolve()):
            raise ValueError("Geometry asset escapes root: " + name)
        if not p.is_file() or digest(p) != asset.get("sha256"):
            raise ValueError("Geometry output absent or SHA mismatch: " + name)
    source = Path(str(manifest.get("source_path", "")))
    if source.is_file() and digest(source) != manifest.get("source_sha256"):
        raise ValueError("Original source hash mismatch")
    return {"path": manifest_file.resolve().relative_to(project_root.resolve()).as_posix(),
            "sha256": digest(manifest_file),
            "schema_version": manifest["schema_version"],
            "source_sha256": manifest.get("source_sha256"),
            "original_source_available": source.is_file()}


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        wr = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def run(project_root: Path, run_id: str, *, references_json: Path | None = None,
        bim_json: Path | None = None, geometry_manifest_json: Path | None = None) -> dict:
    if not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("Run ID must be metric-bim-<unique>")
    if references_json is None and bim_json is None:
        raise ValueError("At least one of references_json or bim_json required")
    root = project_root.resolve()
    inputs = [p.resolve() for p in (references_json, bim_json, geometry_manifest_json) if p is not None]
    if any(not p.is_relative_to(root) for p in inputs):
        raise ValueError("Input paths must be relative to project root for provenance")
    for p in inputs:
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError("Input absent: " + str(p))
    refs = read_json(references_json, "mycon.length_references.v1") if references_json else None
    bim = read_json(bim_json, "mycon.bim_feature_adapter.v1") if bim_json else None
    frame = validate_inputs(refs, bim)
    if bim is not None and bim["model_provenance"]["kind"] != "synthetic_fixture_only":
        model = bim["model_provenance"]
        if not isinstance(model.get("asset"), dict):
            raise ValueError("Real BIM model requires asset path and SHA256")
        asset = model["asset"]
        a = (root / str(asset.get("path", ""))).resolve()
        if not a.is_relative_to(root) or not a.is_file() or not SHA_PATTERN.fullmatch(
                str(asset.get("sha256", ""))) or digest(a) != asset["sha256"]:
            raise ValueError("Real BIM source asset missing/outside root or SHA256 mismatch")
    geometry = (verify_geometry(geometry_manifest_json, root, frame)
                if geometry_manifest_json is not None else None)
    scale = length_scale(refs)
    alignment, tx = build_transform(bim, scale)
    alignment["source_frame"] = frame
    alignment["target_frame"] = "bim_world"
    alignment["source_units"] = "reconstruction_units"
    alignment["target_units"] = "m"
    rows, controls = evaluate_controls(refs, bim, tx)
    heat = deviation(bim, tx)
    target = root / "runs" / run_id / "metric_bim"
    if target.exists() or target.parent.exists():
        # A prior partial run ID must never be re-used, even if its leaf is absent.
        raise FileExistsError("Refusing to replace/reuse run: " + str(target.parent))
    target.parent.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: {"path": p.relative_to(root).as_posix(),
                               "sha256": digest(p)} for p in inputs}
    for item in (scale, alignment):
        item["input_sha256"] = source_hashes
    with tempfile.TemporaryDirectory(prefix=".metric-bim-staging-", dir=root / "runs") as tempdir:
        stage = Path(tempdir)
        _write(stage / "scale_estimate.json", scale)
        _write(stage / "alignment_transform.json", alignment)
        _csv(stage / "alignment_report.csv", rows)
        _csv(stage / "heldout_control_errors.csv", controls)
        with (stage / "deviation_map.csv").open("x", newline="", encoding="utf-8") as stream:
            fields = ["sample_id", "bim_plane_id", "ifc_global_id", "source_state",
                      "xyz_bim_m", "valid_mask", "signed_distance_m",
                      "unsigned_distance_m", "reason"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in heat:
                writer.writerow({**row, "xyz_bim_m": json.dumps(row["xyz_bim_m"])})
        _write(stage / "deviation_heatmap.json", {
            "schema_version": "mycon.deviation_heatmap.v1", "run_id": run_id,
            "coordinate_frame": "bim_world", "units": "m",
            "mask_policy": "no-data is null and never a zero-distance or not-built label",
            "sign_convention": "positive means observed point lies on positive BIM normal side",
            "viewer_encoding": {"color_by": "signed_distance_m", "alpha_by": "valid_mask",
                                "coordinates": "xyz_bim_m", "missing_color": None},
            "points": heat, "valid_sample_count": sum(x["valid_mask"] for x in heat),
            "total_sample_count": len(heat), "metric_validation": "pending"})
        assets = {p.name: {"path": (Path("runs") / run_id / "metric_bim" / p.name).as_posix(),
                           "sha256": digest(p), "bytes": p.stat().st_size}
                  for p in sorted(stage.iterdir())}
        manifest = {"schema_version": SCHEMA, "run_id": run_id,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_frame": frame, "target_frame": "bim_world",
                    "source_units": "reconstruction_units", "target_units": "m",
                    "metric_validation": "pending", "metric_scale_verified": False,
                    "registration_verified": False, "can_feed_stage8": False,
                    "physical_progress_state": "unknown",
                    "source": {"inputs": source_hashes, "geometry_manifest": geometry},
                    "method": "independent_length_fit_or_reviewed_bim_feature_sim3",
                    "holdout_policy": "never used during fitting",
                    "confidence": "not_calibrated",
                    "software": {"python": platform.python_version(), "numpy": np.__version__},
                    "outputs": assets,
                    "blockers": ["Coordinator ownership approval for future Stage8 opt-in consumer",
                                 "Independent field control and BIM revision approval",
                                 "Real-case, GPU and external metric validation not executed"]}
        _write(stage / "metric_bim_manifest.json", manifest)
        # Stage is complete. Only rename own new directory, never existing artifacts.
        target.parent.mkdir(exist_ok=False)
        os.rename(stage, target)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent MyCON civil scale/BIM candidate alignment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--references-json", type=Path)
    parser.add_argument("--bim-json", type=Path)
    parser.add_argument("--geometry-manifest-json", type=Path)
    options = parser.parse_args(argv)
    try:
        result = run(options.project_root, options.run_id,
                     references_json=options.references_json,
                     bim_json=options.bim_json,
                     geometry_manifest_json=options.geometry_manifest_json)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "candidate_created", "run_id": result["run_id"],
                      "metric_validation": "pending", "can_feed_stage8": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
