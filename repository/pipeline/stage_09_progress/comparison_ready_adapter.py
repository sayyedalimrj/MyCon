"""Reference-agnostic, audit-only reader for mycon.comparison_ready_geometry.v1.

This module never derives IFC ownership, construction completion, observed
surface area, metric scale, or registration from reconstruction features.
Fail closed for comparisons; independently approved quantities remain separate.
No Open3D or particular MAX/IFC model is required.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "mycon.comparison_ready_geometry.v1"
LINK_SCHEMA = "mycon.geometry_feature_ifc_mapping.v1"
STATES = {"observed_supported", "candidate_extrapolated", "unknown"}
VISIBILITY = {"visible", "occluded", "out_of_fov", "ambiguous", "unknown"}
FEATURES = ("planes", "edges", "corners")


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vec(value: Any, n: int = 3) -> bool:
    return isinstance(value, list) and len(value) == n and all(_num(v) for v in value)


def _matrix(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 4 and all(_vec(row, 4) for row in value)
            and all(abs(a-b) < 1e-9 for a, b in zip(value[3], (0, 0, 0, 1))))


def _det3(a: list[float], b: list[float], c: list[float]) -> float:
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def _under_root(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or "\\" in relative:
        raise ValueError("unsafe_project_relative_path")
    candidate = root / relative
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("unsafe_or_missing_sidecar")
    return candidate


def _sidecar(root: Path, descriptor: Any, extension: str, *, size_cap: int = 256_000_000) -> tuple[Path, str]:
    if not isinstance(descriptor, dict) or not _hash(descriptor.get("sha256")):
        raise ValueError("invalid_sidecar_descriptor")
    path = _under_root(root, descriptor.get("path"))
    if path.suffix != extension or path.stat().st_size > size_cap:
        raise ValueError("invalid_sidecar_extension_or_size")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != descriptor["sha256"].lower():
        raise ValueError("sidecar_sha256_mismatch")
    return path, digest.hexdigest()


def _records(path: Path, key: str) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get(key), list) or len(obj[key]) > 100_000:
        raise ValueError("invalid_"+key+"_records")
    if not all(isinstance(x, dict) for x in obj[key]):
        raise ValueError("non_object_feature_record")
    return obj[key]


@dataclass(frozen=True)
class ComparisonReadyBundle:
    valid: bool
    reasons: tuple[str, ...]
    manifest_sha256: str | None
    coordinate_frame: str | None
    units: str | None
    metric_scale_verified: bool
    registration_verified: bool
    features: Mapping[str, Mapping[str, dict[str, Any]]]
    provenance: Mapping[str, Any]
    camera_views_qa: int
    cloud_point_count: int | None
    quality_metrics: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "diagnostic_quality_gate": "pass" if self.valid else "blocked",
            "comparison_gate": "blocked_without_independent_ifc_mapping",
            "reason_codes": list(self.reasons),
            "manifest_sha256": self.manifest_sha256,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "metric_scale_verified": self.metric_scale_verified,
            "registration_verified": self.registration_verified,
            "feature_counts": {k: len(self.features.get(k, {})) for k in FEATURES},
            "audited_feature_counts": {k: sum(v["audit_state"] == "observed_supported" for v in self.features.get(k, {}).values()) for k in FEATURES},
            "camera_qa_view_count": self.camera_views_qa,
            "clean_cloud_point_count": self.cloud_point_count,
            "quality_metrics": dict(self.quality_metrics),
            "provenance": dict(self.provenance),
            "physical_progress_estimate": None,
        }


def load_comparison_ready_geometry(manifest_path: str | Path, *, project_root: str | Path,
                                   expected_project_id: str | None = None) -> ComparisonReadyBundle:
    """Validate hashed reconstruction/feature artifacts, without requiring BIM.

    Non-metric units are valid for *diagnostics*. Invalid/missing sidecars or
    duplicate feature IDs invalidate the bundle. Unsupported observed corners
    downgrade to unknown, not to verified construction.
    """
    unknown = lambda reasons: ComparisonReadyBundle(False, tuple(reasons), None, None, None, False, False, {}, {}, 0, None, {})
    try:
        root = Path(project_root).resolve()
        manifest_file = Path(manifest_path)
        if not manifest_file.is_absolute():
            manifest_file = root / manifest_file
        manifest_file = manifest_file.resolve()
        if not manifest_file.is_relative_to(root) or not manifest_file.is_file():
            raise ValueError("manifest_outside_project")
        raw = manifest_file.read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA:
            raise ValueError("invalid_comparison_schema")
        if not str(manifest.get("run_id") or "").startswith("geom-ready-"):
            raise ValueError("invalid_geom_ready_run_id")
        if not manifest.get("project_id") or (expected_project_id and manifest["project_id"] != expected_project_id):
            raise ValueError("project_id_missing_or_mismatch")
        coords = manifest.get("coordinates")
        if not isinstance(coords, dict) or not coords.get("coordinate_frame") or coords.get("units") not in {"m", "cm", "mm", "reconstruction_unit", "arbitrary"}:
            raise ValueError("unknown_geometry_frame_or_units")
        metric = manifest.get("metric_scale_verified") is True
        reg = manifest.get("registration_verified") is True
        prov = manifest.get("provenance")
        if not isinstance(prov, dict) or not prov.get("source_model_sha256") or not _hash(prov.get("source_model_sha256")) or not _hash(prov.get("config_sha256")) or not prov.get("code_commit") or not prov.get("software_versions") or "seed" not in prov:
            raise ValueError("missing_geometry_provenance")
        quality = manifest.get("quality")
        if not isinstance(quality, dict) or not _num(quality.get("removed_point_ratio")) or not (0 <= quality["removed_point_ratio"] <= 1):
            raise ValueError("invalid_cleaning_quality_metrics")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("missing_files_index")
        paths = {name: _sidecar(root, files.get(name), ext)[0] for name, ext in (
            ("clean_cloud_ply", ".ply"), ("clean_cloud_npy", ".npy"),
            ("planes", ".json"), ("edges", ".json"), ("corners", ".json"))}
        # NPY is inspected without pickle/object deserialization and without
        # allocating a large point cloud in RAM.
        import numpy as np
        cloud = np.load(paths["clean_cloud_npy"], allow_pickle=False, mmap_mode="r")
        if cloud.ndim != 2 or cloud.shape[1] < 3 or cloud.shape[0] == 0 or not np.issubdtype(cloud.dtype, np.number) or not np.isfinite(cloud[:, :3]).all():
            raise ValueError("invalid_clean_point_cloud")
        if not paths["clean_cloud_ply"].open("rb").read(4) == b"ply\n":
            raise ValueError("invalid_clean_ply_header")
        features: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in FEATURES}
        for name in FEATURES:
            for r in _records(paths[name], name):
                key = {"planes":"plane_id", "edges":"edge_id", "corners":"corner_id"}[name]
                fid, state = r.get(key), r.get("state")
                if not isinstance(fid, str) or not fid or fid in features[name] or state not in STATES:
                    raise ValueError("invalid_or_duplicate_"+name+"_id_or_state")
                rec = dict(r)
                rec["audit_state"] = state
                if name == "planes":
                    normal = r.get("normal")
                    if not _vec(normal) or not _num(r.get("offset")) or abs(sum(x*x for x in normal)-1) > 1e-3:
                        raise ValueError("invalid_normalized_finite_plane")
                    vertices = r.get("extent_vertices")
                    if not isinstance(vertices, list) or len(vertices) < 3 or not all(_vec(x) for x in vertices):
                        raise ValueError("infinite_or_invalid_plane_extent")
                    if not isinstance(r.get("support_sample_ids"), list) or not isinstance(r.get("support_count"), int) or r["support_count"] < 0 or not _num(r.get("residual")) or r["residual"] < 0:
                        raise ValueError("invalid_plane_support_or_residual")
                    if state == "observed_supported" and (r["support_count"] < 3 or r.get("quality_gate") != "pass"):
                        rec["audit_state"] = "unknown"
                        rec["audit_reason"] = "plane_observation_unverified"
                elif name == "edges":
                    pts = r.get("endpoints")
                    parent = r.get("plane_ids")
                    if not isinstance(pts, list) or len(pts) != 2 or not all(_vec(p) for p in pts) or pts[0] == pts[1] or not isinstance(parent, list) or len(parent) != 2 or any(p not in features["planes"] for p in parent):
                        raise ValueError("invalid_edge_local_geometry_or_plane_id")
                    if state == "observed_supported" and (r.get("support_count",0)<2 or r.get("quality_gate")!="pass"):
                        rec["audit_state"] = "unknown"
                        rec["audit_reason"] = "edge_observation_unverified"
                else:
                    plane_ids = r.get("plane_ids")
                    if not _vec(r.get("xyz")) or not isinstance(plane_ids,list) or len(plane_ids)!=3 or len(set(plane_ids))!=3 or any(p not in features["planes"] for p in plane_ids):
                        raise ValueError("invalid_corner_xyz_or_plane_ids")
                    normals=[features["planes"][p]["normal"] for p in plane_ids]
                    if abs(_det3(*normals)) < 1e-3:
                        rec["audit_state"] = "unknown"
                        rec["audit_reason"] = "degenerate_trihedral_false_corner"
                    if state == "observed_supported" and (rec["audit_state"] == "unknown" or r.get("quality_gate") != "pass" or r.get("support_count",0)<3 or not r.get("source_image_ids")):
                        rec["audit_state"] = "unknown"
                        rec["audit_reason"] = rec.get("audit_reason", "unverified_observed_corner")
                features[name][fid] = rec
        view_count=0
        if files.get("cameras") is not None:
            camera_path,_=_sidecar(root,files["cameras"],".json")
            cameras=_records(camera_path,"cameras")
            ids=set()
            for c in cameras:
                vid=c.get("view_id")
                if not vid or vid in ids or not _hash(c.get("image_sha256")):
                    raise ValueError("invalid_or_duplicate_camera_id")
                ids.add(vid)
                if c.get("qa_verified") is True and c.get("intrinsics") and _matrix(c.get("world_to_camera_4x4")):
                    view_count+=1
        return ComparisonReadyBundle(True, (), hashlib.sha256(raw).hexdigest(), coords["coordinate_frame"], coords["units"], metric, reg, features, prov, view_count, int(cloud.shape[0]), quality)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        return unknown([f"comparison_ready_bundle_invalid:{type(exc).__name__}:{exc}"])


def comparison_for_element(bundle: ComparisonReadyBundle | None, ifc_global_id: str, *,
                           linkage: Mapping[str, Any] | None = None,
                           expected_ifc_sha256: str | None = None,
                           expected_registration_sha256: str | None = None) -> dict[str, Any]:
    """Only an independently validated IFC linkage can support an agreement label.

    Feature identifiers are *not* IFC GlobalIds. Without linkage, return
    unconditionally unknown, even for beautiful corners and high density.
    """
    base = {
        "comparison_ready_schema": SCHEMA,
        "comparison_ready_manifest_sha256": bundle.manifest_sha256 if bundle else None,
        "comparison_ready_valid": bool(bundle and bundle.valid),
        "comparison_coordinate_units": bundle.units if bundle and bundle.valid else None,
        "comparison_coordinate_frame": bundle.coordinate_frame if bundle and bundle.valid else None,
        "comparison_metric_scale_verified": bool(bundle and bundle.metric_scale_verified),
        "comparison_registration_verified": bool(bundle and bundle.registration_verified),
        "comparison_ifc_quality_gate": "blocked",
        "comparison_feature_ids": "[]",
        "geometric_agreement_state": "unknown",
        "comparison_visibility_state": "unknown",
        "comparison_evidence_reason_codes": "missing_independently_verified_feature_ifc_linkage",
    }
    if not bundle or not bundle.valid:
        base["comparison_evidence_reason_codes"] = "comparison_ready_bundle_invalid_or_absent"
        return base
    if not linkage:
        return base
    if not (bundle.metric_scale_verified and bundle.registration_verified and
            _hash(expected_ifc_sha256) and _hash(expected_registration_sha256) and
            linkage.get("schema_version") == LINK_SCHEMA and
            linkage.get("comparison_ready_manifest_sha256") == bundle.manifest_sha256 and
            linkage.get("ifc_sha256") == expected_ifc_sha256 and
            linkage.get("registration_report_sha256") == expected_registration_sha256 and
            linkage.get("metric_control_ref") and linkage.get("registration_quality_gate") == "pass" and
            _matrix(linkage.get("reconstruction_to_ifc_4x4")) and
            linkage.get("independent_review_id") and
            (bundle.units == "m" or (
                _num(linkage.get("metric_scale_to_m")) and linkage["metric_scale_to_m"] > 0 and
                linkage.get("independent_scale_qa_ref")
            ))):
        base["comparison_evidence_reason_codes"] = "scale_registration_or_linkage_not_verified"
        return base
    all_links = linkage.get("feature_links")
    if not isinstance(all_links,list) or any(not isinstance(l,dict) for l in all_links):
        base["comparison_evidence_reason_codes"] = "invalid_feature_link_list"
        return base
    ownership: dict[tuple[str, str], set[str]] = {}
    for link in all_links:
        ownership.setdefault((link.get("feature_type"),link.get("feature_id")),set()).add(link.get("ifc_global_id"))
    if any(len(owners)>1 for owners in ownership.values()):
        base["comparison_evidence_reason_codes"] = "ambiguous_cross_element_feature_ownership"
        return base
    matches = [l for l in all_links if l.get("ifc_global_id") == ifc_global_id]
    if not matches:
        base["comparison_evidence_reason_codes"] = "no_verified_ifc_feature_correspondence"
        return base
    trusted=[]
    for link in matches:
        kind, fid=link.get("feature_type"),link.get("feature_id")
        feat=bundle.features.get(kind,{}).get(fid)
        if (not feat or feat.get("audit_state") != "observed_supported" or
            link.get("quality_gate") != "pass" or
            link.get("association_method") not in {"independent_survey_correspondence","audited_ifc_surface_match"} or
            not link.get("association_qa_ref")):
            continue
        trusted.append(link)
    if len(trusted)!=len(matches) or len({(l["feature_type"],l["feature_id"]) for l in trusted})!=len(matches):
        base["comparison_evidence_reason_codes"] = "candidate_or_duplicate_or_unverified_feature_mapping"
        return base
    base["comparison_ifc_quality_gate"]="pass"
    base["comparison_feature_ids"]=json.dumps(sorted(l["feature_type"]+":"+l["feature_id"] for l in trusted))
    base["comparison_evidence_reason_codes"]=""
    if bundle.camera_views_qa <= 0 or any(
        l.get("visibility_method") not in {"raycast_zbuffer", "audited_manual"}
        or not l.get("visibility_qa_ref") for l in trusted
    ):
        base["comparison_visibility_state"]="unknown"
    elif any(l.get("visibility_state") in {"ambiguous","unknown",None} for l in trusted):
        base["comparison_visibility_state"]="unknown"
    else:
        vis={l["visibility_state"] for l in trusted}
        base["comparison_visibility_state"]=next(iter(vis)) if len(vis)==1 else "unknown"
    deviations=[]
    for link in trusted:
        deviation,tolerance=link.get("deviation_m"),link.get("tolerance_m")
        if not _num(deviation) or not _num(tolerance) or deviation<0 or tolerance<=0:
            break
        deviations.append(deviation > tolerance)
    if len(deviations)==len(trusted) and base["comparison_visibility_state"]=="visible":
        base["geometric_agreement_state"]="nonconforming" if any(deviations) else "conforming"
    return base
