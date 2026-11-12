"""Read-only A->B evidence v1 validator. Fail closed; never infer completion.

The validator only checks integrity, provenance and schema, not empirical
truth. An empty visibility file means NO observability, not zero progress.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from .bim_geometry_evidence import SCHEMA, STATES, sha256_file


def _read_records(root: Path, entry: dict, key: str) -> tuple[list[dict], str | None]:
    if not isinstance(entry, dict) or not all(k in entry for k in ("path", "sha256", "records")):
        return [], f"missing_{key}_descriptor"
    relative = Path(str(entry["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        return [], f"unsafe_{key}_path"
    file = (root / relative).resolve()
    if not file.is_relative_to(root.resolve()) or not file.is_file():
        return [], f"missing_{key}_file"
    if sha256_file(file) != entry["sha256"]:
        return [], f"{key}_hash_mismatch"
    try:
        records = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (ValueError, UnicodeError):
        return [], f"invalid_{key}_jsonl"
    if len(records) != entry["records"]:
        return [], f"{key}_record_count_mismatch"
    return records, None


def validate_manifest(manifest_path: Path, project_root: Path, *, require_verified: bool = True) -> dict:
    """Validate persisted A->B files; reject unknown, corrupt or unaligned inputs.

    A true result means contract integrity / gate acceptance, NEVER physical
    construction completion or independently proven surface accuracy.
    """
    reasons = []
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"valid": False, "reasons": ["manifest_missing_or_invalid"]}
    if m.get("schema_version") != SCHEMA or m.get("coordinate_frame") != "bim_world" or m.get("units") != "m":
        reasons.append("invalid_schema_frame_or_units")
    if not str(m.get("run_id", "")).startswith("geom-") or not m.get("project_id"):
        reasons.append("invalid_run_or_project")
    for key in ("ifc_sha256", "scan_sha256", "registration_report_sha256"):
        sha = m.get(key)
        if not isinstance(sha, str) or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
            reasons.append("invalid_"+key)
    scan_to_bim = m.get("scan_to_bim_4x4")
    try:
        from .research_frames import validate_scan_to_bim_transform
        transform = validate_scan_to_bim_transform(scan_to_bim)
        if not transform["matrix_valid_sim3"]:
            reasons.append("invalid_sim3")
    except (TypeError, ValueError, KeyError):
        reasons.append("invalid_sim3")
    qa = m.get("registration_quality_gate") or {}
    if require_verified and (m.get("metric_scale_verified") is not True or qa.get("status") != "pass"
                             or not (qa.get("independent_controls") or {}).get("passed")
                             or not (qa.get("stage8") or {}).get("passed")):
        reasons.append("metric_registration_not_independently_verified")
    payloads = {}
    for key in ("surface_samples", "camera_poses", "element_visibility"):
        payloads[key], error = _read_records(project_root, m.get(key), key)
        if error:
            reasons.append(error)
    samples, cameras, visibility = (payloads.get(k, []) for k in ("surface_samples", "camera_poses", "element_visibility"))
    sample_ids = [s.get("sample_id") for s in samples]
    if any(not s for s in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        reasons.append("duplicate_or_missing_sample_ids")
    owners = {s.get("global_id") for s in samples if s.get("global_id")}
    camera_ids = {c.get("view_id") for c in cameras}
    if len(camera_ids) != len(cameras):
        reasons.append("duplicate_camera_view_ids")
    for s in samples:
        p = s.get("xyz_m")
        n = s.get("normal")
        if not (isinstance(p, list) and len(p) == 3 and all(isinstance(x, (int,float)) and math.isfinite(x) for x in p)):
            reasons.append("invalid_surface_metric_xyz")
            break
        if n is not None and not (isinstance(n, list) and len(n)==3 and all(isinstance(x, (int,float)) and math.isfinite(x) for x in n)):
            reasons.append("invalid_surface_normals")
            break
        if not s.get("surface_id") or not s.get("geometry_version") or not s.get("triangle_id") is not None:
            reasons.append("missing_triangle_geometry_identity")
            break
        if s.get("global_id") is None and s.get("ownership") not in ("ambiguous_coincident_ifc_surfaces", "unassigned_missing_ifc_global_id"):
            reasons.append("missing_explicit_unknown_ownership")
            break
    pairs=[]
    for c in cameras:
        if require_verified and (not c.get("image_sha256") or c.get("calibration_state") != "verified_camera_metadata"
                                 or not c.get("capture_session_id")):
            reasons.append("unverified_camera_pose_or_image")
            break
    for row in visibility:
        pairs.append((row.get("global_id"),row.get("view_id")))
        if row.get("global_id") not in owners or row.get("view_id") not in camera_ids:
            reasons.append("visibility_references_missing_ifc_or_camera")
            break
        if row.get("state") not in STATES:
            reasons.append("invalid_visibility_state")
            break
        if row.get("state") in ("unknown", "ambiguous") and row.get("visible_sample_count") is not None:
            reasons.append("unknown_cannot_have_zero_visible_count")
            break
        if require_verified and row.get("method") != "raycast_zbuffer":
            reasons.append("unverified_visibility_method")
            break
    if len(set(pairs))!=len(pairs):
        reasons.append("duplicate_element_view_pairs")
    if require_verified and (not samples or not cameras or not visibility):
        reasons.append("insufficient_surface_camera_visibility_evidence")
    return {"valid": not reasons, "reasons": sorted(set(reasons)),
            "counts": {"samples": len(samples), "cameras": len(cameras), "visibility": len(visibility)},
            "meaning": "interface_consistency_only_not_physical_completion"}


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description="Verify immutable A-to-B geometry evidence manifest")
    p.add_argument("--project-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--allow-unknown",action="store_true",help="Schema validation only; not suitable for B observability")
    a=p.parse_args(argv)
    result=validate_manifest(a.manifest,a.project_root,require_verified=not a.allow_unknown)
    print(json.dumps(result,indent=2))
    return 0 if result["valid"] else 2

if __name__=="__main__":
    raise SystemExit(main())
