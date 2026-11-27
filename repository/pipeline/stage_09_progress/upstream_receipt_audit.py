"""Read-only receipt audit for A camera/geometry and D metric/BIM handoffs.

Current producer contracts are diagnostic only. Even hash-valid receipts cannot
promote candidates to IFC visibility or executed quantities. Only the frozen
Stage8 bim_geometry_evidence.v1 contract can authorize per-element geometry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .evidence_metrics import valid_sha256

SCHEMAS = {
    "A": "mycon.civil_geometry_viewer_handoff.v1",
    "D": "mycon.metric_bim_alignment.v1",
}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _confined(root: Path, path: Path) -> Path:
    p = path.resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError("missing_or_outside_project")
    return p


def audit_receipt(owner: str, manifest: Path | None, expected_sha256: str | None,
                  project_root: Path) -> dict[str, Any]:
    result = {"owner": owner, "schema_version": SCHEMAS[owner],
              "supplied": manifest is not None, "integrity_valid": False,
              "manifest_sha256": None, "run_id": None, "verified_artifact_count": 0,
              "geometry_authority": False, "reason_codes": []}
    if manifest is None:
        result["reason_codes"] = ["receipt_not_supplied"]
        return result
    try:
        root = project_root.resolve()
        path = _confined(root, manifest)
        digest = _digest(path)
        result["manifest_sha256"] = digest
        if not valid_sha256(expected_sha256) or digest != expected_sha256.lower():
            raise ValueError("independent_receipt_hash_missing_or_mismatch")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMAS[owner]:
            raise ValueError("receipt_schema_mismatch")
        run = doc.get("run_id")
        if not isinstance(run, str) or not run.startswith("geom-" if owner == "A" else "metric-bim-"):
            raise ValueError("invalid_producer_run_id")
        result["run_id"] = run
        outputs = doc.get("outputs")
        required = ({"registration_qa.json", "feature_evidence.json", "unknown_regions.json"}
                    if owner == "A" else {"scale_estimate.json", "alignment_transform.json",
                    "heldout_control_errors.csv", "deviation_heatmap.json", "viewer_handoff.json"})
        if not isinstance(outputs, dict) or not required.issubset(outputs):
            raise ValueError("required_producer_artifacts_missing")
        for desc in outputs.values():
            if not isinstance(desc, dict):
                raise ValueError("invalid_artifact_descriptor")
            rel = desc.get("path_relative_to_project" if owner == "A" else "path")
            if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or "\\" in rel:
                raise ValueError("unsafe_artifact_path")
            asset = _confined(root, root / rel)
            if (not valid_sha256(desc.get("sha256")) or
                    _digest(asset) != desc["sha256"].lower() or
                    type(desc.get("bytes")) is not int or asset.stat().st_size != desc["bytes"]):
                raise ValueError("artifact_hash_or_size_mismatch")
            result["verified_artifact_count"] += 1
        result["integrity_valid"] = True
        if owner == "A":
            result["source_civil_manifest_sha256"] = doc.get("source_civil_manifest_sha256")
            result["registration_status"] = doc.get("registration_status")
            result["source_frame"] = doc.get("coordinate_frame_depth")
            result["source_units"] = doc.get("units_depth")
            if (doc.get("camera_and_depth_co_registered") is not True or
                    doc.get("feature_photo_projection_allowed") is not True or
                    doc.get("metric_scale_verified") is not True):
                result["reason_codes"].append("A_camera_depth_or_metric_not_verified")
        else:
            result["metric_validation"] = doc.get("metric_validation")
            result["source_frame"] = doc.get("source_frame")
            result["source_units"] = doc.get("source_units")
            if (doc.get("metric_validation") != "approved" or
                    doc.get("metric_scale_verified") is not True or
                    doc.get("registration_verified") is not True or
                    doc.get("can_feed_stage8") is not True or
                    doc.get("target_frame") != "bim_world" or doc.get("target_units") != "m"):
                result["reason_codes"].append("D_metric_registration_or_stage8_not_approved")
        # No approved cross-team binding exists in these producer schemas.
        # Do not silently accept even a modified receipt claiming approved flags.
        result["reason_codes"].append("diagnostic_receipt_requires_coordinator_approved_stage8_binding")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["reason_codes"] = ["receipt_invalid:" + str(exc)]
    return result


def audit_upstream_receipts(*, project_root: Path, a_manifest: Path | None = None,
                            a_sha256: str | None = None, d_manifest: Path | None = None,
                            d_sha256: str | None = None) -> dict[str, Any]:
    a = audit_receipt("A", a_manifest, a_sha256, project_root)
    d = audit_receipt("D", d_manifest, d_sha256, project_root)
    return {"schema_version": "mycon.progress_upstream_audit.v1", "A": a, "D": d,
            "direct_geometry_gate": "blocked",
            "policy": "A/D receipts are diagnostic; use a separately approved IFC-linked Stage8 manifest for visibility. Never derive quantities from receipts."}
