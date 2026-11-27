"""Field-pilot CLI: read-only evidence, explicit IFC selections, no approval gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from . import core
from .ifc_inventory import local_file, write_new, extract_inventory, selected_vertex, selected_triangle


def required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing " + label)
    return value.strip()


def evidence(root, obj):
    if not isinstance(obj, dict):
        raise ValueError("Evidence receipt required")
    sha = required_text(obj.get("sha256"), "evidence SHA256")
    if not core.SHA_PATTERN.fullmatch(sha):
        raise ValueError("Invalid evidence SHA256")
    return local_file(root, required_text(obj.get("path"), "evidence path"), sha)


def load(root, rel, schema):
    path = local_file(root, rel)
    return path, core.read_json(path, schema)


def geometry_input(root, rel):
    path, doc = load(root, rel, "mycon.comparison_ready_geometry.v1")
    if not doc.get("outputs"):
        raise ValueError("Nonempty hashed geometry outputs required")
    for asset in doc["outputs"].values():
        evidence(root, asset)
    core.verify_geometry(path, root, doc["coordinate_frame"])
    return path, doc


def reviewed(root, record):
    if not core.approved(record):
        raise ValueError("Manual reviewer and evidence_id required")
    evidence(root, record.get("field_evidence"))


def geometry_binding(document, geom_path, geom):
    if document.get("geometry_manifest_sha256") != core.digest(geom_path):
        raise ValueError("Field selection not bound to exact geometry manifest")
    if document.get("source_frame") != geom["coordinate_frame"]:
        raise ValueError("Field selection source frame mismatch")


def validate_survey(root, survey, geom_path, geom):
    geometry_binding(survey, geom_path, geom)
    core.validate_inputs(survey, None)
    groups, endpoints = set(), set()
    counts = {"fit": 0, "control": 0}
    for r in survey["references"]:
        # Provisional/nominal suggestions may be retained, but never counted.
        if (not core.approved(r) or r.get("actual_length_m") is None or
                r.get("confidence", {}).get("grade") != "high_reviewed"):
            continue
        reviewed(root, r)
        for key in ("measured_at", "instrument", "measurement_method", "operator"):
            required_text(r.get(key), key)
        core.positive(r["scan_distance_uncertainty_units"], "scan uncertainty")
        group = required_text(r["physical_source_id"], "physical source group")
        ids = r.get("endpoint_ids")
        if not isinstance(ids, list) or len(ids) != 2:
            raise ValueError("Two physical endpoint IDs required")
        ids = [required_text(x, "endpoint ID") for x in ids]
        if group in groups or len(set(ids)) != 2 or endpoints.intersection(ids):
            raise ValueError("Field spans must use separate physical groups and endpoint IDs")
        groups.add(group)
        endpoints.update(ids)
        counts[r["role"]] += 1
    if counts["fit"] < 3 or counts["control"] < 2:
        raise ValueError("Field pilot requires >=3 independently measured fit spans and >=2 independent holdouts")
    result = core.length_scale(survey)
    if result["scale_m_per_reconstruction_unit"] is None:
        raise ValueError("No consistent independent measured scale; investigate fit references")
    return result


def build_ifc_adapter(root, inv_path, inv, selection, geom_path, geom):
    geometry_binding(selection, geom_path, geom)
    if selection.get("inventory_sha256") != core.digest(inv_path):
        raise ValueError("Selection not bound to exact IFC inventory")
    if inv.get("coordinate_frame") != "bim_world" or inv.get("units") != "m":
        raise ValueError("IFC inventory must use BIM engineering world metres")
    model = inv["model"]
    evidence(root, model)
    required_text(model.get("revision"), "IFC revision")
    elements = {x["ifc_global_id"]: x for x in inv["elements"]}
    if not elements or len(elements) != len(inv["elements"]):
        raise ValueError("Empty or duplicate IFC elements")
    matches, planes, samples = [], [], []
    for r in selection["matches"]:
        reviewed(root, r)
        element = elements[r["ifc_global_id"]]
        if r.get("kind") != "corner":
            raise ValueError("Field correspondence must be an operator-identified physical corner")
        if r.get("observation_state") not in core.FIT_STATES:
            raise ValueError("Unsupported or extrapolated corner")
        required_text(r.get("physical_feature_id"), "physical_feature_id")
        p = selected_vertex(element, r["vertex_index"])
        matches.append({**r, "bim_xyz_m": p})
    feature_ids = [r["physical_feature_id"] for r in matches]
    targets = [(r["ifc_global_id"], r["vertex_index"]) for r in matches]
    if len(set(feature_ids)) != len(feature_ids) or len(set(targets)) != len(targets):
        raise ValueError("Repeated physical feature or IFC vertex across fit/control")
    for r in selection.get("planes", []):
        reviewed(root, r)
        element = elements[r["ifc_global_id"]]
        poly, n = selected_triangle(element, r["triangle_index"], r["normal_sign"])
        planes.append({**r, "point_xyz_m": poly[0], "normal_xyz": n,
                       "finite_vertices_xyz_m": poly, "observation_support": "reviewed_finite"})
    for r in selection.get("samples", []):
        if r.get("observation_state") == "observed":
            reviewed(root, r)
        samples.append(r)
    adapter = {"schema_version": "mycon.bim_feature_adapter.v1",
               "source_frame": geom["coordinate_frame"], "source_units": "reconstruction_units",
               "target_frame": "bim_world", "target_units": "m",
               "model_provenance": {"kind": "IFC", "revision": model["revision"],
                                    "asset": {"path": model["path"], "sha256": model["sha256"]}},
               "inventory_sha256": core.digest(inv_path),
               "geometry_manifest_sha256": core.digest(geom_path),
               "matches": matches, "planes": planes, "samples": samples}
    if "registration" in selection:
        adapter["registration"] = selection["registration"]
    core.validate_inputs(None, adapter)
    return adapter


def run_field(root, run_id, geometry_rel, survey_rel=None, inventory_rel=None,
              selection_rel=None, length_tolerance_m=None, position_tolerance_m=None):
    root = root.resolve()
    if not core.RUN_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid metric-bim run ID")
    if bool(inventory_rel) != bool(selection_rel) or not (survey_rel or inventory_rel):
        raise ValueError("Supply survey and/or paired IFC inventory and selection")
    gp, geom = geometry_input(root, geometry_rel)
    inputs = {"geometry": gp}
    refs = None
    scale = core.length_scale(None)
    if survey_rel:
        sp, refs = load(root, survey_rel, "mycon.length_references.v1")
        inputs["survey"] = sp
        scale = validate_survey(root, refs, gp, geom)
    adapter = None
    if inventory_rel:
        ip, inv = load(root, inventory_rel, "mycon.ifc_inventory.v1")
        xp, sel = load(root, selection_rel, "mycon.ifc_field_selection.v1")
        inputs.update(inventory=ip, selection=xp)
        adapter = build_ifc_adapter(root, ip, inv, sel, gp, geom)
        minimum = 3 if refs else 4
        fit = sum(r["role"] == "fit" for r in adapter["matches"])
        hold = sum(r["role"] == "control" for r in adapter["matches"])
        if fit < minimum or hold < 2:
            raise ValueError("IFC pilot needs >=%d fit corners and >=2 independent holdouts" % minimum)
        _, tx = core.build_transform(adapter, scale)
        if tx is None:
            raise ValueError("IFC registration is degenerate or has no consensus")
    if survey_rel:
        length_tolerance_m = core.positive(length_tolerance_m, "operator length tolerance metres")
    if adapter:
        position_tolerance_m = core.positive(position_tolerance_m, "operator position tolerance metres")
    inputs_hash = {k: {"path": v.relative_to(root).as_posix(), "sha256": core.digest(v)}
                   for k, v in inputs.items()}
    dst = root / "field_inputs" / run_id
    if not dst.resolve().is_relative_to(root) or (root / "runs" / run_id).exists():
        raise ValueError("Unsafe or reused run ID")
    dst.mkdir(parents=True, exist_ok=False)
    bp = None
    if adapter:
        bp = dst / "reviewed_ifc_adapter.json"
        write_new(bp, adapter)
    manifest = core.run(root, run_id, references_json=inputs.get("survey"),
                        bim_json=bp, geometry_manifest_json=gp)
    base = root / "runs" / run_id / "metric_bim"
    with (base / "heldout_control_errors.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checks = []
    for row in rows:
        kind = row["kind"]
        limit = length_tolerance_m if kind == "length" else position_tolerance_m
        if limit is None:
            continue
        error = float(row["unsigned_error_m"]) if row["unsigned_error_m"] else None
        checks.append({"record_id": row["record_id"], "kind": kind, "error_m": error,
                       "tolerance_m": limit, "within_candidate_tolerance": error is not None and error <= limit})
    receipt = {"schema_version": "mycon.metric_bim_field_pilot.v1", "run_id": run_id,
               "inputs": inputs_hash,
               "alignment_manifest": {"path": (base / "metric_bim_manifest.json").relative_to(root).as_posix(),
                                      "sha256": core.digest(base / "metric_bim_manifest.json")},
               "checks": checks, "all_candidate_controls_within_tolerance": bool(checks) and all(c["within_candidate_tolerance"] for c in checks),
               "tolerance_note": "Operator diagnostic limits, not engineering approval; no certified measurements",
               "metric_validation": "pending", "metric_scale_verified": False,
               "registration_verified": False, "can_feed_stage8": False,
               "metric_coloring_allowed": False}
    write_new(dst / "field_pilot_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract-ifc")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ifc", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--guid", action="append", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("run")
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--run-id", required=True)
    q.add_argument("--geometry", required=True)
    q.add_argument("--survey")
    q.add_argument("--inventory")
    q.add_argument("--selection")
    q.add_argument("--length-tolerance-m", type=float)
    q.add_argument("--position-tolerance-m", type=float)
    args = parser.parse_args()
    try:
        if args.command == "extract-ifc":
            result = extract_inventory(args.root, args.ifc, args.revision, args.guid, args.out)
            print(json.dumps({"elements": len(result["elements"]), "metric_validation": "pending"}))
        else:
            result = run_field(args.root, args.run_id, args.geometry, args.survey,
                               args.inventory, args.selection, args.length_tolerance_m, args.position_tolerance_m)
            print(json.dumps(result, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "metric_validation": "pending"}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
