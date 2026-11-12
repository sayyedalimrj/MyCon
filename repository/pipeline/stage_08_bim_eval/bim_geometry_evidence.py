"""Opt-in, create-only A->B BIM surface, camera and visibility evidence v1.

Ownership is IFC triangle-based, never an overlapping AABB association. Image
render quality and as-built completion are NOT inferred from BIM visibility.
Requires independent surveyed controls to mark metric registration as verified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np

from .research_frames import validate_scan_to_bim_transform, transform_points

SCHEMA = "mycon.bim_geometry_evidence.v1"
STATES = ("visible", "occluded", "out_of_fov", "ambiguous", "unknown")
RUN_RE = re.compile(r"^geom-[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def verify_independent_controls(path: Path | None, transform: Any, *, limit_m: float = 0.05) -> dict:
    result = {"passed": False, "reason": "missing_independent_survey", "rmse_m": None,
              "max_error_m": None, "control_count": 0, "control_ids": [], "sha256": None,
              "survey_id": None, "distance_scale_relative_error_max": None}
    if path is None or not path.is_file():
        return result
    result["sha256"] = sha256_file(path)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("source") != "independent_survey" or obj.get("units") != "m" or not obj.get("survey_id"):
            result["reason"] = "controls_not_independent_surveyed_meters"
            return result
        records = [r for r in obj.get("controls", []) if r.get("role") == "validation" and r.get("used_for_registration") is False]
        ids = [str(r["id"]) for r in records]
        if len(records) < 3 or len(set(ids)) != len(records):
            result["reason"] = "need_three_unique_unused_validation_controls"
            return result
        src = np.asarray([r["scan_xyz"] for r in records], dtype=float)
        dst = np.asarray([r["bim_xyz"] for r in records], dtype=float)
        if src.shape != dst.shape or src.shape[1:] != (3,) or not np.isfinite(src).all() or not np.isfinite(dst).all():
            result["reason"] = "invalid_control_coordinates"
            return result
        if np.linalg.matrix_rank(src - src.mean(axis=0), tol=1e-8) < 2:
            result["reason"] = "degenerate_control_baseline"
            return result
        predicted = transform_points(src, transform)
        distances = np.linalg.norm(predicted - dst, axis=1)
        rmse = float(np.sqrt(np.mean(distances ** 2)))
        from itertools import combinations
        pair_errors = []
        for a, b in combinations(range(len(src)), 2):
            measured = float(np.linalg.norm(dst[a] - dst[b]))
            if measured >= 0.05:
                pair_errors.append(abs(float(np.linalg.norm(predicted[a]-predicted[b])) / measured - 1.0))
        relative_scale_error = max(pair_errors) if pair_errors else math.inf
        result.update({"control_count": len(records), "control_ids": ids,
                       "survey_id": str(obj["survey_id"]), "rmse_m": rmse,
                       "max_error_m": float(np.max(distances)),
                       "distance_scale_relative_error_max": relative_scale_error})
        if rmse <= limit_m and max(distances) <= 2 * limit_m and relative_scale_error <= 0.02:
            result.update({"passed": True, "reason": "independent_survey_validated"})
        else:
            result["reason"] = "independent_residual_or_scale_failed"
    except (ValueError, KeyError, TypeError) as exc:
        result["reason"] = f"invalid_control_file:{type(exc).__name__}"
    return result


def sample_ifc_triangles(vertices: Any, faces: Any, elements: list[dict], ifc_hash: str | None,
                         *, max_samples_per_element: int = 10000) -> tuple[list[dict], dict]:
    """Deterministic one centroid per IFC-owned triangle; no AABB point reassignment.

    Face ordering must match the concatenated per-element extraction mesh and
    `face_count` metadata. Validate all index ranges before assigning ownership.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("Invalid IFC mesh vertices/faces")
    if (f < 0).any() or (f >= len(v)).any():
        raise ValueError("Triangle indices outside mesh")
    serial = v.astype("<f8").tobytes() + f.astype("<i8").tobytes()
    version = _digest((ifc_hash or "no-ifc").encode() + serial)
    samples: list[dict] = []
    element_info = []
    face_start = 0
    seen_ids = set()
    for element in elements:
        gid = str(element.get("global_id") or "")
        classname = str(element.get("ifc_class") or "unknown")
        count = int(element.get("face_count", 0))
        vstart = int(element.get("vertex_start", 0))
        vcount = int(element.get("vertex_count", 0))
        if count < 0 or vcount < 0 or face_start + count > len(f):
            raise ValueError("Element face offsets are inconsistent")
        if gid and gid in seen_ids:
            raise ValueError(f"Duplicate IFC GlobalId: {gid}")
        if gid:
            seen_ids.add(gid)
        assigned = f[face_start:face_start+count]
        if len(assigned) and not np.all((assigned >= vstart) & (assigned < vstart+vcount)):
            raise ValueError(f"Faces for {gid!r} cross element vertex partition")
        element_info.append({"global_id": gid or None, "ifc_class": classname,
                             "triangle_count": count, "geometry_status": "ok" if count else "unknown_no_triangles"})
        for local_face in range(min(count, max_samples_per_element)):
            tri_id = face_start + local_face
            triangle = v[f[tri_id]]
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            area2 = float(np.linalg.norm(normal))
            if area2 <= 1e-12:
                continue
            normal /= area2
            point = triangle.mean(axis=0)
            sample_id = _digest(f"{version}|{gid}|{local_face}|centroid".encode())[:32]
            geometry_id = _digest(f"{ifc_hash}|{gid}|{classname}|{version}".encode())[:24]
            samples.append({"sample_id": sample_id, "global_id": gid or None,
                            "ifc_class": classname, "geometry_id": geometry_id,
                            "geometry_version": version, "triangle_id": tri_id,
                            "surface_id": f"{gid or 'unassigned'}:tri:{local_face}",
                            "xyz_m": [float(x) for x in point],
                            "normal": [float(x) for x in normal],
                            "triangle_area_m2": area2 / 2.0,
                            "ownership": "ifc_triangle_face" if gid else "unassigned_missing_ifc_global_id",
                            "candidate_global_ids": []})
        face_start += count
    if face_start != len(f):
        raise ValueError("Some mesh faces are not assigned to an IFC element")
    # Exact geometric coincidence across separate IFC elements has no unique
    # ownership. Mark it ambiguous, do NOT duplicate positive ownership.
    by_position = {}
    for s in samples:
        key = tuple(round(x, 6) for x in s["xyz_m"])
        by_position.setdefault(key, []).append(s)
    for overlapping in by_position.values():
        owners = sorted({s["global_id"] for s in overlapping if s["global_id"]})
        if len(owners) > 1:
            for s in overlapping:
                s["candidate_global_ids"] = owners
                s["global_id"] = None
                s["ownership"] = "ambiguous_coincident_ifc_surfaces"
    return samples, {"geometry_version": version, "element_info": element_info,
                     "sampling_policy": "one_deterministic_triangle_centroid_per_face_capped_per_element",
                     "coincident_ownership_tolerance_m": 1e-6,
                     "sample_count": len(samples)}


def parse_camera_manifest(path: Path | None, project_root: Path) -> list[dict]:
    if path is None or not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("coordinate_frame") != "bim_world" or raw.get("units") != "m":
        raise ValueError("Camera poses must be independently calibrated in bim_world metres")
    result = []
    seen = set()
    for c in raw.get("cameras", []):
        view = str(c["view_id"])
        if view in seen:
            raise ValueError("Duplicate camera view_id")
        seen.add(view)
        entry = {"view_id": view, "capture_session_id": str(c.get("capture_session_id") or ""),
                 "image_sha256": None, "width": c.get("width"), "height": c.get("height"),
                 "intrinsics": c.get("intrinsics"),
                 "extrinsics_convention": "world_to_camera_4x4",
                 "world_to_camera_4x4": c.get("world_to_camera_4x4"),
                 "qa": c.get("qa") or {}, "calibration_state": "unknown"}
        try:
            image = Path(str(c["image_path"]))
            if not image.is_absolute():
                image = project_root / image
            if not image.is_file() or not entry["capture_session_id"]:
                raise ValueError("Missing image or capture session")
            entry["image_sha256"] = sha256_file(image)
            if c.get("image_sha256") and c["image_sha256"] != entry["image_sha256"]:
                raise ValueError("Image hash mismatch")
            K = np.asarray(entry["intrinsics"], dtype=float)
            T = np.asarray(entry["world_to_camera_4x4"], dtype=float)
            if K.shape != (3, 3) or T.shape != (4, 4) or not np.isfinite(K).all():
                raise ValueError("Camera matrix shape invalid")
            validate_scan_to_bim_transform(T, allow_scale=False)
            if K[0, 0] <= 0 or K[1, 1] <= 0 or K[2, 2] != 1 or int(entry["width"]) < 16 or int(entry["height"]) < 16:
                raise ValueError("Invalid intrinsics or resolution")
            if not entry["qa"].get("calibrated", False) or entry["qa"].get("frame_verified") is not True:
                raise ValueError("Calibration QA not confirmed")
            entry["calibration_state"] = "verified_camera_metadata"
        except (ValueError, KeyError, TypeError, OSError):
            entry["calibration_state"] = "unknown_invalid_camera_or_provenance"
        result.append(entry)
    return result


def _raycast_bruteforce(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray,
                        destination: np.ndarray) -> tuple[float, int]:
    """CPU reference raycast for small fixtures; production uses Open3D."""
    direction = destination - origin
    target_range = np.linalg.norm(direction)
    if target_range <= 1e-12:
        return math.inf, -1
    direction /= target_range
    closest, chosen = math.inf, -1
    for i, f in enumerate(faces):
        a, b, c = vertices[f]
        e1, e2 = b-a, c-a
        p = np.cross(direction, e2)
        det = np.dot(e1, p)
        if abs(det) < 1e-10:
            continue
        inv = 1.0 / det
        tv = origin-a
        u = inv*np.dot(tv, p)
        if u < -1e-7 or u > 1+1e-7:
            continue
        q = np.cross(tv, e1)
        v = inv*np.dot(direction, q)
        if v < -1e-7 or u+v > 1+1e-7:
            continue
        t = inv*np.dot(e2, q)
        if 1e-7 < t < closest:
            closest, chosen = float(t), i
    return closest, chosen


def classify_visibility(samples: list[dict], cameras: list[dict], vertices: Any, faces: Any,
                        *, metric_ready: bool, z_tolerance_m: float = 0.02) -> list[dict]:
    """Triangle raycast, never scan-to-BIM AABB transfer. No camera => no records."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) > 2000:
        try:
            import open3d as o3d
            scene = o3d.t.geometry.RaycastingScene()
            tensor_mesh = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v.astype(np.float32)),
                                                       o3d.core.Tensor(f.astype(np.uint32)))
            scene.add_triangles(tensor_mesh)
        except (ImportError, RuntimeError, AttributeError):
            scene = None
    else:
        scene = "cpu_reference"
    owners = sorted({s["global_id"] for s in samples if s["global_id"]})
    records = []
    for cam in cameras:
        per_owner = {gid: [] for gid in owners}
        valid = metric_ready and cam["calibration_state"] == "verified_camera_metadata" and scene is not None
        if valid:
            extrinsic = np.asarray(cam["world_to_camera_4x4"], dtype=float)
            R, t = extrinsic[:3, :3], extrinsic[:3, 3]
            center = -(R.T @ t)
            K = np.asarray(cam["intrinsics"], dtype=float)
        for s in samples:
            gid = s["global_id"]
            if gid is None:
                continue
            if not valid:
                per_owner[gid].append("unknown")
                continue
            p = np.asarray(s["xyz_m"], dtype=float)
            c = R @ p + t
            if c[2] <= 1e-6:
                state = "out_of_fov"
            else:
                uv = K @ (c / c[2])
                if not (0 <= uv[0] < int(cam["width"]) and 0 <= uv[1] < int(cam["height"])):
                    state = "out_of_fov"
                else:
                    target_distance = float(np.linalg.norm(p-center))
                    if scene == "cpu_reference":
                        hit_distance, hit_id = _raycast_bruteforce(v, f, center, p)
                    else:
                        direction = (p-center) / target_distance
                        rays = o3d.core.Tensor([[*center.astype(float), *direction.astype(float)]], dtype=o3d.core.Dtype.Float32)
                        hit = scene.cast_rays(rays)
                        hit_distance = float(hit["t_hit"].numpy()[0])
                        hit_id = int(hit["primitive_ids"].numpy()[0])
                    if not math.isfinite(hit_distance):
                        state = "unknown"
                    elif hit_distance < target_distance - z_tolerance_m:
                        state = "occluded"
                    elif abs(hit_distance - target_distance) <= z_tolerance_m:
                        state = "visible" if hit_id == s["triangle_id"] else "ambiguous"
                    else:
                        state = "unknown"
            per_owner[gid].append(state)
        for gid in owners:
            states = per_owner[gid]
            counts = {k: states.count(k) for k in STATES}
            if not valid:
                state = "unknown"
                visible, eligible = None, None
            elif counts["visible"]:
                state = "visible"
                visible, eligible = counts["visible"], len(states)-counts["out_of_fov"]
            elif counts["ambiguous"]:
                state = "ambiguous"
                visible, eligible = None, None
            elif counts["unknown"]:
                state = "unknown"
                visible, eligible = None, None
            elif counts["occluded"]:
                state = "occluded"
                visible, eligible = 0, len(states)-counts["out_of_fov"]
            else:
                state = "out_of_fov"
                visible, eligible = 0, 0
            records.append({"global_id": gid, "view_id": cam["view_id"],
                            "capture_session_id": cam["capture_session_id"],
                            "state": state, "visible_sample_count": visible,
                            "eligible_sample_count": eligible,
                            "state_sample_counts": counts if valid else None,
                            "method": "raycast_zbuffer" if valid else None,
                            "qa": {"metric_registration_approved": metric_ready,
                                   "camera_calibration_state": cam["calibration_state"],
                                   "raycast_available": scene is not None}})
    return records


def _write_jsonl_new(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for r in rows:
            stream.write(_json_bytes(r).decode() + "\n")


def create_bim_geometry_evidence(*, project_root: Path, run_id: str, project_id: str,
    ifc_path: Path, scan_path: Path, registration_report_path: Path, transform: Any,
    vertices: Any, triangles: Any, elements: list[dict], ifc_units: str,
    extraction_source: str, camera_path: Path | None = None,
    independent_controls_path: Path | None = None, seed: int = 42,
    control_rmse_limit_m: float = 0.05, write: bool = True) -> dict:
    """Optional producer. Reject any overwrite and emit untrusted/unknown states.

    All source inputs are read-only; no metric claim without independent survey.
    """
    if not RUN_RE.fullmatch(run_id) or not project_id:
        raise ValueError("Require geom-* run ID and project_id")
    root = project_root.resolve()
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(triangles, dtype=np.int64)
    manifest_path = root / "runs" / run_id / "reports" / "bim_geometry_evidence.v1.json"
    if manifest_path.exists() or any((manifest_path.parent / name).exists() for name in (
        "bim_surface_samples.v1.jsonl", "bim_camera_poses.v1.jsonl", "bim_element_visibility.v1.jsonl")):
        raise FileExistsError("Refusing to overwrite existing A->B geometry evidence")
    ifc_hash = sha256_file(ifc_path) if ifc_path.is_file() else None
    scan_hash = sha256_file(scan_path) if scan_path.is_file() else None
    registration_hash = sha256_file(registration_report_path) if registration_report_path.is_file() else None
    report = json.loads(registration_report_path.read_text(encoding="utf-8")) if registration_hash else {}
    q = report.get("quality_gate") or {}
    try:
        transform_qa = validate_scan_to_bim_transform(transform)
    except (ValueError, TypeError) as exc:
        transform_qa = {"matrix_valid_sim3": False, "error": str(exc)}
    survey = verify_independent_controls(independent_controls_path, transform,
                                         limit_m=control_rmse_limit_m) if transform_qa.get("matrix_valid_sim3") else {
        "passed": False, "reason": "invalid_transform", "sha256": None}
    units_ok = ifc_units.strip().upper() in {"METRE", "METER", "METERS", "METRES"}
    source_ok = extraction_source == "ifcopenshell" and ifc_hash is not None
    registration_pass = q.get("status") == "pass" and q.get("passed") is True
    metric_ready = bool(units_ok and source_ok and registration_pass and survey.get("passed")
                        and scan_hash and transform_qa.get("matrix_valid_sim3")
                        and registration_hash)
    if source_ok and units_ok:
        samples, sample_qa = sample_ifc_triangles(v, f, elements, ifc_hash)
    else:
        samples, sample_qa = [], {"geometry_version": None, "sample_count": 0,
                                  "sampling_policy": "disabled_unverified_ifc_units_or_source"}
    cameras = parse_camera_manifest(camera_path, root)
    visibility = classify_visibility(samples, cameras, v, f, metric_ready=metric_ready)
    if not cameras:
        # No fake cameras, hence no false "not visible" records.
        visibility = []
    manifest = {
        "schema_version": SCHEMA, "run_id": run_id, "project_id": project_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "coordinate_frame": "bim_world", "units": "m",
        "ifc_sha256": ifc_hash, "scan_sha256": scan_hash,
        "registration_report_sha256": registration_hash,
        "registration_quality_gate": {"stage8": q, "independent_controls": survey,
                                      "transform": transform_qa, "status": "pass" if metric_ready else "unknown_unverified"},
        "scan_to_bim_4x4": np.asarray(transform, dtype=float).tolist(),
        "metric_scale_verified": metric_ready,
        "independent_control_evidence_ref": str(independent_controls_path) if independent_controls_path else None,
        "ifc_unit_label_original": ifc_units,
        "source": extraction_source,
        "surface_samples": None, "camera_poses": None, "element_visibility": None,
        "provenance": {"code_commit": None, "config_hash": None, "seed": seed,
                       "hardware": platform.machine(), "python": platform.python_version(),
                       "numpy": np.__version__, "dataset_capture_ids": [c["capture_session_id"] for c in cameras],
                       "independent_survey_id": survey.get("survey_id"),
                       "note": "B must reject as visibility evidence unless all gates and hashes validate"},
        "element_inventory": sample_qa.get("element_info", []),
        "sampling": {k: val for k, val in sample_qa.items() if k != "element_info"},
        "status": "geometry_and_visibility_evaluable" if metric_ready and cameras and visibility else "unknown_unverified_geometry_or_cameras",
        "counts": {"surface_samples": len(samples), "camera_poses": len(cameras),
                   "element_visibility_records": len(visibility)},
    }
    payloads = {"surface_samples": ("bim_surface_samples.v1.jsonl", samples),
                "camera_poses": ("bim_camera_poses.v1.jsonl", cameras),
                "element_visibility": ("bim_element_visibility.v1.jsonl", visibility)}
    if not write:
        return {"manifest": manifest, "records": {key: rows for key, (_, rows) in payloads.items()}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for key, (filename, rows) in payloads.items():
        path = manifest_path.parent / filename
        _write_jsonl_new(path, rows)
        manifest[key] = {"path": path.relative_to(root).as_posix(),
                         "sha256": sha256_file(path), "records": len(rows),
                         "format": "jsonl"}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
    manifest["manifest_path"] = manifest_path.as_posix()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in Stage 8 A-to-B geometry evidence v1 (new run only)")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--ifc", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--registration-report", type=Path, required=True)
    parser.add_argument("--transform-json", type=Path, required=True)
    parser.add_argument("--mesh-ply", type=Path, help="Stage 8 real IFC triangle mesh PLY")
    parser.add_argument("--mesh-json", type=Path, help="Explicit CPU test mesh JSON with vertices and triangles")
    parser.add_argument("--elements-jsonl", type=Path, required=True)
    parser.add_argument("--ifc-units", required=True)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--controls-json", type=Path)
    parser.add_argument("--inspect", action="store_true", help="Validate without writing any output")
    args = parser.parse_args(argv)
    try:
        if bool(args.mesh_ply) == bool(args.mesh_json):
            raise ValueError("Exactly one of --mesh-ply or --mesh-json is required")
        if args.mesh_json:
            mesh_data = json.loads(args.mesh_json.read_text(encoding="utf-8"))
            vertices, triangles = mesh_data["vertices"], mesh_data["triangles"]
        else:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(args.mesh_ply))
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        meta = [json.loads(line) for line in args.elements_jsonl.read_text().splitlines() if line.strip()]
        mat = json.loads(args.transform_json.read_text())["matrix_4x4_scan_to_bim"]
        result = create_bim_geometry_evidence(
            project_root=args.project_root, run_id=args.run_id, project_id=args.project_id,
            ifc_path=args.ifc, scan_path=args.scan, registration_report_path=args.registration_report,
            transform=mat, vertices=vertices, triangles=triangles,
            elements=meta, ifc_units=args.ifc_units, extraction_source="ifcopenshell",
            camera_path=args.camera_json, independent_controls_path=args.controls_json,
            write=not args.inspect)
        print(json.dumps({"status": result.get("status") if not args.inspect else result["manifest"]["status"],
                          "manifest_path": result.get("manifest_path"),
                          "metric_scale_verified": (result if not args.inspect else result["manifest"])["metric_scale_verified"]}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(json.dumps({"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
