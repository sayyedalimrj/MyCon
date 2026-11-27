"""Conservative camera and capture coverage audit for a reconstruction run.

Angles are relative to a reconstructed COLMAP world frame, NEVER room surface
visibility, metric scale, or completeness of a wall/ceiling/doorway.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .workflow import Blocked, checked_run, emit_json, sha256


def coverage(run_dir: Path) -> dict:
    manifest = checked_run(run_dir)
    out = run_dir / "coverage_report.json"
    if out.exists():
        raise Blocked("Existing coverage report; use a new run ID, never overwrite")
    pose_path = run_dir / "camera_poses.json"
    poses = None
    if pose_path.is_file():
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "mycon.reconstruction.camera_poses.v1" or payload.get("run_id") != run_dir.name:
            raise Blocked("Bad camera_poses schema/run ID")
        poses = payload["poses"]
        expected = {f["view_id"]: f for f in manifest["frames"]}
        if len(poses) != len(expected) or {p["view_id"] for p in poses} != set(expected):
            raise Blocked("Camera/selected image identity set mismatch")
        for p in poses:
            f = expected[p["view_id"]]
            if p.get("image_sha256") != f["image_sha256"] or p.get("split") != f["split"]:
                raise Blocked("Camera SHA/split mismatch")
    selected = manifest["frames"]
    times = {}
    for session in sorted({f["capture_session_id"] for f in selected}):
        rows = sorted(f["timestamp_sec"] for f in selected if f["capture_session_id"] == session)
        gaps = np.diff(rows)
        times[session] = {
            "selected_frames": len(rows),
            "sample_span_sec": float(rows[-1] - rows[0]) if len(rows) > 1 else 0,
            "largest_selected_gap_sec": float(gaps.max()) if len(gaps) else None,
            "median_selected_gap_sec": float(np.median(gaps)) if len(gaps) else None,
        }
    registered = []
    missing = []
    per_view = []
    for f in selected:
        p = next((x for x in poses if x["view_id"] == f["view_id"]), None) if poses else None
        ok = bool(p and p.get("registered"))
        if ok:
            cam = p.get("camera")
            m = np.asarray(p.get("world_to_camera_4x4"), dtype=np.float64)
            if not cam or m.shape != (4, 4) or not np.isfinite(m).all():
                raise Blocked("Registered camera has no valid extrinsics/intrinsics")
            if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(m[:3, :3].T@m[:3, :3], np.eye(3), atol=1e-3) or np.linalg.det(m[:3, :3]) < 0.999:
                raise Blocked("Invalid rigid COLMAP pose")
            optical_axis = m[:3,:3].T @ np.array([0., 0., 1.])
            center = np.asarray(p["camera_center_world"], dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise Blocked("Invalid camera center")
            registered.append((f, optical_axis, center))
        else:
            missing.append(f["view_id"])
        per_view.append({"view_id": f["view_id"], "split": f["split"], "registered": ok,
                         "blur_var": f["blur_var"], "exposure_mean": f["luma_mean"],
                         "clipped_fraction": f["clipped_fraction"], "timestamp_sec": f["timestamp_sec"],
                         "angular_diversity_from_camera_pose": "available" if ok else "unknown",
                         "civil_visibility": "unknown_not_raytraced"})
    angles = []
    if len(registered) >= 2:
        directions = np.stack([r[1] for r in registered])
        sim = np.clip(directions @ directions.T, -1, 1)
        sep = np.degrees(np.arccos(sim))
        angles = [float(x) for x in sep[np.triu_indices(len(sep), 1)]]
        sep[np.diag_indices_from(sep)] = np.nan
        nearest = np.nanmin(sep, axis=1)
        nearest_median = float(np.nanmedian(nearest))
    else:
        nearest_median = None
    known = len(registered)
    result = {
        "schema_version": "mycon.reconstruction.coverage.v1", "run_id": run_dir.name,
        "dataset_manifest_sha256": sha256(run_dir / "dataset_manifest.json"),
        "camera_poses_sha256": sha256(pose_path) if poses else None,
        "coordinate_frame": "colmap_opencv_world_to_camera" if poses else "unknown_pending_fresh_colmap",
        "units": "reconstruction_units_unscaled" if poses else "unknown",
        "metric_scale_verified": False, "confidence": "diagnostic_only_not_surface_visibility",
        "frame_selection": manifest["counts"], "source_capture_sessions": times,
        "registered_pose_count": known, "unregistered_view_ids": missing,
        "angular_coverage": {
            "state": "registered_camera_directions_relative_world" if known >= 2 else "not_executed_insufficient_registered_poses",
            "pair_count": len(angles), "max_pairwise_look_direction_deg": max(angles) if angles else None,
            "median_nearest_neighbor_look_direction_deg": nearest_median,
            "room_wall_floor_ceiling_door_coverage": None,
            "note": "Look-direction spread cannot prove coverage of structural surfaces; audit against reviewed masks/3D surfaces separately."
        },
        "per_view": per_view,
        "known_gaps": ["unregistered viewpoints" if missing else "none_from_registered_id_check",
                       "room surface visibility unknown without calibrated geometry and raycasting",
                       "occluded/missed source-video regions not determined"],
        "provenance": {"source_video_sha256": {k:v["source_sha256"] for k,v in manifest["source_videos"].items()}},
    }
    emit_json(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = coverage(args.run_dir)
        print(json.dumps({"state":report["angular_coverage"]["state"], "registered": report["registered_pose_count"]}))
        return 0
    except (OSError, ValueError, KeyError, Blocked) as ex:
        print(json.dumps({"status": "blocked", "reason": str(ex)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
