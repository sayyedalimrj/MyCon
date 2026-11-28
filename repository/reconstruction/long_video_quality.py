"""Temporal coverage audit; never a claim of reconstruction/surface quality."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .workflow import Blocked, checked_run, emit_json, sha256


def audit_long_video(run_dir: Path, review_id: str, *, window_sec: float = 10.) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", review_id):
        raise Blocked("Invalid review ID")
    if not math.isfinite(window_sec) or window_sec <= 0:
        raise Blocked("Invalid window duration")
    root = run_dir.resolve()
    m = checked_run(root)
    pose_path = root / "camera_poses.json"
    p = json.loads(pose_path.read_text(encoding="utf-8"))
    if p.get("schema_version") != "mycon.reconstruction.camera_poses.v1" or p.get("run_id") != root.name:
        raise Blocked("Pose run/schema mismatch")
    pose_by_id = {v["view_id"]: v for v in p["poses"]}
    if len(pose_by_id) != len(p["poses"]) or set(pose_by_id) != {f["view_id"] for f in m["frames"]}:
        raise Blocked("Selected/pose IDs differ")
    for f in m["frames"]:
        v = pose_by_id[f["view_id"]]
        if any(v.get(k) != f.get(k) for k in ("image_sha256", "split", "capture_session_id", "image_relpath")):
            raise Blocked("Source view identity mismatch")
    sources = {}
    for session, src in m["source_videos"].items():
        fps, count = float(src["fps"]), int(src["frames_reported"])
        if not math.isfinite(fps) or fps <= 0 or count < 2:
            raise Blocked("Invalid source duration")
        duration = count / fps
        views = sorted((f for f in m["frames"] if f["capture_session_id"] == session),
                       key=lambda f: f["timestamp_sec"])
        train = [f for f in views if f["split"] == "train"]
        registered = [f for f in train if pose_by_id[f["view_id"]].get("registered") is True]
        windows = []
        for i in range(math.ceil(duration / window_sec)):
            a, z = i * window_sec, min((i + 1) * window_sec, duration)
            chosen = [f for f in views if a <= float(f["timestamp_sec"]) < z]
            tr = [f for f in chosen if f["split"] == "train"]
            te = [f for f in chosen if f["split"] == "test"]
            reg = [f for f in tr if pose_by_id[f["view_id"]].get("registered") is True]
            test_registered = sum(pose_by_id[f["view_id"]].get("registered") is True for f in te)
            windows.append({"start_sec": a, "end_sec": z, "train_selected": len(tr),
                            "train_registered": len(reg), "test_selected": len(te),
                            "test_localized": test_registered,
                            "train_registration_ratio": len(reg) / len(tr) if tr else None,
                            "state": "no_train_selected" if not tr else ("no_train_pose" if not reg else "partial_or_registered"),
                            "surface_coverage": "unknown"})
        def max_gap(frames):
            times = [0.] + [float(f["timestamp_sec"]) for f in frames] + [duration]
            return max(b - a for a, b in zip(times[:-1], times[1:]))
        sources[session] = {"duration_sec_from_decoder_metadata": duration, "source_sha256": src["source_sha256"],
                            "frames_reported": count, "frames_decoded": src["frames_decoded"],
                            "selected_train": len(train), "registered_train": len(registered),
                            "longest_selected_train_gap_including_ends_sec": max_gap(train),
                            "longest_registered_train_gap_including_ends_sec": max_gap(registered),
                            "windows": windows}
    report = {"schema_version": "mycon.long_video_temporal_diagnostics.v1", "run_id": root.name,
              "dataset_sha256": sha256(root / "dataset_manifest.json"), "poses_sha256": sha256(pose_path),
              "window_seconds": window_sec, "train_test_protocol": m.get("train_test_protocol"),
              "sources": sources, "quality_status": "NOT_VALIDATED",
              "note": "Temporal camera coverage is not surface coverage. Heldout-only windows intentionally may lack training poses; prefer a distinct same-scene capture for full-span training. Times use decoder FPS/index, not verified Recorder PTS."}
    out = root / f"long-video-quality-{review_id}.json"
    emit_json(out, report)
    return {"status": "diagnostics_created_not_quality_validated", "path": out.name, "sha256": sha256(out)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--review-id", required=True)
    p.add_argument("--window-sec", type=float, default=10.)
    args = p.parse_args()
    print(json.dumps(audit_long_video(args.run_dir, args.review_id, window_sec=args.window_sec), indent=2))


if __name__ == "__main__":
    main()
