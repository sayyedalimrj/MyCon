"""Create-only video-to-COLMAP-to-civil-viewer pilot with resumable review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .general_pilot import stage
from .workflow import (
    Blocked, IMAGE_KEYS, checked_run, emit_json, prepare, run_colmap, sha256, stamp,
)


def build_review(run_dir: Path, review_id: str, *, threshold: float) -> dict:
    from pipeline.stage_07_cleanup.r_frame_geometry import generate
    from pipeline.stage_09_progress.upstream_receipt_audit import audit_upstream_receipts
    from viewer_civil.core import build_preview
    from viewer_civil.producer_bridge import convert_r_run

    dataset = checked_run(run_dir)
    poses_path = run_dir / "camera_poses.json"
    document = json.loads(poses_path.read_text(encoding="utf-8"))
    ranked = sorted((p for p in document["poses"] if p.get("registered") is True and p["split"] == "train"
                     and p["observations_3d"] >= 30), key=lambda p: (-p["observations_3d"], p["view_id"]))
    if len(ranked) < 3:
        raise Blocked("Fewer than three registered train cameras have usable sparse observations")
    model = run_dir / "colmap/train_text"
    products = generate(
        run_dir, cameras=model / IMAGE_KEYS[0], images=model / IMAGE_KEYS[1],
        points=model / IMAGE_KEYS[2], photos=[run_dir / p["image_relpath"] for p in ranked[:5]],
        run_id=f"geom-civil-r-frame-{review_id}", threshold=threshold,
        r_dataset=run_dir / "dataset_manifest.json", r_poses=poses_path,
    )
    if not products["R_shared_world_frame"]:
        raise Blocked("Sparse geometry could not be bound to the frozen R train model")
    manifest = Path(products.pop("output_dir")) / "civil_r_frame_manifest.json"
    bundle = run_dir / f"viewer-{review_id}"
    convert_r_run(run_dir, bundle, project_root=run_dir,
                  a_manifest=manifest.relative_to(run_dir).as_posix())
    preview = build_preview(bundle, run_dir / f"offline-{review_id}")
    audit = audit_upstream_receipts(project_root=run_dir, a_manifest=manifest, a_sha256=sha256(manifest))
    return {
        "status": "sparse_candidates_and_viewer_created",
        "source_capture_count": len(dataset["source_videos"]),
        "candidates": products,
        "a_manifest": {"path": manifest.relative_to(run_dir).as_posix(), "sha256": sha256(manifest)},
        "viewer_manifest_sha256": sha256(bundle / "viewer_manifest.json"),
        "offline_viewer": preview.relative_to(run_dir).as_posix(),
        "progress_input_audit": audit,
    }


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", required=True, type=Path)
    cli.add_argument("--review-id", required=True, help="Unique ID for each review of a completed SfM run")
    cli.add_argument("--video", type=Path, help="New run. Omit to review a completed run without changing it.")
    cli.add_argument("--capture-id")
    cli.add_argument("--expected-video-sha256")
    cli.add_argument("--test-video", type=Path)
    cli.add_argument("--test-capture-id")
    cli.add_argument("--executable", default="colmap")
    cli.add_argument("--sample-hz", type=float, default=2.)
    cli.add_argument("--max-frames", type=int, default=64)
    cli.add_argument("--max-image-size", type=int, default=960)
    cli.add_argument("--geometry-threshold", required=True, type=float,
                     help="Exploratory plane threshold in unscaled COLMAP world units")
    cli.add_argument("--attempt-dense", action="store_true", help="Run CUDA PatchMatch and fusion if available")
    args = cli.parse_args(argv)
    run = args.run_dir.resolve()
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", args.review_id)
            or not 0 < args.geometry_threshold < float("inf")):
        cli.error("Invalid review ID or nonpositive/nonfinite geometry threshold")
    if ((args.video is None) != (args.capture_id is None)
            or ((args.test_video or args.test_capture_id or args.expected_video_sha256) and not args.video)):
        cli.error("New runs require both --video and --capture-id")
    receipt = run / f"pilot-{args.review_id}.json"
    if receipt.exists():
        cli.error("Review already exists. Use a new review ID.")
    if args.video and run.exists():
        cli.error("New reconstruction requires a new run directory. Omit --video to review completed SfM.")
    report = {"schema_version": "mycon.consolidated_pilot.v1", "run_id": run.name,
              "started_at_utc": stamp(), "stages": {},
              "field_validation": {"metric": "NOT_EVALUATED", "bim": "NOT_EVALUATED",
                                   "semantic_accuracy": "NOT_EVALUATED",
                                   "physical_progress": None, "progress_state": "unknown",
                                   "paired_scientific_metrics": "NOT_EVALUATED"}}
    stages = report["stages"]
    products = run / "same_run_products/av_handoff.json"
    step = "selection"
    try:
        if args.video:
            if args.expected_video_sha256 and sha256(args.video) != args.expected_video_sha256.lower():
                raise Blocked("Source video SHA256 mismatch")
            selected = prepare(run_root=run.parent, run_id=run.name, train_video=args.video,
                               train_session=args.capture_id, test_video=args.test_video,
                               test_session=args.test_capture_id, sample_hz=args.sample_hz,
                               max_frames=args.max_frames, min_blur=35.,
                               min_mean=25., max_mean=230., max_clip=.38)
            stages[step] = {"status": "created", "counts": selected["counts"],
                            "dataset_sha256": sha256(run / "dataset_manifest.json")}
            step = "sfm"
            stages[step] = run_colmap(run, args.executable, execute=True)
        else:
            step = "sfm"
            execution = run / "colmap/colmap_execution.json"
            audit = json.loads(execution.read_text(encoding="utf-8"))
            if (audit.get("status") != "colmap_executed" or audit.get("run_id") != run.name
                    or audit.get("input_sha256") != sha256(run / "dataset_manifest.json")):
                raise Blocked("Resume requires a completed SfM receipt for the exact dataset")
            stages[step] = {"status": "reused", "execution_sha256": sha256(execution)}
        step = "sparse_geometry_and_viewer"
        stages[step] = build_review(run, args.review_id, threshold=args.geometry_threshold)
        step = "sfm_products"
        if products.exists():
            # An earlier review can be inspected independently of these derivative assets.
            stages[step] = {"status": "existing_receipt", "receipt_sha256": sha256(products)}
        else:
            result = stage(run, args.executable, args.max_image_size, run_dense=args.attempt_dense)
            stages[step] = {
                "status": "created", "sparse_vertices": result["sparse_vertices"],
                "dense_status": result["dense_status"], "gs_training": result["gs_training"],
                "receipt_sha256": sha256(products),
            }
        report["status"] = "exploratory_pilot_completed"
        exit_code = 0
    except (Blocked, ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        stages[step] = {"status": "BLOCKED", "reason": str(exc)}
        report["status"] = "partial_or_blocked"
        exit_code = 2
    report["finished_at_utc"] = stamp()
    if run.is_dir():
        emit_json(receipt, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr if exit_code else sys.stdout)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
