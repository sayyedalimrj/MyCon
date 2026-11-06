#!/usr/bin/env python3
"""CPU-only MyCON frame-quality and temporal-coverage QA.

Never changes Recorder, frame images, pose.csv, IMU or checkpoints.
Scores *derived* images, not ARCore pose, metric scale, or true parallax.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

FRAME_RE = re.compile(r"^frame_(\d{6})\.jpg$")
POSE_FRAME_RE = re.compile(r"\b(frame_\d{6}\.jpg)\b")


def registered_frame_names(images_txt: Path) -> list[str]:
    """COLMAP text output has pose headers alternating with 2D observations."""
    if not images_txt.is_file():
        raise RuntimeError("MYCON_COLMAP_IMAGE_LIST_MISSING")
    names = POSE_FRAME_RE.findall(images_txt.read_text(encoding="utf-8"))
    if not names or len(names) != len(set(names)):
        raise RuntimeError("MYCON_COLMAP_IMAGE_LIST_EMPTY_OR_DUPLICATED")
    return names


def temporal_coverage(images_txt: Path, total_frames: int, *,
                      fps: float = 2.0, bins: int = 8,
                      min_bin_fraction: float = 0.25,
                      max_gap_s: float = 12.0) -> dict:
    """Check registered coverage *throughout* the source, not just total fraction.

    These conservative coverage thresholds are preflight gates, not proof of
    geometric connectivity, loop closure, correct 3D or BIM metric scale.
    """
    if total_frames < 1 or fps <= 0 or bins < 2:
        raise ValueError("MYCON_TEMPORAL_COVERAGE_ARGUMENTS_INVALID")
    names = registered_frame_names(images_txt)
    frame_ids = []
    for name in names:
        m = FRAME_RE.fullmatch(name)
        if not m:
            raise RuntimeError("MYCON_NONCANONICAL_COLMAP_FRAME_NAME")
        idx = int(m.group(1))
        if not 1 <= idx <= total_frames:
            raise RuntimeError("MYCON_COLMAP_FRAME_OUTSIDE_VERIFIED_SOURCE")
        frame_ids.append(idx)
    ids = sorted(frame_ids)
    segments = []
    failures = []
    for i in range(bins):
        lo = (i * total_frames) // bins + 1
        hi = ((i + 1) * total_frames) // bins
        expected = max(0, hi - lo + 1)
        count = sum(lo <= p <= hi for p in ids)
        fraction = count / expected if expected else 0.0
        ok = expected > 0 and fraction >= min_bin_fraction
        segments.append({"segment": i + 1, "first_source_frame": lo,
                         "last_source_frame": hi, "expected": expected,
                         "registered": count, "fraction": round(fraction, 4),
                         "passed": ok})
        if not ok:
            failures.append("UNDERCOVERED_SEGMENT_" + str(i + 1))
    # Include missing intervals at the beginning and end of the recording.
    spans = [ids[0] - 1, *(b - a - 1 for a, b in zip(ids, ids[1:])),
             total_frames - ids[-1]]
    gap_frames = max(spans)
    gap_s = gap_frames / fps
    if gap_s > max_gap_s:
        failures.append("UNREGISTERED_CONTIGUOUS_GAP")
    return {"schema": "mycon.temporal.coverage.v1",
            "status": "PASS" if not failures else "FAIL",
            "source_frames": total_frames, "registered_frames": len(ids),
            "registered_fraction": round(len(ids) / total_frames, 4),
            "time_axis": "DERIVED_2FPS_FRAME_ORDINAL_NOT_NATIVE_POSE_TIME",
            "fps": fps, "segments": segments,
            "longest_unregistered_gap_frames": gap_frames,
            "longest_unregistered_gap_s": round(gap_s, 3),
            "fail_reasons": failures,
            "spatial_connectivity_verified": False,
            "metric_scale_verified": False,
            "bim_progress_authorized": False}


def frame_quality_candidates(images: Path, *, fps: float = 2.0,
                             segments: int = 8, max_candidates: int = 200) -> dict:
    """Image QA with bounded-resolution CV operations. Candidates are *not* input.

    Never delete or overwrite the original baseline frame set. Pixel change
    is an appearance-change heuristic; it must never be called translation,
    parallax, native ARCore evidence, or a successful novel-view evaluation.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MYCON_CPU_OPENCV_REQUIRED_FOR_FRAME_QUALITY_QA") from exc
    files = sorted(images.glob("frame_*.jpg"))
    if not files or any(FRAME_RE.fullmatch(p.name) is None for p in files):
        raise RuntimeError("MYCON_FRAME_QUALITY_INPUT_MISSING_OR_BAD_NAMES")
    if fps <= 0 or segments < 2 or max_candidates < segments:
        raise ValueError("MYCON_FRAME_QUALITY_CONFIG_INVALID")
    rows = []
    previous = None
    for i, path in enumerate(files):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.size == 0:
            raise RuntimeError("MYCON_UNREADABLE_DERIVED_FRAME:" + path.name)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        brightness = float(small.mean())
        # Small normalized grid avoids classifying resolution as frame quality.
        blur_laplacian = float(cv2.Laplacian(small, cv2.CV_64F).var())
        pixel_change = (float(cv2.absdiff(small, previous).mean())
                        if previous is not None else None)
        previous = small
        too_dark = brightness < 30.0
        too_bright = brightness > 230.0
        low_detail = blur_laplacian < 12.0
        near_duplicate = pixel_change is not None and pixel_change < 3.0
        # Preservation: poor images may carry unique scene evidence, so never
        # filter an entire time interval based on a single global threshold.
        score = (min(blur_laplacian / 80.0, 2.0)
                 + min((pixel_change or 0.0) / 15.0, 2.0)
                 - 1.0 * int(too_dark or too_bright)
                 - 0.6 * int(near_duplicate))
        rows.append({"frame": path.name, "source_frame_ordinal": i + 1,
                     "time_s_estimate": round(i / fps, 3),
                     "brightness": round(brightness, 3),
                     "laplacian_var_160x90": round(blur_laplacian, 3),
                     "pixel_change_from_previous": (round(pixel_change, 3)
                         if pixel_change is not None else None),
                     "low_detail_warning": low_detail,
                     "exposure_warning": too_dark or too_bright,
                     "near_duplicate_warning": near_duplicate,
                     "score": round(score, 5),
                     "segment": min(segments - 1, i * segments // len(files))})
    # Balanced highest-scoring candidates: reserve one in *every* interval.
    grouped = [[r for r in rows if r["segment"] == b]
               for b in range(segments)]
    if any(not g for g in grouped):
        raise RuntimeError("MYCON_FRAME_QUALITY_EMPTY_TEMPORAL_SEGMENT")
    selected = {max(g, key=lambda r: r["score"])["frame"] for g in grouped}
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    for row in ranked:
        if len(selected) >= max_candidates:
            break
        selected.add(row["frame"])
    return {"schema": "mycon.derived_frame_quality.v1",
            "source_frame_count": len(files),
            "fps_estimate": fps,
            "image_analysis_only": True,
            "native_pose_or_translation_verified": False,
            "frame_names_for_optional_future_candidate_dataset":
                [r["frame"] for r in rows if r["frame"] in selected],
            "candidate_count": len(selected),
            "all_original_frames_preserved": True,
            "frames": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--colmap-images-txt", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args(argv)
    if args.colmap_images_txt:
        result = temporal_coverage(args.colmap_images_txt,
                                   len(list(args.images.glob("frame_*.jpg"))),
                                   fps=args.fps)
    else:
        result = frame_quality_candidates(args.images, fps=args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("MYCON_QA_OUTPUT_EXISTS_PRESERVE")
    args.output.write_text(json.dumps(result, ensure_ascii=False,
                                      sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result.get("status", "DIAGNOSTIC_ONLY"),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if result.get("status") != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
