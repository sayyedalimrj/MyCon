"""Create-only multiview capture preparation, COLMAP reconstruction and scoring.

WARNING: all original captures/6999 checkpoints are strictly read-only.
Heavy Gaussian training is not implemented or started by this CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "mycon.reconstruction.dataset.v1"
RUN_PATTERN = re.compile(r"^recon-[A-Za-z0-9][A-Za-z0-9._-]{3,80}$")
COLMAP_FRAME = "colmap_opencv_world_to_camera"
IMAGE_KEYS = ("cameras.txt", "images.txt", "points3D.txt")


class Blocked(RuntimeError):
    """Safety / provenance / integration gate failed."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def emit_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path, root: Path) -> str:
    root = root.resolve()
    path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as e:
        raise Blocked(f"File outside run root: {path}") from e


def checked_run(run_dir: Path) -> dict:
    path = run_dir / "dataset_manifest.json"
    if not path.is_file():
        raise Blocked(f"Missing dataset_manifest: {path}")
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("schema_version") != SCHEMA or m.get("run_id") != run_dir.name:
        raise Blocked("Dataset schema/version/run ID mismatch")
    if not RUN_PATTERN.fullmatch(run_dir.name):
        raise Blocked("Unsafe Run ID")
    seen: set[str] = set()
    for entry in m["frames"]:
        path = run_dir / entry["image_relpath"]
        if not path.resolve().is_relative_to(run_dir.resolve()) or not path.is_file():
            raise Blocked(f"Missing/unsafe frame path: {path}")
        digest = sha256(path)
        if digest != entry["image_sha256"] or digest in seen:
            raise Blocked(f"Frame SHA mismatch or duplicate image: {path}")
        seen.add(digest)
    if not any(x["split"] == "train" for x in m["frames"]):
        raise Blocked("No training frames")
    if not any(x["split"] == "test" for x in m["frames"]):
        raise Blocked("No genuinely held-out test frames")
    return m


def _dhash(gray: np.ndarray) -> int:
    import cv2
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    ans = 0
    for bit in (small[:, 1:] > small[:, :-1]).flatten():
        ans = (ans << 1) | int(bit)
    return ans


def _video_candidates(video: Path, *, source_id: str, sample_hz: float, min_blur: float,
                      min_mean: float, max_mean: float, max_clip: float, max_frames: int) -> tuple[list[dict], dict]:
    import cv2
    if not video.is_file() or video.stat().st_size <= 0:
        raise Blocked(f"Source video absent/empty: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count < 2:
            raise Blocked("Video FPS/frame count cannot be trusted")
        stride = max(1, round(fps / sample_hz))
        sampled: list[dict] = []
        prev_hashes: list[int] = []
        idx = 0
        while len(sampled) < max_frames * 12:
            ok, bgr = capture.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            mean = float(gray.mean())
            clipped = float(np.mean((gray < 5) | (gray > 250)))
            dh = _dhash(gray)
            duplicate = bool(prev_hashes and min((dh ^ h).bit_count() for h in prev_hashes[-12:]) < 5)
            reasons = []
            if blur < min_blur: reasons.append("blur")
            if mean < min_mean or mean > max_mean: reasons.append("exposure")
            if clipped > max_clip: reasons.append("clipping")
            if duplicate: reasons.append("near_duplicate")
            if not reasons:
                prev_hashes.append(dh)
            sampled.append({"source": source_id, "frame_index": idx, "timestamp_sec": round(idx / fps, 6),
                            "blur_var": round(blur, 4), "luma_mean": round(mean, 4),
                            "clipped_fraction": round(clipped, 6), "dhash": f"{dh:016x}",
                            "reason": ";".join(reasons) or "candidate", "accepted": not reasons,
                            "image_bgr": bgr if not reasons else None, "width": int(bgr.shape[1]),
                            "height": int(bgr.shape[0])})
            idx += 1
        return sampled, {"fps": fps, "frames_reported": count, "frames_decoded": idx,
                         "stride": stride, "sampled": len(sampled), "source_sha256": sha256(video),
                         "source_size_bytes": video.stat().st_size}
    finally:
        capture.release()


def _spread_select(rows: list[dict], max_count: int) -> list[dict]:
    good = [x for x in rows if x["accepted"]]
    if len(good) <= max_count:
        return good
    # Uniform temporal coverage, never take the highest-sharpness frames only.
    cuts = np.linspace(0, len(good), max_count + 1, dtype=int)
    chosen = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        chosen.append(max(good[a:b], key=lambda x: (x["blur_var"], -x["timestamp_sec"])))
    chosen_ids = {(x["source"], x["frame_index"]) for x in chosen}
    for row in good:
        if (row["source"], row["frame_index"]) not in chosen_ids:
            row["reason"], row["accepted"] = "coverage_cap", False
    return sorted(chosen, key=lambda x: (x["source"], x["timestamp_sec"]))


def prepare(*, run_root: Path, run_id: str, train_video: Path, train_session: str,
            test_video: Path | None, test_session: str | None, sample_hz: float,
            max_frames: int, min_blur: float, min_mean: float, max_mean: float,
            max_clip: float, embargo_sec: float = 1.0) -> dict:
    if not RUN_PATTERN.fullmatch(run_id):
        raise Blocked("Run ID must be unique and match recon-... (min 4-character suffix)")
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,70}", train_session):
        raise Blocked("Invalid capture_session_id")
    if test_video and (not test_session or test_session == train_session):
        raise Blocked("Independent test video needs a distinct session ID")
    if test_video and train_video.resolve() == test_video.resolve():
        raise Blocked("Train and test video cannot be the same file under different session IDs")
    if sample_hz <= 0 or max_frames < 8 or min_blur < 0 or not (0 <= min_mean < max_mean <= 255) or not (0 <= max_clip <= 1):
        raise Blocked("Invalid sampling/quality parameters")
    inputs = [(train_video, train_session), *(([(test_video, test_session)] if test_video else []))]
    candidates = {}
    source_stats = {}
    for v, session in inputs:
        a, stat = _video_candidates(v, source_id=session, sample_hz=sample_hz, min_blur=min_blur,
                                    min_mean=min_mean, max_mean=max_mean, max_clip=max_clip,
                                    max_frames=max_frames)
        candidates[session] = a
        source_stats[session] = {"video_parent_external": str(v.resolve().parent),
                                 "video_relpath": v.name, **stat}
    if test_video and source_stats[train_session]["source_sha256"] == source_stats[test_session]["source_sha256"]:
        raise Blocked("Identical source video bytes cannot be labeled independent sessions")
    chosen = {session: _spread_select(candidates[session], max_frames) for _, session in inputs}
    if test_video:
        train = chosen[train_session]
        test = chosen[test_session]
        split_method = "capture_session_heldout"
    else:
        good = chosen[train_session]
        if len(good) < 10:
            raise Blocked("Need >=10 selected frames for single-session exploratory held-out split")
        # Contiguous temporal test block, with an embargo on both sides.
        start = int(len(good) * 0.68)
        end = max(start + 2, int(len(good) * 0.86))
        test = good[start:end]
        window_start = test[0]["timestamp_sec"] - embargo_sec
        window_end = test[-1]["timestamp_sec"] + embargo_sec
        train = [x for x in good if not window_start <= x["timestamp_sec"] <= window_end]
        retained_ids = {id(x) for x in test + train}
        for x in good:
            if id(x) not in retained_ids:
                x["accepted"], x["reason"] = False, "split_embargo"
        split_method = "within_session_temporal_block_exploratory_not_site_independent"
    if len(train) < 5 or len(test) < 2:
        raise Blocked(f"Insufficient quality/diversity: train={len(train)} test={len(test)}")
    # Cross-split frame collision check; exact and perceptual duplicate detection.
    train_hashes = [int(x["dhash"], 16) for x in train]
    if any(min((int(t["dhash"], 16) ^ h).bit_count() for h in train_hashes) < 4 for t in test):
        raise Blocked("Perceptually duplicate image crosses train/test: recollect coverage")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        (run_dir / "images").mkdir()
        for session in candidates:
            (run_dir / "images" / session).mkdir()
        all_selected = {id(x): "train" for x in train} | {id(x): "test" for x in test}
        frames = []
        for session, rows in candidates.items():
            for item in rows:
                split = all_selected.get(id(item))
                if not split:
                    continue
                fname = f"{session}_{item['frame_index']:08d}.png"
                path = run_dir / "images" / session / fname
                import cv2
                if not cv2.imwrite(str(path), item["image_bgr"], [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise Blocked(f"PNG write failed: {path}")
                frames.append({"view_id": fname[:-4], "capture_session_id": session,
                               "source_frame_index": item["frame_index"], "timestamp_sec": item["timestamp_sec"],
                               "image_relpath": rel(path, run_dir), "image_sha256": sha256(path),
                               "width": item["width"], "height": item["height"],
                               "blur_var": item["blur_var"], "luma_mean": item["luma_mean"],
                               "clipped_fraction": item["clipped_fraction"], "dhash": item["dhash"],
                               "split": split, "pose_status": "unknown_pending_fresh_colmap"})
        frames.sort(key=lambda x: (x["capture_session_id"], x["timestamp_sec"]))
        image_names = [str(Path(x["image_relpath"]).relative_to("images")).replace("\\", "/") for x in frames if x["split"] == "train"]
        (run_dir / "train_image_names.txt").write_text("\n".join(image_names) + "\n", encoding="utf-8")
        selection = [{k: x[k] for k in ("source", "frame_index", "timestamp_sec", "blur_var", "luma_mean", "clipped_fraction", "dhash", "reason")} |
                     {"selected_split": all_selected.get(id(x), "rejected")} for rows in candidates.values() for x in rows]
        emit_csv(run_dir / "frame_selection.csv", ["source", "frame_index", "timestamp_sec", "blur_var", "luma_mean", "clipped_fraction", "dhash", "reason", "selected_split"], selection)
        manifest = {"schema_version": SCHEMA, "run_id": run_id, "created_at_utc": stamp(),
                    "coordinate_frame": "unknown_pending_fresh_colmap", "units": "unknown_unscaled",
                    "metric_scale_verified": False, "registration_verified": False,
                    "train_test_protocol": split_method,
                    "test_localization_may_use_test_pixels": True,
                    "training_must_exclude_test_views": True,
                    "source_videos": source_stats, "frame_selection_relpath": "frame_selection.csv",
                    "frame_selection_sha256": sha256(run_dir / "frame_selection.csv"),
                    "train_list_sha256": sha256(run_dir / "train_image_names.txt"),
                    "selection_parameters": {"sample_hz": sample_hz, "max_frames_per_video": max_frames,
                      "min_blur": min_blur, "min_mean": min_mean, "max_mean": max_mean,
                      "max_clip": max_clip, "embargo_sec": embargo_sec}, "frames": frames,
                    "counts": {"train": len(train), "test": len(test), "sampled": sum(len(x) for x in candidates.values()),
                               "rejected": sum(len(x) for x in candidates.values())-len(frames)},
                    "provenance": {"selector": "opencv_laplacian_luma_dhash_temporal_coverage_v1",
                                   "software": {"python": sys.version.split()[0], "opencv": cv2.__version__},
                                   "seed": None, "model_id": None, "gpu_executed": False}}
        emit_json(run_dir / "dataset_manifest.json", manifest)
        return manifest
    except BaseException:
        # Preserve forensic evidence if partial write fails. Never delete the run.
        raise


def _run_colmap(args: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"command": args, "started_at_utc": stamp()}) + "\n")
        f.flush()
        cp = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=False)
        f.write(json.dumps({"exit_code": cp.returncode, "ended_at_utc": stamp()}) + "\n")
    if cp.returncode:
        raise Blocked(f"COLMAP step failed ({cp.returncode}); read {log_path}")


def _colmap_cpu_options(executable: str) -> tuple[str, str]:
    """Probe installed COLMAP; modern and legacy CLI use different GPU flags."""
    choices=[]
    for subcommand in ("feature_extractor", "exhaustive_matcher"):
        try:
            proc=subprocess.run([executable, subcommand, "-h"], capture_output=True,
                                text=True, check=False, timeout=30)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise Blocked(f"COLMAP help probe failed: {exc}") from exc
        output=(proc.stdout or "")+(proc.stderr or "")
        candidates=("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu") if subcommand == "feature_extractor" else ("--FeatureMatching.use_gpu", "--SiftMatching.use_gpu")
        flag=next((x for x in candidates if x in output), None)
        if flag is None:
            raise Blocked(f"Cannot verify CPU option for {subcommand} in installed COLMAP CLI")
        choices.append(flag)
    return tuple(choices)


def run_colmap(run_dir: Path, executable: str, *, execute: bool = False) -> dict:
    m = checked_run(run_dir)
    output = run_dir / "colmap"
    plan = {"schema_version": "mycon.reconstruction.colmap_plan.v1", "run_id": run_dir.name,
            "test_features_enter_database_after_train_mapping": True,
            "train_sparse_sha_verified_after_test_feature_matching": True,
            "fresh_cameras_required": True, "existing_6999_poses_must_not_be_reused": True,
            "input_sha256": sha256(run_dir / "dataset_manifest.json"),
            "train_only_mapper": True, "test_localization_only_after_mapper": True,
            "output_relpath": "colmap", "status": "not_executed"}
    if not execute:
        return plan
    if output.exists():
        raise Blocked("COLMAP output already exists: create a new Run ID; no force overwrite")
    if not shutil.which(executable) and not Path(executable).is_file():
        raise Blocked(f"COLMAP not installed: {executable}")
    feature_cpu_flag, matching_cpu_flag = _colmap_cpu_options(executable)
    output.mkdir(parents=False, exist_ok=False)
    log = output / "colmap.log"
    db = output / "database.db"
    sparse = output / "sparse"
    sparse.mkdir()
    trainlist = run_dir / "train_image_names.txt"
    testlist = output / "test_image_names.txt"
    # Never put test features or matches into the mapper database until AFTER
    # train-only mapping has completed and the resulting model is frozen.
    test_names = [str(Path(f["image_relpath"]).relative_to("images")).replace("\\", "/")
                  for f in m["frames"] if f["split"] == "test"]
    testlist.write_text("\n".join(test_names) + "\n", encoding="utf-8")
    cmds = [
      [executable, "feature_extractor", "--database_path", str(db), "--image_path", str(run_dir / "images"),
       "--image_list_path", str(trainlist), "--ImageReader.single_camera_per_folder", "1", feature_cpu_flag, "0"],
      [executable, "sequential_matcher", "--database_path", str(db), matching_cpu_flag, "0"],
      [executable, "mapper", "--database_path", str(db), "--image_path", str(run_dir / "images"),
       "--output_path", str(sparse)],
    ]
    for cmd in cmds:
        _run_colmap(cmd, log)
    models = sorted([p for p in sparse.iterdir() if p.is_dir() and (p / "images.bin").is_file()],
                    key=lambda p: -sum((p / n).stat().st_size for n in ("cameras.bin", "images.bin", "points3D.bin") if (p / n).is_file()))
    if not models:
        raise Blocked("Fresh COLMAP mapping produced zero sparse models")
    model = models[0]
    train_txt = output / "train_text"
    train_txt.mkdir()
    _run_colmap([executable, "model_converter", "--input_path", str(model), "--output_path", str(train_txt), "--output_type", "TXT"], log)
    if not all((train_txt / n).is_file() and (train_txt / n).stat().st_size for n in IMAGE_KEYS):
        raise Blocked("Fresh mapper incomplete text model")
    # Now localize TEST views against the frozen train reconstruction.
    # Feature extraction/matching modifies only the NEW run's database;
    # train sparse is read-only and never bundle adjusted using test pixels.
    frozen_train_sha = {n: sha256(model / n) for n in ("cameras.bin", "images.bin", "points3D.bin")}
    _run_colmap([executable, "feature_extractor", "--database_path", str(db),
                 "--image_path", str(run_dir / "images"), "--image_list_path", str(testlist),
                 "--ImageReader.single_camera_per_folder", "1", feature_cpu_flag, "0"], log)
    _run_colmap([executable, "exhaustive_matcher", "--database_path", str(db),
                 matching_cpu_flag, "0"], log)
    if any(sha256(model / n) != frozen_train_sha[n] for n in frozen_train_sha):
        raise Blocked("Train model changed during test localization; invalid run")
    # Registrator works on a COPY of the model; never bundle-adjust after adding test views.
    registered = output / "registered_with_test"
    registered.mkdir()
    _run_colmap([executable, "image_registrator", "--database_path", str(db),
                 "--input_path", str(model), "--output_path", str(registered)], log)
    test_txt = output / "registered_text"
    test_txt.mkdir()
    _run_colmap([executable, "model_converter", "--input_path", str(registered), "--output_path", str(test_txt), "--output_type", "TXT"], log)
    audit = report_poses(run_dir, train_txt, test_txt)
    from .coverage import coverage
    coverage(run_dir)
    plan.update({"status": "colmap_executed", "train_model_relpath": rel(model, run_dir),
                 "registered_model_relpath": rel(registered, run_dir), "logs_relpath": rel(log, run_dir),
                 "quality_report_relpath": "registration_quality.json", "coverage_report_relpath": "coverage_report.json",
                 "registration_count_train": audit["registered_train"],
                 "registration_count_test": audit["registered_test"],
                 "train_model_sha256": {n: sha256(model / n) for n in ("cameras.bin", "images.bin", "points3D.bin")},
                 "test_registered_model_sha256": {n: sha256(registered / n) for n in ("cameras.bin", "images.bin", "points3D.bin")}})
    emit_json(output / "colmap_execution.json", plan)
    return plan


def parse_colmap_text(model_dir: Path) -> tuple[dict[int, dict], dict[str, dict]]:
    """Read trusted COLMAP text; two-line image records (including empty 2D line)."""
    cam: dict[int, dict] = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        parts = line.split()
        camera_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        cam[camera_id] = {"model": model, "width": w, "height": h, "parameters": [float(x) for x in parts[4:]]}
    raw = (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
    images = {}
    i = 0
    while i < len(raw):
        first = raw[i].strip()
        if not first or first.startswith("#"):
            i += 1
            continue
        if i + 1 >= len(raw): raise Blocked("COLMAP images.txt missing second line")
        p = first.split(maxsplit=9)
        if len(p) < 10: raise Blocked("Malformed COLMAP images.txt image row")
        w, x, y, z = map(float, p[1:5]); tx, ty, tz = map(float, p[5:8])
        norm = math.sqrt(w*w+x*x+y*y+z*z)
        if not math.isfinite(norm) or abs(norm-1) > 0.02:
            raise Blocked(f"Invalid COLMAP quaternion norm: {norm}")
        w, x, y, z = [v / norm for v in (w, x, y, z)]
        rot = np.array([
          [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        mat = np.eye(4); mat[:3,:3] = rot; mat[:3,3] = [tx,ty,tz]
        observations = raw[i+1].split()
        if len(observations) % 3: raise Blocked("Malformed COLMAP 2D observation triplets")
        valid = sum(1 for j in range(2,len(observations),3) if int(observations[j]) >= 0)
        images[p[9]] = {"camera_id": int(p[8]), "world_to_camera_4x4": mat.tolist(),
                         "camera_center_world": (-rot.T @ np.array([tx,ty,tz])).tolist(),
                         "observations_2d": len(observations)//3, "observations_3d": valid}
        i += 2
    return cam, images


def report_poses(run_dir: Path, train_text: Path, registered_text: Path) -> dict:
    m = checked_run(run_dir)
    cameras, train_poses = parse_colmap_text(train_text)
    test_cameras, all_poses = parse_colmap_text(registered_text)
    poses = []
    wrong_split = set(train_poses).intersection({Path(x["image_relpath"]).relative_to("images").as_posix() for x in m["frames"] if x["split"] == "test"})
    if wrong_split:
        raise Blocked(f"TEST LEAKAGE: test image appears in train COLMAP mapper: {sorted(wrong_split)}")
    for frame in m["frames"]:
        name = Path(frame["image_relpath"]).relative_to("images").as_posix()
        entry = (train_poses if frame["split"] == "train" else all_poses).get(name)
        camera = (cameras if frame["split"] == "train" else test_cameras).get(entry["camera_id"]) if entry else None
        if entry and not camera:
            raise Blocked(f"Unknown camera ID in fresh model: {name}")
        if camera:
            if (camera["width"], camera["height"]) != (frame["width"], frame["height"]):
                raise Blocked(f"Camera dimensions disagree with selected frame: {name}")
            intrinsics = camera["parameters"]
            valid_models = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
                            "RADIAL": 5, "OPENCV": 8, "OPENCV_FISHEYE": 8}
            if camera["model"] not in valid_models or len(intrinsics) != valid_models[camera["model"]]:
                raise Blocked(f"Unsupported/unverified camera intrinsics model: {name}")
            focal = intrinsics[:2] if camera["model"] in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE") else intrinsics[:1]
            if not all(math.isfinite(v) for v in intrinsics) or min(focal) <= 0:
                raise Blocked(f"Invalid camera intrinsics: {name}")
        poses.append({"view_id": frame["view_id"], "image_relpath": frame["image_relpath"],
                      "image_sha256": frame["image_sha256"], "capture_session_id": frame["capture_session_id"],
                      "split": frame["split"], "registered": bool(entry),
                      "coordinate_frame": COLMAP_FRAME if entry else "unknown",
                      "units": "reconstruction_units_unscaled" if entry else "unknown",
                      "metric_scale_verified": False, "camera": camera, **(entry or {})})
    if (run_dir / "camera_poses.json").exists() or (run_dir / "registration_quality.json").exists():
        raise Blocked("Pose/report already exists; refuse overwrite")
    emit_json(run_dir / "camera_poses.json", {"schema_version": "mycon.reconstruction.camera_poses.v1",
              "run_id": run_dir.name, "train_colmap_model_sha256": sha256(train_text / "images.txt"),
              "registered_colmap_model_sha256": sha256(registered_text / "images.txt"),
              "test_pose_method": "colmap_image_registrator_frozen_train_sparse_no_test_bundle_adjustment",
              "test_pixels_used_only_for_pose_localization": True, "poses": poses})
    rtrain = sum(1 for p in poses if p["split"] == "train" and p["registered"])
    rtest = sum(1 for p in poses if p["split"] == "test" and p["registered"])
    counts = {"train": sum(p["split"] == "train" for p in poses), "test": sum(p["split"] == "test" for p in poses)}
    rep = {"schema_version": "mycon.reconstruction.registration_quality.v1", "run_id": run_dir.name,
           "coordinate_frame": COLMAP_FRAME, "units": "reconstruction_units_unscaled",
           "metric_scale_verified": False, "registration_verified": False,
           "registered_train": rtrain, "registered_test": rtest,
           "selected_train": counts["train"], "selected_test": counts["test"],
           "train_registration_ratio": rtrain/counts["train"],
           "test_localization_ratio": rtest/counts["test"],
           "per_view": [{"view_id": p["view_id"], "split": p["split"],
                         "registered": p["registered"], "observations_2d": p.get("observations_2d"),
                         "observations_3d": p.get("observations_3d"),
                         "reprojection_error_px": None, "reason": None if p["registered"] else "not_registered"}
                        for p in poses],
           "limitations": ["No independent scale or GT", "No independent per-view reprojection residual available",
                           "Test camera pose localization uses test RGB; no test images in map optimisation"]}
    emit_json(run_dir / "registration_quality.json", rep)
    return rep


def score(run_dir: Path, render_manifest: Path, *, compute_lpips: bool = False) -> dict:
    """Compare real rendered TEST views against immutable selected-image SHA references."""
    import cv2
    from skimage.metrics import structural_similarity
    m = checked_run(run_dir)
    poses_file = run_dir / "camera_poses.json"
    if not poses_file.is_file(): raise Blocked("No fresh COLMAP cameras; cannot verify view/pose identity")
    pose_report = json.loads(poses_file.read_text(encoding="utf-8"))
    if pose_report.get("schema_version") != "mycon.reconstruction.camera_poses.v1" or pose_report.get("run_id") != run_dir.name:
        raise Blocked("Pose schema / run mismatch")
    poses = {x["view_id"]: x for x in pose_report["poses"]}
    provided = json.loads(render_manifest.read_text(encoding="utf-8"))
    if provided.get("schema_version") != "mycon.reconstruction.render_manifest.v1" or provided.get("run_id") != run_dir.name:
        raise Blocked("Render manifest schema/run mismatch")
    if provided.get("camera_poses_sha256") != sha256(poses_file):
        raise Blocked("Renderer was not tied to this run's fresh camera_poses SHA")
    ckpt = Path(provided["checkpoint_path"]).resolve()
    if not ckpt.is_file() or sha256(ckpt) != provided.get("checkpoint_sha256"):
        raise Blocked("Checkpoint missing or SHA differs; no score without verified checkpoint")
    by_id = {x["view_id"]: x for x in m["frames"]}
    rows = []
    scorer = None
    lpips_reason = "missing_optional_lpips_dependency"
    if compute_lpips:
        try:
            import torch
            import lpips
            scorer = lpips.LPIPS(net="alex").eval()
            lpips_reason = None
        except (ImportError, OSError) as exc:
            raise Blocked(f"LPIPS was requested but pretrained model unavailable: {exc}") from exc
    else:
        lpips_reason = "not_executed_enable_compute_lpips_with_verified_weights"
    seen = set()
    for pred in provided["renders"]:
        view = pred["view_id"]
        if view in seen: raise Blocked(f"Duplicate render view: {view}")
        seen.add(view)
        source = by_id.get(view)
        pose = poses.get(view)
        if not source or source["split"] != "test" or not pose or not pose["registered"]:
            raise Blocked(f"Rendered view is not a held-out registered TEST image: {view}")
        if pred.get("image_sha256") != source["image_sha256"] or pose["image_sha256"] != source["image_sha256"]:
            raise Blocked(f"Source image SHA mismatch in render/pose: {view}")
        p = (render_manifest.parent / pred["render_relpath"]).resolve()
        if not p.is_relative_to(render_manifest.parent.resolve()) or not p.is_file() or sha256(p) != pred.get("render_sha256"):
            raise Blocked(f"Unsafe/unverified render: {p}")
        gt = cv2.imread(str(run_dir / source["image_relpath"]), cv2.IMREAD_COLOR)
        rendered = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if gt is None or rendered is None or gt.shape != rendered.shape:
            raise Blocked(f"Render resolution/shape does not match source: {view}")
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pr_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        mse = float(np.mean((gt_rgb.astype(np.float64) - pr_rgb.astype(np.float64))**2))
        psnr = float(10*math.log10(255.0**2/mse)) if mse > 0 else None
        ssim = float(structural_similarity(gt_rgb, pr_rgb, channel_axis=2, data_range=255))
        lpips_value = None
        if scorer is not None:
            import torch
            with torch.no_grad():
                g = torch.from_numpy(gt_rgb.copy()).permute(2,0,1).unsqueeze(0).float()/127.5 - 1
                r = torch.from_numpy(pr_rgb.copy()).permute(2,0,1).unsqueeze(0).float()/127.5 - 1
                lpips_value = float(scorer(g,r).item())
        rows.append({"view_id": view, "image_sha256": source["image_sha256"],
                     "capture_session_id": source["capture_session_id"], "split": "test",
                     "registered": True, "psnr_db": psnr, "ssim": ssim, "lpips": lpips_value,
                     "lpips_state": "measured" if lpips_value is not None else lpips_reason,
                     "checkpoint_sha256": provided["checkpoint_sha256"], "render_sha256": pred["render_sha256"],
                     "mse": mse, "psnr_state": "identical_inf" if mse == 0 else "measured"})
    if not rows: raise Blocked("No held-out rendered images to evaluate")
    columns = ["view_id", "image_sha256", "capture_session_id", "split", "registered", "psnr_db",
               "ssim", "lpips", "lpips_state", "checkpoint_sha256", "render_sha256", "mse", "psnr_state"]
    emit_csv(run_dir / "per_view_metrics.csv", columns, rows)
    summary = {"schema_version": "mycon.reconstruction.heldout_scores.v1", "run_id": run_dir.name,
               "evaluated_views": len(rows), "selected_test_views": sum(x["split"] == "test" for x in m["frames"]),
               "test_only": True, "psnr_mean_db": None if any(x["psnr_db"] is None for x in rows) else float(np.mean([x["psnr_db"] for x in rows])),
               "ssim_mean": float(np.mean([x["ssim"] for x in rows])),
               "lpips_mean": float(np.mean([x["lpips"] for x in rows])) if all(x["lpips"] is not None for x in rows) else None,
               "render_manifest_sha256": sha256(render_manifest),
               "checkpoint_sha256": provided["checkpoint_sha256"], "gt_is_selected_frame_only": True,
               "metric_scale_verified": False}
    emit_json(run_dir / "heldout_scores.json", summary)
    return summary


def compare_6999(run_dir: Path, baseline_per_view: Path | None = None) -> dict:
    """Comparison is valid only for the same view/image hashes and test split."""
    checked_run(run_dir)
    new = run_dir / "per_view_metrics.csv"
    if (run_dir / "comparison_6999.json").exists():
        raise Blocked("Comparison already exists")
    result = {"schema_version": "mycon.reconstruction.baseline_comparison.v1",
              "run_id": run_dir.name, "baseline": "historical_6999", "comparable": False,
              "historical_reported_log_only": {"psnr_db": 20.736, "ssim": 0.6176, "lpips": 0.522},
              "historical_dataset_and_split_verified": False,
              "reason": "Historical 6999 view hashes/splits/pose lineage unverified; cannot claim superiority",
              "paired_test_views": 0, "delta": None}
    if baseline_per_view:
        if not baseline_per_view.is_file() or not new.is_file():
            raise Blocked("Both measured per-view tables required for paired comparison")
        with baseline_per_view.open(newline="", encoding="utf-8") as f: base = list(csv.DictReader(f))
        with new.open(newline="", encoding="utf-8") as f: latest = list(csv.DictReader(f))
        key = lambda x: (x["view_id"], x["image_sha256"])
        b, n = {key(r):r for r in base}, {key(r):r for r in latest}
        if len(b) != len(base) or len(n) != len(latest) or set(b) != set(n) or any(r["split"] != "test" for r in base+latest):
            result["reason"] = "Unmatched or duplicated view+SHA/test split; paired comparison blocked"
        elif not all(r.get("psnr_db") and r.get("ssim") and r.get("lpips") for r in base+latest):
            result["reason"] = "Missing PSNR/SSIM/LPIPS for some paired views"
        else:
            result["paired_test_views"] = len(n)
            result["view_sha_pairs_match"] = True
            result["descriptive_delta_only_unverified_camera_baseline"] = {
                field: round(sum(float(n[k][field])-float(b[k][field]) for k in n)/len(n), 6)
                for field in ("psnr_db", "ssim", "lpips")
            }
            result["reason"] = "Paired view/image SHA match, but baseline pose/checkpoint/split provenance is not independently certified; scientific comparability remains FALSE"
    emit_json(run_dir / "comparison_6999.json", result)
    return result


def train_plan(run_dir: Path, *, prior_checkpoint: Path | None = None) -> dict:
    m = checked_run(run_dir)
    ckpt = {"provided": False}
    if prior_checkpoint:
        if not prior_checkpoint.is_file(): raise Blocked("Checkpoint not present")
        ckpt = {"provided": True, "external_path": str(prior_checkpoint.resolve()),
                "sha256": sha256(prior_checkpoint), "bytes": prior_checkpoint.stat().st_size}
    result = {"schema_version": "mycon.reconstruction.gs_training_plan.v1", "run_id": run_dir.name,
              "dataset_manifest_sha256": sha256(run_dir / "dataset_manifest.json"),
              "train_view_sha256": {x["view_id"]: x["image_sha256"] for x in m["frames"] if x["split"] == "train"},
              "test_view_sha256": {x["view_id"]: x["image_sha256"] for x in m["frames"] if x["split"] == "test"},
              "requires_new_fresh_colmap_train_model": True,
              "existing_6999_pose_and_checkpoint_not_transferable_without_explicit_transform_and_compatibility_validation": True,
              "legacy_checkpoint_read_only": ckpt,
              "training_status": "not_executed_explicit_gpu_opt_in_required",
              "requested_heavy_training": False, "trained_checkpoint": None, "new_gaussian_ply": None,
              "renders": None, "trajectory": None,
              "training_backend_command": None,
              "required_validation": ["official gsplat install + CUDA", "fresh poses", "train-only dataset",
                                      "recovery snapshot", "held-out test-only raster", "checkpoint and PLY SHA256"]}
    emit_json(run_dir / "gs_training_plan.json", result)
    return result


def receipt(run_dir: Path) -> dict:
    """Append-only audit of existing artifacts; excludes mutable/log source video files."""
    m = checked_run(run_dir)
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "artifact_receipt.json":
            continue
        files.append({"relative_path": rel(path, run_dir), "sha256": sha256(path),
                      "size_bytes": path.stat().st_size})
    result = {"schema_version": "mycon.reconstruction.artifact_receipt.v1",
              "run_id": m["run_id"], "created_at_utc": stamp(),
              "coordinate_frame": "colmap_world_unscaled_if_registered_else_unknown",
              "units": "reconstruction_units_unscaled_or_unknown",
              "metric_scale_verified": False, "quality_gate": "no_independent_metric_validation",
              "source_provenance_manifest_sha256": sha256(run_dir / "dataset_manifest.json"),
              "artifacts": files}
    emit_json(run_dir / "artifact_receipt.json", result)
    return {"status": "receipt_created", "run_id": run_dir.name, "file_count": len(files)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MyCON Reconstruction (never modifies originals, no heavy training)")
    subs = p.add_subparsers(dest="action", required=True)
    a = subs.add_parser("select", help="Create new run and select frames (video read-only)")
    a.add_argument("--run-root", type=Path, required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--train-video", type=Path, required=True)
    a.add_argument("--train-session", required=True)
    a.add_argument("--test-video", type=Path)
    a.add_argument("--test-session")
    a.add_argument("--sample-hz", type=float, default=2.)
    a.add_argument("--max-frames", type=int, default=180)
    a.add_argument("--min-blur", type=float, default=35.)
    a.add_argument("--min-mean", type=float, default=25.)
    a.add_argument("--max-mean", type=float, default=230.)
    a.add_argument("--max-clip", type=float, default=.38)
    a.add_argument("--embargo-sec", type=float, default=1.)
    b = subs.add_parser("colmap", help="Dry-run by default; execute fresh train-only mapper and test localization")
    b.add_argument("--run-dir", type=Path, required=True)
    b.add_argument("--executable", default="colmap")
    b.add_argument("--execute-colmap", action="store_true")
    c = subs.add_parser("score", help="Score independently rendered registered test views")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--render-manifest", type=Path, required=True)
    c.add_argument("--compute-lpips", action="store_true", help="Explicit opt-in: may require pretrained weights")
    d = subs.add_parser("compare", help="Compare to historical 6999 only if paired rows match")
    d.add_argument("--run-dir", type=Path, required=True)
    d.add_argument("--baseline-per-view", type=Path)
    e = subs.add_parser("plan-gs", help="Create audit-only plan, NEVER start training")
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--prior-checkpoint", type=Path)
    f = subs.add_parser("receipt", help="Create immutable SHA256 audit of existing run outputs")
    f.add_argument("--run-dir", type=Path, required=True)
    opts = p.parse_args(argv)
    try:
        if opts.action == "select":
            res = prepare(run_root=opts.run_root, run_id=opts.run_id, train_video=opts.train_video,
                          train_session=opts.train_session, test_video=opts.test_video, test_session=opts.test_session,
                          sample_hz=opts.sample_hz, max_frames=opts.max_frames, min_blur=opts.min_blur,
                          min_mean=opts.min_mean, max_mean=opts.max_mean, max_clip=opts.max_clip,
                          embargo_sec=opts.embargo_sec)
        elif opts.action == "colmap": res = run_colmap(opts.run_dir, opts.executable, execute=opts.execute_colmap)
        elif opts.action == "score": res = score(opts.run_dir, opts.render_manifest, compute_lpips=opts.compute_lpips)
        elif opts.action == "compare": res = compare_6999(opts.run_dir, opts.baseline_per_view)
        elif opts.action == "plan-gs": res = train_plan(opts.run_dir, prior_checkpoint=opts.prior_checkpoint)
        else: res = receipt(opts.run_dir)
    except (Blocked, FileExistsError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked_or_failed", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(res if opts.action != "select" else {"status":"selected", "counts":res["counts"], "run_id":res["run_id"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
