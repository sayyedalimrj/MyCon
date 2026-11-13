"""Real-train-image pairwise SIFT diagnostic, not COLMAP, geometry or camera calibration.

Read-only on SHA-verified extracted train frames. Never loads test images:
protected TEST must not be used to tune training-frame selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from .workflow import Blocked, checked_run, sha256, emit_json


def audit(run_dir: Path, *, max_width: int = 960, features: int = 1300) -> dict:
    import cv2

    manifest = checked_run(run_dir)
    output = run_dir / "train_pairwise_overlap_diagnostic.json"
    if output.exists():
        raise Blocked("Audit exists; use a new run ID, never overwrite")
    rows = sorted(
        (f for f in manifest["frames"] if f["split"] == "train"),
        key=lambda r: (r["capture_session_id"], r["timestamp_sec"]),
    )
    if len(rows) < 3:
        raise Blocked("Too few training images")
    sift = cv2.SIFT_create(nfeatures=features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    image_features = []
    for frame in rows:
        path = run_dir / frame["image_relpath"]
        if sha256(path) != frame["image_sha256"]:
            raise Blocked("Frame SHA mismatch")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise Blocked("Cannot decode training image")
        scale = min(1., max_width / image.shape[1])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        image_features.append((keypoints, descriptors, len(keypoints)))
    pairs = []
    for i in range(len(rows) - 1):
        left, right = rows[i], rows[i + 1]
        k1, d1, n1 = image_features[i]
        k2, d2, n2 = image_features[i + 1]
        if left["capture_session_id"] != right["capture_session_id"]:
            pairs.append({
                "left": left["view_id"], "right": right["view_id"],
                "reason": "different_session_not_compared",
                "fundamental_inliers": None, "good_matches": None,
            })
            continue
        dt = right["timestamp_sec"] - left["timestamp_sec"]
        good = []
        if d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:
            good = [m for m, n in matcher.knnMatch(d1, d2, k=2)
                    if m.distance < .75 * n.distance]
        finliers, hinliers, displacement = 0, 0, None
        if len(good) >= 8:
            p1 = np.float32([k1[m.queryIdx].pt for m in good])
            p2 = np.float32([k2[m.trainIdx].pt for m in good])
            displacement = round(float(np.median(np.linalg.norm(p1 - p2, axis=1))), 2)
            try:
                F, fm = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, .99)
                if F is not None and fm is not None:
                    finliers = int(fm.astype(bool).sum())
                H, hm = cv2.findHomography(p1, p2, cv2.RANSAC, 3.)
                if H is not None and hm is not None:
                    hinliers = int(hm.astype(bool).sum())
            except cv2.error:
                pass
        pairs.append({
            "left": left["view_id"], "right": right["view_id"],
            "session": left["capture_session_id"],
            "dt_sec": round(dt, 6),
            "features_left": n1, "features_right": n2,
            "good_matches": len(good),
            "fundamental_inliers": finliers, "homography_inliers": hinliers,
            "fundamental_inlier_fraction": round(finliers / max(len(good), 1), 4),
            "median_scaled_displacement_px": displacement,
            "flag": "weak_match_proxy" if finliers < 20 else "diagnostic_inliers_present",
        })
    valid = [p for p in pairs if p.get("fundamental_inliers") is not None]
    result = {
        "schema_version": "mycon.reconstruction.real_train_overlap_diagnostic.v1",
        "run_id": run_dir.name,
        "dataset_manifest_sha256": sha256(run_dir / "dataset_manifest.json"),
        "train_images_processed": len(rows), "test_images_processed": 0,
        "sift_max_features": features, "max_width_px": max_width,
        "method": "SIFT_ratio_075_F_RANSAC_1p5px_H_RANSAC_3px_diagnostic_only",
        "pair_count": len(valid),
        "weak_pair_count_under20_F_inliers": sum(p["fundamental_inliers"] < 20 for p in valid),
        "median_good_matches": float(np.median([p["good_matches"] for p in valid])),
        "median_F_inliers": float(np.median([p["fundamental_inliers"] for p in valid])),
        "max_temporal_gap_between_train_sec": max(p.get("dt_sec", 0) for p in valid),
        "per_pair": pairs, "new_colmap_executed": False,
        "metric_scale_verified": False,
        "limitations": [
            "Feature correspondences are not calibrated poses, COLMAP SfM or geometry",
            "F/H counts cannot establish physical surface coverage or parallax",
            "Only TRAIN images analyzed; no TEST used to tune frame selection",
        ],
    }
    emit_json(output, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.run_dir)
        print(json.dumps({k: result[k] for k in (
            "run_id", "pair_count", "weak_pair_count_under20_F_inliers",
            "median_good_matches", "median_F_inliers",
            "max_temporal_gap_between_train_sec",
        )}))
        return 0
    except (Blocked, ValueError, OSError, ImportError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
