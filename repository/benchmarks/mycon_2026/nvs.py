"""Independent held-out RGB novel-view scoring; no geometry/BIM claim.

The two JSON files contain view records with view_id, image (relative path),
sha256, and (for GT) split=test and held_out=true. No auto-resize/cropping.
Optional LPIPS requires an explicitly provided preloaded evaluator; this module
does not fetch model weights or run training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean

from .protocol import resolve_input, sha256


def score_nvs_pairs(reference_path: Path, prediction_path: Path, *, lpips_evaluator=None):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity

    def load(path):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("schema_version") != "mycon.nvs_views.v1" or not isinstance(obj.get("views"), list):
            raise ValueError("Expected mycon.nvs_views.v1 JSON")
        result = {}
        for view in obj["views"]:
            key = str(view.get("view_id") or "").strip()
            if not key or key in result:
                raise ValueError("Duplicate/missing NVS view_id")
            result[key] = view
        return result

    reference, prediction = load(reference_path), load(prediction_path)
    if not reference:
        raise ValueError("No independent held-out NVS ground truth")
    if not all(v.get("split") == "test" and v.get("held_out") is True for v in reference.values()):
        raise ValueError("NVS reference views must be expressly held out")
    seen = set()
    for v in reference.values():
        if not str(v.get("capture_session_id") or "").strip():
            raise ValueError("NVS missing capture session identity")
        if v.get("sha256") in seen:
            raise ValueError("Duplicate GT image across held-out views")
        seen.add(v.get("sha256"))
    scores, used = [], []
    for key in sorted(reference.keys() & prediction.keys()):
        gt, pred = reference[key], prediction[key]
        a = resolve_input(reference_path.parent, gt["image"])
        b = resolve_input(prediction_path.parent, pred["image"])
        for p, item in ((a, gt), (b, pred)):
            if not p.is_file() or sha256(p) != item.get("sha256"):
                raise ValueError("Missing/mismatched NVS image digest: " + key)
        if sha256(a) == sha256(b) and gt.get("image") == pred.get("image"):
            raise ValueError("Prediction reuses the GT image as its own output")
        with Image.open(a) as src:
            x = np.asarray(src.convert("RGB"), dtype=np.float64) / 255.0
        with Image.open(b) as src:
            y = np.asarray(src.convert("RGB"), dtype=np.float64) / 255.0
        if x.shape != y.shape or min(x.shape[:2]) < 7:
            raise ValueError("NVS GT/prediction size mismatch or image too small")
        mse = float(np.mean((x - y) ** 2))
        psnr = float(-10 * math.log10(mse)) if mse > 0 else None
        ssim = float(structural_similarity(x, y, data_range=1.0, channel_axis=2))
        lpips = None if lpips_evaluator is None else float(lpips_evaluator(x, y))
        if lpips is not None and (not math.isfinite(lpips) or lpips < 0):
            raise ValueError("Invalid LPIPS evaluator output")
        scores.append({"view_id": key, "psnr_db": psnr, "ssim": ssim, "lpips": lpips})
        used.extend([{"role": "heldout_nvs_gt", "path": str(a), "sha256": gt["sha256"]},
                     {"role": "nvs_render", "path": str(b), "sha256": pred["sha256"]}])
    if not scores:
        raise ValueError("No paired held-out NVS views")
    finite_psnr = [r["psnr_db"] for r in scores if r["psnr_db"] is not None]
    lpips_values = [r["lpips"] for r in scores if r["lpips"] is not None]
    return {
        "n_gt_views": len(reference), "n_pred_views": len(prediction),
        "n_paired_views": len(scores), "n_missing_prediction": len(reference) - len(scores),
        "psnr_db_mean_finite": mean(finite_psnr) if finite_psnr else None,
        "psnr_n_infinite_perfect": len(scores) - len(finite_psnr),
        "ssim_mean": mean(r["ssim"] for r in scores),
        "lpips_mean": mean(lpips_values) if lpips_values else None,
        "lpips_status": "evaluated" if lpips_values else "not_executed",
        "per_view": scores, "image_hashes": used,
        "note": "NVS photometric quality is separate from metric geometry and BIM accuracy",
    }
