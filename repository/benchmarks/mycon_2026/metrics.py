"""Independent metrics. Missing/abstaining evidence never becomes a negative label.

Distances use metres, probabilities/fractions use [0, 1]. No pipeline imports.
"""
from __future__ import annotations

import csv
import math
import re
import struct
from pathlib import Path
from statistics import mean, median


PLY_TYPES = {
    "char": "b", "uchar": "B", "short": "h", "ushort": "H",
    "int": "i", "uint": "I", "float": "f", "double": "d",
    "int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
    "int32": "i", "uint32": "I", "float32": "f", "float64": "d",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _finite_xyz(v) -> tuple[float, float, float]:
    xyz = tuple(float(n) for n in v)
    if len(xyz) != 3 or not all(math.isfinite(n) for n in xyz):
        raise ValueError("Point coordinates must contain three finite values")
    return xyz


def read_ply_xyz(path: Path, *, max_points: int = 20000, seed: int = 42):
    """Read vertex xyz from ASCII or little-endian binary PLY, deterministic sample.

    Reject unsupported property layout rather than silently returning bad geometry.
    Sampling is uniform without replacement and never reorders source files.
    """
    import random

    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        fmt, count, properties, element = None, None, [], None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            tokens = line.decode("ascii").strip().split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                element = tokens[1]
                if element == "vertex":
                    count = int(tokens[2])
            elif tokens[0] == "property" and element == "vertex":
                if tokens[1] == "list":
                    raise ValueError("Vertex list properties not supported")
                properties.append((tokens[2], tokens[1]))
            elif tokens[0] == "end_header":
                break
        if count is None or count <= 0 or max_points <= 0:
            raise ValueError("Missing/empty PLY vertices or invalid sample cap")
        names = [x[0] for x in properties]
        if not all(k in names for k in ("x", "y", "z")):
            raise ValueError("PLY missing x/y/z vertex properties")
        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError("Unsupported PLY format: " + str(fmt))
        rng = random.Random(seed)
        selected = set(rng.sample(range(count), min(count, max_points)))
        coords = []
        if fmt == "binary_little_endian":
            layout = "<" + "".join(PLY_TYPES[typ] for _, typ in properties)
            width = struct.calcsize(layout)
        for index in range(count):
            if fmt == "ascii":
                values = stream.readline().split()
                if len(values) < len(properties):
                    raise ValueError("Truncated ASCII vertex")
            else:
                data = stream.read(width)
                if len(data) != width:
                    raise ValueError("Truncated binary vertex")
                values = struct.unpack(layout, data)
            if index in selected:
                coords.append(_finite_xyz(values[names.index(key)] for key in ("x", "y", "z")))
        return coords, count


def nearest_distances(source, target, *, max_bruteforce_pairs: int = 2_000_000):
    if not source or not target:
        raise ValueError("Cannot score empty point clouds")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        if len(source) * len(target) > max_bruteforce_pairs:
            raise RuntimeError("scipy_required_for_large_point_clouds")
        return [min(math.dist(a, b) for b in target) for a in source]
    distances, _ = cKDTree(target).query(source, k=1, workers=1)
    return [float(x) for x in distances]


def reconstruction_metrics(pred, truth, *, tau_m: float = 0.05):
    if not math.isfinite(tau_m) or tau_m <= 0:
        raise ValueError("tau_m must be positive and finite")
    d_pred = nearest_distances(pred, truth)
    d_true = nearest_distances(truth, pred)
    precision = sum(x <= tau_m for x in d_pred) / len(d_pred)
    recall = sum(x <= tau_m for x in d_true) / len(d_true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    curve = {}
    for threshold in (0.01, 0.02, 0.05, 0.10):
        acc = sum(x <= threshold for x in d_pred) / len(d_pred)
        comp = sum(x <= threshold for x in d_true) / len(d_true)
        score = 2 * acc * comp / (acc + comp) if acc + comp else 0.0
        label = f"{threshold:.2f}m"
        curve[f"accuracy_at_{label}"] = acc
        curve[f"completeness_at_{label}"] = comp
        curve[f"fscore_at_{label}"] = score
    return {
        **curve,
        "precision_at_tau": precision, "recall_at_tau": recall,
        "accuracy_at_tau": precision, "completeness_at_tau": recall,
        "fscore_at_tau": f1, "tau_m": tau_m,
        "chamfer_l1_m": (mean(d_pred) + mean(d_true)) / 2,
        "chamfer_l2_m2": (mean(x * x for x in d_pred) + mean(x * x for x in d_true)) / 2,
        "n_pred_sampled": len(pred), "n_truth_sampled": len(truth),
    }


def compare_anchors(pred_rows, truth_rows):
    """Paired independent surveyed target-frame control points, not ICP fitness."""
    def index(rows):
        result = {}
        for r in rows:
            key = r["anchor_id"].strip()
            if not key or key in result:
                raise ValueError("Empty or duplicate anchor_id")
            result[key] = _finite_xyz((r["x_m"], r["y_m"], r["z_m"]))
        return result
    a, b = index(pred_rows), index(truth_rows)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError("No matched survey anchors")
    errors = sorted(math.dist(a[k], b[k]) for k in keys)
    # Ratio of pairwise distances detects scale drift even if rigid translation is present.
    ratios = []
    for i, key in enumerate(keys):
        for other in keys[i + 1:]:
            truth_d = math.dist(b[key], b[other])
            if truth_d > 1e-8:
                ratios.append(math.dist(a[key], a[other]) / truth_d)
    scale = median(ratios) if ratios else None
    orientation_angles = []
    for i, key in enumerate(keys):
        for other in keys[i + 1:]:
            u = tuple(a[other][j] - a[key][j] for j in range(3))
            v = tuple(b[other][j] - b[key][j] for j in range(3))
            lu, lv = math.dist((0, 0, 0), u), math.dist((0, 0, 0), v)
            if min(lu, lv) > 1e-8:
                cosine = max(-1., min(1., sum(u[j] * v[j] for j in range(3)) / lu / lv))
                orientation_angles.append(math.degrees(math.acos(cosine)))
    return {
        "n_matched_anchors": len(keys), "tre_mean_m": mean(errors),
        "tre_rmse_m": math.sqrt(mean(e * e for e in errors)),
        "tre_p95_m": errors[math.ceil(0.95 * len(errors)) - 1],
        "scale_ratio_pred_over_truth": scale,
        "scale_error_pct": 100 * abs(scale - 1) if scale is not None else None,
        "n_scale_pairs": len(ratios),
        "orientation_pairwise_median_deg": median(orientation_angles) if orientation_angles else None,
        "n_orientation_pairs": len(orientation_angles),
    }


def _status(status: str):
    label = str(status or "").strip().lower()
    mapping = {
        "built": "built", "complete": "built", "likely_completed": "built",
        "complete_candidate": "built", "not_built": "not_built",
        "partial": "partial", "partially_observed": None,
        "partial_candidate": "partial", "unknown": None,
    }
    # 'not_evidenced' and 'insufficient_coverage' MUST be abstentions.
    return mapping.get(label)


def _fraction(value):
    if value is None or str(value).strip() == "":
        return None
    x = float(value)
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("Fractions/confidences must be finite in [0,1]")
    return x


def _confidence(value):
    if value is None or str(value).strip().lower() in {"", "high", "medium", "low", "uncertain", "unknown"}:
        return None
    return _fraction(value)


def progress_metrics(pred_rows, truth_rows, *, bins=10):
    """Three-class selective evaluation, unknown/abstention retained in denominator.

    Ground truth: global_id,status,completion_fraction,weight.
    Prediction: global_id,completion_state/status,physical_progress_estimate,
                evidence_confidence/confidence. No invented progress fractions.
    """
    gt, predictions = {}, {}
    for r in truth_rows:
        key = str(r.get("global_id", "")).strip()
        if not key or key in gt:
            raise ValueError("Duplicate/empty truth global_id")
        if r.get("status") not in {"built", "not_built", "partial"}:
            raise ValueError("Ground truth requires independent built/not_built/partial label")
        gt[key] = r
    for r in pred_rows:
        key = str(r.get("global_id", "")).strip()
        if not key or key in predictions:
            raise ValueError("Duplicate/empty prediction global_id")
        predictions[key] = r
    if not gt:
        raise ValueError("No independent ground truth rows")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    reported_observability = []
    cm = {c: {k: 0 for k in ("built", "not_built", "partial")} for c in ("built", "not_built", "partial")}
    known = missing = prediction_missing = explicit_abstentions = 0
    numeric_gt_available = numeric_prediction_available = 0
    confidence_correct, absolute_errors = [], []
    total_weight = observed_weight = weighted_pred = weighted_true_known = 0.0
    for key, r in gt.items():
        p = predictions.get(key, {})
        predicted = _status(p.get("completion_state") or p.get("status", ""))
        if p.get("observability_ratio") is not None and str(p["observability_ratio"]).strip():
            reported_observability.append(_fraction(p["observability_ratio"]))
        true = r["status"]
        weight = float(r.get("weight") or 1.0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("GT weights must be positive")
        total_weight += weight
        truth_fraction = _fraction(r.get("completion_fraction"))
        # Independent estimands: a categorical abstention cannot discard a verified
        # numeric fraction (and a categorical success cannot invent one).
        if key not in predictions:
            prediction_missing += 1
        if truth_fraction is not None:
            numeric_gt_available += 1
        fraction = _fraction(p.get("physical_progress_estimate"))
        if fraction is not None:
            numeric_prediction_available += 1
        if fraction is not None and truth_fraction is not None:
            absolute_errors.append((abs(fraction - truth_fraction), weight))
            observed_weight += weight
            weighted_pred += fraction * weight
            weighted_true_known += truth_fraction * weight
        if predicted is None:
            missing += 1
            if key in predictions:
                explicit_abstentions += 1
        else:
            known += 1
            cm[true][predicted] += 1
            conf = _confidence(p.get("evidence_confidence") if p.get("evidence_confidence") not in (None, "") else p.get("confidence"))
            if conf is not None:
                confidence_correct.append((conf, float(predicted == true)))
    scores = {}
    for label in cm:
        tp = cm[label][label]
        fp = sum(cm[t][label] for t in cm if t != label)
        fn = sum(cm[label][t] for t in cm if t != label)
        pre = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores[label] = {"precision": pre, "recall": rec,
                         "f1": (2 * pre * rec / (pre + rec) if pre + rec else 0.0),
                         "support_evaluated": sum(cm[label].values())}
    ece = None
    brier = None
    if confidence_correct:
        ece = 0.0
        for i in range(bins):
            selected = [(c, y) for c, y in confidence_correct
                        if (i / bins <= c < (i + 1) / bins if i < bins - 1 else i / bins <= c <= 1)]
            if selected:
                ece += len(selected) / len(confidence_correct) * abs(mean(c for c, _ in selected) - mean(y for _, y in selected))
        brier = mean((c - y) ** 2 for c, y in confidence_correct)
    total_error_weight = sum(w for _, w in absolute_errors)
    return {
        "n_truth": len(gt), "n_predictions_matched": sum(k in predictions for k in gt),
        "n_evaluated": known, "n_unknown": missing, "unknown_rate": missing / len(gt),
        "n_categorical_missing_output": prediction_missing,
        "n_categorical_abstained": explicit_abstentions,
        "categorical_abstention_rate_all_gt": explicit_abstentions / len(gt),
        "classification_coverage": known / len(gt),
        "n_numeric_gt_available": numeric_gt_available,
        "n_numeric_prediction_available": numeric_prediction_available,
        "n_numeric_missing_or_unpaired": len(gt) - len(absolute_errors),
        "numeric_pair_coverage": len(absolute_errors) / len(gt),
        "observability_ratio_mean_reported": mean(reported_observability) if reported_observability else None,
        "confusion_matrix": cm,
        "per_class": scores, "macro_f1_selective": mean(v["f1"] for v in scores.values()),
        "micro_accuracy_selective": sum(cm[c][c] for c in cm) / known if known else None,
        "micro_precision_selective": sum(cm[c][c] for c in cm) / known if known else None,
        "micro_recall_selective": sum(cm[c][c] for c in cm) / known if known else None,
        "micro_f1_selective": sum(cm[c][c] for c in cm) / known if known else None,
        "progress_mae_fraction_known": (sum(e * w for e, w in absolute_errors) / total_error_weight if total_error_weight else None),
        "n_progress_pairs": len(absolute_errors),
        "progress_weight_coverage": observed_weight / total_weight,
        "project_progress_lower_bound_fraction": weighted_pred / total_weight,
        "project_progress_upper_bound_fraction": (weighted_pred + total_weight - observed_weight) / total_weight,
        "ece_10bin_selective" if bins == 10 else "ece_selective": ece,
        "brier_selective": brier, "n_calibration": len(confidence_correct),
    }


def ifc_element_metrics(pred_rows, truth_rows):
    """Check identity and class against an independently approved IFC inventory."""
    def convert(rows):
        d = {}
        for row in rows:
            gid = str(row.get("global_id") or row.get("GlobalId") or "").strip()
            if not gid or gid in d:
                raise ValueError("Invalid/duplicate IFC element global_id")
            d[gid] = str(row.get("ifc_class") or row.get("type") or "").strip()
        return d
    predicted, truth = convert(pred_rows), convert(truth_rows)
    hits = predicted.keys() & truth.keys()
    ifc_guid = re.compile(r"[0-9A-Za-z_$]{22}")
    bad_guid = sum(ifc_guid.fullmatch(k) is None for k in predicted)
    missing_class = sum(not klass.startswith("Ifc") for klass in predicted.values())
    return {
        "n_gt_elements": len(truth), "n_pred_elements": len(predicted),
        "n_invalid_ifc_globalid_format_pred": bad_guid,
        "n_missing_or_invalid_ifc_class_pred": missing_class,
        "identity_precision": len(hits) / len(predicted) if predicted else None,
        "identity_recall": len(hits) / len(truth) if truth else None,
        "ifc_class_accuracy_matched": sum(predicted[x] == truth[x] for x in hits) / len(hits) if hits else None,
        "n_class_matched": len(hits),
    }
