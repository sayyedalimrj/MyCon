"""Read-only, leakage-guarded benchmark runner; never launches upstream stages."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import time
import importlib.metadata
import platform
import random
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .metrics import (compare_anchors, ifc_element_metrics, progress_metrics,
                      read_csv, read_ply_xyz, reconstruction_metrics)
from .adapters import (load_stage9_v2, load_scientific_progress_v2,
                       activity_progress_metrics, project_progress_metrics)
from .protocol import resolve_input, sha256, validate_manifest
from .nvs import score_nvs_pairs

TASK_INPUTS = {
    "reconstruction": ("reconstruction_gt_ply", "reconstruction_ply"),
    "bim_geometry": ("asbuilt_gt_ply", "aligned_scan_ply"),
    "registration": ("survey_anchors_csv", "registered_anchors_csv"),
    "ifc": ("ifc_elements_csv", "ifc_elements_csv"),
    "progress": ("progress_csv", "element_metrics_csv"),
    "activity_progress": ("activity_progress_gt_csv", "scientific_progress_json"),
    "project_progress": ("project_progress_gt_csv", "scientific_progress_json"),
    "nvs": ("nvs_reference_manifest_json", "nvs_prediction_manifest_json"),
    "performance": (None, "telemetry_json"),
}


def _summary(measured: list[dict], seed: int, *, repetitions: int = 2000):
    """Cluster bootstrap on independent buildings, never on individual points."""
    by_key = defaultdict(list)
    for r in measured:
        if isinstance(r.get("value"), (int, float)) and math.isfinite(r["value"]):
            by_key[(r["method"], r["task"], r["metric"])].append(r)
    result = []
    for (method, task, metric), rows in sorted(by_key.items()):
        groups = defaultdict(list)
        for row in rows:
            groups[row["site_id"]].append(row["value"])
        values = [mean(v) for v in groups.values()]
        ci = [None, None]
        if len(values) >= 2:
            rng = random.Random(f"{seed}:{method}:{task}:{metric}")
            draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
            ci = [draws[int(0.025 * repetitions)], draws[min(repetitions - 1, int(0.975 * repetitions))]]
        result.append({"method": method, "task": task, "metric": metric,
                       "mean": mean(values), "ci95_low": ci[0], "ci95_high": ci[1],
                       "n_sites": len(values), "n_buildings": len({r["building_key"] for r in rows}), "n_cases": len(rows),
                       "unit": "m" if metric.endswith("_m") else ("fraction" if "fraction" in metric or "fscore" in metric else "see_metric")})
    return result


def _paired_differences(measured: list[dict], seed: int, *, repetitions: int = 2000):
    """Paired method deltas evaluated only on exactly matched case IDs.

    Deltas are descriptive. Positive/negative values do NOT imply significance.
    """
    indexed = defaultdict(dict)
    for item in measured:
        indexed[(item["task"], item["metric"], item["method"])][item["sample_id"]] = item
    pairs = []
    tasks = sorted({(r["task"], r["metric"]) for r in measured})
    for task, metric in tasks:
        methods = sorted({k[2] for k in indexed if k[:2] == (task, metric)})
        for pos, left in enumerate(methods):
            for right in methods[pos + 1:]:
                first, second = indexed[(task, metric, left)], indexed[(task, metric, right)]
                shared = sorted(first.keys() & second.keys())
                by_building = defaultdict(list)
                for case in shared:
                    if first[case]["building_key"] != second[case]["building_key"]:
                        raise ValueError("Paired benchmark building mismatch")
                    by_building[first[case]["site_id"]].append(second[case]["value"] - first[case]["value"])
                values = [mean(x) for x in by_building.values()]
                if not values:
                    continue
                ci = [None, None]
                if len(values) >= 2:
                    rng = random.Random(f"{seed}:{task}:{metric}:{left}:{right}")
                    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
                    ci = [draws[int(repetitions * 0.025)], draws[min(repetitions - 1, int(repetitions * 0.975))]]
                pairs.append({"task": task, "metric": metric, "reference_method": left,
                              "candidate_method": right, "candidate_minus_reference": mean(values),
                              "ci95_low": ci[0], "ci95_high": ci[1],
                              "n_paired_cases": len(shared), "n_sites": len(values),
                              "n_buildings": len({first[c]["building_key"] for c in shared}),
                              "cohort_ids": shared, "claim_approved": False})
    return pairs


def _load_ifc_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rows.append({"global_id": obj.get("global_id") or obj.get("GlobalId"),
                             "ifc_class": obj.get("ifc_class") or obj.get("type")})
    return rows


def evaluate_one(task, root, gt, pred, *, tau_m, max_points, seed, heldout_anchor_ids=None):
    a, b = TASK_INPUTS[task]
    if (a is not None and a not in gt) or b not in pred:
        return "not_executed", "missing_ground_truth_or_prediction_path", None, []
    truth = resolve_input(root, gt[a]) if a is not None else None
    prediction = resolve_input(root, pred[b])
    if (truth is not None and not truth.is_file()) or not prediction.is_file():
        return "not_executed", "missing_ground_truth_or_prediction_file", None, []
    inputs = [{"role": "prediction", "path": str(prediction), "sha256": sha256(prediction)}]
    if truth is not None:
        inputs.append({"role": "ground_truth", "path": str(truth), "sha256": sha256(truth)})
    if task == "performance":
        telem = json.loads(prediction.read_text(encoding="utf-8"))
        metrics = {}
        for target, alternatives in {
            "wall_time_sec": ("wall_time_sec", "elapsed_sec"),
            "peak_vram_mb": ("peak_vram_mb", "peak_gpu_memory_mb"),
            "peak_rss_mb": ("peak_rss_mb",),
        }.items():
            for label in alternatives:
                if label in telem and telem[label] is not None:
                    value = float(telem[label])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid telemetry value " + label)
                    metrics[target] = value
                    break
        if "wall_time_sec" not in metrics:
            return "not_executed", "missing_measured_wall_time", None, inputs
        metrics["run_failed"] = int(str(telem.get("status", "")).lower() in ("failed", "error"))
    elif task in ("reconstruction", "bim_geometry"):
        p, pn = read_ply_xyz(prediction, max_points=max_points, seed=seed)
        t, tn = read_ply_xyz(truth, max_points=max_points, seed=seed)
        metrics = reconstruction_metrics(p, t, tau_m=tau_m)
        metrics.update(n_pred_total=pn, n_truth_total=tn)
    elif task == "registration":
        predicted_anchors, truth_anchors = read_csv(prediction), read_csv(truth)
        if heldout_anchor_ids is not None:
            ids = set(heldout_anchor_ids)
            for rows in (predicted_anchors, truth_anchors):
                if not ids.issubset({r.get("anchor_id") for r in rows}):
                    raise ValueError("Missing held-out survey anchor correspondence")
            predicted_anchors = [r for r in predicted_anchors if r.get("anchor_id") in ids]
            truth_anchors = [r for r in truth_anchors if r.get("anchor_id") in ids]
        metrics = compare_anchors(predicted_anchors, truth_anchors)
    elif task == "ifc":
        rows = _load_ifc_jsonl(prediction) if prediction.suffix.lower() == ".jsonl" else read_csv(prediction)
        metrics = ifc_element_metrics(rows, read_csv(truth))
    elif task == "nvs":
        metrics = score_nvs_pairs(truth, prediction)
        inputs.extend(metrics.pop("image_hashes"))
    elif task == "activity_progress":
        metrics = activity_progress_metrics(load_scientific_progress_v2(prediction), read_csv(truth))
    elif task == "project_progress":
        metrics = project_progress_metrics(load_scientific_progress_v2(prediction), read_csv(truth))
    else:
        raw = read_csv(prediction)
        rows = load_stage9_v2(prediction) if any(r.get("scientific_metrics_schema") == "mycon.element_progress.v2" for r in raw) else raw
        metrics = progress_metrics(rows, read_csv(truth))
    return "evaluated", None, metrics, inputs


def _export_tables(out: Path, cases: list[dict], summary: list[dict], paired: list[dict]):
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "task", "metric", "mean", "ci95_low", "ci95_high", "n_sites", "n_buildings", "n_cases", "unit"])
        w.writeheader()
        w.writerows(summary)
    lines = ["# MyCON benchmark — research tables", "",
             "Descriptive method results below may have UNMATCHED cohorts: do NOT compare rows directly.",
             "Empty CI = fewer than two independent buildings. Synthetic tests are excluded.", "",
             "| Task | Method | Metric | Mean | 95% site-cluster CI | Sites |", "|---|---|---|---:|---|---:|"]
    for r in summary:
        ci = "—" if r["ci95_low"] is None else f"[{r['ci95_low']:.4g}, {r['ci95_high']:.4g}]"
        lines.append(f"| {r['task']} | {r['method']} | {r['metric']} | {r['mean']:.4g} | {ci} | {r['n_sites']} |")
    lines.extend(["", "## Matched-cohort pairwise differences (candidate minus reference; no superiority inferred)", "",
                  "| Task | Metric | Reference | Candidate | Delta | Site cluster CI 95% | Paired cases | Sites |",
                  "|---|---|---|---|---:|---|---:|---:|"])
    for p in paired:
        ci = "—" if p["ci95_low"] is None else f"[{p['ci95_low']:.4g}, {p['ci95_high']:.4g}]"
        lines.append(f"| {p['task']} | {p['metric']} | {p['reference_method']} | {p['candidate_method']} | {p['candidate_minus_reference']:.4g} | {ci} | {p['n_paired_cases']} | {p['n_sites']} |")
    # Export LaTeX rows for manuscript inclusion. Absent CIs are explicit NA.
    def tex(s):
        return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    latex = [r"\begin{tabular}{llllrr}", r"\toprule",
             r"Task & Method & Metric & Mean & CI 95\% & Sites \\", r"\midrule"]
    for r in summary:
        ci = "NA" if r["ci95_low"] is None else f"[{r['ci95_low']:.3g}, {r['ci95_high']:.3g}]"
        latex.append(" & ".join([tex(r["task"]), tex(r["method"]), tex(r["metric"]),
                                 f"{r['mean']:.4g}", tex(ci), str(r["n_sites"])]) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    (out / "thesis_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    lines.extend(["", "## Missing experiments / failures", "",
                  "| Sample | Method | Task | Status | Reason |", "|---|---|---|---|---|"])
    for case in cases:
        if case["status"] != "evaluated":
            lines.append(f"| {case['sample_id']} | {case['method']} | {case['task']} | {case['status']} | {case.get('reason', '')} |")
    lines.extend(["", "## Interpretation", "",
                  "Do not claim a method outperforms another unless both methods are evaluated on exactly the same independent test cases and ground truth. Report denominators, failure rate and hardware. No results from fixture data enter thesis tables."])
    (out / "thesis_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_anchor_partition(sample):
    part = sample.get("anchor_partition")
    if not isinstance(part, dict):
        return False
    fit = part.get("fit_anchor_ids")
    heldout = part.get("heldout_tre_anchor_ids")
    if not isinstance(fit, list) or not isinstance(heldout, list):
        return False
    if not fit or len(heldout) < 2:
        return False
    if any(not isinstance(x, str) or not x.strip() for x in fit + heldout):
        return False
    return len(fit) == len(set(fit)) and len(heldout) == len(set(heldout)) and not set(fit) & set(heldout)


def _verified_gt_digest(root, sample, gt_key):
    """Require precommitted independent-GT hash; actual runtime hashing alone is insufficient."""
    truth_map = sample.get("ground_truth") or {}
    expected = (sample.get("ground_truth_sha256") or {}).get(gt_key)
    if not isinstance(expected, str) or len(expected) != 64 or gt_key not in truth_map:
        return False
    if any(c not in "0123456789abcdef" for c in expected.lower()):
        return False
    try:
        path = resolve_input(root, truth_map[gt_key])
        return path.is_file() and sha256(path).lower() == expected.lower()
    except (OSError, ValueError, TypeError):
        return False


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _git_head():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=False, timeout=5,
                                cwd=Path(__file__).resolve().parents[2])
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def run(manifest_path: Path, output_root: Path, *, run_id: str, include_fixtures=False, seed=42,
        tau_m=0.05, max_points=20000):
    if not run_id.startswith("bench-") or any(x in run_id for x in ("/", "\\", "..")):
        raise ValueError("run-id must start with bench- and be a single path component")
    manifest_path = manifest_path.resolve()
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(doc)
    root = manifest_path.parent
    # No benchmark artifact can be stored inside the input dataset directory.
    output_root = output_root.resolve()
    if output_root == root or output_root.is_relative_to(root):
        raise ValueError("Benchmark outputs must be outside input dataset root")
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / run_id
    out.mkdir(exist_ok=False)  # never overwrite old experiment results
    started = time.perf_counter()
    cases, measured = [], []
    provenance = [{"role": "manifest", "path": str(manifest_path), "sha256": sha256(manifest_path)}]
    for sample in doc["samples"]:
        for method, pred in sorted(sample["methods"].items()):
            for task in TASK_INPUTS:
                task_started = time.perf_counter()
                result = {"data_kind": sample.get("data_kind", "unverified"), "sample_id": sample["id"], "site_id": sample["site_id"],
                          "building_id": sample["building_id"], "split": sample["split"],
                          "method": method, "task": task, "status": "not_executed"}
                if sample["split"] != "test":
                    result["reason"] = "not_test_split"
                elif sample.get("data_kind") != "real" and not include_fixtures:
                    result["reason"] = "synthetic_or_unverified_fixture_excluded"
                elif sample.get("data_kind") == "real" and not (
                    sample.get("gt_independently_verified") is True
                    and isinstance(sample.get("gt_review"), dict)
                    and sample["gt_review"].get("reviewer")
                    and sample["gt_review"].get("review_date_utc")
                    and sample.get("source_path")
                ):
                    result["reason"] = "real_sample_missing_independent_gt_provenance"
                elif sample.get("data_kind") == "real" and not (
                    resolve_input(root, sample["source_path"]).is_file()
                    and sha256(resolve_input(root, sample["source_path"])) == sample["source_sha256"]
                ):
                    result["reason"] = "capture_missing_or_sha256_mismatch"
                elif (sample.get("data_kind") == "real" and TASK_INPUTS[task][0] is not None
                      and not _verified_gt_digest(root, sample, TASK_INPUTS[task][0])):
                    result["reason"] = "missing_or_mismatched_independent_gt_sha256"
                elif task == "registration" and sample.get("data_kind") == "real" and not _valid_anchor_partition(sample):
                    result["reason"] = "missing_disjoint_heldout_anchor_partition"
                elif not isinstance(pred, dict):
                    result["reason"] = "invalid_prediction_manifest"
                else:
                    try:
                        status, reason, metrics, inputs = evaluate_one(
                            task, root, sample.get("ground_truth", {}), pred,
                            tau_m=tau_m, max_points=max_points, seed=seed,
                            heldout_anchor_ids=(sample["anchor_partition"]["heldout_tre_anchor_ids"]
                                                if task == "registration" and sample.get("data_kind") == "real"
                                                else None))
                        result.update(status=status, reason=reason, metrics=metrics)
                        provenance.extend(inputs)
                        # Explicitly forbid fixture outputs from joining scientific results.
                        if status == "evaluated" and sample.get("data_kind") == "real":
                            for metric, value in metrics.items():
                                if isinstance(value, (float, int)) and not isinstance(value, bool):
                                    measured.append({"method": method, "task": task, "metric": metric,
                                                     "value": value, "sample_id": sample["id"],
                                                     "building_key": (sample["site_id"], sample["building_id"]),
                                                     "site_id": sample["site_id"]})
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError, csv.Error) as e:
                        result.update(status="failed", reason=f"{type(e).__name__}: {e}")
                result["evaluation_wall_time_sec"] = time.perf_counter() - task_started
                result["exit_code"] = (0 if result["status"] == "evaluated" else
                                       1 if result["status"] == "failed" else None)
                cases.append(result)
    summary = _summary(measured, seed)
    paired = _paired_differences(measured, seed)
    # Failure/availability denominators are all eligible verified-real test cases.
    eligible = [c for c in cases if c["data_kind"] == "real" and c["split"] == "test" and c.get("reason") not in {
        "real_sample_missing_independent_gt_provenance", "capture_missing_or_sha256_mismatch",
        "synthetic_or_unverified_fixture_excluded"}]
    failure_summary = []
    buckets = defaultdict(list)
    for c in eligible:
        buckets[(c["method"], c["task"])].append(c)
    for (method, task), bucket in sorted(buckets.items()):
        attempted = sum(c["status"] in ("failed", "evaluated") for c in bucket)
        failed = sum(c["status"] == "failed" for c in bucket)
        failure_summary.append({"method": method, "task": task, "n_planned": len(bucket),
                                "n_attempted": attempted, "n_not_executed": len(bucket) - attempted,
                                "n_failed": failed, "failure_rate_attempted": failed / attempted if attempted else None,
                                "execution_coverage": attempted / len(bucket)})
    metadata = {
        "schema_version": "mycon.benchmark_result.v1", "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest_path), "dataset_split": "test",
        "seed": seed, "tau_m": tau_m, "max_points": max_points,
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "gpu": "not_measured",
        "model_versions": doc.get("model_versions", {}),
        "hardware_declaration": doc.get("hardware", {}),
        "software_declaration": doc.get("software_versions", {}),
        "n_test_samples": sum(s["split"] == "test" for s in doc["samples"]),
        "statistical_unit": "site", "ci_method": "percentile_site_cluster_bootstrap_2000",
        "benchmark_evaluation_wall_time_sec": time.perf_counter() - started,
        "host_cpu_count": os.cpu_count(), "git_commit_sha": _git_head(),
        "numpy_version": _pkg_version("numpy"), "scipy_version": _pkg_version("scipy"),
        "pillow_version": _pkg_version("pillow"), "scikit_image_version": _pkg_version("scikit-image"),
        "scientific_status": "evaluated" if summary else "not_executed",
        "note": "fixture results are tool-only and never included in scientific comparison",
    }
    (out / "failure_summary.json").write_text(json.dumps(failure_summary, indent=2) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "paired_comparisons.json").write_text(json.dumps(paired, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "scientific_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "input_hashes.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    _export_tables(out, cases, summary, paired)
    return {"output": str(out), "scientific_status": metadata["scientific_status"],
            "n_cases": len(cases), "n_scientific_metric_cells": len(summary)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent, read-only MyCON v1 benchmark")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-m", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--include-fixtures", action="store_true", help="tool self-test only; never scientific")
    args = parser.parse_args(argv)
    ident = args.run_id or "bench-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    info = run(args.manifest, args.output_root, run_id=ident,
               include_fixtures=args.include_fixtures, seed=args.seed,
               tau_m=args.tau_m, max_points=args.max_points)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
