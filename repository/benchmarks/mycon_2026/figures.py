"""Non-fabricated publication figure suite; refuses synthetic scientific claims."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .plots import render

PLOTS = (
    ("reconstruction", "reconstruction", "fscore_at_0.05m"),
    ("bim_registration", "registration", "tre_rmse_m"),
    ("physical_progress", "progress", "progress_mae_fraction_known"),
    ("activity_progress", "activity_progress", "physical_progress_mae_pp"),
    ("heldout_nvs", "nvs", "psnr_db_mean_finite"),
)


def render_suite(result_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = result_dir.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    verified = metadata.get("scientific_status") == "evaluated"
    status = {}
    for name, task, metric in PLOTS:
        if not verified:
            status[name] = {"status": "not_executed", "reason": "no_real_verified_benchmark"}
            continue
        try:
            paths = render(root, task=task, metric=metric)
            status[name] = {"status": "rendered", "files": paths}
        except ValueError as exc:
            status[name] = {"status": "not_executed", "reason": str(exc)}
    # Reliability points are descriptive across eligible real cases; the
    # scientific CI remains site-clustered, not a per-element binomial CI.
    if verified:
        cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for case in cases:
            if (case.get("data_kind") != "real" or case.get("split") != "test"
                    or case.get("status") != "evaluated" or case.get("task") != "progress"):
                continue
            for point in (case.get("metrics") or {}).get("reliability_bins_selective", []):
                if point.get("n", 0) > 0:
                    grouped[point["bin"]].append(point)
        if grouped:
            xs, ys, ns = [], [], []
            for key, entries in sorted(grouped.items()):
                n = sum(v["n"] for v in entries)
                xs.append(sum(v["confidence"] * v["n"] for v in entries) / n)
                ys.append(sum(v["accuracy"] * v["n"] for v in entries) / n)
                ns.append(n)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", label="ideal calibration")
            ax.scatter(xs, ys, label="measured bins")
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(n), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Reported confidence",
                   ylabel="Observed correct-label frequency",
                   title="Selective classification reliability (descriptive)")
            ax.legend()
            paths = _save(fig, root, "figure_calibration")
            status["calibration"] = {"status": "rendered", "files": paths,
                                     "n_classification_confidence_pairs": sum(ns)}
        else:
            status["calibration"] = {"status": "not_executed", "reason": "no_calibrated_real_predictions"}
        risk = defaultdict(list)
        for case in cases:
            if (case.get("data_kind") == "real" and case.get("split") == "test"
                    and case.get("status") == "evaluated" and case.get("task") == "progress"):
                for pt in (case.get("metrics") or {}).get("risk_coverage_selective", []):
                    if pt.get("n", 0) and pt.get("error_rate") is not None:
                        risk[pt["threshold"]].append(pt)
        if risk:
            curve = []
            for threshold, pts in sorted(risk.items()):
                n = sum(pt["n"] for pt in pts)
                curve.append((sum(pt["coverage"] * pt["n"] for pt in pts) / n,
                              sum(pt["error_rate"] * pt["n"] for pt in pts) / n))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([p[0] for p in curve], [p[1] for p in curve], marker="o")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Selective prediction coverage",
                   ylabel="Error among retained classifications", title="Selective risk versus coverage (descriptive)")
            status["selective_risk"] = {"status": "rendered", "files": _save(fig, root, "figure_selective_risk")}
        else:
            status["selective_risk"] = {"status": "not_executed", "reason": "no_calibrated_real_predictions"}
        failures = json.loads((root / "failure_summary.json").read_text(encoding="utf-8"))
        eligible = [x for x in failures if x.get("n_attempted", 0) > 0]
        if eligible:
            fig, ax = plt.subplots(figsize=(max(6, len(eligible) * .8), 4.5))
            ax.bar(range(len(eligible)), [x["failure_rate_attempted"] for x in eligible])
            ax.set_xticks(range(len(eligible)),
                          [x["method"] + "/" + x["task"] for x in eligible],
                          rotation=45, ha="right")
            ax.set(ylabel="Failed / attempted", ylim=(0, 1),
                   title="Real-case failure rate (unattempted cases disclosed separately)")
            paths = _save(fig, root, "figure_failure_rate")
            status["failure_rate"] = {"status": "rendered", "files": paths}
        else:
            status["failure_rate"] = {"status": "not_executed", "reason": "no_attempted_verified_real_cases"}
    else:
        status["calibration"] = {"status": "not_executed", "reason": "no_real_verified_benchmark"}
        status["failure_rate"] = {"status": "not_executed", "reason": "no_real_verified_benchmark"}
        status["selective_risk"] = {"status": "not_executed", "reason": "no_real_verified_benchmark"}
    output = root / "figures_status.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "mycon.figures.v1", "scientific_status": metadata.get("scientific_status"),
                   "figures": status}, stream, indent=2, allow_nan=False)
    return status


def _save(fig, root, stem):
    import matplotlib.pyplot as plt
    files = []
    for ext in ("svg", "png"):
        path = root / (stem + "." + ext)
        if path.exists():
            raise FileExistsError("Refusing to overwrite existing figure")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        files.append(str(path))
    plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_suite(args.result_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
