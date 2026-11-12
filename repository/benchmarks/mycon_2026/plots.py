"""Optional publication figures (PNG 300 dpi + vector SVG), from verified results only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(result_dir: Path, *, metric="fscore_at_tau", task="reconstruction") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = result_dir.resolve()
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("scientific_status") != "evaluated":
        raise ValueError("No verified real benchmark to plot; fixture results prohibited")
    rows = json.loads((result_dir / "scientific_results.json").read_text(encoding="utf-8"))
    relevant = [r for r in rows if r["task"] == task and r["metric"] == metric]
    if not relevant:
        raise ValueError("Metric unavailable in verified real results")
    labels, values = [r["method"] for r in relevant], [r["mean"] for r in relevant]
    bounds = [(r["ci95_low"], r["ci95_high"]) for r in relevant]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4.5))
    bars = ax.bar(range(len(labels)), values, edgecolor="black", linewidth=0.8)
    for i, (val, (lo, hi), r) in enumerate(zip(values, bounds, relevant)):
        if lo is not None and hi is not None:
            ax.errorbar(i, val, yerr=[[max(0, val - lo)], [max(0, hi - val)]],
                        color="black", fmt="none", capsize=3)
        ax.annotate(f"n={r.get('n_sites', 0)} independent sites / {r['n_buildings']} buildings", (i, val),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{task}: independent held-out test (descriptive; check cohort pairing)")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    stem = "figure_" + "".join(c if c.isalnum() or c == "_" else "_" for c in (task + "_" + metric))
    outputs = []
    for ext in ("svg", "png"):
        dest = result_dir / (stem + "." + ext)
        if dest.exists():
            raise FileExistsError(f"Refuse to overwrite existing figure: {dest}")
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        outputs.append(str(dest))
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--task", default="reconstruction")
    parser.add_argument("--metric", default="fscore_at_tau")
    x = parser.parse_args()
    print("\n".join(render(x.result_dir, metric=x.metric, task=x.task)))


if __name__ == "__main__":
    main()
