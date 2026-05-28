#!/usr/bin/env python3
"""Drive an ablation study over a Cartesian grid of config overrides.

Composition:

- :mod:`pipeline.common.ablation` enumerates the grid and produces overlays.
- This script writes one overlay YAML per cell to a working directory, runs
  the configured stage(s) via ``scripts/run_stage.py`` for each cell, then
  invokes ``scripts/aggregate_run_metrics.py`` per cell, and finally writes a
  single ``ablation_summary.csv`` with one row per cell.

The script intentionally has no ICP / Open3D / COLMAP imports of its own;
those are the responsibility of the per-stage runners. The runner is
side-effect-isolated to a working directory passed via ``--out-dir``.

Heavy stages (``stage_05_dense``, full ``stage_08_bim_registration``) can take
many minutes per cell on real data. Use ``--dry-run`` to preview which cells
would be executed and the resulting commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.ablation import AblationGrid, apply_overlay, build_grid, grid_summary


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML file does not parse to a mapping: {path}")
    return data


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"[ablation] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _aggregate_for_cell(reports_dir: Path) -> dict[str, Any]:
    """Run the per-stage aggregator for a cell and read its JSON back."""
    out_json = reports_dir / "run_metrics.json"
    out_csv = reports_dir / "run_metrics.csv"
    rc = _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "aggregate_run_metrics.py"),
         "--reports-dir", str(reports_dir),
         "--out-json", str(out_json),
         "--out-csv", str(out_csv)],
        cwd=PROJECT_ROOT,
    )
    if rc != 0 or not out_json.exists():
        return {"aggregator_status": "failed", "rc": rc}
    try:
        return {"aggregator_status": "ok", "records": json.loads(out_json.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"aggregator_status": f"unreadable:{exc}"}


def run_grid(
    base_cfg_path: Path,
    grid_yaml_path: Path,
    out_dir: Path,
    stage: str,
    *,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> dict[str, Any]:
    base_cfg = _load_yaml(base_cfg_path)
    grid_spec = _load_yaml(grid_yaml_path)
    name = str(grid_spec.get("name") or grid_yaml_path.stem)
    axes = grid_spec.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("grid YAML must contain a top-level 'axes' mapping")

    grid: AblationGrid = build_grid(name, axes)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ablation_summary.json"
    summary_csv = out_dir / "ablation_summary.csv"

    rows: list[dict[str, Any]] = []
    for cell in grid.cells:
        cell_dir = out_dir / cell.name
        cell_dir.mkdir(parents=True, exist_ok=True)
        cell_cfg = apply_overlay(base_cfg, cell.overlay)
        cell_cfg_path = cell_dir / "config.yaml"
        cell_cfg_path.write_text(yaml.safe_dump(cell_cfg, sort_keys=False), encoding="utf-8")

        # Each cell uses its own run_id-derived reports dir to avoid clobbering.
        # If the user has not pinned project.run_id in the overlay, we suffix
        # the cell name to whatever run_id the base config carries.
        original_run_id = (((cell_cfg.get("project") or {}) or {}).get("run_id") or "ablation_run")
        cell_run_id = f"{original_run_id}__{cell.name}"
        cell_cfg.setdefault("project", {})["run_id"] = cell_run_id
        cell_cfg_path.write_text(yaml.safe_dump(cell_cfg, sort_keys=False), encoding="utf-8")

        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_stage.py"),
            stage, "--config", str(cell_cfg_path), "--log-level", log_level, "--force",
        ]
        row: dict[str, Any] = {"cell": cell.name, "label": cell.short_label, **{f"axis.{k}": v for k, v in cell.overlay.items()}}

        if dry_run:
            row["status"] = "dry_run"
            row["command"] = " ".join(cmd)
            rows.append(row)
            continue

        rc = _run(cmd, cwd=PROJECT_ROOT)
        row["stage_rc"] = rc

        # Aggregate this cell's reports if the run produced any.
        reports_dir = PROJECT_ROOT / "runs" / cell_run_id / "reports"
        if reports_dir.exists():
            agg = _aggregate_for_cell(reports_dir)
            row["aggregator_status"] = agg.get("aggregator_status")
            # Surface a few cell-level signals onto the row.
            for rec in agg.get("records", []) or []:
                if rec.get("stage") == stage:
                    for key in (
                        "icp_fitness",
                        "icp_inlier_rmse_m",
                        "icp_robust_loss_applied",
                        "bidirectional_accuracy",
                        "bidirectional_completeness",
                        "bidirectional_f_score",
                    ):
                        if key in rec:
                            row[f"metric.{key}"] = rec[key]
        else:
            row["aggregator_status"] = "no_reports_dir"
        rows.append(row)

    summary = {
        "grid": grid_summary(grid),
        "stage": stage,
        "base_config": str(base_cfg_path),
        "out_dir": str(out_dir),
        "rows": rows,
        "dry_run": dry_run,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    if fieldnames:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an ablation grid over a single pipeline stage.")
    p.add_argument("--config", required=True, type=Path, help="Base YAML config")
    p.add_argument("--grid", required=True, type=Path, help="Ablation grid YAML (with 'name' and 'axes')")
    p.add_argument("--stage", required=True, help="Stage to run for each cell, e.g. stage_08_bim_registration")
    p.add_argument("--out-dir", required=True, type=Path, help="Per-cell working directory")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing stages")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_grid(
        base_cfg_path=args.config,
        grid_yaml_path=args.grid,
        out_dir=args.out_dir,
        stage=args.stage,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )
    grid = summary["grid"]
    print(
        f"ABLATION_RUN_OK grid={grid['name']} cells={grid['cell_count']} "
        f"dry_run={summary['dry_run']} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
