from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .metric_alignment import build_metric_alignment_report


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg_get(cfg, "project.root", "."))).expanduser().resolve()


def resolve_path(cfg: dict[str, Any], dotted: str, default: str) -> Path:
    raw = cfg_get(cfg, dotted, default)
    p = Path(str(raw))
    return p if p.is_absolute() else project_root(cfg) / p


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate metric scan-to-BIM Sim3 alignment from anchors and known distances.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"MISSING_CONFIG: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"INVALID_CONFIG: {cfg_path}")

    anchors_csv = resolve_path(cfg, "metric_alignment.metric_anchors_csv", "data/bim/design/metric_anchors.csv")
    known_distances_csv = resolve_path(cfg, "metric_alignment.known_distances_csv", "data/bim/design/known_distances.csv")
    report_json = resolve_path(cfg, "metric_alignment.report_json", "runs/2026-04-30_site01_baseline/reports/metric_alignment_report.json")

    if report_json.exists() and not args.force:
        raise SystemExit(f"OUTPUT_EXISTS_USE_FORCE: {report_json}")

    report = build_metric_alignment_report(
        anchors_csv=anchors_csv,
        known_distances_csv=known_distances_csv,
        output_json=report_json,
        min_registration_anchors=int(cfg_get(cfg, "metric_alignment.min_registration_anchors", 3)),
        residual_warn_m=float(cfg_get(cfg, "metric_alignment.residual_warn_m", 0.05)),
        residual_fail_m=float(cfg_get(cfg, "metric_alignment.residual_fail_m", 0.15)),
    )

    print(
        "STAGE_08_METRIC_ALIGNMENT_OK "
        f"status={report.get('status')} "
        f"confidence={report.get('confidence')} "
        f"usable_anchors={report.get('usable_registration_anchor_count')} "
        f"report={report_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
