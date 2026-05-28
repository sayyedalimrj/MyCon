#!/usr/bin/env python3
"""End-to-end walkthrough runner for the Phase 4 + Phase 5 finishing layer.

Invokes every dependency-free Phase 4 module on the synthetic fixture
in :file:`examples/end_to_end/inputs/`, and writes a single
``walkthrough_summary.json`` under ``--output-dir`` that links every
output by filename + sha-256. The runner is the canonical entry point
for thesis-defence reproducibility runs.

Modules exercised (in the documented order)
-------------------------------------------

1. ``pipeline.common.schedule_io.load_schedule_csv``
2. ``pipeline.common.bim_schedule_mapping.load_mapping_csv``
3. ``pipeline.stage_11_schedule_variance.run_schedule_variance.main``
4. ``pipeline.common.hitl.CorrectionStore``
5. ``pipeline.common.hitl.build_calibration_records``
6. ``pipeline.common.calibration.calibration_report``
7. ``pipeline.stage_10_copilot.grounding_guard.ground_answer``

CLI
---

::

    python3 scripts/run_end_to_end_walkthrough.py \\
        --output-dir runs/example_walkthrough/ \\
        --data-date-utc 2026-04-16 \\
        [--inputs-dir examples/end_to_end/inputs/]

Print contract
--------------

On success the script prints exactly one line beginning with
``WALKTHROUGH_OK`` so test harnesses can match deterministically.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common import calibration  # noqa: E402
from pipeline.common.hitl import CorrectionStore, build_calibration_records  # noqa: E402
from pipeline.stage_10_copilot.grounding_guard import ground_answer  # noqa: E402
from pipeline.stage_11_schedule_variance.run_schedule_variance import (  # noqa: E402
    main as stage11_main,
)


# ---------------------------------------------------------------------------
# Default fixture location (overridable via --inputs-dir)
# ---------------------------------------------------------------------------

_DEFAULT_INPUTS = PROJECT_ROOT / "examples" / "end_to_end" / "inputs"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Step runners — each returns the path it wrote, plus a short status dict.
# ---------------------------------------------------------------------------


def _run_stage_11(
    *,
    inputs_dir: Path,
    output_dir: Path,
    data_date_iso: str,
) -> dict[str, Any]:
    """Step 1 — Stage 11 schedule variance."""
    schedule_csv = inputs_dir / "schedule.csv"
    mapping_csv = inputs_dir / "bim_schedule_mapping.csv"
    element_metrics_csv = inputs_dir / "element_metrics.csv"
    activity_json = output_dir / "activity_progress.json"
    variance_json = output_dir / "schedule_variance.json"
    dashboard_json = output_dir / "dashboard_summary.json"
    rc = stage11_main(
        [
            "--schedule-csv", str(schedule_csv),
            "--mapping-csv", str(mapping_csv),
            "--element-metrics-csv", str(element_metrics_csv),
            "--activity-progress-json", str(activity_json),
            "--schedule-variance-json", str(variance_json),
            "--dashboard-summary-json", str(dashboard_json),
            "--data-date-utc", data_date_iso,
        ]
    )
    if rc != 0:
        raise RuntimeError(f"stage_11 returned non-zero exit code: {rc}")
    return {
        "step": "stage_11_schedule_variance",
        "outputs": {
            "activity_progress_json": str(activity_json),
            "schedule_variance_json": str(variance_json),
            "dashboard_summary_json": str(dashboard_json),
        },
    }


def _run_calibration(
    *,
    inputs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Steps 2–4 — HITL replay → calibration report."""
    corrections_path = inputs_dir / "hitl_corrections.jsonl"
    if not corrections_path.exists():
        raise FileNotFoundError(
            f"walkthrough fixture is missing the HITL log: {corrections_path}"
        )
    store = CorrectionStore(corrections_path)
    replay = store.replay(target_kinds=["element_acceptance"])
    cal_records = build_calibration_records(replay)
    report = calibration.calibration_report(
        cal_records,
        n_bins=5,
        strategy="equal_mass",
    )
    report["walkthrough_provenance"] = {
        "input_path": str(corrections_path.resolve()),
        "input_sha256": _sha256_file(corrections_path),
        "n_replayed_records": replay.n_total_records,
        "n_effective_records": len(replay.effective),
        "n_conflicts": len(replay.conflicts),
    }
    out_path = output_dir / "calibration_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "step": "hitl_replay_and_calibration",
        "outputs": {"calibration_report_json": str(out_path)},
        "metrics": report["metrics"],
    }


# ---------------------------------------------------------------------------
# Demonstration VLM answers covering the three claim kinds. The runner
# does *not* call a real VLM; it shows the grounding-guard verdict on
# pre-canned answers so the dashboard can render the failure modes.
# ---------------------------------------------------------------------------

def _build_demo_vlm_answers(actual_percent: float) -> tuple[tuple[str, str], ...]:
    """Build the three demo answers from the dashboard's actual percent.

    The well-grounded answer cites the dashboard's actual percent
    *exactly* (rounded to one decimal) so it sits inside the
    grounding-guard's per-quantity tolerance for ``completion``. The
    hallucinated answer fails on a numeric claim; the
    unsupported-entity answer fails on a named-entity claim. Together
    the three answers cover the three failure modes the guard is
    designed to catch.
    """
    actual_pct_int = round(actual_percent)
    return (
        (
            "well_grounded",
            f"Element IfcWall has {actual_pct_int}% completion. Accept.",
        ),
        (
            "hallucinated_numeric",
            "Element IfcWall has 99% completion. Approve.",
        ),
        (
            "unsupported_named_entity",
            "Activity A0432 is behind schedule. Reject.",
        ),
    )


def _run_grounding_guard(
    *,
    output_dir: Path,
    dashboard_json: Path,
) -> dict[str, Any]:
    """Step 5 — VLM grounding-guard demonstration."""
    if not dashboard_json.exists():
        raise FileNotFoundError(
            f"dashboard summary is missing; Stage 11 must run first: {dashboard_json}"
        )
    dashboard = json.loads(dashboard_json.read_text(encoding="utf-8"))
    actual_percent = dashboard["activities"][1]["actual_percent_complete"]
    # Build a tiny evidence package mimicking what Stage 10 would assemble.
    evidence = {
        "metrics": {"completion": actual_percent / 100.0},
        "confidence_flags": ["evidence_package_complete_enough_for_mock_answer"],
        "selected_context": {"element_global_id": "IfcWall"},
    }
    demo_answers = _build_demo_vlm_answers(actual_percent)
    results: list[dict[str, Any]] = []
    for label, answer in demo_answers:
        verdict = ground_answer(answer, evidence)
        results.append(
            {
                "label": label,
                "answer": answer,
                "passed": verdict.passed,
                "n_claims": verdict.n_claims,
                "n_matched": verdict.n_matched,
                "n_unsupported": verdict.n_unsupported,
                "n_unverifiable": verdict.n_unverifiable,
                "risk_tokens": list(verdict.risk_tokens),
            }
        )
    out_path = output_dir / "grounding_guard_demo.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "grounding_guard_demo.v1",
                "dashboard_input_sha256": _sha256_file(dashboard_json),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "step": "grounding_guard_demo",
        "outputs": {"grounding_guard_demo_json": str(out_path)},
        "n_passed": sum(1 for r in results if r["passed"]),
        "n_total": len(results),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory under which all walkthrough outputs are written.",
    )
    p.add_argument(
        "--inputs-dir",
        type=Path,
        default=_DEFAULT_INPUTS,
        help=f"Override input fixture directory (default: {_DEFAULT_INPUTS}).",
    )
    p.add_argument(
        "--data-date-utc",
        default="2026-04-16",
        help="ISO date for Stage 11 (default: %(default)s).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    inputs_dir: Path = args.inputs_dir.resolve()
    if not inputs_dir.is_dir():
        print(f"WALKTHROUGH_FAILED: inputs_dir not found: {inputs_dir}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    try:
        steps.append(
            _run_stage_11(
                inputs_dir=inputs_dir,
                output_dir=output_dir,
                data_date_iso=args.data_date_utc,
            )
        )
        steps.append(_run_calibration(inputs_dir=inputs_dir, output_dir=output_dir))
        steps.append(
            _run_grounding_guard(
                output_dir=output_dir,
                dashboard_json=output_dir / "dashboard_summary.json",
            )
        )
    except Exception as exc:  # pragma: no cover - failure-path message only
        print(f"WALKTHROUGH_FAILED: {exc!r}", file=sys.stderr)
        return 2

    # Final summary linking every output by filename + sha-256.
    file_index: dict[str, dict[str, Any]] = {}
    for step in steps:
        for tag, path_str in step.get("outputs", {}).items():
            path = Path(path_str)
            if path.exists():
                file_index[tag] = {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }

    summary = {
        "schema_version": "walkthrough_summary.v1",
        "generated_at_utc": _utc_iso_now(),
        "data_date_utc": args.data_date_utc,
        "inputs_dir": str(inputs_dir),
        "output_dir": str(output_dir),
        "steps": steps,
        "files": file_index,
    }
    summary_path = output_dir / "walkthrough_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "WALKTHROUGH_OK "
        f"output_dir={output_dir} "
        f"steps={len(steps)} "
        f"files={len(file_index)} "
        f"summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
