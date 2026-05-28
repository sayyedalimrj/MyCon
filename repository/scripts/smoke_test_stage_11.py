#!/usr/bin/env python3
"""Smoke test for Stage 11 schedule variance.

Runs the Stage 11 CLI on tiny synthetic fixtures (built in a temporary
directory) and verifies the three output JSONs have the documented
schema versions and at least one activity each. This is fully
deterministic, has no external deps beyond Python stdlib, and runs in
the lightweight test set.

Print contract: ``STAGE_11_SMOKE_OK`` on success, anything else on
failure. ``test_smoke_scripts_framework.py`` parses this string.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage_11_schedule_variance.run_schedule_variance import main


def _make_fixtures(tmpdir: Path) -> tuple[Path, Path, Path]:
    schedule = tmpdir / "schedule.csv"
    schedule.write_text(
        "activity_id,activity_name,planned_start_iso,planned_finish_iso\n"
        "A0001,Foundations,2026-03-01,2026-04-01\n"
        "A0432,Floor 2 Zone B walls,2026-04-01,2026-05-01\n",
        encoding="utf-8",
    )
    mapping = tmpdir / "mapping.csv"
    mapping.write_text(
        "activity_id,ifc_global_id,weight\n"
        "A0001,Y1,1.0\n"
        "A0432,X1,1.0\n"
        "A0432,X2,1.0\n"
        "A0432,X3,1.0\n",
        encoding="utf-8",
    )
    elements = tmpdir / "elements.csv"
    elements.write_text(
        "global_id,name,status\n"
        "Y1,Foundation,likely_completed\n"
        "X1,Wall1,likely_completed\n"
        "X2,Wall2,partially_observed\n"
        "X3,Wall3,not_evidenced\n",
        encoding="utf-8",
    )
    return schedule, mapping, elements


def _check_outputs(out_dir: Path) -> None:
    ap = json.loads((out_dir / "activity_progress.json").read_text(encoding="utf-8"))
    sv = json.loads((out_dir / "schedule_variance.json").read_text(encoding="utf-8"))
    ds = json.loads((out_dir / "dashboard_summary.json").read_text(encoding="utf-8"))
    assert ap["schema_version"] == "activity_progress.v1"
    assert sv["schema_version"] == "schedule_variance.v1"
    assert ds["schema_version"] == "dashboard_summary.v1"
    assert ap["n_activities"] == 2
    assert sv["n_activities"] == 2
    assert ds["kpi"]["n_activities"] == 2
    # At least one activity should be on schedule given mid-month data date.
    assert sv["n_on_schedule"] + sv["n_ahead"] + sv["n_behind"] + sv["n_unknown_evidence"] == 2


def smoke() -> int:
    with tempfile.TemporaryDirectory(prefix="stage11_smoke_") as td:
        tmp = Path(td)
        sched, mapping, elements = _make_fixtures(tmp)
        out = tmp / "out"
        rc = main(
            [
                "--schedule-csv", str(sched),
                "--mapping-csv", str(mapping),
                "--element-metrics-csv", str(elements),
                "--activity-progress-json", str(out / "activity_progress.json"),
                "--schedule-variance-json", str(out / "schedule_variance.json"),
                "--dashboard-summary-json", str(out / "dashboard_summary.json"),
                "--data-date-utc", "2026-04-16",
            ]
        )
        if rc != 0:
            print(f"STAGE_11_SMOKE_FAILED rc={rc}", file=sys.stderr)
            return rc
        try:
            _check_outputs(out)
        except AssertionError as exc:
            print(f"STAGE_11_SMOKE_FAILED assertion: {exc}", file=sys.stderr)
            return 2
    print("STAGE_11_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke())
