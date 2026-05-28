"""Optional 4D BIM schedule-aware filtering for Stage 8.

This module is intentionally permissive. If the schedule file or mapping columns
are missing, it keeps all IFC elements and reports why. That keeps Stage 8 robust
for early experiments while enabling thesis-grade as-planned registration later.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_access import cfg_bool, cfg_float, cfg_get, cfg_list, resolve_project_path


GLOBAL_ID_COLUMNS = ["global_id", "GlobalId", "ifc_global_id", "element_global_id", "ifc_guid", "guid"]
STATUS_COLUMNS = ["status", "planned_status", "state"]
START_DAY_COLUMNS = ["start_day", "planned_start_day", "start_project_day", "day_start"]
FINISH_DAY_COLUMNS = ["finish_day", "planned_finish_day", "end_day", "planned_end_day"]


@dataclass
class ScheduleFilter:
    enabled: bool
    keep_unmatched: bool = True
    current_project_day: float | None = None
    allowed_statuses: set[str] = field(default_factory=set)
    rules_by_global_id: dict[str, dict[str, str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    rows_loaded: int = 0
    matched_elements: int = 0
    skipped_elements: int = 0

    def allow(self, element: Any) -> bool:
        if not self.enabled:
            return True
        gid = str(getattr(element, "GlobalId", "") or "")
        row = self.rules_by_global_id.get(gid)
        if row is None:
            return self.keep_unmatched
        self.matched_elements += 1
        if self._row_is_due(row):
            return True
        self.skipped_elements += 1
        return False

    def _row_is_due(self, row: dict[str, str]) -> bool:
        status = _first_value(row, STATUS_COLUMNS).strip().lower()
        if status and status in self.allowed_statuses:
            return True
        if self.current_project_day is None:
            # If no current day is configured, status columns are the only active filter.
            return True if not status else status in self.allowed_statuses
        start_raw = _first_value(row, START_DAY_COLUMNS)
        if start_raw:
            try:
                return float(start_raw) <= float(self.current_project_day)
            except ValueError:
                self.warnings.append(f"Could not parse schedule start day {start_raw!r}; keeping row.")
                return True
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "keep_unmatched": self.keep_unmatched,
            "current_project_day": self.current_project_day,
            "allowed_statuses": sorted(self.allowed_statuses),
            "rows_loaded": self.rows_loaded,
            "matched_elements": self.matched_elements,
            "skipped_elements": self.skipped_elements,
            "warnings": list(dict.fromkeys(self.warnings)),
        }


def _first_value(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name])
    return ""


def _resolve_schedule_path(cfg: Any) -> Path:
    configured = cfg_get(cfg, "bim.schedule_filter_csv", None)
    if configured:
        return resolve_project_path(cfg, "bim.schedule_filter_csv")
    return resolve_project_path(cfg, "inputs.schedule", "data/bim/design/schedule.csv")


def build_schedule_filter(cfg: Any) -> ScheduleFilter:
    enabled = cfg_bool(cfg, "bim.schedule_filter_enabled", False)
    filt = ScheduleFilter(
        enabled=enabled,
        keep_unmatched=cfg_bool(cfg, "bim.schedule_filter_keep_unmatched", True),
        current_project_day=None,
        allowed_statuses={str(x).strip().lower() for x in cfg_list(cfg, "bim.schedule_filter_allowed_statuses", ["done", "complete", "completed", "in_progress", "started", "active"])},
    )
    if not enabled:
        return filt
    day_value = cfg_get(cfg, "bim.current_project_day", None)
    if day_value is not None:
        try:
            filt.current_project_day = float(day_value)
        except (TypeError, ValueError):
            filt.warnings.append(f"Invalid bim.current_project_day={day_value!r}; schedule day filtering disabled.")
            filt.current_project_day = None
    path = _resolve_schedule_path(cfg)
    if not path.exists() or path.stat().st_size <= 0:
        filt.warnings.append(f"Schedule filter enabled but schedule CSV is missing or empty: {path}. Keeping all elements.")
        return filt
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gid = _first_value(row, GLOBAL_ID_COLUMNS).strip()
                if not gid:
                    continue
                filt.rules_by_global_id[gid] = {str(k): str(v) for k, v in row.items() if k is not None and v is not None}
                filt.rows_loaded += 1
    except Exception as exc:  # noqa: BLE001
        filt.warnings.append(f"Could not read schedule filter CSV {path}: {exc}. Keeping all elements.")
    if filt.rows_loaded == 0:
        filt.warnings.append("Schedule filter enabled but no GlobalId mapping rows were found. Keeping all elements.")
    return filt
