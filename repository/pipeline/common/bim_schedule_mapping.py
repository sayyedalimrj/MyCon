"""BIM <-> schedule activity mapping for the finishing layer.

We need to know **which IFC elements belong to which scheduled
activity**. This module loads, validates, and queries that mapping.

Two paths are supported, in priority order:

1. **Explicit mapping CSV** — the recommended path. Three columns:

       activity_id, ifc_global_id, weight

   where ``weight`` defaults to 1.0 and lets the user weight elements
   when one element is partially associated with multiple activities
   (e.g. an MEP duct that crosses two scheduled trades).

2. **Convention-based fallback** — when an explicit mapping is absent,
   a future revision can match ``IfcBuildingElement.Tag`` or
   ``IfcRelAssignsToGroup`` against ``activity_id`` patterns. That
   fallback is *not* implemented in this commit; this module raises a
   clear error if the explicit CSV is requested but missing, so the
   caller can decide how to handle the gap.

See ``docs/end_to_end_finishing_plan.md`` Section 4 for the full spec.

Design notes
------------

- Pure stdlib. Lightweight test set safe.
- The mapping is many-to-many: one activity can map to many elements,
  one element can map to many activities (with weights).
- Validators record their findings as structured records, not by
  raising; the caller decides whether mismatches are fatal. This
  matches the rest of the codebase's "loud but resumable" error
  philosophy.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "MAPPING_SCHEMA_VERSION",
    "REQUIRED_MAPPING_COLUMNS",
    "MappingEntry",
    "BimScheduleMapping",
    "MappingValidationReport",
    "load_mapping_csv",
    "validate_mapping",
]


MAPPING_SCHEMA_VERSION = "bim_schedule_mapping.v1"

REQUIRED_MAPPING_COLUMNS: tuple[str, ...] = ("activity_id", "ifc_global_id")


@dataclass(frozen=True)
class MappingEntry:
    """One ``(activity_id, ifc_global_id, weight)`` tuple."""

    activity_id: str
    ifc_global_id: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BimScheduleMapping:
    """Loaded mapping with two index views: by activity, by element."""

    entries: tuple[MappingEntry, ...]
    source_path: str = ""
    source_sha256: str = ""
    schema_version: str = MAPPING_SCHEMA_VERSION
    by_activity: Mapping[str, tuple[MappingEntry, ...]] = field(default_factory=dict)
    by_element: Mapping[str, tuple[MappingEntry, ...]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def elements_for_activity(self, activity_id: str) -> tuple[MappingEntry, ...]:
        return self.by_activity.get(activity_id, ())

    def activities_for_element(self, ifc_global_id: str) -> tuple[MappingEntry, ...]:
        return self.by_element.get(ifc_global_id, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "n_entries": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass(frozen=True)
class MappingValidationReport:
    """Outcome of validating a mapping against schedule + BIM element list.

    Mismatches are reported as separate counts so the caller can choose
    a policy (fail-loud vs warn).
    """

    n_entries: int
    n_mapped_elements: int
    n_unique_activities_in_mapping: int
    activities_in_mapping_not_in_schedule: tuple[str, ...]
    elements_in_mapping_not_in_bim: tuple[str, ...]
    bim_elements_not_in_mapping: tuple[str, ...]
    coverage_ratio: float  # mapped_bim_elements / total_bim_elements

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_entries": self.n_entries,
            "n_mapped_elements": self.n_mapped_elements,
            "n_unique_activities_in_mapping": self.n_unique_activities_in_mapping,
            "activities_in_mapping_not_in_schedule": list(self.activities_in_mapping_not_in_schedule),
            "elements_in_mapping_not_in_bim": list(self.elements_in_mapping_not_in_bim),
            "bim_elements_not_in_mapping": list(self.bim_elements_not_in_mapping),
            "coverage_ratio": self.coverage_ratio,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_weight(raw: Any) -> float:
    if raw is None:
        return 1.0
    s = str(raw).strip()
    if not s:
        return 1.0
    try:
        v = float(s)
    except ValueError:
        return 1.0
    if v < 0.0:
        return 0.0
    return v


def load_mapping_csv(path: Path | str) -> BimScheduleMapping:
    """Load a mapping CSV. See module docstring for column spec.

    Rows missing either required column are skipped silently; the
    caller can run :func:`validate_mapping` to surface mismatches.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the CSV has no header or is missing a required column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BIM<->schedule mapping CSV not found: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"mapping CSV is empty: {path}")
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"mapping CSV has no header: {path}")
    field_set = {(f or "").strip() for f in reader.fieldnames}
    missing = [c for c in REQUIRED_MAPPING_COLUMNS if c not in field_set]
    if missing:
        raise ValueError(f"mapping CSV {path} missing required columns: {missing}")

    entries: list[MappingEntry] = []
    for row in reader:
        a = (row.get("activity_id") or "").strip()
        e = (row.get("ifc_global_id") or "").strip()
        if not a or not e:
            continue
        entries.append(MappingEntry(activity_id=a, ifc_global_id=e, weight=_parse_weight(row.get("weight"))))

    by_activity: dict[str, list[MappingEntry]] = defaultdict(list)
    by_element: dict[str, list[MappingEntry]] = defaultdict(list)
    for entry in entries:
        by_activity[entry.activity_id].append(entry)
        by_element[entry.ifc_global_id].append(entry)

    return BimScheduleMapping(
        entries=tuple(entries),
        source_path=str(path.resolve()),
        source_sha256=sha,
        schema_version=MAPPING_SCHEMA_VERSION,
        by_activity={k: tuple(v) for k, v in by_activity.items()},
        by_element={k: tuple(v) for k, v in by_element.items()},
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_mapping(
    mapping: BimScheduleMapping,
    *,
    schedule_activity_ids: Iterable[str],
    bim_element_global_ids: Iterable[str],
) -> MappingValidationReport:
    """Cross-check the mapping against schedule + BIM element lists."""
    sched_set = set(schedule_activity_ids)
    bim_set = set(bim_element_global_ids)

    activities_in_mapping = {e.activity_id for e in mapping.entries}
    elements_in_mapping = {e.ifc_global_id for e in mapping.entries}

    activities_orphan = tuple(sorted(activities_in_mapping - sched_set))
    elements_orphan = tuple(sorted(elements_in_mapping - bim_set))
    bim_uncovered = tuple(sorted(bim_set - elements_in_mapping))

    if bim_set:
        coverage = len(bim_set & elements_in_mapping) / len(bim_set)
    else:
        coverage = 0.0

    return MappingValidationReport(
        n_entries=len(mapping.entries),
        n_mapped_elements=len(elements_in_mapping),
        n_unique_activities_in_mapping=len(activities_in_mapping),
        activities_in_mapping_not_in_schedule=activities_orphan,
        elements_in_mapping_not_in_bim=elements_orphan,
        bim_elements_not_in_mapping=bim_uncovered,
        coverage_ratio=coverage,
    )
