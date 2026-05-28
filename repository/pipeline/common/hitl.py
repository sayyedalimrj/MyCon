"""Human-in-the-loop (HITL) corrections: a first-class structured signal.

Why this module exists
----------------------

Several components in the pipeline emit decisions whose ground truth is only
knowable from a human reviewer: per-element acceptance in
:mod:`stage_09_progress`, per-activity completion in
:mod:`stage_09_progress.decision_enrichment`, and per-question VLM answers
in :mod:`stage_10_copilot`. Up to now those reviews lived only in informal
notes. This module makes them first-class:

- a typed, schema-validated **record format** with a stable ``schema_version``;
- an **append-only JSONL store** with conflict detection and last-write-wins
  semantics that are auditable rather than silent;
- a **replay** API that converts the store into the (confidence, correctness)
  pairs consumed by :mod:`pipeline.common.calibration` and into per-element
  evidence weights that can be fed into the multi-view fusion of
  :mod:`pipeline.stage_09_progress.multiview_fusion`.

Design choices
--------------

- **Append-only storage.** Corrections are never silently overwritten. When
  the same ``(target_kind, target_id, predicted_value)`` triple is corrected
  twice, both records are kept; the *latest by timestamp* wins on replay,
  but the conflict is reported in :class:`ReplayResult.conflicts` so the
  reviewer can audit the disagreement.
- **No external deps.** Pure stdlib + dataclasses. Runs in the lightweight
  test set; safe to call from CI.
- **Forward-compatible schema.** Records are tagged with
  ``schema_version="hitl_correction.v1"``. Future schema bumps add new keys
  with defaults instead of breaking the loader.

Literature grounding
--------------------

- Beck et al., *Beyond Active Learning: Leveraging the Full Potential of
  Human Interaction*, WACV 2024 — argues that HITL signals should be
  retained as structured, replayable artefacts rather than collapsed into
  one-shot label updates. The replay API here implements that idea.
- Rožanec et al., *Human in the AI Loop via xAI and Active Learning for
  Visual Inspection*, 2023 (arXiv 2307.05508) — pairs explainability with
  HITL acceptance for industrial inspection; the per-correction
  ``rationale`` and ``evidence_refs`` fields are designed to support that
  workflow.
- Bosché, *Automated recognition of 3D CAD model objects in laser scans
  and calculation of as-built dimensions*, ASEM 2010 — the original
  scan-to-BIM acceptance review whose semantics this store is faithful to.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "CORRECTION_SCHEMA_VERSION",
    "VALID_TARGET_KINDS",
    "VALID_DECISION_VALUES",
    "Correction",
    "CorrectionStore",
    "ReplayResult",
    "ConflictRecord",
    "build_calibration_records",
]


CORRECTION_SCHEMA_VERSION = "hitl_correction.v1"

VALID_TARGET_KINDS = frozenset(
    {
        "element_acceptance",      # per-BIM-element acceptance (Stage 9)
        "activity_completion",     # per-schedule-activity completion (Stage 9)
        "vlm_answer",              # Stage 10 copilot answer
        "anchor_validation",       # metric anchor validation (Stage 8)
        "registration_quality",    # Stage 8 registration accept/reject
    }
)

VALID_DECISION_VALUES = frozenset({"accept", "reject", "uncertain", "rework"})


@dataclass(frozen=True)
class Correction:
    """One reviewer correction.

    Fields
    ------
    schema_version : str
        Always ``"hitl_correction.v1"`` for records written by this module.
    target_kind : str
        One of :data:`VALID_TARGET_KINDS`. What kind of decision the
        reviewer is correcting.
    target_id : str
        Stable identifier for the corrected target. Conventions:

        - ``element_acceptance`` → IFC GlobalId.
        - ``activity_completion`` → activity ID from the schedule CSV.
        - ``vlm_answer`` → ``run_id::query_id`` (or any deterministic key
          built by the caller).
        - ``anchor_validation`` → ``anchor_id``.
        - ``registration_quality`` → ``run_id``.

    predicted_value : str
        The decision the pipeline produced (``accept`` / ``reject`` /
        ``uncertain`` / ``rework``).
    predicted_confidence : str
        The discrete confidence the pipeline reported alongside the
        prediction. One of ``high`` / ``medium`` / ``low`` /
        ``low_to_medium`` / ``unverified``.
    corrected_value : str
        The reviewer's authoritative decision. Same vocabulary as
        ``predicted_value``.
    reviewer_id : str
        Opaque identifier for the reviewer. We do not impose any structure;
        teams can use email, employee ID, or pseudonym.
    timestamp_utc : str
        ISO-8601 UTC timestamp. Used for conflict resolution (last-write-wins).
    rationale : str
        Free-text explanation of *why* the reviewer corrected. Empty string
        is allowed but discouraged; the field is captured because it is
        valuable for downstream rule mining (Beck et al., WACV 2024).
    evidence_refs : tuple[str, ...]
        Pointers to artefacts that support the correction (frame paths,
        report files, etc). Captured so the corrected decision is itself
        evidence-linked, mirroring the pipeline's own provenance discipline.
    run_id : str
        Run that produced the prediction being corrected. Optional but
        strongly recommended; it allows :func:`replay` to bucket corrections
        per run.
    record_id : str
        Stable ID of this correction record. Auto-generated by
        :meth:`CorrectionStore.append` from a hash of the payload + the
        timestamp; callers may override for migrations.

    Notes
    -----
    The dataclass is frozen so individual records are immutable in memory.
    The store is append-only on disk. Any *change* to a correction is a
    new record with a new ``timestamp_utc``; the older record is preserved
    and surfaces as a :class:`ConflictRecord` on replay.
    """

    schema_version: str
    target_kind: str
    target_id: str
    predicted_value: str
    predicted_confidence: str
    corrected_value: str
    reviewer_id: str
    timestamp_utc: str
    rationale: str = ""
    evidence_refs: tuple[str, ...] = ()
    run_id: str = ""
    record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_refs"] = list(self.evidence_refs)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Correction":
        """Build from a dict; validates required keys and value-vocabularies.

        Unknown keys are *ignored* (forward-compat). Missing optional keys
        default to empty. Missing required keys or invalid values raise
        :class:`ValueError`.
        """
        try:
            target_kind = str(data["target_kind"]).strip().lower()
            target_id = str(data["target_id"])
            predicted_value = str(data["predicted_value"]).strip().lower()
            corrected_value = str(data["corrected_value"]).strip().lower()
            reviewer_id = str(data["reviewer_id"])
            timestamp_utc = str(data["timestamp_utc"])
        except KeyError as exc:
            raise ValueError(f"missing required correction key: {exc}") from exc

        if target_kind not in VALID_TARGET_KINDS:
            raise ValueError(
                f"target_kind must be one of {sorted(VALID_TARGET_KINDS)}, got {target_kind!r}"
            )
        if predicted_value not in VALID_DECISION_VALUES:
            raise ValueError(
                f"predicted_value must be one of {sorted(VALID_DECISION_VALUES)}, "
                f"got {predicted_value!r}"
            )
        if corrected_value not in VALID_DECISION_VALUES:
            raise ValueError(
                f"corrected_value must be one of {sorted(VALID_DECISION_VALUES)}, "
                f"got {corrected_value!r}"
            )

        evidence_refs_raw = data.get("evidence_refs", ()) or ()
        if isinstance(evidence_refs_raw, str):
            evidence_refs = (evidence_refs_raw,)
        else:
            evidence_refs = tuple(str(x) for x in evidence_refs_raw)

        return cls(
            schema_version=str(data.get("schema_version", CORRECTION_SCHEMA_VERSION)),
            target_kind=target_kind,
            target_id=target_id,
            predicted_value=predicted_value,
            predicted_confidence=str(data.get("predicted_confidence", "")).strip().lower(),
            corrected_value=corrected_value,
            reviewer_id=reviewer_id,
            timestamp_utc=timestamp_utc,
            rationale=str(data.get("rationale", "")),
            evidence_refs=evidence_refs,
            run_id=str(data.get("run_id", "")),
            record_id=str(data.get("record_id", "")),
        )


@dataclass(frozen=True)
class ConflictRecord:
    """Two corrections that disagree on the same ``(target_kind, target_id)``."""

    target_kind: str
    target_id: str
    earlier_record_id: str
    earlier_corrected_value: str
    earlier_reviewer_id: str
    earlier_timestamp_utc: str
    later_record_id: str
    later_corrected_value: str
    later_reviewer_id: str
    later_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayResult:
    """Output of :meth:`CorrectionStore.replay`.

    Fields
    ------
    effective : tuple[Correction, ...]
        One winning correction per ``(target_kind, target_id)`` after
        last-write-wins resolution.
    conflicts : tuple[ConflictRecord, ...]
        Pairs where two corrections disagreed on the same target. Always
        keeps the *immediately preceding* correction as the conflict
        partner; longer histories surface as multiple ConflictRecords (one
        per adjacent pair).
    n_total_records : int
        Total number of records in the store (before resolution).
    """

    effective: tuple[Correction, ...]
    conflicts: tuple[ConflictRecord, ...]
    n_total_records: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective": [c.to_dict() for c in self.effective],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "n_total_records": self.n_total_records,
        }


class CorrectionStore:
    """Append-only JSONL store of HITL corrections.

    The store is filesystem-backed for durability and grep-ability. All
    writes are *atomic per record*: each record is one ``json.dumps`` line
    appended under a process-local lock. Two processes appending to the
    same file simultaneously is safe under POSIX append semantics for
    individual writes smaller than ``PIPE_BUF``; longer records may
    interleave. For multi-writer workflows the recommended pattern is one
    file per reviewer host, then a periodic merge — but that is left to
    the operator and not enforced here.

    Reads are tolerant of malformed lines (skipped with a counter) so a
    half-written tail line cannot brick the store.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(self, correction: Correction | Mapping[str, Any]) -> Correction:
        """Append a correction record. Returns the canonical Correction.

        Auto-fills ``timestamp_utc`` and ``record_id`` if missing.
        """
        if isinstance(correction, Correction):
            data = correction.to_dict()
        else:
            data = dict(correction)
        data.setdefault("schema_version", CORRECTION_SCHEMA_VERSION)
        if not data.get("timestamp_utc"):
            data["timestamp_utc"] = _utc_iso_now()
        if not data.get("record_id"):
            data["record_id"] = _stable_record_id(data)
        record = Correction.from_dict(data)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
                f.write("\n")
        return record

    def append_many(self, corrections: Iterable[Correction | Mapping[str, Any]]) -> list[Correction]:
        """Bulk append. Returns the canonical Correction objects in order.

        Each record is appended one at a time so a partial failure leaves
        the store readable up to the failure point.
        """
        out: list[Correction] = []
        for c in corrections:
            out.append(self.append(c))
        return out

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Correction]:
        return self._iter_records()

    def _iter_records(self) -> Iterator[Correction]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                try:
                    yield Correction.from_dict(obj)
                except ValueError:
                    # Skip schema-invalid records but do not crash; an
                    # operator can run a separate validator to find them.
                    continue

    def all(self) -> list[Correction]:
        """Eager list of all valid records in disk order."""
        return list(self._iter_records())

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        *,
        target_kinds: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> ReplayResult:
        """Resolve the store to one effective correction per target.

        Filters
        -------

        - ``target_kinds`` — keep only records whose ``target_kind`` is in
          the set. ``None`` keeps all kinds.
        - ``run_id`` — keep only records that came from the given run. The
          common case for batch recalibration: replay the corrections that
          apply to the run we are calibrating against.

        Resolution
        ----------

        For each ``(target_kind, target_id)`` group, the record with the
        latest ``timestamp_utc`` wins. Adjacent disagreements are emitted
        as :class:`ConflictRecord` so the operator sees disagreement
        rather than silent overwrite.
        """
        kept: list[Correction] = []
        for r in self._iter_records():
            if target_kinds is not None and r.target_kind not in set(target_kinds):
                continue
            if run_id is not None and r.run_id != run_id:
                continue
            kept.append(r)

        # Group by (target_kind, target_id), sort each group by timestamp.
        groups: dict[tuple[str, str], list[Correction]] = {}
        for r in kept:
            groups.setdefault((r.target_kind, r.target_id), []).append(r)

        effective: list[Correction] = []
        conflicts: list[ConflictRecord] = []
        for key, group in groups.items():
            group_sorted = sorted(group, key=lambda c: c.timestamp_utc)
            for prev, cur in zip(group_sorted, group_sorted[1:]):
                if prev.corrected_value != cur.corrected_value:
                    conflicts.append(
                        ConflictRecord(
                            target_kind=key[0],
                            target_id=key[1],
                            earlier_record_id=prev.record_id,
                            earlier_corrected_value=prev.corrected_value,
                            earlier_reviewer_id=prev.reviewer_id,
                            earlier_timestamp_utc=prev.timestamp_utc,
                            later_record_id=cur.record_id,
                            later_corrected_value=cur.corrected_value,
                            later_reviewer_id=cur.reviewer_id,
                            later_timestamp_utc=cur.timestamp_utc,
                        )
                    )
            effective.append(group_sorted[-1])

        # Stable order: by (target_kind, target_id) so output is diffable.
        effective.sort(key=lambda c: (c.target_kind, c.target_id))
        conflicts.sort(key=lambda c: (c.target_kind, c.target_id, c.later_timestamp_utc))

        return ReplayResult(
            effective=tuple(effective),
            conflicts=tuple(conflicts),
            n_total_records=len(kept),
        )


# ---------------------------------------------------------------------------
# Bridges to other modules
# ---------------------------------------------------------------------------


def build_calibration_records(
    replay: ReplayResult,
    *,
    target_kinds: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a :class:`ReplayResult` into records consumable by
    :func:`pipeline.common.calibration.calibration_report`.

    Each effective correction maps to a calibration record with:

    - ``confidence`` — the *predicted* confidence label.
    - ``correct``   — ``True`` iff predicted_value == corrected_value.

    Filtering by ``target_kinds`` lets callers compute per-decision-kind
    calibration (e.g. element acceptance vs activity completion separately,
    which is the right granularity for thesis tables).
    """
    out: list[dict[str, Any]] = []
    kinds = set(target_kinds) if target_kinds is not None else None
    for c in replay.effective:
        if kinds is not None and c.target_kind not in kinds:
            continue
        out.append(
            {
                "target_kind": c.target_kind,
                "target_id": c.target_id,
                "confidence": c.predicted_confidence or "unknown",
                "correct": (c.predicted_value == c.corrected_value),
                "record_id": c.record_id,
                "run_id": c.run_id,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _utc_iso_now() -> str:
    """Return ISO-8601 UTC 'Z' timestamp at second resolution.

    Second resolution is sufficient for HITL conflict resolution; using
    higher resolution pollutes diffs and does not change the algorithm.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_record_id(data: Mapping[str, Any]) -> str:
    """Deterministic ID = first 12 hex chars of SHA-256 of the canonicalised payload.

    We deliberately exclude ``record_id`` itself from the hashed payload to
    avoid the chicken-and-egg problem.
    """
    import hashlib

    payload = {k: v for k, v in data.items() if k != "record_id"}
    serialised = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()[:12]
