"""Independent, provenance-aware scientific element metrics for Stage 9.

Physical quantity is measured by an approved source. Geometry/visibility
cannot substitute for quantities. A reported fractional progress must have a
traceable computation, unit and as-of date; absent audit is null.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Mapping

APPROVED_QUANTITY_SOURCES = {'approved_measurement', 'signed_site_log', 'survey_qa'}
CALIBRATED_CONFIDENCE_SOURCES = {'heldout_calibration', 'audited_quality_model'}
AUDITED_VISIBILITY_SOURCES = {'raycast_zbuffer', 'audited_manual', 'manual_occlusion_audit'}
ALLOWED_UNITS = {'m', 'm2', 'm3', 'count', 'kg', 'tonne', 'lump_sum'}


def finite_number(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == '':
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


def unit_ratio(value: Any) -> float | None:
    x = finite_number(value)
    return x if x is not None and 0 <= x <= 1 else None


def as_utc(value: Any) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        val = value
    elif isinstance(value, str) and value.strip():
        try:
            val = dt.datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError:
            return None
    else:
        return None
    if val.tzinfo is None:
        return None  # no implicit local timezone in audit records
    return val.astimezone(dt.timezone.utc)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value.lower())


def element_scientific_fields(
    row: Mapping[str, Any], metadata: Mapping[str, Any] | None = None,
    *, data_date: dt.datetime | None = None, require_audit: bool = False,
) -> dict[str, Any]:
    """Return independent fractions, categorical geometry state and audit gate.

    require_audit=False preserves old Stage9 enrichment behavior, which is
    *not* sufficient for authoritative v2. Stage11 ALWAYS calls with True.
    """
    meta = metadata or {}

    def get(key: str) -> Any:
        val = row.get(key)
        return val if val is not None and str(val).strip() != '' else meta.get(key)

    planned = finite_number(get('planned_quantity'))
    done = finite_number(get('executed_quantity'))
    unit = str(get('quantity_unit') or '').strip()
    planned_unit = str(get('planned_quantity_unit') or unit).strip()
    executed_unit = str(get('executed_quantity_unit') or unit).strip()
    verified = str(get('quantity_verified') or '').strip().lower() in {'true', 'yes', '1'}
    qsource = str(get('quantity_source') or '').strip().lower()
    reasons: list[str] = []
    basic = (verified and qsource in APPROVED_QUANTITY_SOURCES and
             planned is not None and planned > 0 and done is not None and
             0 <= done <= planned and unit in ALLOWED_UNITS and
             unit == planned_unit == executed_unit)
    if not basic:
        reasons.append('quantity_missing_invalid_or_not_approved')
    audit = {
        'quantity_record_id':get('quantity_record_id'),
        'quantity_approval_id':get('quantity_approval_id'),
        'quantity_approved_by':get('quantity_approved_by'),
        'quantity_approved_at_utc':get('quantity_approved_at_utc'),
        'quantity_as_of_utc':get('quantity_as_of_utc'),
        'quantity_document_sha256':get('quantity_document_sha256'),
        'quantity_source':qsource or None,
        'quantity_unit':unit or None,
        'planned_quantity':planned,
        'executed_quantity':done,
    }
    if require_audit:
        required = ('quantity_record_id','quantity_approval_id','quantity_approved_by')
        approved = as_utc(audit['quantity_approved_at_utc'])
        asof = as_utc(audit['quantity_as_of_utc'])
        if (not all(isinstance(audit[k], str) and audit[k].strip() for k in required)
            or not valid_sha256(audit['quantity_document_sha256'])
            or approved is None or asof is None):
            reasons.append('quantity_provenance_incomplete')
        elif data_date is not None:
            date = as_utc(data_date)
            if date is None or approved > date or asof > date or asof.date() != date.date():
                reasons.append('quantity_approval_or_snapshot_outside_data_date')
    physical = done / planned if basic and not reasons else None

    geom_valid = str(get('geometry_adapter_valid') or '').lower() in {'true','1','yes'}
    state = str(get('geometry_evidence_state') or '').strip()
    if not state or not geom_valid:
        state = 'unknown_geometry_identity'
    obs_source = str(get('observability_source') or '').strip().lower()
    obs = unit_ratio(get('observability_ratio')) if geom_valid and obs_source in AUDITED_VISIBILITY_SOURCES else None
    reg = str(get('registration_confidence') or '').lower().strip()
    synthetic = str(get('synthetic_bim_fallback_used') or '').lower() in {'true','1','yes'}
    if synthetic or reg in {'low', 'failed', 'invalid'}:
        state = 'uncertain_registration'
        obs = None
    confidence_source = str(get('evidence_confidence_source') or '').strip().lower()
    confidence = unit_ratio(get('evidence_confidence')) if (
        geom_valid and reg in {'high','medium'} and not synthetic
        and confidence_source in CALIBRATED_CONFIDENCE_SOURCES
        and get('evidence_calibration_ref')) else None
    return {
        'physical_progress_estimate': physical,
        'physical_progress_status': 'verified_quantity' if physical is not None else 'unknown',
        'physical_progress_reason_codes': ';'.join(reasons),
        'quantity_audit_gate': 'pass' if physical is not None else 'blocked',
        'quantity_audit': audit,
        'observability_ratio':obs,
        'evidence_confidence':confidence,
        'geometry_evidence_state':state,
        'geometry_conformance_state': get('geometry_conformance_state') or 'unknown',
        'scientific_metrics_schema':'mycon.element_progress.v2',
    }
