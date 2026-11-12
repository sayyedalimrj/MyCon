"""Stage 11 research v2: as-of approved BOQ quantity progress and schedule.

A reported number is either fully traceable or is null; partial bounds are
set-identification under unknown fractions in [0,1], never 95% CI.
Legacy acceptance statuses and scan proximity never become executed quantity.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from pipeline.stage_09_progress.evidence_metrics import (
    as_utc, element_scientific_fields, finite_number, valid_sha256,
)

SCHEMA = 'mycon.scientific_progress.v2'
APPROVED_WEIGHTS = {'approved_boq', 'approved_cost', 'approved_quantity'}


def _audit_weight(row: Mapping[str, Any], *, key: str, date: dt.datetime) -> tuple[float | None, dict[str, Any], list[str]]:
    if key == 'element':
        weight = finite_number(row.get('weight'))
        basis = str(row.get('weight_basis') or '').lower()
        prefix = 'weight'
    else:
        weight = finite_number(row.get('activity_weight'))
        basis = str(row.get('activity_weight_basis') or '').lower()
        prefix = 'activity_weight'
    audit = {'value':weight, 'basis':basis,
             'approval_id':row.get(prefix+'_approval_id'),
             'approved_by':row.get(prefix+'_approved_by'),
             'approved_at_utc':row.get(prefix+'_approved_at_utc'),
             'document_sha256':row.get(prefix+'_document_sha256')}
    reasons: list[str] = []
    when = as_utc(audit['approved_at_utc'])
    if weight is None or weight <= 0 or basis not in APPROVED_WEIGHTS:
        reasons.append('weight_missing_or_invalid')
    if not (audit['approval_id'] and audit['approved_by'] and valid_sha256(audit['document_sha256']) and when):
        reasons.append('weight_approval_provenance_incomplete')
    elif when > date:
        reasons.append('weight_approval_after_data_date')
    return (weight if not reasons else None, audit, reasons)


def _baseline(schedule_rows: list[Mapping[str, Any]], activity_id: str, date: dt.datetime) -> tuple[float | None, dict[str,Any], list[str]]:
    """Use the approval for EXACT as-of day; old or future points cannot be reused."""
    candidates = [r for r in schedule_rows if str(r.get('activity_id') or '').strip() == activity_id]
    seen = []
    for record in candidates:
        when = as_utc(record.get('planned_as_of_utc'))
        if when is not None and when.date() == date.date():
            seen.append(record)
    if len(seen) != 1:
        return None, {}, ['missing_or_duplicate_dated_baseline']
    row = seen[0]
    pct = finite_number(row.get('approved_planned_percent'))
    approved = as_utc(row.get('baseline_approved_at_utc'))
    audit = {
        'baseline_id':row.get('baseline_id'),
        'planned_as_of_utc':row.get('planned_as_of_utc'),
        'approved_at_utc':row.get('baseline_approved_at_utc'),
        'approved_by':row.get('baseline_approved_by'),
        'document_sha256':row.get('baseline_document_sha256'),
        'source':row.get('baseline_source'),
        'planned_percent':pct,
    }
    reasons=[]
    if pct is None or not 0<=pct<=100 or row.get('baseline_source') != 'approved_baseline':
        reasons.append('invalid_approved_baseline_value')
    if not (audit['baseline_id'] and audit['approved_by'] and valid_sha256(audit['document_sha256']) and approved):
        reasons.append('baseline_approval_provenance_incomplete')
    elif approved > date:
        reasons.append('baseline_approved_after_data_date')
    return pct if not reasons else None, audit, reasons


def _status(planned: float | None, actual: float | None, band: float) -> str:
    if planned is None: return 'unknown_schedule_baseline'
    if actual is None: return 'unknown_evidence'
    if actual-planned > band: return 'ahead'
    if planned-actual > band: return 'behind'
    return 'on_schedule'


def _aggregate(parts: list[dict[str,Any]], weights: list[float] | None) -> dict[str,Any]:
    if not weights or len(weights)!=len(parts) or any(w<=0 or not math.isfinite(w) for w in weights):
        return {'physical_progress_estimate':None,'physical_progress_lower_bound_pct':None,
                'physical_progress_upper_bound_pct':None,'planned_percent_complete':None,
                'schedule_variance_percent':None,'observability_ratio':None,'evidence_confidence':None}
    total = sum(weights)
    def avg(field: str) -> float | None:
        values = [r.get(field) for r in parts]
        return sum(w*v for w,v in zip(weights,values))/total if all(v is not None for v in values) else None
    return {k:avg(k) for k in ('physical_progress_estimate','physical_progress_lower_bound_pct',
            'physical_progress_upper_bound_pct','planned_percent_complete','schedule_variance_percent',
            'observability_ratio','evidence_confidence')}


def build_scientific_progress_report(
    *, schedule: Any, mapping: Any, element_rows: Iterable[Mapping[str,Any]],
    data_date: dt.datetime, schedule_rows: Iterable[Mapping[str,Any]] = (),
    on_schedule_band_pct: float = 5.0, run_id: str = 'progress-unspecified',
) -> dict[str,Any]:
    if not run_id.startswith('progress-'):
        raise ValueError('run id must start with progress-')
    date = as_utc(data_date)
    if date is None:
        raise ValueError('data date must have timezone')
    if not math.isfinite(on_schedule_band_pct) or on_schedule_band_pct < 0:
        raise ValueError('invalid schedule tolerance')
    all_rows = list(element_rows)
    by_gid: dict[str,list[Mapping[str,Any]]] = {}
    for row in all_rows:
        gid = str(row.get('global_id') or row.get('GlobalId') or '').strip()
        if gid: by_gid.setdefault(gid,[]).append(row)
    baseline_rows = list(schedule_rows)
    activity_records: list[dict[str,Any]] = []
    project_weights: list[float] = []
    project_bases: set[str] = set()
    for activity in schedule.activities:
        aid = activity.activity_id
        entries = mapping.elements_for_activity(aid)
        warnings: list[str] = []
        seen_gid: set[str] = set()
        element_records: list[dict[str,Any]] = []
        element_weights: list[float] = []
        element_bases: set[str] = set()
        n_verified=0
        for entry in entries:
            gid=entry.ifc_global_id
            recs=by_gid.get(gid,[])
            if gid in seen_gid:
                warnings.append('duplicate_global_id_mapping')
            seen_gid.add(gid)
            weight_info = {k:getattr(entry,k,None) for k in (
                'weight','weight_basis','weight_approval_id','weight_approved_by',
                'weight_approved_at_utc','weight_document_sha256')}
            weight, weight_audit, wr = _audit_weight(weight_info,key='element',date=date)
            if wr: warnings.extend(wr)
            if weight is not None:
                element_weights.append(weight)
                element_bases.add(weight_audit['basis'])
            if len(recs) != 1:
                warnings.append('missing_or_duplicate_element_record:'+gid)
                parsed={'physical_progress_estimate':None,'physical_progress_status':'unknown',
                        'physical_progress_reason_codes':'missing_or_duplicate_element_record',
                        'geometry_evidence_state':'unknown', 'observability_ratio':None,
                        'evidence_confidence':None,'quantity_audit_gate':'blocked','quantity_audit':{}}
            else:
                parsed=element_scientific_fields(recs[0],data_date=date,require_audit=True)
            if parsed['physical_progress_estimate'] is not None:
                n_verified+=1
            element_records.append({
                'ifc_global_id':gid,'activity_id':aid,'wbs_code':activity.wbs_code,
                'physical_progress_estimate':parsed['physical_progress_estimate'],
                'physical_progress_status':parsed['physical_progress_status'],
                'physical_progress_reason_codes':parsed['physical_progress_reason_codes'],
                'geometry_evidence_state':parsed['geometry_evidence_state'],
                'geometry_conformance_state':parsed.get('geometry_conformance_state','unknown'),
                'observability_ratio':parsed['observability_ratio'],
                'evidence_confidence':parsed['evidence_confidence'],
                'quantity_audit_gate':parsed['quantity_audit_gate'],
                'quantity_audit':parsed['quantity_audit'],
                'mapping_weight':weight,'weight_audit':weight_audit,
                'geometry_manifest_sha256': recs[0].get('geometry_manifest_sha256') if len(recs)==1 else None,
            })
        valid_weights = (bool(entries) and len(seen_gid)==len(entries)
                         and len(element_weights)==len(entries) and len(element_bases)==1)
        if not valid_weights: warnings.append('invalid_or_incomparable_element_weights')
        total = sum(element_weights) if valid_weights else None
        if valid_weights:
            sum_known=sum(w*e['physical_progress_estimate'] for w,e in zip(element_weights,element_records)
                          if e['physical_progress_estimate'] is not None)
            weight_unknown=sum(w for w,e in zip(element_weights,element_records)
                               if e['physical_progress_estimate'] is None)
            low=100*sum_known/total
            high=100*(sum_known+weight_unknown)/total
            actual=low if weight_unknown==0 else None
            obs = (sum(w*e['observability_ratio'] for w,e in zip(element_weights,element_records))/total
                   if all(e['observability_ratio'] is not None for e in element_records) else None)
            conf = (sum(w*e['evidence_confidence'] for w,e in zip(element_weights,element_records))/total
                    if all(e['evidence_confidence'] is not None for e in element_records) else None)
            measured_weight = (total-weight_unknown)/total
        else:
            actual=low=high=obs=conf=measured_weight=None
        baseline, baseline_audit, baseline_reasons = _baseline(baseline_rows,aid,date)
        warnings.extend(baseline_reasons)
        variance=actual-baseline if actual is not None and baseline is not None else None
        var_bounds=[low-baseline, high-baseline] if low is not None and baseline is not None else None
        st=_status(baseline,actual,on_schedule_band_pct)
        if actual is None: warnings.append('unknown_physical_progress')
        if st.startswith('unknown'): warnings.append('schedule_decision_blocked')
        row={
            'activity_id':aid,'activity_name':activity.activity_name,'wbs_code':activity.wbs_code,
            'physical_progress_estimate':actual,
            'physical_progress_status':'verified_quantity' if actual is not None else 'unknown',
            'physical_progress_lower_bound_pct':low,'physical_progress_upper_bound_pct':high,
            'progress_interval_type':'partial_identification_not_confidence_interval',
            'observability_ratio':obs,'evidence_confidence':conf,
            'verified_quantity_weight_ratio':measured_weight,
            'planned_percent_complete':baseline,'baseline_audit':baseline_audit,
            'provisional_linear_planned_percent':activity.planned_percent_complete_at(date),
            'planner_reported_percent':activity.percent_complete,
            'schedule_variance_percent':variance,'schedule_variance_bounds_pct':var_bounds,
            'schedule_variance_status':st,'schedule_decision_quality_gate':'pass' if not st.startswith('unknown') else 'blocked',
            'element_weights_approved':valid_weights,'mapping_weight_basis':next(iter(element_bases)) if valid_weights else None,
            'n_mapped_elements':len(entries),'n_verified_quantity_elements':n_verified,
            'warnings':sorted(set(warnings)),'elements':element_records,
        }
        activity_records.append(row)
        approved_weight_rec = next((r for r in baseline_rows if r.get('activity_id')==aid and
                                    as_utc(r.get('planned_as_of_utc')) and as_utc(r['planned_as_of_utc']).date()==date.date()),{})
        pw,pa,pr=_audit_weight(approved_weight_rec,key='activity',date=date)
        if pw is not None:
            project_weights.append(pw);project_bases.add(pa['basis'])
        row['activity_weight_audit']=pa
        row['activity_weight_quality_gate']='pass' if pw is not None else 'blocked'
        row['activity_weight_reason_codes']=pr
    valid_project = bool(activity_records) and len(project_weights)==len(activity_records) and len(project_bases)==1
    overall = _aggregate(activity_records,project_weights if valid_project else None)
    overall['schedule_variance_status']=_status(overall['planned_percent_complete'],
                                                 overall['physical_progress_estimate'],on_schedule_band_pct)
    overall['activity_weights_approved']=valid_project
    overall['schedule_decision_quality_gate']='pass' if not overall['schedule_variance_status'].startswith('unknown') else 'blocked'
    wbs: dict[str,dict[str,Any]]={}
    for wbs_code in sorted({a['wbs_code'] for a in activity_records}):
        subset=[a for a in activity_records if a['wbs_code']==wbs_code]
        ws=[finite_number(a['activity_weight_audit'].get('value')) for a in subset]
        qualified=all(a['activity_weight_quality_gate']=='pass' for a in subset) and len(set(a['activity_weight_audit']['basis'] for a in subset))==1
        agg=_aggregate(subset,ws if qualified else None)
        agg['schedule_variance_status']=_status(agg['planned_percent_complete'],agg['physical_progress_estimate'],on_schedule_band_pct)
        agg['activity_ids']=[a['activity_id'] for a in subset]
        agg['quality_gate']='pass' if qualified and not agg['schedule_variance_status'].startswith('unknown') else 'blocked'
        wbs[wbs_code]=agg
    return {
        'schema_version':SCHEMA,'run_id':run_id,'data_date_utc':date.isoformat(timespec='seconds'),
        'status':'verified' if overall['schedule_decision_quality_gate']=='pass' else 'unknown',
        'units':{'element_progress':'fraction_0_1','activity_and_project_percent':'percent_0_100',
                 'observability':'fraction_0_1','evidence_confidence':'unit_interval_quality_score'},
        'uncertainty_method':'deterministic_partial_identification_not_statistical_CI',
        'legacy_warning':'v1 acceptance-score and Wilson interval are not physical progress estimates',
        'n_activities':len(activity_records), 'overall':overall,'wbs':wbs,
        'activity_status_counts':dict(Counter(a['schedule_variance_status'] for a in activity_records)),
        'activities':activity_records,
        'quality_gates':{'physical_progress':'pass' if all(a['physical_progress_estimate'] is not None for a in activity_records) and activity_records else 'blocked',
                         'approved_schedule':'pass' if all(a['planned_percent_complete'] is not None for a in activity_records) and activity_records else 'blocked',
                         'project_weights':'pass' if valid_project else 'blocked',
                         'schedule_decision':overall['schedule_decision_quality_gate']},
    }
