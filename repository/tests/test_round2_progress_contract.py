"""A->B contract, B->C scientific semantics and no-fabricated-progress gates."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from pipeline.stage_09_progress.geometry_evidence_adapter import load_geometry_evidence_manifest
from pipeline.stage_09_progress.enrich_scientific_evidence import enrich_scientific_evidence_rows, main as stage9_main
from pipeline.stage_09_progress.evidence_metrics import element_scientific_fields
from pipeline.common.bim_schedule_mapping import load_mapping_csv
from pipeline.common.schedule_io import load_schedule_csv
from pipeline.stage_11_schedule_variance.scientific_progress import build_scientific_progress_report
from pipeline.stage_11_schedule_variance.run_scientific_progress import main as stage11_main

pytestmark=pytest.mark.lightweight
ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'configs/research_progress/round2_contract_fixture'
E1='0AbCdeFgHiJkLmNoPqRsTu'
E2='1BcDefGhIjKlMnOpQrStUv'
DATE=dt.datetime(2026,1,16,tzinfo=dt.timezone.utc)


def _rows(name):
    with (FIX/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _bundle(path=FIX/'bim_geometry_evidence.v1.json',root=ROOT,**kw):
    return load_geometry_evidence_manifest(path,project_root=root,
         expected_global_ids={E1,E2},expected_project_id='mycon-round2-synthetic',
         expected_ifc_sha256='1'*64,expected_scan_sha256='2'*64,
         expected_registration_sha256='3'*64,**kw)


def _make_mutable(tmp_path):
    root=tmp_path/'repository'
    p=root/'configs/research_progress/round2_contract_fixture'
    p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(FIX,p)
    return root,p


def _modify(tmp_path, *, manifest_changes=None, visibility_edit=None, surface_edit=None):
    root,p=_make_mutable(tmp_path)
    manifest_path=p/'bim_geometry_evidence.v1.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest_changes:manifest_changes(manifest)
    for key,edit in [('element_visibility',visibility_edit),('surface_samples',surface_edit)]:
        if edit:
            side=p/f'{key}.jsonl' if key=='element_visibility' else p/'surface_samples.jsonl'
            records=[json.loads(x) for x in side.read_text().splitlines() if x]
            edit(records)
            side.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf-8')
            manifest[key]['sha256']=hashlib.sha256(side.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    return root,manifest_path


def test_manifest_valid_and_observability_by_union():
    b=_bundle()
    assert b.valid,b.reason_codes
    assert b.rows[E1]['observability_ratio']==0.5
    assert b.rows[E2]['observability_ratio']==0
    assert b.rows[E2]['geometry_evidence_state']=='occluded'
    assert b.for_element(E1,None)['geometry_evidence_state']=='observed_geometry_consistent'
    assert b.for_element(E1,0.8)['geometry_evidence_state']=='partial_built_verified'
    assert b.for_element(E1,1.0)['geometry_evidence_state']=='built_observed_verified'


def test_manifest_hash_mismatch_blocks_all_visibility(tmp_path):
    root,p=_make_mutable(tmp_path)
    with (p/'element_visibility.jsonl').open('a') as fh:fh.write('{}\n')
    b=_bundle(p/'bim_geometry_evidence.v1.json',root)
    assert not b.valid
    assert b.for_element(E1)['observability_ratio'] is None
    assert any('sidecar_hash_mismatch' in reason for reason in b.reason_codes)


def test_manifest_bad_scale_blocks(tmp_path):
    root,p=_modify(tmp_path,manifest_changes=lambda m:m.update(metric_scale_verified=False))
    b=_bundle(p,root)
    assert not b.valid
    assert 'metric_scale_not_independently_verified' in b.reason_codes


def test_manifest_failed_registration_blocks(tmp_path):
    root,p=_modify(tmp_path,manifest_changes=lambda m:m.update(registration_quality_gate={'status':'fail'}))
    b=_bundle(p,root)
    assert not b.valid
    assert 'registration_gate_not_passed' in b.reason_codes


def test_unknown_ifc_id_blocks(tmp_path):
    root,p=_modify(tmp_path,surface_edit=lambda r:r[0].update(global_id='NO_SUCH_IFC'))
    b=_bundle(p,root)
    assert not b.valid
    assert 'unrecognized_global_id' in b.reason_codes


def test_duplicate_stable_sample_id_blocks(tmp_path):
    root,p=_modify(tmp_path,surface_edit=lambda r:r[1].update(sample_id='s1'))
    b=_bundle(p,root)
    assert not b.valid
    assert 'invalid_or_duplicate_surface_sample' in b.reason_codes


def test_cross_element_visibility_sample_identity_blocks(tmp_path):
    def corrupt(records):
        records[0]['eligible_sample_ids'].append('s5')
    root,p=_modify(tmp_path,visibility_edit=corrupt)
    b=_bundle(p,root)
    assert not b.valid
    assert 'visibility_cross_element_sample_id' in b.reason_codes


def test_segmentation_failure_blocks(tmp_path):
    root,p=_modify(tmp_path,visibility_edit=lambda r:r[0].update(segmentation_status='failed'))
    b=_bundle(p,root)
    assert not b.valid
    assert 'segmentation_failure_detected' in b.reason_codes


def test_out_of_fov_alone_is_null_not_zero(tmp_path):
    def change(r):
        r[1]['state']='out_of_fov'
        r[1]['eligible_sample_ids']=[]
    root,p=_modify(tmp_path,visibility_edit=change)
    b=_bundle(p,root)
    assert b.valid
    assert b.rows[E2]['geometry_evidence_state']=='out_of_fov'
    assert b.rows[E2]['observability_ratio'] is None


def test_unverified_deviation_does_not_claim_nonconformance(tmp_path):
    def change(r):r[0]['geometry_assessment']['quality_gate']='not_evaluated'
    root,p=_modify(tmp_path,visibility_edit=change)
    b=_bundle(p,root)
    assert b.valid
    assert b.rows[E1]['geometry_conformance_state']=='unknown'
    assert b.rows[E1]['geometry_evidence_state']=='observed_geometry_unassessed'


def test_qualified_geometric_mismatch_is_not_zero_construction(tmp_path):
    def change(r):r[0]['geometry_assessment']['max_deviation_m']=0.15
    root,p=_modify(tmp_path,visibility_edit=change)
    b=_bundle(p,root)
    assert b.valid
    assert b.rows[E1]['geometry_evidence_state']=='geometric_nonconformance'
    assert b.rows[E1]['observability_ratio']==0.5


def test_invalid_path_traversal_blocked(tmp_path):
    root,p=_modify(tmp_path,manifest_changes=lambda m:m['surface_samples'].update(path='../../../../etc/passwd'))
    b=_bundle(p,root)
    assert not b.valid


def test_no_geometry_does_not_erase_approved_quantity():
    rows=_rows('elements_stage9.csv')
    upgraded,summary=enrich_scientific_evidence_rows(rows,manifest_path=None,project_root=ROOT,data_date=DATE)
    assert summary['geometry_adapter_valid'] is False
    assert upgraded[0]['physical_progress_estimate']==0.8
    assert upgraded[0]['observability_ratio'] is None
    assert upgraded[0]['geometry_evidence_state']=='unknown_geometry_identity'


def test_missing_quantity_signature_blocks_even_high_coverage():
    row=_rows('elements_stage9.csv')[0]
    row.update(quantity_approved_by='',coverage='1.0',status='likely_completed')
    out=element_scientific_fields(row,data_date=DATE,require_audit=True)
    assert out['physical_progress_estimate'] is None
    assert out['quantity_audit_gate']=='blocked'


def test_future_approval_blocks_and_unit_mismatch_blocks():
    row=_rows('elements_stage9.csv')[0]
    row['quantity_approved_at_utc']='2026-01-17T00:00:00+00:00'
    assert element_scientific_fields(row,data_date=DATE,require_audit=True)['physical_progress_estimate'] is None
    row['quantity_approved_at_utc']='2026-01-15T00:00:00+00:00'
    row['executed_quantity_unit']='m3'
    assert element_scientific_fields(row,data_date=DATE,require_audit=True)['physical_progress_estimate'] is None


def _report(*,element_rows=None,schedule_rows=None,mapping=None,date=DATE):
    return build_scientific_progress_report(
        schedule=load_schedule_csv(FIX/'schedule.csv'),
        mapping=mapping if mapping is not None else load_mapping_csv(FIX/'mapping.csv'),
        element_rows=element_rows if element_rows is not None else _rows('elements_stage9.csv'),
        schedule_rows=schedule_rows if schedule_rows is not None else _rows('schedule.csv'),
        data_date=date,run_id='progress-test',
    )


def test_partial_identification_and_boq_weights_not_ci():
    report=_report()
    a=report['activities'][0]
    assert a['physical_progress_estimate'] is None
    assert a['physical_progress_lower_bound_pct']==pytest.approx(60)
    assert a['physical_progress_upper_bound_pct']==pytest.approx(85)
    assert a['schedule_variance_bounds_pct']==pytest.approx([10,35])
    assert a['progress_interval_type']=='partial_identification_not_confidence_interval'
    assert a['schedule_variance_status']=='unknown_evidence'
    assert a['observability_ratio']==pytest.approx(0.375)
    assert report['wbs']['1.1']['physical_progress_estimate'] is None
    assert report['quality_gates']['schedule_decision']=='blocked'


def test_complete_approved_quantities_give_real_ahead_classification():
    rows=_rows('elements_stage9.csv')
    rows[1].update({k:rows[0][k] for k in (
        'quantity_unit','quantity_source','quantity_verified','quantity_record_id',
        'quantity_approval_id','quantity_approved_by','quantity_approved_at_utc',
        'quantity_as_of_utc','quantity_document_sha256')})
    rows[1].update(planned_quantity='10',executed_quantity='2')
    a=_report(element_rows=rows)['activities'][0]
    assert a['physical_progress_estimate']==pytest.approx(65)
    assert a['schedule_variance_percent']==pytest.approx(15)
    assert a['schedule_variance_status']=='ahead'
    assert a['schedule_decision_quality_gate']=='pass'


@pytest.mark.parametrize('percent,expected',[('95','behind'),('90','on_schedule'),('70','ahead')])
def test_behind_on_ahead_date_baseline(percent,expected):
    rows=_rows('elements_stage9.csv')
    rows[1].update({k:rows[0][k] for k in (
        'quantity_unit','quantity_source','quantity_verified','quantity_record_id',
        'quantity_approval_id','quantity_approved_by','quantity_approved_at_utc',
        'quantity_as_of_utc','quantity_document_sha256')})
    rows[1].update(planned_quantity='10',executed_quantity='10')
    bs=_rows('schedule.csv')
    bs[0]['approved_planned_percent']=percent
    status=_report(element_rows=rows,schedule_rows=bs)['activities'][0]['schedule_variance_status']
    # physical = 85%; deltas -10, -5, +15 percentage points
    assert status==expected


def test_future_baseline_cannot_enter_schedule_decision():
    bs=_rows('schedule.csv');bs[0]['planned_as_of_utc']='2026-01-17T00:00:00+00:00'
    r=_report(schedule_rows=bs)
    assert r['activities'][0]['planned_percent_complete'] is None
    assert r['activities'][0]['schedule_variance_status']=='unknown_schedule_baseline'


def test_duplicate_element_record_unknown_fraction():
    rows=_rows('elements_stage9.csv')
    assert _report(element_rows=rows+[rows[0]])['activities'][0]['physical_progress_lower_bound_pct']==0


def test_low_registration_retains_independent_verified_quantity():
    rows=_rows('elements_stage9.csv')
    rows[0]['registration_confidence']='low'
    assert _report(element_rows=rows)['activities'][0]['physical_progress_lower_bound_pct']==pytest.approx(60)
    assert element_scientific_fields(rows[0],data_date=DATE,require_audit=True)['geometry_evidence_state']=='uncertain_registration'


def test_research_cli_chain_real_exit_codes(tmp_path):
    step9=tmp_path/'progress-9.csv';stage9_summary=tmp_path/'progress-9.json'
    a9=[
        '--element-metrics-csv',str(FIX/'elements_stage9.csv'),
        '--geometry-manifest',str(FIX/'bim_geometry_evidence.v1.json'),
        '--project-root',str(ROOT),'--out-element-csv',str(step9),
        '--out-summary-json',str(stage9_summary),
        '--data-date-utc','2026-01-16',
        '--expected-ifc-sha256','1'*64,
        '--expected-scan-sha256','2'*64,
        '--expected-registration-sha256','3'*64,
        '--expected-project-id','mycon-round2-synthetic',
    ]
    assert stage9_main(a9)==0
    updated=list(csv.DictReader(step9.open()))
    assert updated[0]['geometry_evidence_state']=='partial_built_verified'
    assert updated[1]['geometry_evidence_state']=='occluded'
    assert updated[0]['physical_progress_estimate']=='0.8'
    target=tmp_path/'progress-11.json'
    a11=[
        '--schedule-csv',str(FIX/'schedule.csv'),
        '--mapping-csv',str(FIX/'mapping.csv'),
        '--element-metrics-csv',str(step9),
        '--out-json',str(target), '--data-date-utc','2026-01-16',
        '--research-run-id','progress-round2-chain',
    ]
    assert stage11_main(a11)==0
    output=json.loads(target.read_text())
    assert output['schema_version']=='mycon.scientific_progress.v2'
    assert output['activities'][0]['physical_progress_lower_bound_pct']==pytest.approx(60)
    assert output['activities'][0]['observability_ratio']==pytest.approx(0.375)
    assert output['provenance']['input_sha256']['element_metrics_csv']==hashlib.sha256(step9.read_bytes()).hexdigest()
    assert stage11_main(a11)==2 # forbid overwriting research report


def test_wrong_ifc_hash_gate_fails_closed_but_quantity_remains():
    rows=_rows('elements_stage9.csv')
    updated,summary=enrich_scientific_evidence_rows(rows,manifest_path=FIX/'bim_geometry_evidence.v1.json',
         project_root=ROOT,expected_ifc_sha256='a'*64,data_date=DATE)
    assert not summary['geometry_adapter_valid']
    assert updated[0]['physical_progress_estimate']==0.8
    assert updated[0]['observability_ratio'] is None


def test_at_as_of_date_does_not_reuse_stale_quantity():
    rows=_rows('elements_stage9.csv')
    r=_report(element_rows=rows,date=dt.datetime(2026,1,17,tzinfo=dt.timezone.utc))
    assert r['activities'][0]['physical_progress_estimate'] is None
    assert r['activities'][0]['physical_progress_lower_bound_pct']==0


def test_low_image_count_downgrades_geometry_qa_not_quantity():
    b=_bundle()
    assert b.valid
    assert b.rows[E1]['geometry_n_views']==1
    assert b.rows[E1]['geometry_quality_gate']=='limited_view_support_or_unassessed_geometry'
    rows=_rows('elements_stage9.csv')
    upgraded,report=enrich_scientific_evidence_rows(rows,manifest_path=FIX/'bim_geometry_evidence.v1.json',
        project_root=ROOT,expected_ifc_sha256='1'*64,expected_scan_sha256='2'*64,
        expected_registration_sha256='3'*64,expected_project_id='mycon-round2-synthetic',
        data_date=DATE)
    assert upgraded[0]['physical_progress_estimate']==0.8
    assert upgraded[0]['geometry_quality_gate']!='pass'


def test_b_to_c_example_separates_fraction_null_and_approved_baseline():
    example=json.loads((FIX/'scientific_progress.v2.synthetic.json').read_text())
    assert example['schema_version']=='mycon.scientific_progress.v2'
    assert example['fixture_type'].startswith('synthetic_only')
    assert example['units']['element_progress']=='fraction_0_1'
    assert example['units']['activity_and_project_percent']=='percent_0_100'
    a=example['activities'][0]
    assert a['physical_progress_estimate'] is None
    assert a['physical_progress_lower_bound_pct']==pytest.approx(60)
    assert a['physical_progress_upper_bound_pct']==pytest.approx(85)
    assert a['schedule_decision_quality_gate']=='blocked'
    assert a['baseline_audit']['baseline_id']=='SYNTH-BL1'
    e1,e2=a['elements']
    assert e1['ifc_global_id']==E1 and e1['physical_progress_estimate']==0.8
    assert e2['ifc_global_id']==E2 and e2['physical_progress_estimate'] is None
    assert e2['geometry_evidence_state']=='occluded' and e2['observability_ratio']==0
    assert e1['geometry_quality_gate']!='pass'  # one view is not enough
