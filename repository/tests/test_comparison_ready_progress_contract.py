"""Model-agnostic reconstruction feature package is never construction completion."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from pipeline.stage_09_progress.comparison_ready_adapter import (
    SCHEMA, LINK_SCHEMA, load_comparison_ready_geometry, comparison_for_element,
)

F=Path(__file__).resolve().parents[1]/'configs/research_progress/comparison_ready_fixture'
ROOT=Path(__file__).resolve().parents[1]
SHA='1'*64
REG='3'*64
I=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]


def bundle(path=None,root=ROOT):
    return load_comparison_ready_geometry(path or F/'comparison_ready.json',project_root=root,
        expected_project_id='mycon-synthetic-only')


def modify(tmp_path, side=None, edit=None, manifest_change=None):
    root=tmp_path/'repository'
    path=root/'configs/research_progress/comparison_ready_fixture'
    path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(F,path)
    m=path/'comparison_ready.json'
    obj=json.loads(m.read_text())
    if side:
        s=path/side
        data=json.loads(s.read_text())
        edit(data)
        s.write_text(json.dumps(data))
        key=side[:-5]
        obj['files'][key]['sha256']=hashlib.sha256(s.read_bytes()).hexdigest()
    if manifest_change:manifest_change(obj)
    m.write_text(json.dumps(obj))
    return root,m


def verified(tmp_path):
    root,m=modify(tmp_path,manifest_change=lambda x:x.update(metric_scale_verified=True,registration_verified=True))
    b=bundle(m,root)
    assert b.valid,b.reasons
    return b


def link(b,**kw):
    return {**{
        'schema_version':LINK_SCHEMA,'comparison_ready_manifest_sha256':b.manifest_sha256,
        'ifc_sha256':SHA,'registration_report_sha256':REG,
        'metric_control_ref':'independent-survey-control-QA','metric_scale_to_m':0.7,'independent_scale_qa_ref':'survey-QA-1','registration_quality_gate':'pass',
        'independent_review_id':'audited-mapping-01','reconstruction_to_ifc_4x4':I,
        'feature_links':[{'ifc_global_id':'IFC-EXAMPLE-01','feature_type':'edges','feature_id':'edge-1',
             'quality_gate':'pass','association_method':'independent_survey_correspondence',
             'association_qa_ref':'QA-01','visibility_state':'visible',
             'visibility_method':'raycast_zbuffer','visibility_qa_ref':'camera-QA-01',
             'deviation_m':0.002,'tolerance_m':0.02}]
    },**kw}


def test_unscaled_generic_geometry_diagnostic_valid_but_progress_null():
    b=bundle()
    assert b.valid,b.reasons
    assert not b.metric_scale_verified and not b.registration_verified
    assert b.cloud_point_count==4
    assert b.summary()['physical_progress_estimate'] is None
    assert b.summary()['feature_counts']=={'planes':4,'edges':1,'corners':2}
    assert b.summary()['audited_feature_counts']['corners']==0
    assert b.features['corners']['corner-false']['audit_reason']=='degenerate_trihedral_false_corner'
    assert b.features['corners']['corner-candidate']['audit_state']=='candidate_extrapolated'
    res=comparison_for_element(b,'IFC-EXAMPLE-01')
    assert res['geometric_agreement_state']=='unknown'
    assert res['comparison_ifc_quality_gate']=='blocked'
    assert res['comparison_feature_ids']=='[]'


def test_unknown_ifc_even_when_geometry_has_many_points():
    b=bundle()
    x=comparison_for_element(b,'OTHER-IFC')
    assert x['geometric_agreement_state']=='unknown'
    assert x['comparison_visibility_state']=='unknown'


def test_no_scale_rejects_even_claimed_good_mapping():
    b=bundle()
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=link(b),expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['comparison_ifc_quality_gate']=='blocked'


def test_true_metric_and_independent_ifc_mapping_allows_geometry_comparison(tmp_path):
    b=verified(tmp_path)
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=link(b),expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['comparison_ifc_quality_gate']=='pass'
    assert x['comparison_feature_ids']=='["edges:edge-1"]'
    assert x['geometric_agreement_state']=='conforming'
    assert 'physical_progress_estimate' not in x


def test_nonconforming_geom_does_not_imply_not_built(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l['feature_links'][0]['deviation_m']=0.5
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['geometric_agreement_state']=='nonconforming'
    assert 'physical_progress_estimate' not in x


def test_unsupported_false_corner_cannot_be_mapped_as_confirmed(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l['feature_links'][0].update(feature_type='corners',feature_id='corner-false')
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['comparison_ifc_quality_gate']=='blocked'
    assert x['geometric_agreement_state']=='unknown'


def test_unobserved_candidate_corner_cannot_verify_ifc(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l['feature_links'][0].update(feature_type='corners',feature_id='corner-candidate')
    assert comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)['geometric_agreement_state']=='unknown'


def test_wrong_ifc_sha_invalidates_mapping(tmp_path):
    b=verified(tmp_path)
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=link(b),expected_ifc_sha256='9'*64,expected_registration_sha256=REG)
    assert x['comparison_ifc_quality_gate']=='blocked'


def test_failed_registration_invalidates_mapping(tmp_path):
    b=verified(tmp_path)
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=link(b,registration_quality_gate='fail'),expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['geometric_agreement_state']=='unknown'


def test_occluded_does_not_create_false_absence(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l['feature_links'][0]['visibility_state']='occluded'
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['geometric_agreement_state']=='unknown'
    assert x['comparison_visibility_state']=='occluded'


def test_incomplete_or_missing_correspondence_unknown(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l['feature_links'][0]['association_qa_ref']=''
    assert comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)['comparison_ifc_quality_gate']=='blocked'


def test_duplicate_plane_id_fails_closed(tmp_path):
    root,m=modify(tmp_path,side='planes.json',edit=lambda x:x['planes'].append(dict(x['planes'][0])))
    b=bundle(m,root)
    assert not b.valid
    assert b.summary()['physical_progress_estimate'] is None


def test_invalid_plane_normal_fails_closed(tmp_path):
    root,m=modify(tmp_path,side='planes.json',edit=lambda x:x['planes'][0].update(normal=[10,0,0]))
    assert not bundle(m,root).valid


def test_cloud_sha_tamper_fail_closed(tmp_path):
    root,m=modify(tmp_path)
    with (m.parent/'clean_cloud.ply').open('a') as f:f.write('0 0 2\n')
    assert not bundle(m,root).valid


def test_arbitrary_units_are_valid_diagnostics():
    assert bundle().units=='arbitrary'


def test_bad_camera_qa_is_not_metric_or_visibility_support(tmp_path):
    root,m=modify(tmp_path,side='cameras.json',edit=lambda x:x['cameras'][0].update(qa_verified=False))
    b=bundle(m,root)
    assert b.valid and b.camera_views_qa==0
    assert comparison_for_element(b,'IFC-EXAMPLE-01')['comparison_visibility_state']=='unknown'


def test_unknown_source_state_cannot_be_promoted_to_observed(tmp_path):
    root,m=modify(tmp_path,side='edges.json',edit=lambda x:x['edges'][0].update(state='unknown'))
    b=bundle(m,root)
    assert b.valid and b.features['edges']['edge-1']['audit_state']=='unknown'


def test_path_traversal_blocked(tmp_path):
    root,m=modify(tmp_path,manifest_change=lambda x:x['files']['planes'].update(path='../../../../../../etc/passwd'))
    assert not bundle(m,root).valid


def test_cross_element_feature_reuse_is_untrusted(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l['feature_links'].append({**l['feature_links'][0], 'ifc_global_id':'OTHER-IFC'})
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['comparison_ifc_quality_gate']=='blocked'
    assert x['comparison_evidence_reason_codes']=='ambiguous_cross_element_feature_ownership'


def test_unverified_camera_does_not_claim_visibility(tmp_path):
    root,m=modify(tmp_path,side='cameras.json',edit=lambda x:x['cameras'][0].update(qa_verified=False),
                  manifest_change=lambda x:x.update(metric_scale_verified=True,registration_verified=True))
    b=bundle(m,root)
    l=link(b)
    x=comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)
    assert x['geometric_agreement_state']=='unknown'
    assert x['comparison_visibility_state']=='unknown'


def test_missing_independent_metric_conversion_blocks_even_if_flags_true(tmp_path):
    b=verified(tmp_path)
    l=link(b)
    l.pop('metric_scale_to_m')
    assert comparison_for_element(b,'IFC-EXAMPLE-01',linkage=l,expected_ifc_sha256=SHA,expected_registration_sha256=REG)['comparison_ifc_quality_gate']=='blocked'
