"""Round-2 reproducible contract checks; synthetic fixtures only."""
import csv
import json
import pytest
from benchmarks.mycon_2026.metrics import progress_metrics, reconstruction_metrics, compare_anchors
from benchmarks.mycon_2026.adapters import (
    load_stage9_v2, load_scientific_progress_v2, activity_progress_metrics,
    project_progress_metrics, validate_geometry_evidence)
from benchmarks.mycon_2026.protocol import sha256
from benchmarks.mycon_2026.runner import run
pytestmark = pytest.mark.lightweight


def _csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def _b():
    return {"schema_version":"mycon.scientific_progress.v2", "run_id":"progress-test",
        "units":{"element_progress":"fraction_0_1","activity_and_project_percent":"percent_0_100"},
        "activities":[{"activity_id":"a1","physical_progress_estimate":70,
                       "physical_progress_lower_bound_pct":70,"physical_progress_upper_bound_pct":70,
                       "planned_percent_complete":50,"schedule_variance_percent":20}],
        "overall":{"physical_progress_estimate":70,"physical_progress_lower_bound_pct":70,
                   "physical_progress_upper_bound_pct":70,"schedule_variance_percent":20}}


def test_unknown_does_not_drop_numeric_progress():
    gt=[{"global_id":"A","status":"partial","completion_fraction":"0.5","weight":"2"},
        {"global_id":"B","status":"built","completion_fraction":"1","weight":"1"}]
    pred=[{"global_id":"A","status":"unknown","physical_progress_estimate":"0.6"},
          {"global_id":"B","status":"not_evidenced","physical_progress_estimate":"0.9"}]
    x=progress_metrics(pred,gt)
    assert x["n_evaluated"]==0 and x["n_categorical_abstained"]==2
    assert x["n_categorical_missing_output"]==0 and x["n_progress_pairs"]==2
    assert x["progress_mae_fraction_known"]==pytest.approx(0.1)
    assert x["project_progress_lower_bound_fraction"]==pytest.approx(0.7)
    y=progress_metrics([{"global_id":"A","status":"partial"}],gt)
    assert y["n_evaluated"]==1 and y["n_categorical_missing_output"]==1
    assert y["n_progress_pairs"]==0 and y["n_numeric_missing_or_unpaired"]==2


def test_stage9_fraction_and_verified_quantity(tmp_path):
    path=tmp_path/"element.csv"
    row={"global_id":"A","status":"unknown","coverage":"0.98",
         "physical_progress_estimate":"0.7","physical_progress_status":"verified_quantity",
         "planned_quantity":"10","executed_quantity":"7","quantity_verified":"true",
         "quantity_source":"signed_site_log","quantity_unit":"m3"}
    _csv(path,[row])
    assert load_stage9_v2(path)[0]["physical_progress_estimate"]==0.7
    row["physical_progress_estimate"]="70"
    _csv(path,[row])
    with pytest.raises(ValueError,match="Stage9 physical"):
        load_stage9_v2(path)
    row["physical_progress_estimate"]="0.7"
    row["quantity_verified"]="false"
    _csv(path,[row])
    with pytest.raises(ValueError,match="Unverified"):
        load_stage9_v2(path)
    row["physical_progress_estimate"]=""
    _csv(path,[row])
    assert load_stage9_v2(path)[0]["physical_progress_estimate"] is None


def test_stage11_percent_and_baseline(tmp_path):
    p=tmp_path/"v2.json"
    p.write_text(json.dumps(_b()))
    r=load_scientific_progress_v2(p)
    gt=[{"activity_id":"a1","physical_progress_pct":"75","planned_percent_pct":"50","baseline_id":"approved-1"}]
    x=activity_progress_metrics(r,gt)
    assert x["physical_progress_mae_pp"]==5 and x["schedule_variance_mae_pp"]==5
    assert project_progress_metrics(r,[{"baseline_id":"approved-1","physical_progress_pct":"75",
                                        "planned_percent_pct":"50"}])["project_progress_absolute_error_pp"]==5
    r["activities"][0].update(physical_progress_estimate=0.7,
        physical_progress_lower_bound_pct=0.7,physical_progress_upper_bound_pct=0.7)
    p.write_text(json.dumps(r))
    assert activity_progress_metrics(load_scientific_progress_v2(p),gt)["physical_progress_mae_pp"]==pytest.approx(74.3)
    r["units"]["activity_and_project_percent"]="fraction_0_1"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match="units mismatch"):
        load_scientific_progress_v2(p)


def test_scientific_v2_unknown_not_zero():
    r=_b()
    r["activities"][0]["physical_progress_estimate"]=None
    gt=[{"activity_id":"a1","physical_progress_pct":"40","planned_percent_pct":"50","baseline_id":"b"}]
    x=activity_progress_metrics(r,gt)
    assert x["n_numeric_pairs"]==0 and x["n_numeric_missing_prediction"]==1
    assert x["physical_progress_mae_pp"] is None


def test_geometry_thresholds_and_orientation():
    xyz=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]
    m=reconstruction_metrics(xyz,xyz)
    for tau in ("0.01m","0.02m","0.05m","0.10m"):
        assert m["fscore_at_"+tau]==1 and m["accuracy_at_"+tau]==1
        assert m["completeness_at_"+tau]==1
    rows=[{"anchor_id":str(i),"x_m":str(x),"y_m":str(y),"z_m":str(z)}
          for i,(x,y,z) in enumerate(xyz)]
    assert compare_anchors(rows,rows)["orientation_pairwise_median_deg"]==pytest.approx(0)


def test_a_manifest_fail_closed_sha_and_frame(tmp_path):
    assets={}
    for k in ("surface_samples","camera_poses","element_visibility"):
        f=tmp_path/(k+".jsonl")
        f.write_text("{}\n")
        assets[k]={"path":f.name,"sha256":sha256(f)}
    doc={"schema_version":"mycon.bim_geometry_evidence.v1","run_id":"geom-test",
         "project_id":"p","created_at_utc":"2026-11-12","coordinate_frame":"bim_world",
         "units":"m","ifc_sha256":"a"*64,"scan_sha256":"b"*64,
         "registration_report_sha256":"c"*64,"registration_quality_gate":{"status":"pass"},
         "metric_scale_verified":True,"provenance":{"seed":42},
         "scan_to_bim_4x4":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],**assets}
    p=tmp_path/"manifest.json"
    p.write_text(json.dumps(doc))
    assert validate_geometry_evidence(p,root=tmp_path)["asset_hashes_verified"]
    doc["surface_samples"]["sha256"]="0"*64
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="SHA-256"):
        validate_geometry_evidence(p,root=tmp_path)
    doc["surface_samples"]["sha256"]=sha256(tmp_path/"surface_samples.jsonl")
    doc["coordinate_frame"]="camera"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="metres"):
        validate_geometry_evidence(p,root=tmp_path)


def test_runner_bv2_fixture_excluded_from_science(tmp_path):
    inp=tmp_path/"data"
    inp.mkdir()
    (inp/"v2.json").write_text(json.dumps(_b()))
    _csv(inp/"truth.csv",[{"activity_id":"a1","physical_progress_pct":"75",
                            "planned_percent_pct":"50","baseline_id":"b1"}])
    manifest={"schema_version":"mycon.validation.v1","samples":[{
        "id":"case","site_id":"site","building_id":"building","capture_session_id":"session",
        "source_sha256":"f"*64,"split":"test","data_kind":"synthetic",
        "ground_truth":{"activity_progress_gt_csv":"truth.csv"},
        "methods":{"b-v2":{"scientific_progress_json":"v2.json"}}}]}
    (inp/"manifest.json").write_text(json.dumps(manifest))
    r=run(inp/"manifest.json",tmp_path/"runs",run_id="bench-round2",include_fixtures=True)
    cases=json.loads((tmp_path/"runs"/"bench-round2"/"cases.json").read_text())
    assert next(c for c in cases if c["task"]=="activity_progress")["metrics"]["physical_progress_mae_pp"]==5
    assert r["scientific_status"]=="not_executed"
    assert json.loads((tmp_path/"runs"/"bench-round2"/"scientific_results.json").read_text())==[]


def test_nvs_heldout_pair_scoring_and_hash(tmp_path):
    import numpy as np
    from PIL import Image
    from benchmarks.mycon_2026.nvs import score_nvs_pairs
    a=tmp_path/"truth.png"
    b=tmp_path/"render.png"
    Image.fromarray(np.full((12,12,3),180,dtype=np.uint8)).save(a)
    Image.fromarray(np.full((12,12,3),178,dtype=np.uint8)).save(b)
    ref={"schema_version":"mycon.nvs_views.v1","views":[{
        "view_id":"test-1","image":a.name,"sha256":sha256(a),
        "capture_session_id":"session1","split":"test","held_out":True}]}
    pred={"schema_version":"mycon.nvs_views.v1","views":[{
        "view_id":"test-1","image":b.name,"sha256":sha256(b)}]}
    ra=tmp_path/"ref.json";rb=tmp_path/"prediction.json"
    ra.write_text(json.dumps(ref));rb.write_text(json.dumps(pred))
    metrics=score_nvs_pairs(ra,rb)
    assert metrics["n_paired_views"]==1 and metrics["n_missing_prediction"]==0
    assert metrics["psnr_db_mean_finite"]>30 and 0<metrics["ssim_mean"]<=1
    assert metrics["lpips_status"]=="not_executed" and metrics["lpips_mean"] is None
    assert len(metrics["image_hashes"])==2
    ref["views"][0]["held_out"]=False
    ra.write_text(json.dumps(ref))
    with pytest.raises(ValueError,match="held out"):
        score_nvs_pairs(ra,rb)


def test_numeric_gt_without_categorical_gt():
    gt=[{"global_id":"A","status":"unknown","completion_fraction":"0.55"}]
    pred=[{"global_id":"A","status":"unknown","physical_progress_estimate":"0.65"}]
    x=progress_metrics(pred,gt)
    assert x["n_gt_classification_available"]==0
    assert x["classification_coverage"] is None and x["n_progress_pairs"]==1
    assert x["progress_mae_fraction_known"]==pytest.approx(0.1)
