"""Round 3 C-owned model-agnostic geometry readiness and VLM grounding regressions."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from benchmarks.mycon_2026.protocol import sha256
from benchmarks.mycon_2026.model_agnostic import evaluate_ready, load_ready
from benchmarks.mycon_2026.grounding import evaluate_grounding
from benchmarks.mycon_2026.runner import run

pytestmark = pytest.mark.lightweight

def fixture(tmp_path, *, slope=False, corner_state="observed_supported"):
    root=tmp_path
    z=0.3 if slope else 0.
    pts=[(0.,0.,0.),(1.,0.,z),(0.,1.,0.),(1.,1.,z)]
    clean=root/"clean.ply"
    clean.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n"+
        "".join("%s %s %s\n"%p for p in pts))
    n=[-.3,0,1] if slope else [0,0,1]
    patch={"id":"p","normal":n,"offset":0,"centroid":[.5,.5,z/2],
           "support_count":4,"support_extent":[[0,0,0],[1,1,z]]}
    edge={"id":"e","endpoints":[list(pts[0]),list(pts[1])],
          "plane_ids":["p","other"],"support_count":3}
    corner={"id":"c","xyz":list(pts[0]),"state":corner_state,
            "support_count":4 if corner_state=="observed_supported" else 0,
            "feature_ids":["p","other"]}
    refs={}
    for key, rows in (("planes",[patch]),("edges",[edge]),("corners",[corner])):
        dest=root/(key+".json")
        dest.write_text(json.dumps({key:rows}))
        refs[key]={"path":dest.name,"sha256":sha256(dest)}
    refs["clean_cloud"]={"path":clean.name,"sha256":sha256(clean)}
    m={"schema_version":"mycon.comparison_ready_geometry.v1",
       "run_id":"geom-ready-fixture","units":"arbitrary","coordinate_frame":"reconstruction",
       "input_point_count":5,"metric_scale_verified":False,"registration_verified":False,
       "outputs":refs}
    manifest=root/"comparison_ready.json";manifest.write_text(json.dumps(m))
    gt={"schema_version":"mycon.structural_gt.v1","units":"arbitrary",
        "coordinate_frame":"reconstruction",
        "points":[{"xyz":list(p),"class":"structure"} for p in pts]+
                 [{"xyz":[9,9,9],"class":"outlier"}],
        "planes":[patch],"edges":[edge],
        "corners":[{"id":"c_gt","xyz":list(pts[0]),"observable":True}]}
    truth=root/"truth.json";truth.write_text(json.dumps(gt))
    return manifest, truth

def test_cleaning_multimetric_no_bim(tmp_path):
    manifest,gt=fixture(tmp_path)
    x=evaluate_ready(manifest,gt,root=tmp_path)
    assert x["structure_retention_fraction"]==1
    assert x["outlier_removal_fraction"]==1
    assert x["plane_tp"]==x["edge_tp"]==x["corner_tp"]==1
    assert x["corner_unknown_rate"]==0
    assert "chamfer_l1_m" not in x and "chamfer_l1_source_units" in x
    assert not x["metric_scale_verified"] and not x["registration_verified"]

def test_sloped_planes_and_no_fake_corners(tmp_path):
    m,gt=fixture(tmp_path,slope=True,corner_state="candidate_extrapolated")
    scores=evaluate_ready(m,gt,root=tmp_path)
    assert scores["plane_tp"]==1
    assert scores["corner_tp"]==0 and scores["corner_fn"]==1
    assert scores["corner_n_unknown_or_extrapolated"]==1

def test_tampered_source_digest_fails_closed(tmp_path):
    m,gt=fixture(tmp_path)
    obj=json.loads(m.read_text())
    obj["outputs"]["planes"]["sha256"]="0"*64
    m.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="SHA"):
        load_ready(m,tmp_path)

def test_vlm_roi_abstention_and_unsourced_measurement():
    review=[{"view_id":"v1","feature_id":"p1","label":"plane","bbox":[.1,.1,.8,.8]}]
    claims=[{"claim_id":"a","view_id":"v1","feature_id":"p1","label":"plane",
             "bbox":[.1,.1,.8,.8]},
            {"claim_id":"b","view_id":"v1","feature_id":"p1","label":"plane",
             "bbox":[.1,.1,.8,.8],"assertion_type":"dimension","value":100,"unit":"m"},
            {"claim_id":"c","state":"abstained"}]
    x=evaluate_grounding(claims,review)
    assert x["n_supported"]==1 and x["n_hallucinated"]==1 and x["n_abstained"]==1

def test_runner_synthetic_never_publishes_real_results(tmp_path):
    m,gt=fixture(tmp_path)
    manifest={"schema_version":"mycon.validation.v1","samples":[{
        "id":"fixture","site_id":"fictional-site","building_id":"fictional-building",
        "capture_session_id":"fictional-session","source_sha256":"a"*64,
        "split":"test","data_kind":"synthetic",
        "ground_truth":{"structural_gt_json":gt.name},
        "methods":{"A-fixture":{"comparison_ready_manifest_json":m.name}}}]}
    path=tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    result=run(path,tmp_path.parent/"output",run_id="bench-bim-free-smoke",include_fixtures=True)
    assert result["scientific_status"]=="not_executed"
    assert json.loads((tmp_path.parent/"output"/"bench-bim-free-smoke"/"scientific_results.json").read_text())==[]
    cases=json.loads((tmp_path.parent/"output"/"bench-bim-free-smoke"/"cases.json").read_text())
    assert next(r for r in cases if r["task"]=="geometry_ready")["metrics"]["plane_tp"]==1
