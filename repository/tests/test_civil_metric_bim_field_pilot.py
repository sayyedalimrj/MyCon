"""Synthetic field workflow QA; actual IFC backend optional, never field truth."""
import json
from pathlib import Path
import numpy as np
import pytest

from civil_metric_bim import core
from civil_metric_bim.field_run import run_field
from civil_metric_bim.ifc_inventory import mesh_record, selected_triangle, extract_inventory
from test_civil_metric_bim_synthetic import approval, references, span


def save(root, name, doc):
    p = root / name
    p.write_text(json.dumps(doc))
    return p


def asset(root, name):
    return {"path": name, "sha256": core.digest(root / name)}


def inputs(root):
    (root / "cloud.ply").write_text("synthetic placeholder only")
    (root / "survey.txt").write_text("synthetic review evidence only")
    geom = {"schema_version": "mycon.comparison_ready_geometry.v1",
            "coordinate_frame": "colmap_reconstruction_world", "metric_scale_verified": False,
            "outputs": {"cloud": asset(root, "cloud.ply")}}
    gp = save(root, "geometry.json", geom)
    r = references()
    r["references"].append(span("held2", 4, role="control"))
    r["geometry_manifest_sha256"] = core.digest(gp)
    for row in r["references"]:
        row.update(field_evidence=asset(root, "survey.txt"), measured_at="fixture-only",
                   instrument="synthetic", measurement_method="synthetic", operator="test",
                   endpoint_ids=[row["reference_id"] + "a", row["reference_id"] + "b"])
    save(root, "survey.json", r)
    return r, gp


def ifc_inputs(root, gp):
    (root / "room.ifc").write_text("fixture only, not IFC parser output")
    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1,2,1]], float)
    target = 2 * verts + [1,2,3]
    inv = {"schema_version": "mycon.ifc_inventory.v1", "coordinate_frame": "bim_world", "units": "m",
           "model": {**asset(root, "room.ifc"), "revision": "synthetic"},
           "elements": [mesh_record("synthetic-guid", "IfcWall", target, [[0,1,2]])]}
    ip = save(root, "inventory.json", inv)
    m = []
    for i, v in enumerate(verts):
        m.append({"match_id": str(i), "physical_feature_id": "corner"+str(i),
                  "endpoint_id": "corner-end"+str(i), "role": "fit" if i < 4 else "control",
                  "kind": "corner", "scan_xyz": v.tolist(), "vertex_index": i,
                  "ifc_global_id": "synthetic-guid", "observation_state": "observed_supported",
                  "approval": approval(), "field_evidence": asset(root, "survey.txt")})
    sel = {"schema_version": "mycon.ifc_field_selection.v1", "inventory_sha256": core.digest(ip),
           "geometry_manifest_sha256": core.digest(gp), "source_frame": "colmap_reconstruction_world",
           "matches": m, "planes": [{"plane_id":"triangle", "ifc_global_id":"synthetic-guid",
               "triangle_index":0, "normal_sign":1, "approval":approval(),
               "field_evidence":asset(root,"survey.txt")}],
           "samples": [{"sample_id":"observed", "scan_xyz":[.2,.2,.05], "bim_plane_id":"triangle",
                        "observation_state":"observed", "approval":approval(),
                        "field_evidence":asset(root,"survey.txt")},
                       {"sample_id":"unknown", "scan_xyz":[.2,.2,.05], "bim_plane_id":"triangle",
                        "observation_state":"occluded"}]}
    save(root,"selection.json",sel)
    return sel


def test_field_length_only(tmp_path):
    inputs(tmp_path)
    receipt = run_field(tmp_path, "metric-bim-field-01", "geometry.json", "survey.json", length_tolerance_m=.01)
    assert receipt["all_candidate_controls_within_tolerance"]
    assert len(receipt["checks"]) == 2
    assert receipt["metric_validation"] == "pending"
    assert not receipt["metric_scale_verified"]
    assert not receipt["registration_verified"]
    assert not receipt["can_feed_stage8"]


def test_field_control_error_does_not_certify(tmp_path):
    r, gp = inputs(tmp_path)
    r["references"][-1]["actual_length_m"] += .25
    save(tmp_path, "survey.json", r)
    receipt = run_field(tmp_path, "metric-bim-field-02", "geometry.json", "survey.json", length_tolerance_m=.01)
    assert not receipt["all_candidate_controls_within_tolerance"]
    assert receipt["metric_validation"] == "pending"


def test_field_requires_two_controls(tmp_path):
    r, _ = inputs(tmp_path)
    r["references"].pop()
    save(tmp_path,"survey.json",r)
    with pytest.raises(ValueError,match="independent holdouts"):
        run_field(tmp_path,"metric-bim-field-03","geometry.json","survey.json",length_tolerance_m=.01)


def test_tampered_measurement_receipt_rejected(tmp_path):
    inputs(tmp_path)
    (tmp_path/"survey.txt").write_text("changed")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        run_field(tmp_path,"metric-bim-field-04","geometry.json","survey.json",length_tolerance_m=.01)


def test_repeated_endpoint_rejected(tmp_path):
    r,_ = inputs(tmp_path)
    r["references"][-1]["endpoint_ids"][0] = r["references"][0]["endpoint_ids"][0]
    save(tmp_path,"survey.json",r)
    with pytest.raises(ValueError,match="separate physical groups"):
        run_field(tmp_path,"metric-bim-field-05","geometry.json","survey.json",length_tolerance_m=.01)


def test_ifc_combined_candidate_and_mask(tmp_path):
    _,gp = inputs(tmp_path)
    ifc_inputs(tmp_path,gp)
    r = run_field(tmp_path,"metric-bim-field-06","geometry.json","survey.json","inventory.json","selection.json",.01,.01)
    assert len(r["checks"]) == 4 and r["all_candidate_controls_within_tolerance"]
    assert not r["metric_coloring_allowed"]
    h = json.loads((tmp_path/"runs/metric-bim-field-06/metric_bim/deviation_heatmap.json").read_text())
    assert h["points"][0]["signed_distance_m"] == pytest.approx(.1)
    assert h["points"][1]["signed_distance_m"] is None
    assert h["points"][1]["valid_mask"] is False


def test_ifc_only_candidate(tmp_path):
    _,gp = inputs(tmp_path)
    ifc_inputs(tmp_path,gp)
    r = run_field(tmp_path,"metric-bim-field-07","geometry.json",None,"inventory.json","selection.json",None,.01)
    assert len(r["checks"]) == 2 and r["all_candidate_controls_within_tolerance"]
    assert not r["metric_scale_verified"]


def test_ifc_inventory_hash_mismatch(tmp_path):
    _,gp = inputs(tmp_path)
    ifc_inputs(tmp_path,gp)
    (tmp_path/"inventory.json").write_text((tmp_path/"inventory.json").read_text()+" ")
    with pytest.raises(ValueError,match="exact IFC inventory"):
        run_field(tmp_path,"metric-bim-field-08","geometry.json",None,"inventory.json","selection.json",None,.01)


def test_ifc_cross_role_endpoint_leakage(tmp_path):
    r,gp = inputs(tmp_path)
    s = ifc_inputs(tmp_path,gp)
    s["matches"][-1]["endpoint_id"] = r["references"][0]["endpoint_ids"][0]
    save(tmp_path,"selection.json",s)
    with pytest.raises(ValueError,match="cross-role"):
        run_field(tmp_path,"metric-bim-field-09","geometry.json","survey.json","inventory.json","selection.json",.01,.01)


def test_triangle_selection_no_extrapolated_polygon():
    e = mesh_record("x","IfcWall",[[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
    poly,n = selected_triangle(e,0,-1)
    assert len(poly) == 3 and n == [0,0,-1]
    with pytest.raises(ValueError):
        selected_triangle(e,1,1)


def test_native_ifc_backend_if_available(tmp_path):
    # This is a genuine small programmatically authored IFC, not a room survey.
    ifc = pytest.importorskip("ifcopenshell")
    import ifcopenshell.api
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject")
    ifcopenshell.api.run("unit.assign_unit", model)
    context = ifcopenshell.api.run("context.add_context",model,context_type="Model")
    body = ifcopenshell.api.run("context.add_context",model,context_type="Model",
                               context_identifier="Body",target_view="MODEL_VIEW",parent=context)
    wall = ifcopenshell.api.run("root.create_entity",model,ifc_class="IfcWall")
    rep = ifcopenshell.api.run("geometry.add_wall_representation",model,context=body,length=2.,height=3.,thickness=.2)
    ifcopenshell.api.run("geometry.assign_representation",model,product=wall,representation=rep)
    matrix = np.eye(4); matrix[:3,3] = [4,5,6]
    ifcopenshell.api.run("geometry.edit_object_placement",model,product=wall,matrix=matrix)
    model.write(str(tmp_path/"native.ifc"))
    result = extract_inventory(tmp_path,"native.ifc","synthetic",[wall.GlobalId],"native.json")
    v = np.array(result["elements"][0]["vertices_xyz_m"])
    assert np.ptp(v,axis=0) == pytest.approx([2,.2,3])
    assert v.min(axis=0) == pytest.approx([4,5,6])
    assert result["metric_validation"] == "pending"
