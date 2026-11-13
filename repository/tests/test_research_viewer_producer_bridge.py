"""Dedicated synthetic cross-producer contract tests (not a real MyCON site test)."""
import json
import struct
import zlib
import pytest
from viewer_civil.core import ViewerInputError, sha256, load_bundle
from viewer_civil.producer_bridge import convert_r_run, invert_colmap_world_to_camera

def _png(path,color):
    def ch(tag,data):
        return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    pixel=bytes([0,color,24,180])
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+ch(b"IHDR",struct.pack(">IIBBBBB",1,1,8,2,0,0,0))
                     +ch(b"IDAT",zlib.compress(pixel))+ch(b"IEND",b""))

@pytest.fixture
def rrun(tmp_path):
    root=tmp_path/"recon-test1";(root/"images"/"session").mkdir(parents=True)
    frames=[];poses=[]
    for i in range(2):
        path=root/"images"/"session"/f"image{i}.png";_png(path,90+i)
        frame={"view_id":f"view_{i}","image_relpath":f"images/session/image{i}.png","image_sha256":sha256(path),
               "width":1,"height":1,"capture_session_id":"session","split":"train" if i==0 else "test"}
        frames.append(frame)
        pose={k:frame[k] for k in ("view_id","image_relpath","image_sha256","capture_session_id","split")}
        pose.update({"registered":i==0,"coordinate_frame":"colmap_opencv_world_to_camera" if i==0 else "unknown",
              "units":"reconstruction_units_unscaled" if i==0 else "unknown",
              "camera":{"model":"PINHOLE","parameters":[1.,1.,.5,.5],"width":1,"height":1} if i==0 else None,
              "world_to_camera_4x4":[[0.,-1.,0.,2.],[1.,0.,0.,3.],[0.,0.,1.,4.],[0.,0.,0.,1.]] if i==0 else None})
        poses.append(pose)
    (root/"dataset_manifest.json").write_text(json.dumps({"schema_version":"mycon.reconstruction.dataset.v1",
        "run_id":root.name,"synthetic":True,"metric_scale_verified":False,"frames":frames}))
    (root/"camera_poses.json").write_text(json.dumps({"schema_version":"mycon.reconstruction.camera_poses.v1",
        "run_id":root.name,"poses":poses}))
    return root

def _change(path,callback):
    x=json.loads(path.read_text());callback(x);path.write_text(json.dumps(x))

def _a_source(tmp_path):
    root=tmp_path/"a";d=root/"runs"/"a"/"geometry";d.mkdir(parents=True)
    over=d/"viewer_overlay.json"
    over.write_text(json.dumps({"schema_version":"mycon.civil_geometry_review.v1","run_id":"a",
      "coordinate_frame":"colmap_world_unverified","units":"arbitrary_reconstruction_units",
      "points":[[0,0,1]],
      "planes":[{"plane_id":"p1","finite_boundary_xyz":[[0,0,0],[1,0,0],[0,1,0]],
                 "semantic_label":"unknown"}],
      "edges":[{"edge_id":"e1","endpoints_xyz":[[0,0,0],[1,0,0]]}],
      "corners":[{"corner_id":"c1","xyz":[0,0,0],"state":"candidate_extrapolated"}]}))
    man=d/"civil_geometry_manifest.json"
    man.write_text(json.dumps({"schema_version":"mycon.civil_geometry_review.v1",
      "run_id":"a","coordinate_frame":"colmap_world_unverified","units":"arbitrary_reconstruction_units",
      "outputs":{"viewer_overlay.json":{"relative_path":"runs/a/geometry/viewer_overlay.json",
                                         "sha256":sha256(over)}}}))
    return root,"runs/a/geometry/civil_geometry_manifest.json"

def _bridge(root,ap,rrun):
    proof=root/"proof.txt";proof.write_text("SYNTHETIC correspondence; NOT surveyed or physical evidence")
    rec={"schema_version":"mycon.viewer.frame_bridge.v1",
       "r_camera_poses_sha256":sha256(rrun/"camera_poses.json"),
       "a_manifest_sha256":sha256(root/ap),
       "a_coordinate_frame":"colmap_world_unverified",
       "r_pose_convention":"colmap_opencv_world_to_camera",
       "independent_review":True,"reviewer_id":"synthetic_test_only",
       "metric_scale_verified":False,"same_reconstruction_unit_ratio":1,
       "geometry_to_r_colmap_world_4x4":[[1,0,0,2],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
       "independent_evidence":{"path":"proof.txt","sha256":sha256(proof)}}
    (root/"bridge.json").write_text(json.dumps(rec))
    return "bridge.json"

def test_w2c_se3_inverse_round_trip():
    w=[[0,-1,0,2],[1,0,0,3],[0,0,1,4],[0,0,0,1]]
    inv=invert_colmap_world_to_camera(w)
    for i in range(4):
        for j in range(4):
            assert sum(w[i][k]*inv[k][j] for k in range(4))==pytest.approx(int(i==j),abs=1e-9)
    assert [inv[i][3] for i in range(3)]==pytest.approx([-3,2,-4])

@pytest.mark.parametrize("bad",[-1,2,float("nan")])
def test_reject_reflection_scale_nan(bad):
    w=[[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]]
    w[0][0]=bad
    with pytest.raises(ViewerInputError):invert_colmap_world_to_camera(w)

def test_source_frame_bytes_are_linked_to_registered_pose(rrun,tmp_path):
    output=tmp_path/"viewer-r-test";convert_r_run(rrun,output)
    b=load_bundle(output)
    assert b["manifest"]["geometry_alignment_status"]=="no_A_geometry_supplied"
    assert len(b["assets"]["camera_poses"]["cameras"])==1
    assert len(b["assets"]["mesh"]["faces"])==0
    assert b["manifest"]["excluded_views"]==[{"view_id":"view_1","reason":"not_registered"}]
    f=b["assets"]["frame_manifest"]["frames"][0]
    assert f["source_view_id"]=="view_0"
    assert sha256(output/f["path"])==sha256(rrun/"images/session/image0.png")
    assert b["assets"]["camera_poses"]["cameras"][0]["camera_to_world"][0][3]==pytest.approx(-3)
    assert b["deviation_enabled"] is False

def test_corrupted_source_fails_before_output_creation(rrun,tmp_path):
    (rrun/"images/session/image0.png").write_bytes(b"bad")
    target=tmp_path/"new-viewer"
    with pytest.raises(ViewerInputError,match="image SHA256 mismatch"):
        convert_r_run(rrun,target)
    assert not target.exists()

def test_pose_frame_split_mismatch_fails_closed(rrun,tmp_path):
    _change(rrun/"camera_poses.json",lambda d:d["poses"][0].update(split="test"))
    with pytest.raises(ViewerInputError,match="split conflict"):
        convert_r_run(rrun,tmp_path/"target")

def test_registered_pose_must_have_expected_r_convention(rrun,tmp_path):
    _change(rrun/"camera_poses.json",lambda d:d["poses"][0].update(coordinate_frame="unknown"))
    with pytest.raises(ViewerInputError,match="coordinate frame"):
        convert_r_run(rrun,tmp_path/"target")

def test_missing_registered_pose_not_fabricated(rrun,tmp_path):
    _change(rrun/"camera_poses.json",lambda d:d["poses"][0].update(registered=False))
    with pytest.raises(ViewerInputError,match="no registered cameras"):
        convert_r_run(rrun,tmp_path/"target")

def test_existing_bundle_never_overwritten(rrun,tmp_path):
    out=tmp_path/"viewer-r-test";convert_r_run(rrun,out)
    with pytest.raises(FileExistsError):convert_r_run(rrun,out)

def test_A_geometry_withheld_without_approved_transform(rrun,tmp_path):
    root,ap=_a_source(tmp_path)
    dest=tmp_path/"target";convert_r_run(rrun,dest,project_root=root,a_manifest=ap)
    b=load_bundle(dest)
    assert "withheld" in b["manifest"]["geometry_alignment_status"]
    assert b["assets"]["planes"]["planes"]==[]
    assert b["deviation_enabled"] is False

def test_A_source_hash_integrity_and_signed_witness_required(rrun,tmp_path):
    root,ap=_a_source(tmp_path)
    (root/"runs/a/geometry/viewer_overlay.json").write_text("tampered")
    with pytest.raises(ViewerInputError,match="overlay SHA256 mismatch"):
        convert_r_run(rrun,tmp_path/"target",project_root=root,a_manifest=ap)
    assert not (tmp_path/"target").exists()

def test_approved_test_only_nonmetric_A_overlay_keeps_features_unknown(rrun,tmp_path):
    root,ap=_a_source(tmp_path);proof=_bridge(root,ap,rrun)
    dest=tmp_path/"target";convert_r_run(rrun,dest,project_root=root,a_manifest=ap,frame_bridge=proof)
    b=load_bundle(dest)
    assert "not_surveyed" in b["manifest"]["geometry_alignment_status"]
    assert b["assets"]["corners"]["corners"][0]["xyz"]==[2,0,0]
    assert b["assets"]["corners"]["corners"][0]["confidence"] is None
    assert b["assets"]["planes"]["planes"][0]["state"]=="unknown"
    assert len(b["assets"]["point_cloud"]["points"])==1
    assert not b["deviation_enabled"]

def test_false_review_flag_does_not_enable_geometry(rrun,tmp_path):
    root,ap=_a_source(tmp_path);proof=_bridge(root,ap,rrun)
    _change(root/proof,lambda d:d.update(independent_review=False))
    with pytest.raises(ViewerInputError,match="not independently reviewed"):
        convert_r_run(rrun,tmp_path/"target",project_root=root,a_manifest=ap,frame_bridge=proof)

def test_nonzero_lens_distortion_marked_approximate(rrun,tmp_path):
    _change(rrun/"camera_poses.json",lambda d:d["poses"][0]["camera"].update(
      model="SIMPLE_RADIAL",parameters=[1.,.5,.5,.15]))
    dest=tmp_path/"target";convert_r_run(rrun,dest)
    b=load_bundle(dest)
    assert b["assets"]["camera_poses"]["cameras"][0]["frustum_projection"]=="approximate_ignore_distortion"
    assert b["assets"]["camera_poses"]["quality"]["frustum_distortion_approximation"] is True

def test_browser_camera_frame_selection_and_canvas(rrun,tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("playwright.sync_api")
    import shutil
    exe=shutil.which("chromium") or shutil.which("chromium-browser")
    if not exe:pytest.skip("chromium absent")
    from playwright.sync_api import sync_playwright
    from viewer_civil.core import build_preview
    from viewer_civil.browser import load_offline_page
    dest=tmp_path/"target";convert_r_run(rrun,dest)
    page_path=build_preview(dest,tmp_path/"preview")
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,executable_path=exe,
                   args=["--disable-gpu","--disable-webgl"])
        page=browser.new_page(viewport={"width":1440,"height":900})
        load_offline_page(page,page_path)
        assert page.locator("#selected-camera").inner_text()=="CAM-0001"
        assert "FRAME-0001" in page.locator("#frame-meta").inner_text()
        assert page.evaluate("document.querySelector('#selected-frame').naturalWidth")==1
        assert page.evaluate("window.MyCONViewer.state.layers.cameras") is True
        assert page.locator("#scene-canvas").count()==1
        assert page.evaluate("window.MyCONViewer.setLayer('deviation',true)") is False
        browser.close()
