"""Dedicated Civil Viewer CPU tests. No real source/checkpoint imports or writes."""
import hashlib
import json
from pathlib import Path
import shutil
import pytest

from viewer_civil import ViewerInputError, build_preview, load_bundle, make_demo


@pytest.fixture()
def source(tmp_path):
    return make_demo(tmp_path/"capture", "viewer-synthetic-unit-1")


def _retarget(source, role, change):
    path=source/(role+".json")
    data=json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data),encoding="utf-8")
    manifest_path=source/"viewer_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][role]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")


def test_read_only_bundle_and_camera_source_links(source):
    b=load_bundle(source)
    assert b["manifest"]["schema_version"]=="mycon.viewer_bundle.v1"
    assert len(b["assets"]["camera_poses"]["cameras"])==6
    assert len(b["assets"]["frame_manifest"]["frames"])==6
    assert b["deviation_enabled"] is False
    assert b["deviation_reason"]=="synthetic_dataset"


def test_sha256_tampering_rejected(source):
    path=source/"corners.json"
    path.write_text(path.read_text(encoding="utf-8")+" ")
    with pytest.raises(ViewerInputError,match="SHA256 mismatch"):
        load_bundle(source)


def test_path_traversal_rejected(source):
    path=source/"viewer_manifest.json"
    m=json.loads(path.read_text(encoding="utf-8"))
    m["files"]["mesh"]["path"]="../secret.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ViewerInputError,match="unsafe relative path"):
        load_bundle(source)


def test_symlink_escape_rejected(source,tmp_path):
    external=tmp_path/"private.json"
    external.write_text("{}")
    (source/"out.json").symlink_to(external)
    path=source/"viewer_manifest.json"
    m=json.loads(path.read_text(encoding="utf-8"))
    m["files"]["mesh"]["path"]="out.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ViewerInputError,match="path missing or escapes"):
        load_bundle(source)


def test_unit_conflict_rejected(source):
    path=source/"viewer_manifest.json"
    m=json.loads(path.read_text(encoding="utf-8"))
    m["files"]["edges"]["units"]="m"
    path.write_text(json.dumps(m))
    with pytest.raises(ViewerInputError,match="frame/units mismatch"):
        load_bundle(source)


def test_camera_frame_id_conflict_rejected(source):
    _retarget(source,"camera_poses",lambda d: d["cameras"][0].update(frame_id="WRONG"))
    with pytest.raises(ViewerInputError,match="camera/frame identifier"):
        load_bundle(source)


def test_non_rigid_camera_rejected(source):
    def alter(d):
        d["cameras"][0]["camera_to_world"][0][0]*=2
    _retarget(source,"camera_poses",alter)
    with pytest.raises(ViewerInputError,match="rigid 4x4"):
        load_bundle(source)


def test_missing_frame_or_corrupted_pixels_rejected(source):
    frame=source/"frames/frame_001.png"
    frame.write_bytes(frame.read_bytes()+b"x")
    with pytest.raises(ViewerInputError,match="source image SHA256 mismatch"):
        load_bundle(source)


def test_unknown_is_not_zero_or_progress(source):
    b=load_bundle(source)
    assert any(p["state"]=="unknown" and p["confidence"] is None for p in b["assets"]["planes"]["planes"])
    assert not b["deviation_enabled"]
    assert "physical_progress" not in b["manifest"]


def test_no_unintentional_overwrite(source,tmp_path):
    with pytest.raises(FileExistsError):
        make_demo(source)
    pytest.importorskip("plotly")
    out=tmp_path/"preview"
    page=build_preview(source,out)
    assert page.is_file()
    assert (out/"plotly.min.js").stat().st_size>1_000_000
    assert (out/"viewer_build.json").is_file()
    with pytest.raises(FileExistsError):
        build_preview(source,out)


def test_browser_frame_selection_toggles_and_canvas_fallback(source,tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("playwright.sync_api")
    if not (shutil.which("chromium") or shutil.which("chromium-browser")):
        pytest.skip("Chromium binary not available; still run dedicated CPU validation")
    from playwright.sync_api import sync_playwright
    from viewer_civil.browser import load_offline_page
    html=build_preview(source,tmp_path/"preview")
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path=shutil.which("chromium") or shutil.which("chromium-browser"),
                                  args=["--use-angle=swiftshader","--enable-webgl"])
        page=browser.new_page(viewport={"width":1440,"height":900})
        load_offline_page(page,html)
        assert page.locator("#selected-camera").inner_text()=="CAM-001"
        page.locator('button[data-camera="CAM-004"]').click()
        assert page.locator("#selected-camera").inner_text()=="CAM-004"
        assert "FRAME-004" in page.locator("#frame-meta").inner_text()
        page.locator('input[data-layer="corners"]').uncheck()
        assert page.evaluate("window.MyCONViewer.state.layers.corners") is False
        assert page.locator('input[data-layer="deviation"]').is_disabled()
        assert page.evaluate('window.MyCONViewer.setLayer("deviation",true)') is False
        page.locator('button[data-mode="coverage"]').click()
        assert page.evaluate("window.MyCONViewer.state.mode")=="coverage"
        page.locator('input[data-layer="bim"]').check()
        assert page.evaluate("window.MyCONViewer.state.layers.bim") is True
        page.screenshot(path=str(tmp_path/"synthetic_browser_capture.png"))
        browser.close()
