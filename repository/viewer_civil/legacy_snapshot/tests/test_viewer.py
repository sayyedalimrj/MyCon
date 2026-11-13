import json, hashlib
from pathlib import Path
import pytest
from mycon_viewer import make_demo, load_bundle, ViewerInputError, build_preview
from mycon_viewer.browser import load_offline_page

@pytest.fixture
def demo(tmp_path):
    return make_demo(tmp_path/'source','viewer-test-0001')

def test_manifest_receipts_and_frame_links(demo):
    obj=load_bundle(demo)
    assert len(obj['assets']['camera_poses']['cameras'])==6
    assert len(obj['assets']['frame_manifest']['frames'])==6
    assert obj['deviation_enabled'] is False
    assert obj['deviation_reason']=='synthetic_dataset'

def test_fails_closed_on_tampered_file(demo):
    p=demo/'mesh.json';p.write_text(p.read_text()+' ',encoding='utf-8')
    with pytest.raises(ViewerInputError, match='SHA256 mismatch'): load_bundle(demo)

def test_rejects_parent_traversal(demo):
    p=demo/'viewer_manifest.json';m=json.loads(p.read_text())
    m['files']['mesh']['path']='../private/mesh.json'
    p.write_text(json.dumps(m))
    with pytest.raises(ViewerInputError, match='Unsafe relative path'):load_bundle(demo)

def test_rejects_units_mismatch(demo):
    p=demo/'viewer_manifest.json';m=json.loads(p.read_text());m['files']['edges']['units']='m';p.write_text(json.dumps(m))
    with pytest.raises(ViewerInputError,match='Frame/units mismatch'):load_bundle(demo)

def test_rejects_camera_frame_mismatch(demo):
    p=demo/'camera_poses.json';body=json.loads(p.read_text());body['cameras'][0]['frame_id']='MISSING';p.write_text(json.dumps(body))
    manifest=demo/'viewer_manifest.json';m=json.loads(manifest.read_text());m['files']['camera_poses']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();manifest.write_text(json.dumps(m))
    with pytest.raises(ViewerInputError,match='Camera ID or frame link invalid'):load_bundle(demo)

def test_no_implicit_overwrite(demo,tmp_path):
    with pytest.raises(FileExistsError): make_demo(demo)
    out=tmp_path/'preview';html=build_preview(demo,out)
    assert html.is_file()
    with pytest.raises(FileExistsError): build_preview(demo,out)

def test_browser_camera_selection_layers_and_gate(demo,tmp_path):
    from playwright.sync_api import sync_playwright
    html=build_preview(demo,tmp_path/'preview')
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--use-gl=angle','--use-angle=swiftshader','--enable-webgl'])
        page=browser.new_page(viewport={'width':1440,'height':900})
        load_offline_page(page,html)
        assert page.locator('#selected-camera').inner_text()=='CAM-001'
        page.locator('button[data-camera="CAM-004"]').click()
        assert page.locator('#selected-camera').inner_text()=='CAM-004'
        assert 'FRAME-004' in page.locator('#frame-meta').inner_text()
        page.locator('input[data-layer="corners"]').uncheck()
        assert page.evaluate('window.MyCONViewer.state.layers.corners') is False
        assert page.locator('input[data-layer="deviation"]').is_disabled()
        assert page.evaluate('window.MyCONViewer.setLayer("deviation",true)') is False
        page.locator('button[data-mode="coverage"]').click()
        assert page.evaluate('window.MyCONViewer.state.mode') == 'coverage'
        page.locator('input[data-layer="bim"]').check()
        assert page.evaluate('window.MyCONViewer.state.layers.bim') is True
        if page.locator('#scene-canvas').count():
            markers=page.evaluate('window.MyCONViewer.cameraMarkers()')
            marker=next(m for m in markers if m['id']=='CAM-005')
            page.locator('#scene-canvas').click(position={'x':marker['x'],'y':marker['y']})
            assert page.locator('#selected-camera').inner_text()=='CAM-005'
            from PIL import Image
            image_path=tmp_path/'actual_render.png'
            page.screenshot(path=str(image_path))
            assert len(Image.open(image_path).convert('RGB').getcolors(1500000) or []) > 40
        browser.close()
