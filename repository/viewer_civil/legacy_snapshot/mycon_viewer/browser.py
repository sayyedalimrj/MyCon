"""Headless browser driver for restricted/offline environments (no localhost/file navigation)."""
from pathlib import Path
import base64, json

def load_offline_page(page, preview:Path):
    """Load the SAME shipped Plotly, styles and viewer scripts entirely from local bytes.

    In constrained browser sandboxes file:// and localhost navigation may be disabled;
    Playwright set_content and script injection make the actual app inspectable.
    """
    preview=Path(preview)
    html=preview.read_text(encoding='utf-8')
    html=html.replace('<link rel="stylesheet" href="viewer.css">','')
    html=html.replace('<script src="plotly.min.js"></script>','').replace('<script src="viewer.js"></script>','')
    start='<script type="application/json" id="mycon-payload">'
    before, rest=html.split(start,1)
    payload_text,after=rest.split('</script>',1)
    payload=json.loads(payload_text)
    for frame in payload['assets']['frame_manifest']['frames']:
        file=preview.parent/frame['preview_path']
        if not file.resolve().is_relative_to(preview.parent.resolve()):
            raise ValueError('Preview image escaped output folder')
        frame['preview_path']='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode('ascii')
    html=before+start+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script>'+after
    page.set_content(html,wait_until='load')
    page.add_style_tag(content=(preview.parent/'viewer.css').read_text(encoding='utf-8'))
    page.add_script_tag(content=(preview.parent/'plotly.min.js').read_text(encoding='utf-8'))
    page.add_script_tag(content=(preview.parent/'viewer.js').read_text(encoding='utf-8'))
    page.wait_for_function('window.__VIEWER_READY__ === true || Boolean(window.__VIEWER_ERROR__)',timeout=60000)
    error=page.evaluate('window.__VIEWER_ERROR__')
    if error:
        raise RuntimeError(f'Viewer rendering error: {error}')
