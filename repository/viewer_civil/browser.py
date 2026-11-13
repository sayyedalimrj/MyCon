"""Load an offline preview in browsers that cannot navigate file:// or localhost."""
import base64
import json
from pathlib import Path


def load_offline_page(page, html):
    html = Path(html)
    root = html.parent
    markup = html.read_text(encoding="utf-8")
    markup = markup.replace('<link rel="stylesheet" href="viewer.css">', "")
    markup = markup.replace('<script src="plotly.min.js"></script>', "")
    markup = markup.replace('<script src="viewer.js"></script>', "")
    anchor = '<script id="mycon-payload" type="application/json">'
    before, following = markup.split(anchor, 1)
    payload, after = following.split("</script>", 1)
    body = json.loads(payload)
    for frame in body["assets"]["frame_manifest"]["frames"]:
        preview = (root/frame["preview_path"]).resolve()
        if not preview.is_relative_to(root.resolve()):
            raise ValueError("frame outside preview")
        mime = "image/png" if preview.suffix.lower()==".png" else "image/jpeg"
        frame["preview_path"]="data:"+mime+";base64,"+base64.b64encode(preview.read_bytes()).decode("ascii")
    payload=json.dumps(body,ensure_ascii=False).replace("<","\\u003c")
    page.set_content(before+anchor+payload+"</script>"+after,wait_until="load")
    page.add_style_tag(content=(root/"viewer.css").read_text(encoding="utf-8"))
    page.add_script_tag(content=(root/"plotly.min.js").read_text(encoding="utf-8"))
    page.add_script_tag(content=(root/"viewer.js").read_text(encoding="utf-8"))
    page.wait_for_function("window.__VIEWER_READY__ || window.__VIEWER_ERROR__", timeout=60000)
    error = page.evaluate("window.__VIEWER_ERROR__")
    if error:
        raise RuntimeError("Viewer failed: "+error)
