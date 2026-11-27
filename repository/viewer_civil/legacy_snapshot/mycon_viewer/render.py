"""Render standalone offline preview. Strictly reads source; creates a NEW output directory."""
from pathlib import Path
import json, shutil
from PIL import Image
import plotly
from .io import load_bundle, safe_relative

STATIC=Path(__file__).parent/'static'

def build_preview(source: str|Path, destination: str|Path) -> Path:
    bundle=load_bundle(source)
    out=Path(destination)
    out.mkdir(parents=True,exist_ok=False)
    (out/'frames').mkdir()
    frames=bundle['assets']['frame_manifest']['frames']
    for frame in frames:
        path=safe_relative(bundle['root'],frame['path'])
        with Image.open(path) as im:
            im=im.convert('RGB'); im.thumbnail((960,540), Image.Resampling.LANCZOS)
            im.save(out/'frames'/f"{frame['frame_id']}.png", optimize=True)
        frame['preview_path']=f"frames/{frame['frame_id']}.png"
    payload={k:v for k,v in bundle.items() if k!='root'}
    js=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(STATIC/'index.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('__MYCON_PAYLOAD__',js),encoding='utf-8')
    shutil.copy(STATIC/'viewer.js',out/'viewer.js')
    shutil.copy(STATIC/'viewer.css',out/'viewer.css')
    shutil.copy(Path(plotly.__file__).parent/'package_data'/'plotly.min.js',out/'plotly.min.js')
    (out/'viewer_build.json').write_text(json.dumps({'run_id':bundle['manifest']['run_id'], 'schema_version':'mycon.viewer_preview.v1',
        'source_manifest_sha256':__import__('hashlib').sha256((Path(source)/'viewer_manifest.json').read_bytes()).hexdigest(),
        'metric_deviation_enabled':bundle['deviation_enabled'], 'reason':bundle['deviation_reason']},indent=2),encoding='utf-8')
    return out/'index.html'
