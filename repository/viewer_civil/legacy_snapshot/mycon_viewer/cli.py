"""python -m mycon_viewer.cli demo|render|capture|video"""
from pathlib import Path
import argparse, subprocess
from .demo import make_demo
from .render import build_preview
from .browser import load_offline_page

def capture(html:Path, output:Path, video=False, seconds=8):
    from playwright.sync_api import sync_playwright
    if output.exists(): raise FileExistsError(f'Will not overwrite {output}')
    if video:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='mycon-viewer-record-') as tmp:
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--use-gl=angle','--use-angle=swiftshader','--enable-webgl'])
                page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
                load_offline_page(page,html)
                for i in range(seconds*8):
                    page.evaluate('(i)=>{window.MyCONViewer.orbit(i/8)}',i)
                    page.screenshot(path=str(Path(tmp)/f'{i:05d}.png'))
                browser.close()
            cmd=['ffmpeg','-y','-loglevel','error','-framerate','8','-i',str(Path(tmp)/'%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','22',str(output)]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            if proc.returncode: raise RuntimeError('ffmpeg failed: '+proc.stderr)
    else:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--use-gl=angle','--use-angle=swiftshader','--enable-webgl'])
            page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
            load_offline_page(page,html)
            page.screenshot(path=str(output)); browser.close()
    return output

def main():
    ap=argparse.ArgumentParser(description='MyCON Civil 3D Viewer, non-destructive')
    subs=ap.add_subparsers(dest='cmd',required=True)
    p=subs.add_parser('demo');p.add_argument('destination');p.add_argument('--run-id',default='viewer-synthetic-20261113')
    p=subs.add_parser('render');p.add_argument('source');p.add_argument('destination')
    p=subs.add_parser('capture');p.add_argument('html');p.add_argument('output')
    p=subs.add_parser('video');p.add_argument('html');p.add_argument('output');p.add_argument('--seconds',type=int,default=6)
    a=ap.parse_args()
    if a.cmd=='demo': print(make_demo(Path(a.destination),a.run_id))
    elif a.cmd=='render': print(build_preview(a.source,a.destination))
    else: print(capture(Path(a.html).resolve(),Path(a.output),a.cmd=='video',getattr(a,'seconds',6)))
if __name__=='__main__': main()
