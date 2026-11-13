"""Run: python -m viewer_civil demo|render|capture|video (from repository/)."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from .core import build_preview
from .demo import make_demo
from .browser import load_offline_page


def capture(html, target, video=False, seconds=4):
    from playwright.sync_api import sync_playwright
    target=Path(target)
    if target.exists():
        raise FileExistsError(target)
    chrome=shutil.which("chromium") or shutil.which("chromium-browser")
    if video and not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg executable is required for video output")
    with tempfile.TemporaryDirectory(prefix="mycon-viewer-video-") as tmp:
        with sync_playwright() as p:
            options={"headless":True,"args":["--use-angle=swiftshader","--enable-webgl"]}
            if chrome: options["executable_path"]=chrome
            browser=p.chromium.launch(**options)
            page=browser.new_page(viewport={"width":1440,"height":900},device_scale_factor=1)
            load_offline_page(page,html)
            if video:
                for i in range(seconds*8):
                    page.evaluate("(i)=>window.MyCONViewer.orbit(i/8)",i)
                    page.screenshot(path=str(Path(tmp)/f"{i:05d}.png"))
            else:
                page.screenshot(path=str(target))
            browser.close()
        if video:
            cmd=["ffmpeg","-nostdin","-loglevel","error","-framerate","8",
                 "-i",str(Path(tmp)/"%05d.png"),"-c:v","libx264",
                 "-pix_fmt","yuv420p","-crf","24",str(target)]
            subprocess.run(cmd,check=True)
    return target


def main():
    cli=argparse.ArgumentParser(description="Independent read-only MyCON Civil Viewer")
    sub=cli.add_subparsers(dest="mode",required=True)
    d=sub.add_parser("demo");d.add_argument("destination");d.add_argument("--run-id",default="viewer-synthetic-20261114")
    r=sub.add_parser("render");r.add_argument("source");r.add_argument("destination")
    for key in ("capture","video"):
        s=sub.add_parser(key);s.add_argument("html");s.add_argument("destination")
        if key=="video":s.add_argument("--seconds",type=int,default=4)
    args=cli.parse_args()
    if args.mode=="demo": result=make_demo(args.destination,args.run_id)
    elif args.mode=="render": result=build_preview(args.source,args.destination)
    else: result=capture(args.html,args.destination,args.mode=="video",getattr(args,"seconds",4))
    print(result)


if __name__=="__main__": main()
