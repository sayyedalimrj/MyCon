#!/usr/bin/env python3
"""MyCON proven-video fallback: Recorder -> COLMAP -> official gsplat v1.5.3.

Inspired by the *method* evaluated in SpatialScene3D v10.5 (indoor phone video).
Executes original Apache-2.0 gsplat code without copying an unlicensed notebook.
Only nonmetric SfM geometry is offered to BIM until independent control gates pass.
Never overwrites Recorder, other experiments or preexisting model checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from mycon_capture_safe import prepare as prepare_capture
from mycon_thesis_one_run import find_capture, validate_attempt
from mycon_colmap_gsplat_env import setup as setup_gsplat, run_logged

DRIVE = Path("/content/drive/MyDrive/MyCon_R4")
LOCAL = Path("/content/MyCON_ColmapGsplat")
GSPLAT_TAG = "v1.5.3"
EXPECTED_FPS = 2.0
EXPECTED_WIDTH = 640
STEPS = 7000
MIN_REGISTERED = 60
MIN_FRACTION = .65


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()


def save_once(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise RuntimeError("IMMUTABLE_RECEIPT_CONFLICT: "+str(path))
        return
    tmp=path.with_name(path.name+".partial")
    if tmp.exists():raise RuntimeError("PARTIAL_RECEIPT_PRESERVED: "+str(tmp))
    try:
        tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def check_source(root: Path) -> None:
    if not (root/".git").exists() or not (root/"examples/simple_trainer.py").is_file():
        raise RuntimeError("OFFICIAL_GSPLAT_CHECKOUT_MISSING")
    origin=subprocess.check_output(["git","-C",str(root),"remote","get-url","origin"],text=True).strip()
    if origin.rstrip("/") not in ("https://github.com/nerfstudio-project/gsplat.git",
                                 "git@github.com:nerfstudio-project/gsplat.git"):
        raise RuntimeError("UNTRUSTED_GSPLAT_SOURCE")
    tag=subprocess.check_output(["git","-C",str(root),"describe","--tags","--exact-match","HEAD"],text=True).strip()
    if tag != GSPLAT_TAG:raise RuntimeError("GSPLAT_SOURCE_NOT_PINNED_V153")


def ensure_source(root: Path, log: Path) -> None:
    if not root.exists():
        run_logged(["git","clone","--depth","1","--branch",GSPLAT_TAG,
                    "https://github.com/nerfstudio-project/gsplat.git",str(root)],log)
    check_source(root)


def commands_for_scene(scene: Path, database: Path, sparse: Path) -> list[list[str]]:
    return [
        ["colmap","feature_extractor","--database_path",str(database),
         "--image_path",str(scene/"images"),"--ImageReader.camera_model","SIMPLE_RADIAL",
         "--ImageReader.single_camera","1","--SiftExtraction.use_gpu","0"],
        ["colmap","sequential_matcher","--database_path",str(database),
         "--SiftMatching.use_gpu","0"],
        ["colmap","mapper","--database_path",str(database),
         "--image_path",str(scene/"images"),"--output_path",str(sparse)]
    ]


def count_registered_images(images_txt: Path) -> int:
    if not images_txt.is_file():raise RuntimeError("COLMAP_IMAGES_TXT_MISSING")
    # COLMAP alternates camera lines and 2D observations; blank second
    # lines are legal, so simply counting nonempty lines gives wrong totals.
    cameras=[line for line in images_txt.read_text().splitlines()
             if re.match(r"^\s*\d+\s+",line) and
                re.search(r"\bframe_\d{6}\.jpg\s*$",line)]
    return len(cameras)


def colmap_gate(registered: int, frames: int) -> dict:
    if frames < 80:raise RuntimeError("VIDEO_TOO_FEW_PREPARED_FRAMES")
    fraction=registered/frames
    if registered < MIN_REGISTERED or fraction < MIN_FRACTION:
        raise RuntimeError(f"COLMAP_REGISTRATION_INSUFFICIENT: {registered}/{frames} ({fraction:.1%}); revise capture/frames before GPU training")
    return {"status":"COLMAP_REGISTERED","registered":registered,
            "frames":frames,"registration_fraction":round(fraction,4),
            "metric_scale_verified":False,"bim_progress_authorized":False}


def prepare_frames(video: Path, scene: Path, experiment: Path, log: Path) -> int:
    images=scene/"images"
    receipt=experiment/"frames_receipt.json"
    prior=sorted(images.glob("frame_*.jpg")) if images.is_dir() else []
    if receipt.exists():
        state=json.loads(receipt.read_text())
        if len(prior)!=state.get("count"):raise RuntimeError("PREPARED_IMAGES_INCOMPLETE_PRESERVED")
        if any(p.stat().st_size<1000 for p in prior):raise RuntimeError("PREPARED_FRAME_CORRUPT")
        return len(prior)
    if prior:raise RuntimeError("UNRECEIPTED_EXISTING_FRAMES_PRESERVED")
    images.mkdir(parents=True,exist_ok=True)
    cmd=["ffmpeg","-hide_banner","-nostdin","-loglevel","error","-noautorotate",
         "-i",str(video),"-vf",
         f"transpose=clock,fps={EXPECTED_FPS:g},scale={EXPECTED_WIDTH}:-2:flags=lanczos",
         "-q:v","2",str(images/"frame_%06d.jpg")]
    run_logged(cmd,log)
    frames=sorted(images.glob("frame_*.jpg"))
    if not (80<=len(frames)<=240) or any(p.stat().st_size<1000 for p in frames):
        raise RuntimeError("INCOMPLETE_96_SECOND_FRAME_EXTRACTION")
    save_once(receipt,{"count":len(frames),"fps":EXPECTED_FPS,"width":EXPECTED_WIDTH,
                       "first_sha256":digest(frames[0]),"last_sha256":digest(frames[-1]),
                       "capture_video_sha256":digest(video),"orientation":"CW_90_ONCE",
                       "source_recording_changed":False})
    return len(frames)


def prepare_colmap(scene: Path, experiment: Path, count: int, local: Path, log: Path) -> dict:
    receipt=experiment/"colmap_receipt.json"
    if receipt.exists():
        record=json.loads(receipt.read_text())
        for p in ("cameras.bin","images.bin","points3D.bin"):
            if not (scene/"sparse/0"/p).is_file():
                raise RuntimeError("COLMAP_RECEIPT_WITH_MISSING_MODEL")
        return colmap_gate(record["registered"],count)
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP_NOT_INSTALLED: Colab installs apt colmap in notebook bootstrap")
    sparse=scene/"sparse"
    if sparse.exists() and any(sparse.iterdir()):
        raise RuntimeError("PARTIAL_COLMAP_PRESERVED_NEW_ATTEMPT_REQUIRED")
    sparse.mkdir(parents=True,exist_ok=True)
    database=local/"colmap_cache"/(experiment.parent.name+"_"+experiment.name+".db")
    database.parent.mkdir(parents=True,exist_ok=True)
    for cmd in commands_for_scene(scene,database,sparse):
        run_logged(cmd,log)
    model=sparse/"0"
    if not all((model/n).is_file() for n in ("cameras.bin","images.bin","points3D.bin")):
        raise RuntimeError("COLMAP_SPARSE_MODEL_NOT_FOUND")
    txt=experiment/"colmap_text"
    txt.mkdir(parents=True,exist_ok=True)
    run_logged(["colmap","model_converter","--input_path",str(model),
                "--output_path",str(txt),"--output_type","TXT"],log)
    registered=count_registered_images(txt/"images.txt")
    gate=colmap_gate(registered,count)
    save_once(receipt,gate)
    return gate


def gsplat_command(py: Path, source: Path, scene: Path, result: Path) -> list[str]:
    return [str(py),str(source/"examples/simple_trainer.py"),"default",
            "--data_dir",str(scene),"--data_factor","1","--result_dir",str(result),
            "--test_every","8","--max_steps",str(STEPS),
            "--save_steps",str(STEPS),"--eval_steps",str(STEPS),
            "--ply_steps",str(STEPS),"--save_ply","--disable_viewer",
            "--packed"]


def evidence(result: Path) -> dict:
    checkpoints=sorted(result.glob("ckpts/ckpt_*.pt"))
    ply=sorted(result.glob("ply/point_cloud_*.ply"))
    videos=sorted(result.glob("videos/traj_*.mp4"))
    metrics=sorted(result.glob("stats/*.json"))+sorted(result.glob("*.json"))
    return {"checkpoints_present":bool(checkpoints),"checkpoint_paths":[str(p) for p in checkpoints],
            "ply_present":bool(ply),"ply_paths":[str(p) for p in ply],
            "novel_view_video_present":bool(videos),
            "novel_view_videos":[str(p) for p in videos],
            "evaluation_files":[str(p) for p in metrics],
            "metric_BIM_validated":False,
            "note":"Checkpoint/PLY presence alone never certifies model quality or metric geometry"}


def run(args: argparse.Namespace) -> dict:
    attempt=validate_attempt(args.attempt)
    drive=args.drive_root.expanduser().resolve()
    if args.mode=="status":
        folder=drive/"research_colmap_gsplat"
        reports=sorted(folder.glob(f"*/{attempt}/status.json"),key=lambda p:p.stat().st_mtime,reverse=True)
        if not reports:return {"status":"NO_ATTEMPT_YET","mode":"status"}
        doc=json.loads(reports[0].read_text())
        doc["evidence"]=evidence(reports[0].parent/"gsplat_result")
        return doc
    if not drive.is_dir():raise RuntimeError("DRIVE_NOT_MOUNTED")
    archive=find_capture(drive,args.capture)
    local=args.local_root.expanduser().resolve()
    if args.mode=="plan":
        return {"status":"PLAN_ONLY","backend":"COLMAP_GSPLAT_V153",
                "source_capture":str(archive),"fps":EXPECTED_FPS,"width":EXPECTED_WIDTH,
                "training_authorized":False}
    local.mkdir(parents=True,exist_ok=True)
    original=prepare_capture(archive,local/"verified_recorder_input",verified_rotation_cw=90)
    if not 90 <= original["full_video_duration_s"] <= 110:
        raise RuntimeError("WRONG_CAPTURE_EXPECT_96_SECOND_RECORDER")
    experiment=drive/"research_colmap_gsplat"/original["archive_sha256"][:16]/attempt
    experiment.mkdir(parents=True,exist_ok=True)
    statusfile=experiment/"status.json"
    inputrec={"source_sha256":original["archive_sha256"],"method":"COLMAP+gsplat",
              "gsplat_tag":GSPLAT_TAG,"fps":EXPECTED_FPS,"width":EXPECTED_WIDTH,
              "post_steps":STEPS,"metric_geometry_authorized":False}
    save_once(experiment/"input_contract.json",inputrec)
    scene=experiment/"scene"
    log=experiment/"logs"/"pipeline.log"
    n=prepare_frames(local/"verified_recorder_input/raw_capture.mp4",scene,experiment,log)
    gate=prepare_colmap(scene,experiment,n,local,log)
    gs_source=local/"official_gsplat_v153"
    ensure_source(gs_source,log)
    if args.mode=="preflight":
        # Explicit GPU prerequisites are checked now, before training.
        py=setup_gsplat(local/"gsplat_py310_torch24_cu121",local,gs_source,log)
        result={"status":"PREFLIGHT_COLMAP_AND_GSPLAT_CUDA_PASSED_NO_TRAINING",
                "registered":gate["registered"],"frames":n,
                "cuda_python":str(py),"source_sha256":original["archive_sha256"],
                "training_executed":False,"bim_metric_validated":False}
        if statusfile.exists():
            old=json.loads(statusfile.read_text())
            if old.get("source_sha256")!=result["source_sha256"]:
                raise RuntimeError("OLD_STATUS_DIFFERENT_CAPTURE")
        statusfile.write_text(json.dumps(result,indent=2)+"\n")
        return result
    if args.mode!="all":raise RuntimeError("UNKNOWN_MODE")
    if not args.approve_training:raise RuntimeError("TRAINING_NOT_APPROVED_USE_EXPLICIT_ALL")
    py=setup_gsplat(local/"gsplat_py310_torch24_cu121",local,gs_source,log)
    resultdir=experiment/"gsplat_result"
    if resultdir.exists() and any(resultdir.iterdir()):
        raise RuntimeError("EXISTING_GSPLAT_OUTPUT_PRESERVED: use a distinct --attempt; cannot fake resume")
    resultdir.mkdir(parents=True,exist_ok=True)
    run_logged(gsplat_command(py,gs_source,scene,resultdir),experiment/"logs"/"train.log",
               cwd=gs_source/"examples",live=True)
    artifacts=evidence(resultdir)
    if (not artifacts["checkpoints_present"] or not artifacts["ply_present"]
            or not artifacts["novel_view_video_present"]):
        raise RuntimeError("GSPLAT_TRAIN_FINISHED_BUT_CHECKPOINT_PLY_OR_NOVEL_VIEW_MISSING")
    result={"status":"GSPLAT_TRAINING_OUTPUTS_EXIST_NOT_BIM_METRIC",
            "registered":gate["registered"],"frames":n,
            "source_sha256":original["archive_sha256"],
            "evidence":artifacts,"bim_metric_validated":False}
    statusfile.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("plan","preflight","all","status"),default="preflight")
    p.add_argument("--capture",type=Path)
    p.add_argument("--drive-root",type=Path,default=DRIVE)
    p.add_argument("--local-root",type=Path,default=LOCAL)
    p.add_argument("--attempt",default="colmap_gsplat_baseline_01")
    p.add_argument("--approve-training",action="store_true")
    a=p.parse_args(argv)
    print(json.dumps(run(a),indent=2,ensure_ascii=False))
    return 0


if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception as exc:
        print("MYCON_COLMAP_GSPLAT_STOPPED:",type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(2)
