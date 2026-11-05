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
from mycon_gsplat_resume_patch import ensure_trainer
from mycon_gsplat_finish import render_and_evaluate_if_missing

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


def supported_colmap_commands(commands: list[list[str]]) -> list[list[str]]:
    """Adapt CPU SIFT flags to installed COLMAP CLI; reject unknown variant."""
    replacements={
        "feature_extractor":("--SiftExtraction.use_gpu","--FeatureExtraction.use_gpu"),
        "sequential_matcher":("--SiftMatching.use_gpu","--FeatureMatching.use_gpu"),
    }
    resolved=[]
    for cmd in commands:
        step=cmd[1]
        if step in replacements:
            probe=subprocess.run(["colmap",step,"-h"],
                                 capture_output=True,text=True,timeout=30)
            helptext=probe.stdout+"\\n"+probe.stderr
            old,new=replacements[step]
            if old in helptext:
                resolved.append(cmd)
            elif new in helptext:
                resolved.append([new if v==old else v for v in cmd])
            else:
                raise RuntimeError("COLMAP_SIFT_CPU_FLAG_UNSUPPORTED_FOR_INSTALLED_VERSION:"+step)
        else:resolved.append(cmd)
    return resolved


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


def verify_prepared_frames(scene: Path, experiment: Path) -> int:
    """Reuse Drive frames after Runtime reset; NEVER reopen a vanished local MP4."""
    receipt=experiment/"frames_receipt.json"
    if not receipt.is_file():
        raise RuntimeError("CPU_FRAMES_RECEIPT_MISSING_ON_DRIVE")
    row=json.loads(receipt.read_text())
    images=sorted((scene/"images").glob("frame_*.jpg"))
    if len(images)!=row.get("count") or not 80 <= len(images) <= 240:
        raise RuntimeError("CPU_FRAMES_INCOMPLETE_AFTER_RUNTIME_SWITCH")
    if digest(images[0])!=row.get("first_sha256") or digest(images[-1])!=row.get("last_sha256"):
        raise RuntimeError("CPU_FRAMES_SHA_MISMATCH_AFTER_RUNTIME_SWITCH")
    if any(p.stat().st_size<1000 for p in images):
        raise RuntimeError("CPU_FRAMES_ZERO_OR_CORRUPTED")
    return len(images)


def local_training_scene(scene: Path, experiment: Path, local: Path) -> Path:
    """Copy only registered COLMAP images+model onto fast /content local disk.

    Persistent Drive scene remains immutable; a fresh GPU VM is allowed to
    rebuild this disposable cache without repeating COLMAP or video extraction.
    """
    model=scene/"sparse/0"
    if not all((model/n).is_file() for n in ("cameras.bin","images.bin","points3D.bin")):
        raise RuntimeError("CPU_COLMAP_MODEL_MISSING_ON_DRIVE")
    dest=local/"scene_cache"/experiment.parent.name/experiment.name
    dest_images=dest/"images"
    dest_sparse=dest/"sparse/0"
    names=[]
    path=experiment/"colmap_text/images.txt"
    if not path.is_file():raise RuntimeError("CPU_COLMAP_IMAGES_LIST_MISSING")
    for line in path.read_text().splitlines():
        match=re.search(r"\b(frame_\d{6}\.jpg)\s*$",line)
        if match and re.match(r"^\s*\d+\s+",line):
            names.append(match.group(1))
    if len(names)!=len(set(names)) or len(names)<MIN_REGISTERED:
        raise RuntimeError("COLMAP_REGISTERED_IMAGE_NAMES_INVALID")
    # Disk/space check protects against failed half-copy and CUDA env crowding.
    required=sum((scene/"images"/n).stat().st_size for n in names)
    required+=sum((model/n).stat().st_size for n in ("cameras.bin","images.bin","points3D.bin"))
    available=shutil.disk_usage(local).free
    if available < required+18*(1024**3):
        raise RuntimeError(f"INSUFFICIENT_LOCAL_DISK_FOR_CUDA_AND_SCENE: need {(required+18*1024**3)/1024**3:.1f} GiB")
    dest_images.mkdir(parents=True,exist_ok=True)
    dest_sparse.mkdir(parents=True,exist_ok=True)
    for name in names:
        source=scene/"images"/name
        target=dest_images/name
        if target.exists():
            if target.stat().st_size!=source.stat().st_size:
                raise RuntimeError("GPU_LOCAL_CACHE_IMAGE_MISMATCH_PRESERVE")
        else:shutil.copy2(source,target)
    for name in ("cameras.bin","images.bin","points3D.bin"):
        source=model/name;target=dest_sparse/name
        if target.exists():
            if digest(target)!=digest(source):
                raise RuntimeError("GPU_LOCAL_CACHE_COLMAP_MISMATCH_PRESERVE")
        else:shutil.copy2(source,target)
    # pycolmap's images all refer to original registered names
    if set(p.name for p in dest_images.glob("frame_*.jpg"))!=set(names):
        raise RuntimeError("GPU_CACHE_HAS_UNEXPECTED_OR_MISSING_IMAGES")
    return dest


def prepare_colmap(scene: Path, experiment: Path, count: int, local: Path, log: Path) -> dict:
    receipt=experiment/"colmap_receipt.json"
    if receipt.exists():
        record=json.loads(receipt.read_text())
        for p in ("cameras.bin","images.bin","points3D.bin"):
            if not (scene/"sparse/0"/p).is_file():
                raise RuntimeError("COLMAP_RECEIPT_WITH_MISSING_MODEL")
        if not (experiment/"bim_candidate"/"colmap_sparse_unscaled.ply").is_file():
            raise RuntimeError("COLMAP_RECEIPT_WITH_MISSING_BIM_CANDIDATE")
        return colmap_gate(record["registered"],count)
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP_NOT_INSTALLED: Colab installs apt colmap in notebook bootstrap")
    sparse=scene/"sparse"
    if sparse.exists() and any(sparse.iterdir()):
        raise RuntimeError("PARTIAL_COLMAP_PRESERVED_NEW_ATTEMPT_REQUIRED")
    sparse.mkdir(parents=True,exist_ok=True)
    database=local/"colmap_cache"/(experiment.parent.name+"_"+experiment.name+".db")
    database.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env["QT_QPA_PLATFORM"]="offscreen"
    env["OMP_NUM_THREADS"]=str(min(4,max(1,os.cpu_count() or 2)))
    for cmd in supported_colmap_commands(commands_for_scene(scene,database,sparse)):
        run_logged(cmd,log,env=env)
    model=sparse/"0"
    if not all((model/n).is_file() for n in ("cameras.bin","images.bin","points3D.bin")):
        raise RuntimeError("COLMAP_SPARSE_MODEL_NOT_FOUND")
    txt=experiment/"colmap_text"
    txt.mkdir(parents=True,exist_ok=True)
    run_logged(["colmap","model_converter","--input_path",str(model),
                "--output_path",str(txt),"--output_type","TXT"],log)
    registered=count_registered_images(txt/"images.txt")
    gate=colmap_gate(registered,count)
    # This is separate COLMAP SfM geometry, NOT the photorealistic Gaussian PLY.
    # Unit scale is arbitrary until independently surveyed control anchors
    # establish the COLMAP-to-BIM similarity transform and held-out residuals.
    candidate=experiment/"bim_candidate"/"colmap_sparse_unscaled.ply"
    candidate.parent.mkdir(parents=True,exist_ok=True)
    if candidate.exists():raise RuntimeError("BIM_CANDIDATE_ALREADY_EXISTS_WITHOUT_RECEIPT")
    run_logged(["colmap","model_converter","--input_path",str(model),
                "--output_path",str(candidate),"--output_type","PLY"],log)
    if not candidate.is_file() or candidate.stat().st_size<512:
        raise RuntimeError("COLMAP_INDEPENDENT_SFM_CANDIDATE_EMPTY")
    save_once(experiment/"bim_candidate"/"stage8_handoff.json",{
        "schema":"mycon.stage8.geometry_candidate.v1",
        "candidate_ply":str(candidate),
        "origin":"COLMAP_SFM_SPARSE_POINTS_NOT_RENDERER_DEPTH",
        "unit_system":"ARBITRARY_SCALE_NONMETRIC",
        "independent_site_anchors_present":False,
        "stage8_registration_approved":False,
        "stage9_progress_approved":False,
        "required_next_steps":[
            "collect independent site/BIM points",
            "align to BIM with held-out metric validation",
            "run geometric coverage and per-element residual checks"]
    })
    save_once(receipt,gate)
    return gate


def gsplat_command(py: Path, source: Path, scene: Path, result: Path,
                   trainer: Path | None=None) -> list[str]:
    return [str(py),str(trainer or (source/"examples/simple_trainer.py")),"default",
            "--data_dir",str(scene),"--data_factor","1","--result_dir",str(result),
            "--test_every","8","--max_steps",str(STEPS),
            "--save_steps",str(STEPS),"--eval_steps",str(STEPS),
            "--ply_steps",str(STEPS),"--save_ply","--disable_viewer",
            "--packed"]


def real_video_cuda_smoke(py: Path, source: Path, scene: Path,
                          experiment: Path, trainer: Path) -> dict:
    """20 actual optimizer steps on real registered phone frames, not fake Gaussians.

    Uses /content local output and never touches full 7000-step result root.
    This does consume a small amount of GPU time; it is NOT a zero-GPU test.
    """
    smoke=Path("/content/MyCON_ColmapGsplat")/"short_real_video_smoke"/experiment.parent.name/experiment.name
    marker=smoke/"mycon_20_step_result.json"
    if marker.is_file():
        row=json.loads(marker.read_text())
        if row.get("status")!="REAL_20_STEP_VIDEO_T4_PASS":
            raise RuntimeError("CUDA_SMOKE_MARKER_MISMATCH")
        return row
    if smoke.exists() and any(smoke.iterdir()):
        raise RuntimeError("INTERRUPTED_20_STEP_GPU_SMOKE_PRESERVED_CHANGE_RUNTIME_OR_INSPECT_LOG")
    smoke.mkdir(parents=True,exist_ok=True)
    cmd=gsplat_command(py,source,scene,smoke,trainer)
    # Official gsplat's short run is deliberately no PLY/no video/no heldout.
    # The 20-step optimizer loop validates pycolmap camera parsing,
    # rasterization, real pixel batches, backward, and checkpoint writes.
    cmd[cmd.index("--max_steps")+1]="20"
    cmd[cmd.index("--save_steps")+1]="20"
    cmd[cmd.index("--eval_steps")+1]="-1"
    cmd[cmd.index("--ply_steps")+1]="-1"
    cmd.remove("--save_ply")
    cmd.append("--disable_video")
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE="1",MAX_JOBS="2",
               TORCH_CUDA_ARCH_LIST="7.5",MYCON_RESUME_EVERY="0")
    prefix=Path(py).parent.parent
    env["CUDA_HOME"]=str(prefix)
    env["PATH"]=str(prefix/"bin")+os.pathsep+env.get("PATH","")
    run_logged(cmd,experiment/"logs"/"real_video_20_step_smoke.log",
               cwd=source/"examples",env=env,live=True)
    stats=sorted((smoke/"stats").glob("train_step*.json"))
    ckpts=sorted((smoke/"ckpts").glob("ckpt_*_rank0.pt"))
    if not stats or not ckpts:
        raise RuntimeError("GPU_REAL_VIDEO_OPTIMIZER_SMOKE_OUTPUTS_MISSING")
    measured=json.loads(stats[-1].read_text())
    mem=measured.get("mem")
    if not isinstance(mem,(int,float)) or not 0<mem<13.5:
        raise RuntimeError("T4_SMOKE_MEMORY_EXCEEDS_CONSERVATIVE_13_5_GIB")
    result={"status":"REAL_20_STEP_VIDEO_T4_PASS",
            "real_registered_video_images":True,
            "max_gpu_allocated_gib":round(float(mem),3),
            "steps":20,"full_training_executed":False}
    marker.write_text(json.dumps(result,indent=2)+"\n")
    return result


def evidence(result: Path) -> dict:
    checkpoints=sorted(result.glob("ckpts/ckpt_*.pt"))
    ply=sorted(result.glob("ply/point_cloud_*.ply"))
    videos=sorted(result.glob("videos/traj_*.mp4"))
    metrics=sorted(result.glob("stats/val_step*.json"))
    measured={}
    if metrics:
        report=json.loads(metrics[-1].read_text())
        for field in ("psnr","ssim","lpips","num_GS"):
            value=report.get(field)
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                measured[field]=value
    return {"checkpoints_present":bool(checkpoints),"checkpoint_paths":[str(p) for p in checkpoints],
            "ply_present":bool(ply),"ply_paths":[str(p) for p in ply],
            "novel_view_video_present":bool(videos),
            "novel_view_videos":[str(p) for p in videos],
            "heldout_evaluation_present":bool(metrics),
            "heldout_measured_metrics":measured,
            "evaluation_files":[str(p) for p in metrics],
            "metric_BIM_validated":False,
            "note":"Checkpoint/PLY presence alone never certifies model quality or metric geometry"}


def verify_original_archive(archive: Path, local: Path) -> dict:
    """No GPU, no pip, no apt: validate native Recorder contract and disk."""
    import zipfile
    from mycon_capture_safe import source_members
    if archive.stat().st_size < 2_000_000:
        raise RuntimeError("ORIGINAL_RECORDER_ZIP_SUSPICIOUSLY_SMALL")
    if not local.parent.exists():
        # /content should exist in Colab; don't create a mutable workdir
        raise RuntimeError("LOCAL_CONTENT_ROOT_MISSING")
    free=shutil.disk_usage(local.parent).free
    with zipfile.ZipFile(archive) as z:
        _prefix,_entries,_mapping,record=source_members(z)
        video=z.getinfo(record["video_full"])
        if video.file_size < 25_000_000:
            raise RuntimeError("RECORDER_FULL_VIDEO_TOO_SMALL")
        if free < video.file_size*2+2_000_000_000:
            raise RuntimeError("CPU_DISK_NOT_ENOUGH_FOR_VERIFIED_INPUT_AND_FRAMES")
        if z.testzip() is not None:
            raise RuntimeError("RECORDER_ZIP_CRC_FAILURE")
    return {"status":"CPU_SOURCE_AND_RECORDER_SCHEMA_PASS_NOT_GPU_OR_TRAINING",
            "archive":str(archive),"archive_sha256":digest(archive),
            "archive_bytes":archive.stat().st_size,
            "video_bytes":video.file_size,
            "local_free_gib":round(free/2**30,2),
            "training_executed":False,"cuda_compatibility_checked":False,
            "metric_BIM_validated":False}


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
    if args.mode=="verify":
        return verify_original_archive(archive,local)
    if args.mode in ("preflight","all"):
        # CPU geometry persisted the fully verified source receipts on Drive.
        # In a fresh GPU VM avoid decompressing the 436MB video a second time.
        # We still rehash the immutable ZIP to rule out a same-name replacement.
        archive_sha=digest(archive)
        experiment=drive/"research_colmap_gsplat"/archive_sha[:16]/attempt
        cpu_receipt=experiment/"verified_capture_cpu.json"
        if not cpu_receipt.is_file():
            raise RuntimeError("CPU_SOURCE_AND_GEOMETRY_RECEIPT_MISSING_BEFORE_GPU")
        original=json.loads(cpu_receipt.read_text())
        if (original.get("archive_sha256")!=archive_sha
            or not 90 <= original.get("full_video_duration_s",0) <= 110):
            raise RuntimeError("CPU_CAPTURE_SHA_OR_DURATION_CHANGED")
    else:
        local.mkdir(parents=True,exist_ok=True)
        original=prepare_capture(archive,local/"verified_recorder_input",verified_rotation_cw=90)
        if not 90 <= original["full_video_duration_s"] <= 110:
            raise RuntimeError("WRONG_CAPTURE_EXPECT_96_SECOND_RECORDER")
        experiment=drive/"research_colmap_gsplat"/original["archive_sha256"][:16]/attempt
        if args.mode=="geometry":
            save_once(experiment/"verified_capture_cpu.json",original)
    experiment.mkdir(parents=True,exist_ok=True)
    statusfile=experiment/"status.json"
    inputrec={"source_sha256":original["archive_sha256"],"method":"COLMAP+gsplat",
              "gsplat_tag":GSPLAT_TAG,"fps":EXPECTED_FPS,"width":EXPECTED_WIDTH,
              "post_steps":STEPS,"metric_geometry_authorized":False}
    save_once(experiment/"input_contract.json",inputrec)
    scene=experiment/"scene"
    log=experiment/"logs"/"pipeline.log"
    if args.mode in ("preflight","all"):
        # A GPU runtime must NEVER spend time doing CPU-only SfM because the
        # operator forgot to run the free CPU geometry step first.
        if (not (experiment/"colmap_receipt.json").is_file()
            or not (experiment/"bim_candidate"/"colmap_sparse_unscaled.ply").is_file()):
            raise RuntimeError("CPU_GEOMETRY_MUST_PASS_BEFORE_GPU_PREFLIGHT")
    if args.mode=="all":
        # Check the exact experiment and preflight before extracting video,
        # processing COLMAP, downloading source, or installing dependencies.
        if not statusfile.is_file():
            raise RuntimeError("PREFLIGHT_MUST_PASS_BEFORE_GPU_TRAINING")
        accepted=json.loads(statusfile.read_text())
        if (accepted.get("status") not in ("PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING",
                                           "TRAINING_STARTED","TRAINING_OR_RECOVERY_STARTED",
                                           "TRAINING_FAILED_PRESERVE_PARTIAL",
                                           "GSPLAT_TRAINING_OUTPUTS_EXIST_NOT_BIM_METRIC")
            or accepted.get("source_sha256")!=original["archive_sha256"]):
            raise RuntimeError("SAME_CAPTURE_COLMAP_AND_CUDA_PREFLIGHT_REQUIRED")
        prefix=local/"gsplat_py310_torch24_cu121"
        if not (prefix/"bin/python").is_file():
            raise RuntimeError("PREFLIGHT_PYTHON_ENV_NOT_PRESENT_CURRENT_RUNTIME")
    if args.mode in ("preflight","all"):
        n=verify_prepared_frames(scene,experiment)
        # The COLMAP stage has already run on CPU and its receipt survives
        # Colab's full CPU -> GPU VM teardown.
        record=json.loads((experiment/"colmap_receipt.json").read_text())
        gate=colmap_gate(record["registered"],n)
    else:
        n=prepare_frames(local/"verified_recorder_input/raw_capture.mp4",scene,experiment,log)
        gate=prepare_colmap(scene,experiment,n,local,log)
    if args.mode=="geometry":
        # Safe, CPU-only COLMAP stage. Can run before assigning a Colab GPU;
        # all extracted frames / SfM geometry receipts live on Drive and
        # survive a CPU -> T4 runtime switch.
        result={"status":"GEOMETRY_PREPARED_COLMAP_REGISTERED_NO_GPU",
                "registered":gate["registered"],"frames":n,
                "source_sha256":original["archive_sha256"],
                "training_executed":False,"bim_metric_validated":False}
        if statusfile.exists():
            old=json.loads(statusfile.read_text())
            if old.get("source_sha256")!=result["source_sha256"]:
                raise RuntimeError("GEOMETRY_ATTEMPT_SOURCE_CONFLICT")
            if old.get("status") in ("TRAINING_STARTED","TRAINING_FAILED_PRESERVE_PARTIAL",
                                     "GSPLAT_TRAINING_OUTPUTS_EXIST_NOT_BIM_METRIC",
                                     "PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING"):
                return old
        statusfile.write_text(json.dumps(result,indent=2)+"\n")
        return result
    gs_source=local/"official_gsplat_v153"
    if args.mode=="all" and accepted.get("registered")!=gate["registered"]:
        raise RuntimeError("PREFLIGHT_COLMAP_REGISTRATION_CHANGED")
    local.mkdir(parents=True,exist_ok=True)
    training_scene=local_training_scene(scene,experiment,local)
    ensure_source(gs_source,log)
    if args.mode=="preflight":
        # Explicit GPU prerequisites are checked now, before 7000-step training.
        py=setup_gsplat(local/"gsplat_py310_torch24_cu121",local,gs_source,log)
        patched=ensure_trainer(gs_source)
        smoke=real_video_cuda_smoke(py,gs_source,training_scene,experiment,patched)
        result={"status":"PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING",
                "registered":gate["registered"],"frames":n,
                "cuda_python":str(py),"source_sha256":original["archive_sha256"],
                "real_video_smoke":smoke,
                "training_executed":False,"bim_metric_validated":False}
        if statusfile.exists():
            old=json.loads(statusfile.read_text())
            if old.get("source_sha256")!=result["source_sha256"]:
                raise RuntimeError("OLD_STATUS_DIFFERENT_CAPTURE")
            if old.get("status") in ("TRAINING_STARTED","TRAINING_OR_RECOVERY_STARTED",
                                      "TRAINING_FAILED_PRESERVE_PARTIAL",
                                      "GSPLAT_TRAINING_OUTPUTS_EXIST_NOT_BIM_METRIC"):
                return old
        statusfile.write_text(json.dumps(result,indent=2)+"\n")
        return result
    if args.mode!="all":raise RuntimeError("UNKNOWN_MODE")
    if not args.approve_training:raise RuntimeError("TRAINING_NOT_APPROVED_USE_EXPLICIT_ALL")
    # A GPU training request MUST have a same-capture accepted preflight receipt.
    # Never install multi-GB packages just because MODE=all was accidentally set.
    if not statusfile.is_file():
        raise RuntimeError("PREFLIGHT_MUST_PASS_BEFORE_GPU_TRAINING")
    accepted=json.loads(statusfile.read_text())
    if (accepted.get("status") not in ("PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING",
                                           "TRAINING_STARTED","TRAINING_OR_RECOVERY_STARTED",
                                           "TRAINING_FAILED_PRESERVE_PARTIAL",
                                           "GSPLAT_TRAINING_OUTPUTS_EXIST_NOT_BIM_METRIC")
        or accepted.get("source_sha256")!=original["archive_sha256"]
        or accepted.get("registered")!=gate["registered"]):
        raise RuntimeError("SAME_CAPTURE_COLMAP_AND_CUDA_PREFLIGHT_REQUIRED")
    prefix=local/"gsplat_py310_torch24_cu121"
    if not (prefix/"bin/python").is_file():
        raise RuntimeError("PREFLIGHT_PYTHON_ENV_NOT_PRESENT_CURRENT_RUNTIME")
    # Repeat real CUDA kernel checks, but NEVER reinstall during a training run.
    from mycon_colmap_gsplat_env import verify_existing
    py=verify_existing(prefix,gs_source,log)
    patched=ensure_trainer(gs_source)
    resultdir=experiment/"gsplat_result"
    completed_recovery=resultdir/"recovery"/"resume_step_006999.pt"
    existing=resultdir.exists() and any(resultdir.iterdir())
    recoveries=sorted((resultdir/"recovery").glob("resume_step_*.pt"))
    final_ckpt=resultdir/"ckpts"/"ckpt_6999_rank0.pt"
    if existing and not recoveries and not final_ckpt.is_file():
        raise RuntimeError("EARLY_INTERRUPTED_MODEL_BEFORE_FIRST_RECOVERY; preserve this attempt, use distinct ATTEMPT")
    resultdir.mkdir(parents=True,exist_ok=True)
    state={"status":"TRAINING_OR_RECOVERY_STARTED","source_sha256":original["archive_sha256"],
           "registered":gate["registered"],"frames":n,
           "recovery_snapshots_found":len(recoveries),
           "bim_metric_validated":False}
    statusfile.write_text(json.dumps(state,indent=2)+"\n")
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE="1",MAX_JOBS="2",TORCH_CUDA_ARCH_LIST="7.5",
               MYCON_RESUME_EVERY="500")
    env["CUDA_HOME"]=str(prefix)
    env["PATH"]=str(prefix/"bin")+os.pathsep+env.get("PATH","")
    try:
        # A complete recovery state can be finalized without re-running any
        # gradient steps. Otherwise the patched trainer restores ALL optimizer,
        # scheduler and densification states, then continues at the next step.
        training_required=not completed_recovery.is_file() and not final_ckpt.is_file()
        if training_required:
            run_logged(gsplat_command(py,gs_source,training_scene,resultdir,patched),
                       experiment/"logs"/"train_or_resume.log",
                       env=env,cwd=gs_source/"examples",live=True)
        render_and_evaluate_if_missing(py,gs_source,training_scene,resultdir,patched)
        artifacts=evidence(resultdir)
        if (not artifacts["checkpoints_present"] or not artifacts["ply_present"]
                or not artifacts["novel_view_video_present"] or
                not artifacts["heldout_evaluation_present"]
                or not all(k in artifacts["heldout_measured_metrics"] for
                           k in ("psnr","ssim","lpips"))):
            raise RuntimeError("GSPLAT_ACCEPTANCE_ARTIFACTS_INCOMPLETE")
    except Exception as exc:
        state.update(status="TRAINING_FAILED_PRESERVE_PARTIAL",error=str(exc)[-1000:])
        statusfile.write_text(json.dumps(state,indent=2)+"\n")
        raise
    result={"status":"GSPLAT_TRAINING_OUTPUTS_EXIST_NOT_BIM_METRIC",
            "registered":gate["registered"],"frames":n,
            "source_sha256":original["archive_sha256"],
            "recovered_from_snapshot":bool(recoveries),
            "completed_gradient_training_reexecuted":bool(training_required),
            "evidence":artifacts,"bim_metric_validated":False}
    statusfile.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("plan","verify","geometry","preflight","all","status"),default="preflight")
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
