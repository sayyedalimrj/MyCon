#!/usr/bin/env python3
"""MyCON next experiment: safe LongSplat and pose-prior readiness controller.

Read-only baseline. Does not call the COLMAP baseline mapper or old gsplat trainer.
CUDA installation, LongSplat training and rendering are separate, explicit stages.
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

BASE_COMMIT = "abc97337e5268d82e49cec4623c0cf5730386b19"
LONGSPLAT_COMMIT = "19750775a9d19f30aa05a8333c4c6c231b2d5f4a"
RUN_NAME = "longsplat_novelview_v1_20261109"
BASE_ATTEMPT = "mycon_abi_audited_v6_20261106"
BASE_HASH_PREFIX = "811c5a1323e45e2a"
CONFIRM = "RUN_NEW_LONGSPLAT_EXPERIMENT"
FRAME_RE = re.compile(r"^frame_\d{6}\.jpg$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_once(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise RuntimeError("MYCON_EXISTING_RECEIPT_CONFLICT_PRESERVED: " + str(path))
        return
    tmp = path.with_name(path.name + ".tmp_" + str(os.getpid()))
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(body)
    if path.exists():
        tmp.unlink()
        raise RuntimeError("MYCON_CONCURRENT_RECEIPT_EXISTS_PRESERVED")
    tmp.rename(path)


def run_command(cmd: list[str], log: Path, *, cwd: Path | None = None,
                env: dict | None = None, timeout: int | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    clean_env = dict(os.environ if env is None else env)
    clean_env["MPLBACKEND"] = "Agg"
    clean_env["PYTHONUNBUFFERED"] = "1"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n>>> " + json.dumps(cmd) + "\n")
        handle.flush()
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=clean_env,
                                  stdout=handle, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MYCON_COMMAND_TIMEOUT_PRESERVED_LOG:" + str(log)) from exc
        if proc.returncode:
            raise RuntimeError(f"MYCON_COMMAND_FAILED_{proc.returncode}_SEE_LOG:{log}")


def work_paths(root: Path, run_name: str) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", run_name):
        raise ValueError("MYCON_UNSAFE_RUN_NAME")
    original = root / "research_colmap_gsplat" / BASE_HASH_PREFIX / BASE_ATTEMPT
    return {
        "root": root, "baseline": original,
        "new": root / "research_novelview" / run_name,
        "dataset": original / "longsplat_video8fps_512",
        "source": Path("/content/MyCON_LongSplat_official_pinned"),
        "local_data": Path("/content/MyCON_LongSplat_new_data"),
        "env": Path("/content/MyCON_LS_t4_pt241_cuda121"),
    }


def verify(paths: dict[str, Path], *, full_hash: bool = False) -> dict:
    baseline = paths["baseline"]
    canonical = baseline / "gsplat_result" / "canonical"
    eval_path = canonical / "canonical_evaluation.json"
    model_path = canonical / "canonical_model.json"
    manifest_path = paths["dataset"] / "dataset_receipt.json"
    for path in (eval_path, model_path, manifest_path):
        if not path.is_file():
            raise RuntimeError("MYCON_REQUIRED_BASELINE_OR_DATASET_RECEIPT_MISSING:" + str(path))
    ev = json.loads(eval_path.read_text())
    md = json.loads(model_path.read_text())
    receipt = json.loads(manifest_path.read_text())
    if (ev.get("schema") != "mycon.canonical_evaluation.v1"
        or ev.get("ply_sha256") != "afd378d8849b31c1455c1f73e4faaac9cb5188b5d4502355bd2b2fdbcbaa64ec"
        or md.get("canonical_sha256") != ev.get("canonical_checkpoint_sha256")
        or ev.get("metrics", {}).get("num_GS") != 1477725):
        raise RuntimeError("MYCON_FROZEN_BASELINE_IDENTITY_MISMATCH")
    if (receipt.get("fps") != 8 or receipt.get("width") != 512
        or receipt.get("source_recorder_archive_sha256")
           != "811c5a1323e45e2a4ef635eeda9f4335be06ef01ccb74ed3b431328400f862b6"
        or receipt.get("training_executed") is not False):
        raise RuntimeError("MYCON_LONGSPLAT_DATASET_SOURCE_MISMATCH")
    files = receipt.get("manifest", {}).get("files", {})
    if len(files) < 650 or len(files) > 950:
        raise RuntimeError("MYCON_LONGSPLAT_DATASET_WRONG_COUNT")
    image_dir = paths["dataset"] / "images"
    names = sorted(p.name for p in image_dir.glob("*.jpg"))
    if len(names) != len(files) or set(names) != set(files):
        raise RuntimeError("MYCON_LONGSPLAT_IMAGES_MISSING_OR_EXTRA")
    if not all(FRAME_RE.fullmatch(name) for name in names):
        raise RuntimeError("MYCON_LONGSPLAT_BAD_FRAME_NAMES")
    sampled = names if full_hash else sorted(set([names[0], names[-1]]
                                  + [names[i] for i in range(0, len(names), max(1, len(names)//16))]))
    for name in sampled:
        p = image_dir / name
        item = files[name]
        if (p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]):
            raise RuntimeError("MYCON_LONGSPLAT_FRAME_HASH_MISMATCH:" + name)
    result = {"baseline": str(baseline), "dataset_images": str(image_dir),
              "frames": len(files), "fps": 8, "width": 512,
              "source_commit": BASE_COMMIT, "longsplat_commit": LONGSPLAT_COMMIT,
              "checked_hashes": len(sampled), "all_image_hashes_checked": full_hash,
              "training_executed": False, "site_metric_validated": False,
              "gsplat_baseline_metrics": ev["metrics"],
              "source_archive_sha256": receipt["source_recorder_archive_sha256"]}
    return result


def prepare(paths: dict[str, Path]) -> dict:
    checked = verify(paths, full_hash=False)
    out = paths["new"]
    out.mkdir(parents=True, exist_ok=True)
    save_once(out / "input_identity.json", {
        "schema": "mycon.novelview.input.v1",
        "base_model_sha256": "afd378d8849b31c1455c1f73e4faaac9cb5188b5d4502355bd2b2fdbcbaa64ec",
        "baseline_source_pin": BASE_COMMIT,
        "source_archive_sha256": checked["source_archive_sha256"],
        "dataset_frames": checked["frames"], "fps": 8,
        "official_longsplat_commit": LONGSPLAT_COMMIT,
        "metric_anchors_approved": False,
        "ARCore_priors_directly_injected": False,
        "camera_scale_drift_or_time_sync_resolved": False})
    save_once(out / "pose_prior_gate.json", {
        "status": "NOT_AUTHORIZED_TO_INJECT_ARCORE_PRIORS",
        "reason": "Native ARCore vs COLMAP timing/scale drift still requires independent validation.",
        "colmap_pose_prior_mapper_requires": [
            "Verified per-frame timestamp association and camera convention",
            "Independent uncertainty/covariance estimate",
            "Separate copied COLMAP database; never mutate baseline"],
        "longsplat_custom_pipeline_uses_unposed_images": True})
    save_once(out / "baseline_comparison_gate.json", {
        "status": "WAITING_FOR_TRAINED_LONGSPLAT_AND_MATCHED_HOLDOUT",
        "baseline_metrics": checked["gsplat_baseline_metrics"],
        "warning": "Different heldout splits or cameras are not comparable; do not claim improvement from raw PSNR values."})
    return checked


def pinned_source(paths: dict[str, Path], *, fetch: bool) -> Path:
    repo = paths["source"]
    if not (repo / ".git").is_dir():
        if not fetch:
            raise RuntimeError("MYCON_LONGSPLAT_SOURCE_MISSING_ENABLE_FETCH_SOURCE")
        if repo.exists():
            raise RuntimeError("MYCON_UNEXPECTED_SOURCE_DIRECTORY_PRESERVED:" + str(repo))
        run_command(["git","clone","--no-checkout","--depth","1",
                     "https://github.com/NVlabs/LongSplat.git",str(repo)],
                    paths["new"]/"logs"/"clone.log", timeout=1800)
        run_command(["git","fetch","--depth","1","origin",LONGSPLAT_COMMIT],
                    paths["new"]/"logs"/"clone.log",cwd=repo,timeout=900)
        run_command(["git","checkout","--detach",LONGSPLAT_COMMIT],
                    paths["new"]/"logs"/"clone.log",cwd=repo,timeout=300)
        run_command(["git","submodule","update","--init","--recursive","--depth","1"],
                    paths["new"]/"logs"/"submodules.log",cwd=repo,timeout=3600)
    origin = subprocess.check_output(["git","-C",str(repo),"remote","get-url","origin"],text=True).strip()
    rev = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if "NVlabs/LongSplat" not in origin or rev != LONGSPLAT_COMMIT:
        raise RuntimeError("MYCON_OFFICIAL_LONGSPLAT_SOURCE_UNTRUSTED_OR_UNPINNED")
    if not (repo/"train.py").is_file() or not (repo/"submodules/mast3r").is_dir():
        raise RuntimeError("MYCON_LONGSPLAT_REQUIRED_SOURCES_MISSING")
    return repo


def stage_copy(paths: dict[str, Path]) -> Path:
    verify(paths, full_hash=True)
    dst = paths["local_data"] / "MyCON_96s" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    src = paths["dataset"] / "images"
    receipt = json.loads((paths["dataset"]/"dataset_receipt.json").read_text())
    for name, meta in receipt["manifest"]["files"].items():
        d = dst / name
        if d.exists():
            if d.stat().st_size != meta["bytes"] or sha256(d) != meta["sha256"]:
                raise RuntimeError("MYCON_LOCAL_COPY_CONFLICT_PRESERVE:" + str(d))
        else:
            shutil.copy2(src/name,d)
            if d.stat().st_size != meta["bytes"] or sha256(d) != meta["sha256"]:
                raise RuntimeError("MYCON_LOCAL_COPY_HASH_FAILED:" + name)
    return dst


def cuda_environment(paths: dict[str, Path], *, install: bool) -> Path:
    repo = pinned_source(paths,fetch=False)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from mycon_gpu_autoprofile import detect_hardware,profile_for
    from mycon_longsplat_autoenv import install as install_longsplat,verify_cuda,effective_env
    h = detect_hardware()
    prof = profile_for(h)
    if h.sm != 75 or (prof["torch_version"],prof["torchvision_version"],
                       prof["torch_cuda_wheel"]) != ("2.4.1","0.19.1","cu121"):
        raise RuntimeError("MYCON_UNSUPPORTED_LONGSPLAT_HARDWARE_OR_ABI")
    py = paths["env"]/"bin/python"
    log = paths["new"]/"logs"/"longsplat_cuda_install.log"
    os.environ["MPLBACKEND"] = "Agg"
    if install:
        install_longsplat(prof,repo,paths["env"],log)
    if not py.is_file():
        raise RuntimeError("MYCON_LONGSPLAT_ENV_NOT_INSTALLED_CONFIRM_INSTALL")
    verify_cuda(py,prof,paths["new"]/"logs"/"cuda_smoke.log",
                effective_env(prof,paths["env"]))
    save_once(paths["new"]/"cuda_verified.json",{
        "torch": "2.4.1","torchvision": "0.19.1","cuda_wheel": "cu121",
        "gpu": "T4","longsplat_source_commit": LONGSPLAT_COMMIT,
        "model_training_completed": False})
    return py


def do_train(paths: dict[str, Path], confirmation: str, resolution: int,
             post_iter: int) -> None:
    if confirmation != CONFIRM:
        raise RuntimeError("MYCON_LONGSPLAT_TRAINING_NOT_AUTHORIZED")
    if resolution not in (1, 2) or not 5000 <= post_iter <= 40000:
        raise RuntimeError("MYCON_UNSAFE_TRAINING_CONFIG")
    prepared = prepare(paths)
    pinned_source(paths,fetch=False)
    py = cuda_environment(paths,install=False)
    stage_copy(paths)
    model = paths["new"]/"longsplat_model"
    start_receipt = paths["new"]/"training_started.json"
    done_receipt = paths["new"]/"training_completed.json"
    if done_receipt.is_file():
        raise RuntimeError("MYCON_LONGSPLAT_ALREADY_FINISHED_DO_NOT_RETRAIN")
    if start_receipt.exists() or model.exists():
        raise RuntimeError("MYCON_PARTIAL_TRAINING_PRESERVED_MANUAL_RESUME_REVIEW_REQUIRED")
    model.parent.mkdir(parents=True,exist_ok=True)
    save_once(start_receipt,{"status":"TRAINING_STARTED_NOT_RESULT","source_commit":LONGSPLAT_COMMIT,
                            "post_iter":post_iter,"resolution":resolution,
                            "input_frames":prepared["frames"]})
    import socket
    sock=socket.socket()
    sock.bind(("127.0.0.1",0))
    port=sock.getsockname()[1]
    sock.close()
    source=paths["source"]
    data_dir=paths["local_data"]/"MyCON_96s"
    command=[str(py),"-u",str(source/"train.py"),
             "--eval","-s",str(data_dir),"-m",str(model),"--port",str(port),
             "--images","images","--mode","custom","-r",str(resolution),
             "--pose_iteration","100","--local_iter","200","--global_iter","500",
             "--post_iter",str(post_iter)]
    run_command(command,paths["new"]/"logs"/"train.log",cwd=source,
                env=None,timeout=None)
    if not model.exists() or not any(model.rglob("*.ply")):
        raise RuntimeError("MYCON_LONGSPLAT_TRAIN_RETURNED_WITHOUT_PLY_EVIDENCE")
    save_once(done_receipt,{"status":"TRAINING_FINISHED_OUTPUTS_REQUIRE_QA",
                            "source_commit":LONGSPLAT_COMMIT,
                            "model_dir":str(model),
                            "baseline_unchanged":True,
                            "bim_metric_validated":False})


def do_render(paths: dict[str, Path]) -> None:
    if not (paths["new"]/"training_completed.json").is_file():
        raise RuntimeError("MYCON_TRAINING_NOT_COMPLETE_REJECT_RENDER")
    if (paths["new"]/"render_completed.json").exists():
        raise RuntimeError("MYCON_RENDER_ALREADY_DONE")
    py = cuda_environment(paths,install=False)
    source = pinned_source(paths,fetch=False)
    model = paths["new"]/"longsplat_model"
    run_command([str(py),"-u",str(source/"render.py"),"-m",str(model)],
                paths["new"]/"logs"/"render.log",cwd=source)
    save_once(paths["new"]/"render_completed.json",{
        "status":"RENDER_COMPLETE_NOT_HOLDOUT_COMPARABLE",
        "metric_geometry_validated":False})
    run_command([str(py),"-u",str(source/"metrics.py"),"-m",str(model)],
                paths["new"]/"logs"/"metrics.log",cwd=source)
    save_once(paths["new"]/"metrics_completed.json",{
        "status":"RAW_LONGSPLAT_METRICS_EXIST_NOT_DIRECT_BASELINE_COMPARISON",
        "matched_views_and_split_verified":False})


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("plan","prepare","source","install","cuda-check","train","render","status"),
                   default="prepare")
    p.add_argument("--drive-root",type=Path,default=Path("/content/drive/MyDrive/MyCon_R4"))
    p.add_argument("--run-name",default=RUN_NAME)
    p.add_argument("--fetch-source",action="store_true")
    p.add_argument("--allow-install",action="store_true")
    p.add_argument("--train-confirmation",default="")
    p.add_argument("--resolution",type=int,default=2)
    p.add_argument("--post-iter",type=int,default=20000)
    args=p.parse_args(argv)
    paths=work_paths(args.drive_root,args.run_name)
    if args.stage=="status":
        out={"run":str(paths["new"]),"receipts":sorted(p.name for p in paths["new"].glob("*.json"))
             if paths["new"].is_dir() else [],"baseline_modified":False}
    elif args.stage=="plan":
        out={"status":"PLAN_ONLY","source":LONGSPLAT_COMMIT,"run":str(paths["new"]),
             "training_approved":False}
    elif args.stage=="prepare":
        out={"status":"DATASET_PREPARED_NO_TRAINING",**prepare(paths)}
    elif args.stage=="source":
        prepare(paths)
        out={"status":"LONGSPLAT_SOURCE_PINNED_NO_TRAINING",
             "source":str(pinned_source(paths,fetch=args.fetch_source))}
    elif args.stage in ("install","cuda-check"):
        prepare(paths)
        out={"status":"CUDA_VERIFIED_NO_TRAINING",
             "python":str(cuda_environment(paths,install=args.stage=="install" and args.allow_install))}
    elif args.stage=="train":
        do_train(paths,args.train_confirmation,args.resolution,args.post_iter)
        out={"status":"TRAINING_FINISHED_QA_PENDING","run":str(paths["new"])}
    else:
        do_render(paths)
        out={"status":"RENDER_AND_NATIVE_METRICS_COMPLETE_COMPARISON_PENDING",
             "run":str(paths["new"])}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("MYCON_NEXT_EXPERIMENT_STOP:",type(exc).__name__,str(exc),file=sys.stderr)
        raise
