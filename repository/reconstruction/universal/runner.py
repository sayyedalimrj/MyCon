"""MyCON universal reconstruction entry point. Stage directories are immutable checkpoints."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

from reconstruction.workflow import emit_json, sha256

from .inputs import collect, list_sources

PROFILES = {
    "Auto HQ": (.085,.48,4.3,1600,5500,8,720,1600,True),
    "Fast Preview": (.17,.60,6.0,1000,3500,6,512,1024,False),
    "High Quality": (.075,.42,3.8,1920,6500,9,832,1920,True),
    "Ultra Quality": (.06,.38,3.5,2400,8000,10,1024,2560,True),
    "Long Video": (.095,.55,4.7,1480,5200,8,704,1440,True),
    "Low VRAM": (.115,.52,4.5,1260,4800,7,640,1024,True),
    "Scientific Benchmark": (.085,.48,4.3,1600,5500,8,720,1600,True),
}
KEYS=("min_spacing","max_spacing","flow_delta","feature_size","features","neighbors","cpu_size","cuda_size","retry_matching")
STAGES=("intake","features","matches","map","undistort","dense","export")


def hardware(root):
    raw={}
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            k,_,v=line.partition(":")
            if k in ("MemTotal","MemAvailable"):
                raw[k]=int(v.split()[0])*1024
    gpu=[]
    if shutil.which("nvidia-smi"):
        try:
            out=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.free",
                   "--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL,timeout=8)
            for line in out.splitlines():
                name,vr,free=line.split(",")
                gpu.append(dict(name=name.strip(),vram_mib=int(vr),free_mib=int(free)))
        except (OSError,subprocess.SubprocessError,ValueError):
            pass
    return {"device":platform.machine(),"cpu_available":len(os.sched_getaffinity(0)),"ram":raw,
            "free_disk_bytes":shutil.disk_usage(root).free,"gpus":gpu,"python":platform.python_version()}


def file_records(root):
    return {p.relative_to(root).as_posix():{"bytes":p.stat().st_size,"sha256":sha256(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name!="checkpoint.json"}


def verify(root, manifest):
    if not manifest or any(not (root/p).resolve().is_relative_to(root.resolve()) for p in manifest):
        return False
    return manifest == file_records(root)


def copy_checkpoint(source,destination):
    if destination.exists():
        settled=read(destination/"checkpoint.json")
        origin=read(source/"checkpoint.json")
        if settled != origin or not verify(destination,settled["files"]):
            raise ValueError(f"Mirror checkpoint differs: {destination}")
        return
    tmp=destination.with_name("."+destination.name+".copy-"+str(time.time_ns()))
    shutil.copytree(source,tmp)
    receipt=read(tmp/"checkpoint.json")
    if not verify(tmp,receipt["files"]):
        raise ValueError(f"Mirrored files failed integrity: {tmp}")
    tmp.rename(destination)


def run_stage(run,name,action,say,mirror=None):
    final=run/name
    if mirror is not None and not final.exists() and (mirror/name).exists():
        copy_checkpoint(mirror/name,final)
        say(f"{name} · restored from persistent checkpoint")

    if final.exists():
        receipt=read(final/"checkpoint.json")
        if receipt.get("status")!="COMPLETE" or not verify(final,receipt.get("files")):
            raise ValueError(f"Checkpoint integrity failure: {final}. Preserve it and use a clean run or restore its exact bytes.")
        say(f"{name} · resumed verified checkpoint")
        if mirror is not None:copy_checkpoint(final,mirror/name)
        return receipt["result"]
    tmp=run/("."+name+".working")
    if tmp.exists():
        tmp.rename(run/(tmp.name+".interrupted-"+str(time.time_ns())))
    tmp.mkdir()
    t=time.monotonic()
    try:
        result=action(tmp)
        if not isinstance(result,dict):
            raise TypeError("Stage did not return a report dictionary")
        files=file_records(tmp)
        if not files:
            raise ValueError(f"{name} completed without file evidence")
        receipt={"status":"COMPLETE","result":result,"files":files,"wall_s":time.monotonic()-t}
        emit_json(tmp/"checkpoint.json",receipt)
        tmp.rename(final)
        say(f"{name} · complete · {receipt['wall_s']:.1f}s")
    except BaseException as exc:
        if tmp.is_dir():
            emit_json(tmp/"failure.json",{"status":"FAILED","stage":name,"reason":str(exc)})
        raise
    if mirror is not None:
        copy_checkpoint(final,mirror/name)
        say(f"{name} · saved persistent checkpoint")
    return result


def read(path):
    return json.loads(path.read_text())


def reconstruct(args):
    from . import sfm

    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    mirror=args.checkpoint_mirror.resolve() if args.checkpoint_mirror else None
    if mirror:
        if mirror.is_relative_to(root) or root.is_relative_to(mirror):
            raise ValueError("Checkpoint mirror and local workspace must be separate directories")
        mirror.mkdir(parents=True,exist_ok=True)
    profiles=dict(zip(KEYS,PROFILES[args.profile]))
    hw=hardware(root)
    threads=args.threads or min(8,hw["cpu_available"])
    profiles["threads"]=threads
    profiles["cpu_size"]=min(profiles["cpu_size"],640) if hw["ram"].get("MemAvailable",10**10)<5_000_000_000 else profiles["cpu_size"]
    if hw["gpus"] and hw["gpus"][0]["free_mib"]<7500:
        profiles["cuda_size"]=min(profiles["cuda_size"],1200)
    sources=list_sources(args.inputs)
    files=[{"name":p.name,"sha256":sha256(p),"bytes":p.stat().st_size} for p in sources]
    desired={"schema":"mycon.universal.run.v1","inputs":files,"profile":args.profile,"engine":args.dense_engine,
             "config":profiles,"orbit_seconds":args.orbit_seconds,"holds_out_independent_test":False}
    with (root/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=root/"run_identity.json"
        saved_identity=mirror/"run_identity.json" if mirror else None
        if saved_identity and saved_identity.exists() and not identity.exists():
            shutil.copyfile(saved_identity,identity)
        if identity.exists():
            previous=read(identity)
            if (any(previous[key]!=desired[key] for key in ("schema","inputs","profile","engine"))
                    or previous.get("orbit_seconds",5)!=args.orbit_seconds):
                raise ValueError("Resume input or settings differ from the existing run")
            # Retain the frozen image/feature parameters even if the resumed VM differs.
            profiles=previous["config"]
        else:
            emit_json(identity,desired)
        if saved_identity and not saved_identity.exists():
            emit_json(saved_identity,read(identity))
        elif saved_identity and read(saved_identity)!=read(identity):
            raise ValueError("Persistent and local run identity differ")
        stage="intake"
        log=root/"progress.jsonl"

        def say(message):
            record={"stage":stage,"message":message,"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    "peak_host_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            with log.open("a") as f:
                f.write(json.dumps(record,ensure_ascii=False)+"\n")
            print(f"{stage} [{STAGES.index(stage)+1}/{len(STAGES)}] {message}",flush=True)

        emit_json(root/f"environment-{time.time_ns()}.json",hw)

        def intake_action(work):
            m=collect(sources,work,profiles,say)
            emit_json(work/"capture_analysis.json",m)
            return {"frames":len(m["frames"]),"videos":len(m["videos"])}

        run_stage(root,stage,intake_action,say,mirror)
        m=read(root/"intake/capture_analysis.json")
        if args.through==stage:return 0
        stage="features"
        run_stage(root,stage,lambda t:sfm.extract(root/"intake",t,m,profiles,say),say,mirror)
        if args.through==stage:return 0
        stage="matches"
        run_stage(root,stage,lambda t:sfm.match(root/"intake",root/"features",t,m,profiles,say),say,mirror)
        if args.through==stage:return 0
        stage="map"
        sparse=run_stage(root,stage,lambda t:sfm.map_scene(root/"intake",root/"matches",t,m,profiles,say),say,mirror)
        if args.through==stage:return 0
        from . import dense, render

        stage="undistort"
        selection = "cuda_patchmatch" if args.dense_engine in ("auto","cuda") and sfm.pc.has_cuda and hw["gpus"] else "cpu_stereo"
        if args.dense_engine=="cuda" and selection!="cuda_patchmatch":
            say("CUDA unavailable. Proceeding with calibrated CPU stereo.")
        plan=root/"dense_plan.json"
        if mirror is not None and (mirror/"dense_plan.json").exists() and not plan.exists():
            shutil.copyfile(mirror/"dense_plan.json",plan)
        if plan.exists():
            selection=read(plan)["engine"]
        else:
            emit_json(plan,{"engine":selection})
        if mirror is not None and not (mirror/"dense_plan.json").exists():
            emit_json(mirror/"dense_plan.json",read(plan))
        if args.dense_engine!="none":
            run_stage(root,stage,lambda t:dense.undistort(root/"map/model",root/"intake/frames",t,
                         profiles["cuda_size"] if selection=="cuda_patchmatch" else profiles["cpu_size"],profiles),say,mirror)
        if args.through==stage:return 0
        stage="dense"
        quality={"status":"sparse_requested","engine":None}
        if args.dense_engine!="none":
            quality=run_stage(root,stage,lambda t:dense.reconstruct(root/"undistort",t,(mirror or root)/"depth_cache",
                                     m,profiles,selection,say),say,mirror)
        if args.through==stage:return 0
        stage="export"
        result=run_stage(root,stage,lambda t:render.export(root,t,m,sparse,quality,args.orbit_seconds,say),say,mirror)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs",type=Path,nargs="+")
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--profile",choices=PROFILES,default="Auto HQ")
    cli.add_argument("--dense-engine",choices=("auto","cpu","cuda","none"),default="auto")
    cli.add_argument("--through",choices=STAGES,default="export")
    cli.add_argument("--threads",type=int)
    cli.add_argument("--checkpoint-mirror",type=Path)
    cli.add_argument("--orbit-seconds",type=int,default=5)
    a=cli.parse_args()
    if a.threads is not None and a.threads < 1:
        cli.error("--threads must be positive")
    if a.orbit_seconds < 0:
        cli.error("--orbit-seconds must be nonnegative")
    try:
        return reconstruct(a)
    except (Exception,KeyboardInterrupt) as exc:
        print(f"Run stopped · {type(exc).__name__} · {exc}",file=sys.stderr,flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
