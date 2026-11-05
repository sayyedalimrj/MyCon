#!/usr/bin/env python3
"""Reproducible canonical model export and evaluation, never overwrite originals.

Official gsplat saves its ckpt and PLY BEFORE its last optimizer step. MyCON
saves its optimizer recovery AFTER the step. Never evaluate one and display the
other. An immutable canonical directory contains checkpoint, splat PLY, render,
metrics, and a single source-SHA provenance manifest.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from mycon_colmap_gsplat_env import run_logged
from mycon_audit_runtime import sha256, atomic_json, metrics_valid, validate_artifact_file


def _cuda_env(py: Path) -> dict:
    env=os.environ.copy()
    prefix=Path(py).parent.parent
    env.update(PYTHONNOUSERSITE="1",MAX_JOBS="2",
               TORCH_CUDA_ARCH_LIST="7.5",
               CUDA_HOME=str(prefix),MYCON_MAX_GAUSSIANS="1500000",
               MYCON_MAX_CUDA_GIB="13.0")
    if Path("/usr/bin/gcc-12").is_file() and Path("/usr/bin/g++-12").is_file():
        env.update(CC="/usr/bin/gcc-12",CXX="/usr/bin/g++-12",
                   CUDAHOSTCXX="/usr/bin/g++-12")
    env["PATH"]=str(prefix/"bin")+os.pathsep+env.get("PATH","")
    return env


def source_checkpoint(result: Path, step: int=6999) -> tuple[Path,str]:
    recovery=result/"recovery"/f"resume_step_{step:06d}.pt"
    stamp=Path(str(recovery)+".sha256.json")
    official=result/"ckpts"/f"ckpt_{step}_rank0.pt"
    if recovery.is_file():
        if not stamp.is_file():
            raise RuntimeError("MYCON_FINAL_RECOVERY_HASH_RECEIPT_MISSING")
        proof=json.loads(stamp.read_text())
        if proof.get("sha256")!=sha256(recovery) or proof.get("size")!=recovery.stat().st_size:
            raise RuntimeError("MYCON_FINAL_RECOVERY_HASH_OR_SIZE_MISMATCH")
        return recovery,"post_optimizer_recovery"
    if validate_artifact_file(official,2048):
        return official,"official_pre_optimizer"
    raise RuntimeError("MYCON_NO_COMPLETE_FINAL_MODEL_CHECKPOINT")


def make_final_model(py: Path, result: Path, *, expected_step: int=6999) -> Path:
    """Produce canonical checkpoint+PLY from the SAME validated source weights."""
    src,kind=source_checkpoint(result,expected_step)
    target=result/"canonical"
    canonical=target/"ckpts"/f"ckpt_{expected_step}_rank0.pt"
    ply=target/"ply"/f"point_cloud_{expected_step}.ply"
    manifest=target/"canonical_model.json"
    source_sha=sha256(src)
    identity={"schema":"mycon.canonical_model.v1",
              "source_checkpoint":str(src),"source_sha256":source_sha,
              "source_kind":kind,"step":expected_step,
              "gaussian_frame":"COLMAP_SFM_ARBITRARY_SCALE",
              "colmap_to_gaussian_4x4":[[1,0,0,0],[0,1,0,0],
                                       [0,0,1,0],[0,0,0,1]],
              "metric_bim_validated":False}
    if manifest.is_file():
        previous=json.loads(manifest.read_text())
        for k,v in identity.items():
            if previous.get(k)!=v:
                raise RuntimeError("MYCON_CANONICAL_MODEL_PROVENANCE_CHANGED")
        if not (validate_artifact_file(canonical,2048) and
                validate_artifact_file(ply,1024)
                and previous.get("canonical_sha256")==sha256(canonical)
                and previous.get("ply_sha256")==sha256(ply)):
            raise RuntimeError("MYCON_CANONICAL_MODEL_HASH_OR_ARTIFACT_INCOMPLETE")
        return canonical
    if canonical.exists() or ply.exists():
        raise RuntimeError("MYCON_CANONICAL_PARTIAL_OUTPUT_PRESERVED_USE_DIAGNOSTICS")
    # A single Python/CUDA process loads source checkpoint, writes exact model
    # and exports PLY from the SAME tensor dictionary. No official ckpt mixup.
    code=r'''import os,sys,torch
from pathlib import Path
from gsplat import export_splats
source,model,ply,step,kind=map(str,sys.argv[1:])
step=int(step)
data=torch.load(source,map_location="cpu",weights_only=False)
if kind=="post_optimizer_recovery":
  if data.get("patch")!="mycon-resume-v1" or data.get("last_completed_step")!=step:
    raise RuntimeError("FINAL_RECOVERY_IDENTITY_OR_STEP_WRONG")
else:
  if data.get("step")!=step:raise RuntimeError("FINAL_OFFICIAL_STEP_WRONG")
splats=data["splats"]
if not all(k in splats for k in ("means","scales","quats","opacities","sh0","shN")):
  raise RuntimeError("FINAL_SPLAT_PARAMETERS_INCOMPLETE")
if any(not bool(torch.isfinite(x).all()) for x in splats.values()):
  raise RuntimeError("FINAL_SPLAT_PARAMETERS_NOT_FINITE")
model=Path(model);ply=Path(ply)
model.parent.mkdir(parents=True,exist_ok=True)
ply.parent.mkdir(parents=True,exist_ok=True)
m_tmp=model.with_name(model.name+".partial."+str(os.getpid()))
p_tmp=ply.with_name("partial_"+str(os.getpid())+"_"+ply.name)
torch.save({"step":step,"splats":splats},m_tmp)
os.replace(m_tmp,model)
export_splats(means=splats["means"],scales=splats["scales"],
              quats=splats["quats"],opacities=splats["opacities"],
              sh0=splats["sh0"],shN=splats["shN"],
              format="ply",save_to=str(p_tmp))
if not p_tmp.is_file() or p_tmp.stat().st_size<1024:
  raise RuntimeError("FINAL_PLY_EXPORT_EMPTY")
os.replace(p_tmp,ply)
print("MYCON_CANONICAL_MODEL_AND_PLY_FROM_ONE_SPLAT_STATE",step)
'''
    run_logged([str(py),"-c",code,str(src),str(canonical),str(ply),
                str(expected_step),kind],
               result.parent/"logs"/"finish_canonical_model.log",
               env=_cuda_env(py))
    if not (validate_artifact_file(canonical,2048) and validate_artifact_file(ply)):
        raise RuntimeError("MYCON_CANONICAL_EXPORT_INCOMPLETE")
    identity.update(canonical_sha256=sha256(canonical),ply_sha256=sha256(ply))
    atomic_json(manifest,identity)
    return canonical


def render_and_evaluate_if_missing(py: Path, source: Path, scene: Path,
                                   result: Path, trainer: Path,
                                   *,expected_step: int=6999) -> dict:
    ckpt=make_final_model(py,result,expected_step=expected_step)
    canonical=result/"canonical"
    stats=canonical/"stats"/f"val_step{expected_step:04d}.json"
    video=canonical/"videos"/f"traj_{expected_step}.mp4"
    if not (stats.is_file() and video.is_file()):
        # Official --ckpt is EVAL-ONLY, which is why both assets are derived
        # from the new canonical ckpt and its unchanged COLMAP coordinate frame.
        cmd=[str(py),str(trainer),"default",
             "--data_dir",str(scene),"--data_factor","1",
             "--result_dir",str(canonical),"--test_every","8",
             "--max_steps",str(expected_step+1),
             "--disable_viewer","--packed","--ckpt",str(ckpt)]
        run_logged(cmd,result.parent/"logs"/"final_canonical_eval.log",
                   cwd=source/"examples",env=_cuda_env(py),live=True)
    if not (validate_artifact_file(stats,20) and validate_artifact_file(video)):
        raise RuntimeError("MYCON_CANONICAL_EVAL_MISSING")
    measured=json.loads(stats.read_text())
    if not metrics_valid(measured):
        raise RuntimeError("MYCON_CANONICAL_METRICS_NONFINITE_OR_INVALID")
    identity=json.loads((canonical/"canonical_model.json").read_text())
    evaluation={"schema":"mycon.canonical_evaluation.v1",
                "canonical_checkpoint_sha256":identity["canonical_sha256"],
                "ply_sha256":identity["ply_sha256"],
                "stats_sha256":sha256(stats),"video_sha256":sha256(video),
                "metrics":measured,
                "metric_geometry_validated":False}
    marker=canonical/"canonical_evaluation.json"
    if marker.is_file() and json.loads(marker.read_text())!=evaluation:
        raise RuntimeError("MYCON_CANONICAL_EVAL_CHANGED_NEVER_SILENTLY_OVERWRITE")
    atomic_json(marker,evaluation)
    return {"checkpoint":str(ckpt),
            "ply":str(canonical/"ply"/f"point_cloud_{expected_step}.ply"),
            "video":str(video),"metrics":measured,
            "canonical_identity":evaluation,"training_restarted":False}
