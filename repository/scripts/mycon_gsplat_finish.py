#!/usr/bin/env python3
"""Complete missing gsplat artifacts from verified checkpoints WITHOUT training.

Only operates inside a single new experiment's output folder. The original
Recorder ZIP, NVIDIA model/checkpoints and all other experiments remain intact.
This uses gsplat v1.5.3's official exporter and evaluation-only --ckpt path.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from mycon_colmap_gsplat_env import run_logged


def make_final_model(py: Path, result: Path, *, expected_step: int = 6999) -> Path:
    recovery=result/"recovery"/f"resume_step_{expected_step:06d}.pt"
    official=result/"ckpts"/f"ckpt_{expected_step}_rank0.pt"
    ply=result/"ply"/f"point_cloud_{expected_step}.ply"
    if not official.is_file() and not recovery.is_file():
        raise RuntimeError("FINAL_GSPLAT_CHECKPOINT_NOT_FOUND_CANNOT_FINALIZE")
    if ply.is_file() and official.is_file():
        return official
    # Save only absent derived artifacts. Never overwrite a valid checkpoint.
    script=r'''import os,sys,torch
from pathlib import Path
from gsplat import export_splats
recovery,ckpt,ply,step=map(str,sys.argv[1:])
step=int(step)
source=Path(recovery)
official=Path(ckpt)
if not official.exists() and not source.is_file():
  raise RuntimeError("NO_SOURCE_CHECKPOINT_FOR_FINISH")
saved=torch.load(str(source if source.is_file() else official),
                 map_location="cuda",weights_only=False)
if source.is_file():
  if saved.get("last_completed_step")!=step:
    raise RuntimeError("RECOVERY_STEP_NOT_COMPLETE")
  splats=saved["splats"]
else:
  if saved.get("step")!=step:
    raise RuntimeError("OFFICIAL_CHECKPOINT_WRONG_STEP")
  splats=saved["splats"]
if not official.is_file():
  official.parent.mkdir(parents=True,exist_ok=True)
  tmp=official.with_name(official.name+".partial."+str(os.getpid()))
  torch.save({"step":step,"splats":splats},str(tmp))
  os.replace(tmp,official)
if not Path(ply).is_file():
  Path(ply).parent.mkdir(parents=True,exist_ok=True)
  tmp=Path(ply).with_name("partial_"+str(os.getpid())+"_"+Path(ply).name)
  export_splats(means=splats["means"],scales=splats["scales"],
                quats=splats["quats"],opacities=splats["opacities"],
                sh0=splats["sh0"],shN=splats["shN"],
                format="ply",save_to=str(tmp))
  if not tmp.is_file() or tmp.stat().st_size<1024:
    raise RuntimeError("GSPLAT_PLY_EXPORT_EMPTY")
  os.replace(tmp,ply)
print("MYCON_FINAL_GSPLAT_CHECKPOINT_AND_PLY_READY",official,ply)
'''
    log=result.parent/"logs"/"finish_model.log"
    run_logged([str(py),"-c",script,str(recovery),str(official),str(ply),str(expected_step)],
               log)
    return official


def render_and_evaluate_if_missing(py: Path, source: Path,
                                   scene: Path, result: Path,
                                   trainer: Path, *, expected_step: int=6999) -> dict:
    ckpt=make_final_model(py,result,expected_step=expected_step)
    stats=result/"stats"/f"val_step{expected_step:04d}.json"
    video=result/"videos"/f"traj_{expected_step}.mp4"
    if not stats.is_file() or not video.is_file():
        # Official --ckpt is evaluation-only; no training optimizer is invoked.
        cmd=[str(py),str(trainer),"default",
             "--data_dir",str(scene),"--data_factor","1",
             "--result_dir",str(result),"--test_every","8",
             "--max_steps",str(expected_step+1),
             "--disable_viewer","--packed","--ckpt",str(ckpt)]
        run_logged(cmd,result.parent/"logs"/"final_eval_only.log",
                   cwd=source/"examples",live=True)
    if not stats.is_file() or not video.is_file() or video.stat().st_size<1024:
        raise RuntimeError("FINAL_GSPLAT_METRICS_OR_VIDEO_MISSING_AFTER_EVAL")
    measured=json.loads(stats.read_text())
    for field in ("psnr","ssim","lpips"):
        if not isinstance(measured.get(field),(int,float)):
            raise RuntimeError("FINAL_GSPLAT_EVAL_MISSING_METRIC_"+field)
    return {"checkpoint":str(ckpt),"ply":str(result/"ply"/f"point_cloud_{expected_step}.ply"),
            "video":str(video),"metrics":measured,
            "training_restarted":False}


if __name__=="__main__":
    raise SystemExit("Import make_final_model/render_and_evaluate_if_missing via the MyCON controller")
