#!/usr/bin/env python3
"""Finish an existing isolated environment; no model training or source deletion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from mycon_longsplat_autoenv import effective_env,run_logged,verify_cuda


def finish(a):
    reports=a.reports.resolve();reports.mkdir(parents=True,exist_ok=True)
    source=a.source.resolve();root=a.env_root.resolve();py=root/'bin/python'
    profile=json.loads(a.profile.read_text())
    env=effective_env(profile,root)
    env['MAX_JOBS']='2';env['FORCE_CUDA']='1'
    env['PIP_CONSTRAINT']=str(root.parent/(root.name+'_torch_constraints.txt'))
    log=reports/'finish_preflight.log'
    requirements=reports/'requirements_remaining.txt'
    # These packages do not rebuild PyTorch3D and can install while its native build runs.
    requirements.write_text('\n'.join(x for x in (source/'requirements.txt').read_text().splitlines()
                                      if 'pytorch3d' not in x.lower())+'\n')
    run_logged([str(py),'-m','pip','install','--no-build-isolation','-r',str(requirements)],log,env=env,cwd=source)
    if a.pytorch3d_ready_marker:
        deadline=time.monotonic()+a.wait_seconds
        while 'PYTORCH3D_INSTALLED_WITH_CUDA121' not in a.pytorch3d_ready_marker.read_text():
            if 'Traceback' in a.pytorch3d_ready_marker.read_text():
                raise RuntimeError('PYTORCH3D_BUILD_FAILED: read preserved build log')
            if time.monotonic()>deadline:
                raise RuntimeError('DEPENDENCY_WAIT_TIMEOUT: preserve existing build and logs')
            time.sleep(5)
    # Persist the expensive wheel before later, independent preflight steps can fail.
    wheel_cache=reports/'verified_wheels';wheel_cache.mkdir(exist_ok=True)
    if a.pytorch3d_dist:
        for wheel in a.pytorch3d_dist.glob('*.whl'):
            dest=wheel_cache/wheel.name
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).digest()!=hashlib.sha256(wheel.read_bytes()).digest():
                    raise RuntimeError('CACHE_WHEEL_IDENTITY_CHANGED: preserve both')
            else:shutil.copy2(wheel,dest)
    for ext in ('simple-knn','diff-gaussian-rasterization','fused-ssim'):
        run_logged([str(py),'-m','pip','install','--no-build-isolation',str(source/'submodules'/ext)],
                   log,env=env,cwd=source)
    rope=source/'submodules/mast3r/dust3r/croco/models/curope'
    if (rope/'setup.py').is_file():
        run_logged([str(py),'setup.py','build_ext','--inplace'],log,env=env,cwd=rope)
    verify_cuda(py,profile,log,env)
    for entry in ('train.py','render.py','convert_3dgs.py'):
        run_logged([str(py),str(source/entry),'--help'],log,env=env,cwd=source,timeout_s=120)
    frames=sorted(a.frames.glob('frame_*.jpg'))
    if len(frames)<30:raise RuntimeError('Full-video frames missing')
    # Two-image inference tests the real checkpoint and import path without any optimization.
    snippet="""import json,torch
from utils.mast3r_utils import Mast3rMatcher
from dust3r.utils.image import load_images
from dust3r.inference import inference
model=Mast3rMatcher().model
images=load_images(FRAME_PATHS,size=512,verbose=False)
result=inference([(images[0],images[1])],model,device='cuda',batch_size=1,verbose=False)
pts=result['pred1']['pts3d'];assert torch.isfinite(pts).all()
torch.cuda.synchronize()
print(json.dumps({'mast3r_two_image_inference':'PASS','points_shape':list(pts.shape),'max_cuda_mib':torch.cuda.max_memory_allocated()/2**20,'training_executed':False}))
""".replace('FRAME_PATHS',repr([str(frames[0]),str(frames[min(8,len(frames)-1)])]))
    run_logged([str(py),'-c',snippet],log,env=env,cwd=source,timeout_s=300)
    marker={k:profile[k] for k in ('torch_version','torchvision_version','torch_cuda_wheel','torch_cuda_arch')}
    result={**marker,'python':str(py),'cuda_verified':True,'status':'ISOLATED_ENV_CUDA_IMPORT_SMOKE_PASSED',
            'mast3r_inference_verified':True,'model_training_or_quality_verified':False}
    (reports/'cuda_full_preflight.json').write_text(json.dumps(result,indent=2))
    (root.parent/(root.name+'_installed.json')).write_text(json.dumps(result,indent=2))
    frozen=subprocess.check_output([str(py),'-m','pip','freeze'],env=env,text=True)
    (reports/'pip_freeze.txt').write_text(frozen)
    print('FULL_PREFLIGHT_PASSED_NO_TRAINING',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','env-root','profile','frames','reports'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pytorch3d-ready-marker',type=Path)
    p.add_argument('--pytorch3d-dist',type=Path)
    p.add_argument('--wait-seconds',type=int,default=2400)
    a=p.parse_args()
    try:finish(a)
    except Exception:
        a.reports.mkdir(parents=True,exist_ok=True)
        dest=a.reports/('preflight_failure_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.txt')
        dest.write_text(traceback.format_exc());raise

if __name__=='__main__':main()
