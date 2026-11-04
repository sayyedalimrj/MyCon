#!/usr/bin/env python3
"""Detect the ACTUAL CUDA device, select conservative LongSplat settings.

Never claim that choosing a GPU in Colab ensures that CUDA kernels are active.
TPUs, CPU-only runtimes and unsupported compiler arches are explicitly gated.
Profiles are recommendations, NOT benchmark evidence or accuracy guarantees.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hardware:
    model: str
    vram_mib: int
    sm: int
    driver: str
    nvcc: str
    system_ram_gib: float
    free_disk_gib: float
    runtime_cuda_available: bool = False


def _command(args: list[str]) -> tuple[int,str]:
    try:
        p=subprocess.run(args,text=True,capture_output=True,timeout=15)
        return p.returncode,p.stdout.strip()
    except (OSError,subprocess.TimeoutExpired):
        return 1,''


def _nvcc_version() -> str:
    status,output=_command(['nvcc','--version'])
    m=re.search(r'release\s+(\d+\.\d+)',output)
    return m.group(1) if status==0 and m else ''


def detect_hardware() -> Hardware:
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('NO_NVIDIA_GPU: Colab TPU/CPU runtimes cannot train NVIDIA LongSplat. Select a CUDA GPU.')
    fields='name,memory.total,compute_cap,driver_version'
    code,output=_command(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'])
    if code != 0:
        code,output=_command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'])
        if code:
            raise RuntimeError('GPU_NOT_QUERYABLE: nvidia-smi cannot report the active GPU')
        raw=next((ln.split(',') for ln in output.splitlines() if ln.strip()),[])
        if len(raw)<3:raise RuntimeError('GPU_NOT_QUERYABLE: GPU report incomplete')
        name,memory,driver=[x.strip() for x in raw[:3]]
        cc=_guess_sm(name)
    else:
        raw=next((ln.split(',') for ln in output.splitlines() if ln.strip()),[])
        if len(raw)<4:raise RuntimeError('GPU_NOT_QUERYABLE: GPU report incomplete')
        name,memory,arch,driver=[x.strip() for x in raw[:4]]
        try:cc=int(round(float(arch)*10))
        except ValueError:cc=_guess_sm(name)
    mib=int(float(re.findall(r'\d+(?:\.\d+)?',memory)[0]))
    if cc == 0:raise RuntimeError('UNKNOWN_GPU_COMPUTE_CAPABILITY: cannot safely compile CUDA extensions')
    try:
        import psutil
        ram=psutil.virtual_memory().total/2**30
    except ImportError:
        ram=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**30
    disk=shutil.disk_usage('/content' if Path('/content').exists() else '/tmp').free/2**30
    return Hardware(name,mib,cc,driver,_nvcc_version(),round(ram,2),round(disk,2))


def _guess_sm(name: str) -> int:
    n=name.lower()
    if 't4' in n:return 75
    if 'l4' in n:return 89
    if 'a100' in n:return 80
    if 'h100' in n:return 90
    if 'h200' in n:return 90
    if 'b200' in n or 'b100' in n:return 100
    if 'rtx pro 6000' in n or 'g4' in n or 'blackwell' in n:return 120
    if 'v100' in n:return 70
    if 'p100' in n:return 60
    return 0


def profile_for(h: Hardware, quality: str='quality') -> dict:
    if quality not in ('quality','conservative'):
        raise ValueError('Invalid quality policy')
    if h.sm < 70 or h.vram_mib < 12000:
        raise RuntimeError(f'GPU_BELOW_RESEARCH_MINIMUM: {h.model} ({h.vram_mib} MiB); use T4/L4/A100 or better')
    if h.free_disk_gib<18:
        raise RuntimeError(f'DISK_INSUFFICIENT: {h.free_disk_gib} GiB free; need >=18 GiB for source, CUDA dependencies and cache')
    needs_blackwell_toolkit=(h.sm>=100 and (not h.nvcc or tuple(int(x) for x in h.nvcc.split('.')[:2]) < (12,8)))
    if h.sm>=100:
        if needs_blackwell_toolkit and h.free_disk_gib<30:
            raise RuntimeError('BLACKWELL_REQUIRES_CUDA_12_8: at least 30 GiB free disk to install CUDA 12.8 automatically')
        wheel='cu128'
        torch='2.7.1'
        vision='0.22.1'
        family='BLACKWELL_EXPERIMENTAL'
    elif h.sm>=89:
        wheel='cu121';torch='2.5.1';vision='0.20.1';family='ADA_OR_HOPPER'
    else:
        wheel='cu121';torch='2.5.1';vision='0.20.1';family='TURING_AMPERE'
    if h.vram_mib<20000:
        fps,width,post=8.0,512,20000
    elif h.vram_mib<35000:
        fps,width,post=10.0,640,24000
    elif h.vram_mib<70000:
        fps,width,post=10.0,768,28000
    else:
        fps,width,post=12.0,896,32000
    # LongSplat's public README recommends ~10fps, 512px for fast long-video
    # processing; larger models and high angular velocity increase RAM/compute.
    if quality=='conservative':
        fps=max(7.0,fps-2.0)
        width=max(448,width-128)
        post=max(16000,post-4000)
    return {'gpu':asdict(h),'quality_policy':quality,'family':family,
            'fps':fps,'width':width,'post_iter':post,
            'torch_version':torch,'torchvision_version':vision,'torch_cuda_wheel':wheel,
            'torch_cuda_arch':f'{h.sm//10}.{h.sm%10}',
            'num_workers':min(4,max(1,os.cpu_count()//2 if os.cpu_count() else 2)),
            'prune_ratio':0.6,'gpu_is_provisioned_by_user':True,
            'native_longsplat_gpu_training_verified':False,
            'needs_local_cuda_12_8_install':needs_blackwell_toolkit,
            'metric_accuracy_guaranteed':False,
            'notes':'Profile chooses budget; actual memory/quality depend on images and installed CUDA kernels'}


def validate_cuda_stack(h: Hardware, required: list[str] | None = None) -> dict:
    import importlib
    try:
        torch=importlib.import_module('torch')
    except ImportError as e:
        raise RuntimeError('TORCH_NOT_INSTALLED_IN_LONGSPLAT_ENV') from e
    if not torch.cuda.is_available():raise RuntimeError('TORCH_CUDA_UNAVAILABLE: LongSplat CUDA cannot run')
    c=torch.cuda.get_device_capability(0)
    sm=int(c[0])*10+int(c[1])
    if sm!=h.sm:raise RuntimeError(f'DEVICE_MISMATCH: nvidia-smi SM {h.sm}, Torch SM {sm}')
    # Memory allocate + execute, not just cuda.is_available().
    x=torch.ones((128,128),device='cuda')
    y=float((x@x).sum().item())
    del x
    if not math.isfinite(y):raise RuntimeError('CUDA_MATRIX_SMOKE_NONFINITE')
    required=required or ['simple_knn','diff_gaussian_rasterization','fused_ssim','torch_scatter','pytorch3d']
    broken=[]
    for mod in required:
        try:importlib.import_module(mod)
        except Exception as exc:broken.append(f'{mod}: {type(exc).__name__}: {exc}')
    if broken:raise RuntimeError('CUDA_EXTENSIONS_FAILED: '+' | '.join(broken))
    return {'sm':sm,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'smoke_matrix':y,
            'extensions':required,'status':'CUDA_PRECHECK_PASSED_NOT_MODEL_TRAINED'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality',choices=['quality','conservative'],default='quality')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate-stack',action='store_true')
    a=p.parse_args()
    try:
        h=detect_hardware()
        profile=profile_for(h,a.quality)
        if a.validate_stack:profile['cuda_smoke']=validate_cuda_stack(h)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        temp=a.output.with_suffix(a.output.suffix+'.tmp')
        temp.write_text(json.dumps(profile,indent=2,ensure_ascii=False),encoding='utf-8')
        temp.replace(a.output)
        print(json.dumps(profile,ensure_ascii=False,indent=2))
        return 0
    except Exception as e:
        print('GPU_AUTO_PREFLIGHT_REFUSED:',type(e).__name__,str(e),file=__import__('sys').stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
