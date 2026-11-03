#!/usr/bin/env python3
"""Isolated, hardware-aware installer for the OFFICIAL NVIDIA LongSplat stack.

Runs once per actual Colab virtual machine; never overwrites MyCON R4 Python,
notebooks, checkpoints or Recorder metadata. Requires internet, conda-forge,
PyTorch/NVIDIA package servers, GitHub and possibly GitLab. Fail-closed CUDA
smoke is mandatory before launching any paid GPU training.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def run_logged(cmd: list[str], log: Path, env: dict | None=None, cwd: Path | None=None,
               timeout_s: int=3600) -> None:
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\n>>> '+' '.join(cmd)+'\n')
        f.flush()
        try:
            p=subprocess.run(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,
                             timeout=timeout_s,check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError('INSTALL_STEP_TIMEOUT: '+cmd[0]+'; see '+str(log)) from e
        if p.returncode:
            raise RuntimeError(f'INSTALL_STEP_FAILED({p.returncode}): {cmd[0]}; see {log}')


def micromamba(exe: Path, log: Path) -> Path:
    if exe.is_file():return exe
    # Official micromamba distribution; downloaded into temporary local disk.
    exe.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mycon-mamba-') as tmp:
        archive=Path(tmp)/'mamba.tar.bz2'
        try:
            req=urllib.request.Request('https://micro.mamba.pm/api/micromamba/linux-64/latest',
                                       headers={'User-Agent':'MyCON-Research-Installer/1.0'})
            with urllib.request.urlopen(req,timeout=90) as r,archive.open('wb') as dst:
                shutil.copyfileobj(r,dst)
            with tarfile.open(archive,'r:bz2') as tar:
                member=tar.getmember('bin/micromamba')
                if not member.isfile():raise RuntimeError('Official micromamba executable missing')
                with tar.extractfile(member) as source,exe.open('wb') as dst:
                    shutil.copyfileobj(source,dst)
            exe.chmod(0o755)
            run_logged([str(exe),'--version'],log)
        except Exception:
            exe.unlink(missing_ok=True)
            raise
    return exe


def nvcc_info() -> tuple[str,Path | None]:
    executable=shutil.which('nvcc')
    if not executable:return '',None
    p=subprocess.run([executable,'--version'],capture_output=True,text=True,check=False)
    m=re.search(r'release\s+(\d+\.\d+)',p.stdout)
    return (m.group(1) if m else ''), Path(executable).resolve().parent.parent


def effective_env(profile: dict,envroot: Path) -> dict:
    d=os.environ.copy()
    d['MAX_JOBS']=str(min(3,max(1,os.cpu_count()//2 if os.cpu_count() else 2)))
    d['TORCH_CUDA_ARCH_LIST']=str(profile['torch_cuda_arch'])
    version,cuda_home=nvcc_info()
    if (envroot/'bin/nvcc').is_file():
        d['CUDA_HOME']=str(envroot)
        d['PATH']=str(envroot/'bin')+':'+d.get('PATH','')
    elif cuda_home:d['CUDA_HOME']=str(cuda_home)
    d['PYTHONNOUSERSITE']='1'
    d['PIP_DISABLE_PIP_VERSION_CHECK']='1'
    d['MAMBA_ROOT_PREFIX']=str(envroot.parent/'mamba_root')
    return d


def install(profile: dict,source: Path,root: Path,log: Path, *, repair_partial: bool=False) -> dict:
    if not (source/'requirements.txt').is_file() or not (source/'submodules/mast3r').is_dir():
        raise RuntimeError('OFFICIAL_NVIDIA_SOURCE_NOT_VERIFIED: install official checkout first')
    torch=profile['torch_version'];vision=profile['torchvision_version']
    wheel=profile['torch_cuda_wheel'];arch=profile['torch_cuda_arch']
    assert wheel in ('cu121','cu128')
    root=root.expanduser().resolve();root.parent.mkdir(parents=True,exist_ok=True)
    py=root/'bin/python'
    receipt=root.parent/(root.name+'_installed.json')
    marker={k:profile[k] for k in ('torch_version','torchvision_version','torch_cuda_wheel','torch_cuda_arch')}
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if any(old.get(k)!=v for k,v in marker.items()):
            raise RuntimeError('EXISTING_ENV_DIFFERENT_GPU_OR_TORCH: do not overwrite; choose a separate env directory')
        if not py.is_file():raise RuntimeError('ENV_RECEIPT_HAS_NO_PYTHON; preserve environment for diagnosis')
        verify_cuda(py,profile,log,effective_env(profile,root))
        print('REUSED_VERIFIED_CUDA_ENV:',py)
        return old
    partial = root.exists() and any(root.iterdir())
    if partial:
        if not repair_partial:
            raise RuntimeError('PARTIAL_ENV_PRESERVED: explicitly repair packages in place or choose a new prefix')
        if not py.is_file():
            raise RuntimeError('PARTIAL_ENV_NO_PYTHON: preserve this prefix and choose a new one')
        # An explicit package repair must not reinstall Torch into an unrelated env.
        probe = subprocess.run([str(py),'-c',
            "import sys,torch,json;print(json.dumps([sys.version_info[:2],torch.__version__.split('+')[0],torch.version.cuda]))"],
            capture_output=True,text=True,check=False)
        expected_cuda='12.8' if wheel=='cu128' else '12.1'
        if probe.returncode or json.loads(probe.stdout) != [[3,10],torch,expected_cuda]:
            raise RuntimeError('PARTIAL_ENV_IDENTITY_MISMATCH: preserve existing environment')
        print('REPAIRING_PACKAGES_IN_PLACE_NO_DELETION',flush=True)
    mm=micromamba(root.parent/'tools'/'micromamba',log)
    env=effective_env(profile,root)
    if not partial:
        run_logged([str(mm),'create','-y','-p',str(root),'-c','conda-forge',
                    'python=3.10.13','cmake','ninja','pip','setuptools','wheel'],log,env=env)
    # Build CUDA toolkit must support selected physical GPU. Colab images may
    # ship a compiler older than the hardware. Install isolated nvcc, never
    # replace the host Colab CUDA toolkit or corrupt the existing MyCON R4 env.
    version,cuda_home=nvcc_info()
    local_nvcc=root/'bin/nvcc'
    if local_nvcc.is_file():
        compiler=subprocess.run([str(local_nvcc),'--version'],capture_output=True,text=True,check=False)
        match=re.search(r'release\s+(\d+\.\d+)',compiler.stdout)
        version=match.group(1) if match else ''
    sm=int(profile['gpu']['sm'])
    minimum=(12,8) if sm>=100 else (11,8) if sm>=89 else (11,0)
    if not version or tuple(map(int,version.split('.'))) != ((12,8) if wheel=='cu128' else (12,1)):
        target='12.8' if sm>=100 else '12.1'
        # NVIDIA's old meta-package can otherwise resolve a newer nvcc (13.x).
        label='cuda-12.8.1' if wheel=='cu128' else 'cuda-12.1.1'
        run_logged([str(mm),'install','-y','-p',str(root),
                    '-c','nvidia/label/'+label,'-c','conda-forge','--strict-channel-priority',
                    'cuda-toolkit='+target,'cuda-nvcc='+target,'cuda-version='+target],log,env=env)
        env['CUDA_HOME']=str(root)
        env['PATH']=str(root/'bin')+':'+env.get('PATH','')
    # Inspect the binary, never infer compiler compatibility from package names.
    actual=subprocess.run([str(root/'bin/nvcc') if (root/'bin/nvcc').is_file() else 'nvcc','--version'],
                          capture_output=True,text=True,check=False,env=env)
    match=re.search(r'release\s+(\d+\.\d+)',actual.stdout)
    expected='12.8' if wheel=='cu128' else '12.1'
    if actual.returncode or not match or match.group(1)!=expected:
        raise RuntimeError('CUDA_COMPILER_BINARY_MISMATCH: expected '+expected+'; '+actual.stdout[-300:])
    wheels='https://download.pytorch.org/whl/'+wheel
    run_logged([str(py),'-m','pip','install','--index-url',wheels,
                f'torch=={torch}',f'torchvision=={vision}'],log,env=env)
    constraints=root.parent/(root.name+'_torch_constraints.txt')
    constraints.write_text(f'torch=={torch}\ntorchvision=={vision}\n')
    env['PIP_CONSTRAINT']=str(constraints)
    # This installs upstream Python deps, including their upstream PyTorch3D
    # Git reference. CUDA extension builds can fail on some GPUs/Colab images;
    # never declare readiness until actual import and CUDA computation pass.
    # Build against the installed Torch; PEP517 isolation cannot import it.
    # Reuse an official prebuilt scatter kernel when available for this wheel.
    if wheel == 'cu121' and torch == '2.5.1':
        run_logged([str(py),'-m','pip','install','--only-binary=:all:',
                    'torch_scatter','-f','https://data.pyg.org/whl/torch-2.5.1+cu121.html'],log,env=env)
    run_logged([str(py),'-m','pip','install','iopath','fvcore'],log,env=env)
    requirements = source/'requirements.txt'
    pinned = root.parent/(root.name+'_requirements.txt')
    content = requirements.read_text().replace(
        'git+https://github.com/facebookresearch/pytorch3d.git@stable',
        'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47')
    pinned.write_text(content)
    run_logged([str(py),'-m','pip','install','--no-build-isolation','-r',str(pinned)],log,env=env,cwd=source,timeout_s=5400)
    for ext in ('simple-knn','diff-gaussian-rasterization','fused-ssim'):
        run_logged([str(py),'-m','pip','install','--no-build-isolation',str(source/'submodules'/ext)],
                   log,env=env,cwd=source,timeout_s=3600)
    # Test (and run) only after the environment is fully provisioned.
    verify_cuda(py,profile,log,env)
    report={**marker,'python':str(py),'cuda_verified':True,'status':'ISOLATED_ENV_CUDA_IMPORT_SMOKE_PASSED',
            'model_training_or_quality_verified':False}
    tmp=receipt.with_suffix('.partial.json');tmp.write_text(json.dumps(report,indent=2))
    os.replace(tmp,receipt)
    return report


def verify_cuda(py: Path,profile: dict,log: Path,env: dict):
    script=r'''import torch,importlib,json
assert torch.cuda.is_available(),'GPU inaccessible to torch'
c=torch.cuda.get_device_capability(0)
assert c[0]*10+c[1]==EXPECTED_SM,(c,EXPECTED_SM)
x=torch.ones((96,96),device='cuda');assert bool(torch.isfinite(x@x).all())
for name in ('simple_knn','diff_gaussian_rasterization','fused_ssim','torch_scatter','pytorch3d'):
 importlib.import_module(name)
# Importing an extension is NOT proof its CUDA binary supports this actual SM.
from simple_knn._C import distCUDA2
q=torch.randn((64,3),device='cuda',dtype=torch.float32)
out=distCUDA2(q);assert bool(torch.isfinite(out).all()),'simple_knn CUDA failed'
from fused_ssim import fused_ssim
v=torch.rand((1,3,32,32),device='cuda')
y=fused_ssim(v,v);assert bool(torch.isfinite(y).all()),'fused_ssim CUDA failed'
from torch_scatter import scatter_add
src=torch.ones(8,device='cuda');ind=torch.tensor([0,0,1,1,2,2,3,3],device='cuda')
z=scatter_add(src,ind,dim=0);assert z.shape[0]==4 and z.sum().item()==8
from pytorch3d.ops import knn_points
p=torch.rand((1,10,3),device='cuda');assert knn_points(p,p,K=1).dists.isfinite().all()
torch.cuda.synchronize()
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'sm':c,'extension_cuda_kernels':'PASS'}))
'''.replace('EXPECTED_SM',str(profile['gpu']['sm']))
    run_logged([str(py),'-c',script],log,env=env,timeout_s=180)
    run_logged([str(py),str(Path(__file__).with_name('mycon_cuda_raster_smoke.py'))],log,env=env,timeout_s=180)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--nvidia-root',type=Path,required=True)
    p.add_argument('--env-root',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--repair-partial',action='store_true')
    a=p.parse_args()
    print(json.dumps(install(json.loads(a.profile.read_text()),a.nvidia_root,a.env_root,a.log,repair_partial=a.repair_partial),indent=2))

if __name__=='__main__':main()
