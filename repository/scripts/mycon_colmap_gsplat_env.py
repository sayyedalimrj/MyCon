#!/usr/bin/env python3
"""Pinned Python3.10/Torch2.4/CUDA12.1 + gsplat1.5.3 prebuilt T4 runtime.

All packages stay in a new isolated micromamba prefix. Never modify host
PyTorch, LongSplat env or original Recorder / previous checkpoints.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_logged(cmd: list[str], log: Path, *, cwd: Path | None = None,
               env: dict | None = None, live: bool = False,
               timeout_s: int = 3600, idle_timeout_s: int = 2100) -> None:
    """Live watch with bounded stalled-runtime cost, preserving all GPU receipts.

    No shell. A separate reader thread drains stdout byte-by-byte, so tqdm's
    carriage-return-only progress is a valid heartbeat. Python subprocess is
    terminated only if genuinely idle or exceeding the explicit run budget.
    """
    import threading
    import time
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8") as f:
        f.write("\n>>> "+json.dumps(cmd)+"\n")
        f.flush()
        if not live:
            try:
                proc=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,
                                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                    check=False,timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("COMMAND_TIMEOUT_NONLIVE_SAFELY_PRESERVED: "+
                                   str(log)) from exc
            if proc.returncode:
                raise RuntimeError("COMMAND_FAILED(exit="+str(proc.returncode)+"): "+
                                   str(cmd[0])+"; log="+str(log))
            return
        proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,errors="replace",bufsize=1)
        started=time.monotonic()
        last_activity=[started]
        lines=[]
        writer_errors=[]
        def drain():
            current=[]
            try:
                while True:
                    ch=proc.stdout.read(1)
                    if not ch:break
                    last_activity[0]=time.monotonic()
                    f.write(ch)
                    if ch in ("\r","\n"):
                        line="".join(current).strip()
                        if line:
                            lines.append(line[-350:])
                            if len(lines)>15:lines.pop(0)
                            if ("Step:" in line or "error" in line.lower()
                                or "Traceback" in line
                                or "MYCON_IMMUTABLE_HASH_VERIFIED" in line):
                                print("MYCON_GPU:",line[-250:],flush=True)
                        current.clear()
                        f.flush()
                    else:
                        current.append(ch)
                        if len(current)>5000:current=current[-5000:]
            except Exception as exc:
                writer_errors.append(exc)
            finally:
                if proc.stdout is not None:proc.stdout.close()
                f.flush()
        th=threading.Thread(target=drain,daemon=True)
        th.start()
        stopped=None
        while proc.poll() is None:
            time.sleep(2)
            now=time.monotonic()
            if writer_errors:
                stopped="GPU_LOG_READER_FAILED"
                break
            if now-started > timeout_s:
                stopped="GPU_RUN_TOTAL_TIME_GUARD"
                break
            if now-last_activity[0] > idle_timeout_s:
                stopped="GPU_RUN_STALLED_NO_OUTPUT_GUARD"
                break
        if stopped:
            proc.terminate()
            try:proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            proc.wait()
        th.join(timeout=20)
        f.flush()
        if stopped:
            raise RuntimeError(stopped+"; last lines: "+" | ".join(lines[-8:])+
                               "; preserved logs="+str(log))
        if proc.returncode:
            raise RuntimeError("COMMAND_FAILED(exit="+str(proc.returncode)+"): "+
                               " | ".join(lines[-8:])[-800:]+"; log="+str(log))



def gsplat_cuda_smoke(py: Path, source: Path, env: dict, log: Path) -> None:
    code=r'''import torch,gsplat
from gsplat.rendering import rasterization
assert torch.__version__.startswith("2.4.1+cu121"),torch.__version__
assert gsplat.__version__=="1.5.3",gsplat.__version__
from importlib.metadata import version
assert version("gsplat")=="1.5.3+pt24cu121",version("gsplat")
assert torch.cuda.is_available()
assert tuple(torch.cuda.get_device_capability(0))==(7,5)
means=torch.tensor([[0.,0.,3.]],device="cuda",requires_grad=True)
quats=torch.tensor([[1.,0.,0.,0.]],device="cuda")
scales=torch.full((1,3),.5,device="cuda")
opacity=torch.ones(1,device="cuda")*.85
colors=torch.tensor([[.3,.5,.8]],device="cuda")
view=torch.eye(4,device="cuda")[None]
K=torch.tensor([[[40.,0.,16.],[0.,40.,16.],[0.,0.,1.]]],device="cuda")
rgb,alpha,_=rasterization(means,quats,scales,opacity,colors,view,K,
                           width=32,height=32,packed=True)
assert torch.isfinite(rgb).all()
rgb.sum().backward()
assert means.grad is not None and torch.isfinite(means.grad).all()
from fused_ssim import fused_ssim
v=torch.rand(1,3,32,32,device="cuda")
assert torch.isfinite(fused_ssim(v,v)).all()
# gsplat's final hold-out metrics use torchmetrics LPIPS/torchvision AlexNet
# weights. Exercise/download them in preflight rather than after 7000 steps.
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips=LearnedPerceptualImagePatchSimilarity(net_type="alex").to("cuda").eval()
with torch.no_grad():
  assert torch.isfinite(lpips(torch.rand(1,3,64,64,device="cuda"),
                            torch.rand(1,3,64,64,device="cuda"))).all()
torch.cuda.synchronize()
print("MYCON_GSPLAT_REAL_FORWARD_BACKWARD_AND_LPIPS_PASS",torch.__version__,gsplat.__version__)
'''
    run_logged([str(py),"-c",code],log,env=env)
    # Verify trainer imports/CLI against the pinned tag before paid training.
    run_logged([str(py),str(source/"examples/simple_trainer.py"),"default","--help"],
               log,env=env,cwd=source/"examples")


def verify_existing(prefix: Path, source: Path, log: Path) -> Path:
    """Fast fail-closed, NO-PIP/NO-CONDA check immediately before GPU training."""
    prefix=prefix.expanduser().resolve()
    py=prefix/"bin/python"
    receipt=prefix.parent/(prefix.name+"_verified.json")
    if not py.is_file() or not receipt.is_file():
        raise RuntimeError("GPU_TRAINING_REFUSED_ENV_NOT_ALREADY_PREFLIGHT_VERIFIED")
    try:
        evidence=json.loads(receipt.read_text())
    except (OSError,ValueError) as exc:
        raise RuntimeError("GPU_TRAINING_REFUSED_ENV_RECEIPT_UNREADABLE") from exc
    if evidence.get("pinned")!="gsplat-v1.5.3-torch2.4.1-cu121-t4":
        raise RuntimeError("GPU_TRAINING_REFUSED_ENV_WRONG_VERSION")
    args=["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader,nounits"]
    gpu=subprocess.run(args,capture_output=True,text=True,check=False)
    if gpu.returncode or not gpu.stdout.strip().startswith("7.5"):
        raise RuntimeError("GPU_TRAINING_REFUSED_NEED_T4_SM75")
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE="1",MAX_JOBS="2",TORCH_CUDA_ARCH_LIST="7.5")
    if (Path("/usr/bin/gcc-12")).is_file() and (Path("/usr/bin/g++-12")).is_file():
        env.update(CC="/usr/bin/gcc-12",CXX="/usr/bin/g++-12",
                   CUDAHOSTCXX="/usr/bin/g++-12")
    if (prefix/"bin/nvcc").is_file():
        env["CUDA_HOME"]=str(prefix)
        env["PATH"]=str(prefix/"bin")+os.pathsep+env.get("PATH","")
    gsplat_cuda_smoke(py,source,env,log)
    return py


def setup(prefix: Path, work: Path, source: Path, log: Path) -> Path:
    prefix=prefix.expanduser().resolve()
    # Stop before any multi-GB installs if Colab has no NVIDIA T4.
    gpu=subprocess.run(["nvidia-smi","--query-gpu=compute_cap",
                        "--format=csv,noheader,nounits"],
                       capture_output=True,text=True,check=False)
    if gpu.returncode or not gpu.stdout.strip():
        raise RuntimeError("GSPLAT_NO_NVIDIA_GPU_SELECT_T4_IN_COLAB_RUNTIME")
    try:
        capability=float(gpu.stdout.splitlines()[0].strip())
    except ValueError as exc:
        raise RuntimeError("GSPLAT_GPU_COMPUTE_CAPABILITY_UNREADABLE") from exc
    if round(capability*10)!=75:
        raise RuntimeError("GSPLAT_PINNED_BASELINE_REQUIRES_T4_SM75; got SM"+str(round(capability*10)))
    # Disk check before cloning CUDA toolkit / downloading multi-GB Torch wheels.
    available=shutil.disk_usage(str(work)).free if work.is_dir() else shutil.disk_usage(str(work.parent)).free
    if available < 18*(1024**3):
        raise RuntimeError(f"GPU_PREFLIGHT_LOCAL_DISK_LT_18_GIB: {available/1024**3:.1f} GiB")
    py=prefix/"bin/python"
    receipt=prefix.parent/(prefix.name+"_verified.json")
    env=os.environ.copy()
    env["PYTHONNOUSERSITE"]="1"
    env["MAX_JOBS"]="2"
    env["TORCH_CUDA_ARCH_LIST"]="7.5"
    env["PIP_DISABLE_PIP_VERSION_CHECK"]="1"
    # Micromamba -p environments still need a stable root prefix in Colab.
    env["MAMBA_ROOT_PREFIX"]=str(work/"mamba_root")
    if (Path("/usr/bin/g++-12")).is_file():
        env.update(CC="/usr/bin/gcc-12",CXX="/usr/bin/g++-12",CUDAHOSTCXX="/usr/bin/g++-12")
    if (prefix/"bin/nvcc").is_file():
        env["CUDA_HOME"]=str(prefix)
        env["PATH"]=str(prefix/"bin")+os.pathsep+env.get("PATH","")
    if receipt.is_file():
        old=json.loads(receipt.read_text())
        if old.get("pinned")!="gsplat-v1.5.3-torch2.4.1-cu121-t4":
            raise RuntimeError("GSPLAT_ENV_WRONG_IDENTITY")
        gsplat_cuda_smoke(py,source,env,log)
        print("MYCON_GSPLAT_REUSED_CUDA_VERIFIED",flush=True)
        return py
    intent=prefix.parent/(prefix.name+"_install_intent.json")
    partial=prefix.exists() and any(prefix.iterdir())
    if partial:
        # A previously interrupted package installation can finish cleanly.
        # Re-check all real CUDA kernels first, then allow ONE controlled
        # idempotent repair; never delete/reinstall Torch in a known good env.
        if py.is_file():
            try:
                gsplat_cuda_smoke(py,source,env,log)
            except Exception:
                pass
            else:
                receipt.write_text(json.dumps({
                    "pinned":"gsplat-v1.5.3-torch2.4.1-cu121-t4",
                    "python":str(py),"status":"ADOPTED_EXISTING_ALL_REAL_KERNELS_PASSED",
                    "trained_model_verified":False},indent=2)+"\n")
                print("MYCON_GSPLAT_ADOPTED_EXISTING_ENV_AFTER_FULL_CUDA_SMOKE",flush=True)
                return py
            # Refuse to install into a prefix containing a different working
            # PyTorch/CUDA identity; such a prefix is not ours to modify.
            probe=subprocess.run([str(py),"-c",
                "import json,sys;import torch;print(json.dumps([list(sys.version_info[:2]),torch.__version__,torch.version.cuda]))"],
                capture_output=True,text=True)
            if probe.returncode==0:
                try:
                    identity=json.loads(probe.stdout.strip().splitlines()[-1])
                except (ValueError,IndexError):
                    raise RuntimeError("PARTIAL_ENV_TORCH_IDENTITY_UNREADABLE")
                if (identity[0]!=[3,10] or not identity[1].startswith("2.4.1+cu121")
                    or identity[2]!="12.1"):
                    raise RuntimeError("PARTIAL_ENV_WRONG_TORCH_ABI_PRESERVE")
        if not intent.is_file() or json.loads(intent.read_text()).get("pinned")!="gsplat-v1.5.3-torch2.4.1-cu121-t4":
            raise RuntimeError("GSPLAT_PARTIAL_ENV_PRESERVED_UNKNOWN_ORIGIN")
        stamp=prefix.parent/(prefix.name+"_one_repair_attempted.json")
        if stamp.is_file():
            raise RuntimeError("GSPLAT_PARTIAL_ENV_ALREADY_RETRIED_ONCE_PRESERVED")
        stamp.write_text(json.dumps({"status":"REPAIR_IN_PROGRESS_NO_DELETION",
                                     "prefix":str(prefix)},indent=2)+"\n")
        print("MYCON_ONCE_ONLY_REPAIRING_PARTIAL_ENV_WITHOUT_DELETION",flush=True)
    if not partial and not intent.is_file():
        intent.write_text(json.dumps({
            "pinned":"gsplat-v1.5.3-torch2.4.1-cu121-t4",
            "prefix":str(prefix),"source":"MYCON_GSPLAT_ISOLATED_INSTALL"},indent=2)+"\n")
    # CUDA 12.1 C++ extensions REJECT host GCC newer than 12. A prior real
    # MyCON LongSplat run hit this exact failure. Install on the GPU VM
    # BEFORE fetching multi-GB torch wheels or invoking nvcc. Unknown prior
    # environments are rejected before any host apt modification.
    if not (Path("/usr/bin/gcc-12").is_file() and Path("/usr/bin/g++-12").is_file()):
        run_logged(["apt-get","update","-qq"],log)
        run_logged(["apt-get","install","-y","gcc-12","g++-12"],log)
    if not (Path("/usr/bin/gcc-12").is_file() and Path("/usr/bin/g++-12").is_file()):
        raise RuntimeError("GCC12_REQUIRED_BEFORE_CUDA121_NATIVE_EXTENSION")
    # These variables were initialized BEFORE apt. Set them again explicitly
    # now or nvcc still invokes the incompatible default Colab GCC.
    env.update(CC="/usr/bin/gcc-12",CXX="/usr/bin/g++-12",
               CUDAHOSTCXX="/usr/bin/g++-12")
    # Use the already referenced safe micromamba downloader from MyCON.
    from mycon_longsplat_autoenv import micromamba
    mm=micromamba(work/"tools/micromamba",log)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    run_logged([str(mm),"install" if partial else "create","-y","-p",str(prefix),"-c","conda-forge",
                "python=3.10.13","pip","setuptools","wheel","ninja","cmake"],log,env=env)
    run_logged([str(mm),"install","-y","-p",str(prefix),
                "-c","nvidia/label/cuda-12.1.1","-c","conda-forge",
                "--strict-channel-priority","cuda-nvcc=12.1","cuda-version=12.1",
                "cuda-toolkit=12.1"],log,env=env)
    env["CUDA_HOME"]=str(prefix)
    env["PATH"]=str(prefix/"bin")+os.pathsep+env.get("PATH","")
    nvcc=subprocess.run([str(prefix/"bin/nvcc"),"--version"],capture_output=True,text=True)
    if nvcc.returncode or "release 12.1" not in nvcc.stdout:
        raise RuntimeError("GSPLAT_CUDA_COMPILER_MISMATCH_EXPECT_12_1")
    run_logged([str(py),"-m","pip","install","--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.4.1","torchvision==0.19.1"],log,env=env)
    constraints=prefix.parent/(prefix.name+"_constraints.txt")
    constraints.write_text("torch==2.4.1\ntorchvision==0.19.1\nnumpy==1.26.4\n")
    env["PIP_CONSTRAINT"]=str(constraints)
    run_logged([str(py),"-m","pip","install","numpy==1.26.4","ninja","rich",
                "jaxtyping","scipy","scikit-learn","viser==0.2.3",
                "imageio[ffmpeg]","opencv-python-headless==4.10.0.84",
                "Pillow==10.4.0","tqdm","tyro==0.8.14","pyyaml",
                "matplotlib","tensorboard","tensorly","splines",
                "torchmetrics[image]==1.4.3","plyfile"],log,env=env)
    # Verify official July 2025 GitHub release SHA256 BEFORE installing native
    # code. This exact 21.1MB cp310 Linux wheel is published by gsplat v1.5.3.
    # Never install a substituted wheel or unexpectedly built source package.
    from mycon_audit_runtime import sha256
    wheel_sha="0493bab68ed5fc71f4ce8bfc2be03b584d8a41a06a6d9362e09a795340f8c488"
    wheel_dir=work/"verified_wheels"/"gsplat_v153_pt24cu121_cp310"
    wheel_dir.mkdir(parents=True,exist_ok=True)
    wheels=list(wheel_dir.glob("gsplat-1.5.3+pt24cu121-cp310-cp310-linux_x86_64.whl"))
    if not wheels:
        run_logged([str(py),"-m","pip","download","--no-deps",
                    "--only-binary=:all:","--dest",str(wheel_dir),
                    "--index-url","https://docs.gsplat.studio/whl/pt24cu121",
                    "gsplat==1.5.3+pt24cu121"],log,env=env,timeout_s=900)
        wheels=list(wheel_dir.glob("gsplat-1.5.3+pt24cu121-cp310-cp310-linux_x86_64.whl"))
    if len(wheels)!=1 or sha256(wheels[0])!=wheel_sha:
        raise RuntimeError("MYCON_GSPLAT_OFFICIAL_WHEEL_SHA256_MISMATCH")
    run_logged([str(py),"-m","pip","install","--no-deps",str(wheels[0])],
               log,env=env)
    for requirement in (
        "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e",
        "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787",
    ):
        run_logged([str(py),"-m","pip","install",requirement],log,env=env)
    # Prebuild the one remaining upstream fused CUDA SSIM extension with isolation OFF.
    run_logged([str(py),"-m","pip","install","--no-build-isolation",
                "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"],
               log,env=env)
    gsplat_cuda_smoke(py,source,env,log)
    # Freeze the resolved transitive dependency set + driver/compiler identity
    # as evidence. Upstream floating wheels must never be silently mistaken
    # for a byte-for-byte reproducible environment.
    freeze=subprocess.run([str(py),"-m","pip","freeze","--all"],
                          capture_output=True,text=True,env=env,check=True)
    driver=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,compute_cap",
                           "--format=csv,noheader"],
                          capture_output=True,text=True,check=True)
    from mycon_audit_runtime import sha256
    constraints_out=prefix.parent/(prefix.name+"_pip_freeze.txt")
    if constraints_out.exists():
        if constraints_out.read_text()!=freeze.stdout:
            raise RuntimeError("GSPLAT_PIP_FREEZE_CONFLICT_PRESERVE")
    else:
        constraints_out.write_text(freeze.stdout)
    doc={"pinned":"gsplat-v1.5.3-torch2.4.1-cu121-t4",
         "python":str(py),"status":"ACTUAL_GSPLAT_CUDA_FWD_BWD_SMOKE_PASSED",
         "torch_cuda_driver":driver.stdout.strip(),
         "compiler":"GCC 12, nvcc 12.1",
         "gsplat_source":"v1.5.3",
         "pip_freeze_sha256":sha256(constraints_out),
         "trained_model_verified":False}
    tmp=receipt.with_suffix(".partial.json")
    if tmp.exists():raise RuntimeError("GSPLAT_PARTIAL_VERIFICATION_RECEIPT_PRESERVED")
    tmp.write_text(json.dumps(doc,indent=2)+"\n")
    os.replace(tmp,receipt)
    return py
