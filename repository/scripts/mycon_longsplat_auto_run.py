#!/usr/bin/env python3
"""One-run thesis controller: verified Recorder capture -> actual NVIDIA LongSplat.

Consumes the *full* original Recorder ZIP, preserving native pose/IMU/intrinsics.
GPU is detected, not chosen: Colab hardware selection must happen in the UI.
No scientific gate is bypassed. If upstream training is interrupted, original
model files stay untouched and the next run fails with a precise hold reason.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from mycon_gpu_autoprofile import detect_hardware,profile_for
from mycon_capture_safe import prepare as prepare_capture
from mycon_longsplat_autoenv import install as install_env,effective_env
from mycon_fetch_nvidia_longsplat import install as install_nvidia
from mycon_longsplat_weights import prepare_weights
from mycon_pose_qa_recovery import audit_with_recovery as audit_recorder


def atomic_json(path: Path,data: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.partial')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
    os.replace(temp,path)


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def profile_key(p: dict) -> str:
    # Do not silently resume a 512px T4 training session with 896px A100
    # hyperparameters. Separate identical video experiments by exact profile.
    d={k:p[k] for k in ('fps','width','post_iter','prune_ratio','quality_policy')}
    h=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:10]
    return f"{p['quality_policy']}_{int(p['fps'])}fps_{p['width']}w_{h}"


def drive_log(root: Path,name: str,record: dict):
    dest=root/'diagnostics'/name
    atomic_json(dest,record)
    return dest


def invoke_cmd(cmd: list[str],log: Path,env: dict | None=None,cwd: Path | None=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\n'+time.strftime('%FT%TZ',time.gmtime())+' '+json.dumps(cmd)+'\n')
        f.flush()
        p=subprocess.Popen(cmd,env=env,cwd=cwd,stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                           bufsize=1)
        # Live phase log: user sees progress in Colab, not just an idle cell.
        try:
            for line in p.stdout:
                f.write(line)
                f.flush()
                if any(k in line for k in ('TRAINING COMPLETE','NOVEL VIEW','REUSED','Geometry',
                                         'PASSED','ERROR','Traceback','FAILED','EXPORT')):
                    print(line.rstrip(),flush=True)
        finally:
            code=p.wait()
        if code:
            raise RuntimeError(f'PHASE_FAILED(exit={code}) see {log}')


def run(args) -> dict:
    capture=args.capture.expanduser().resolve()
    drive=args.drive_root.expanduser().resolve()
    if not capture.is_file():raise FileNotFoundError('Cannot locate original Recorder ZIP: '+str(capture))
    if not drive.is_dir():raise FileNotFoundError('Drive/MyCon_R4 not mounted: '+str(drive))
    local=args.local_root.expanduser().resolve()
    local.mkdir(parents=True,exist_ok=True)
    status_root=drive/'research_longsplat_auto'
    status_root.mkdir(parents=True,exist_ok=True)
    h=detect_hardware()
    profile=profile_for(h,args.quality)
    profile['checkpoint_contract']='No overwrite, no false training resume, new profile -> new experiment directory'
    # First step has zero GPU model allocations. Fail early on lack of resources.
    initial=drive_log(status_root,'hardware_preflight.json',profile)
    print('GPU DETECTED:',h.model,h.vram_mib,'MiB, SM',h.sm, 'cuda-nvcc',h.nvcc,
          'quality settings:',{k:profile[k] for k in ('fps','width','post_iter')})
    if args.plan_only:
        return {'status':'PLAN_ONLY','gpu_profile':profile,'gpu_report':str(initial)}
    # The source capture ZIP itself remains read-only on MyDrive.
    # Stable local input path also retains a receipt to recover after interrupted
    # unpacking without pretending a broken ZIP is valid.
    recorder_local=local/'verified_recorder_input'
    native=prepare_capture(capture,recorder_local,verified_rotation_cw=90)
    if not (90<=native['full_video_duration_s']<=110):
        raise RuntimeError('WRONG_CAPTURE_OR_CLIPPED_VIDEO: need the full original approx 96s worksite, not a 10-second sample')
    # Read original IMU, camera calibration, ARCore session pose. This is an
    # input-integrity/motion report; NVIDIA LongSplat remains UNPOSED and is
    # never falsely claimed to use ARCore as metric supervision.
    motion=audit_recorder(recorder_local)
    drive_log(status_root,'recorder_motion_preflight.json',motion)
    print('RECORDER QA:',motion['tracked_rows'],'tracked poses, span',round(motion['time_span_s'],1),
          's; rotation-dominant steps:',motion['rotation_dominant_small_baseline_steps'])
    captured_sha=native['archive_sha256']
    key=profile_key(profile)
    experiment_root=status_root/captured_sha[:16]/key
    workspace=experiment_root/'workspace'
    experiment_root.mkdir(parents=True,exist_ok=True)
    stable=experiment_root/'gpu_profile.json'
    if stable.is_file():
        old=json.loads(stable.read_text())
        for k in ('fps','width','post_iter','quality_policy','prune_ratio','torch_cuda_wheel'):
            if old.get(k)!=profile[k]:raise RuntimeError('PROFILE_CHANGED_WITHIN_EXPERIMENT: refuse unsafe resume')
    else:atomic_json(stable,profile)
    native_dest=experiment_root/'recorder_integrity_reference.json'
    if native_dest.is_file():
        old=json.loads(native_dest.read_text())
        if old.get('archive_sha256')!=captured_sha:raise RuntimeError('RECORDER_SOURCE_CHANGED')
    else:atomic_json(native_dest,native)
    # Upstream always remains a checked-out NVIDIA original, not a copied fork.
    nvidia_root=(getattr(args,'nvidia_root',None) or local/'NVIDIA_LongSplat').expanduser().resolve()
    official=install_nvidia(nvidia_root)
    atomic_json(experiment_root/'nvidia_original_source.json',official)
    # NVIDIA's training utilities otherwise download the multi-GB pretrained
    # MASt3R weights mid-GPU run. Pre-download/verify once and save a Drive
    # receipt for future Colab VMs. This is separate from the Recorder ZIP.
    weights=prepare_weights(nvidia_root,drive/'model_cache'/'nvidia_longsplat_mast3r')
    atomic_json(experiment_root/'pretrained_weight_receipt.json',weights)
    env_root=(getattr(args,'env_root',None) or local/f"longsplat_py310_sm{h.sm}_{profile['torch_cuda_wheel']}").expanduser().resolve()
    env_log=experiment_root/'logs'/'cuda_install.log'
    env_report=install_env(profile,nvidia_root,env_root,env_log)
    stackenv=effective_env(profile,env_root)
    # Independent CUDA smoke is part of acceptance; missing deps stop the run
    # BEFORE the costliest MASt3R + novel-view optimization.
    # The isolated LongSplat interpreter is CUDA-smoke-tested by install_env.
    # Colab host Python 3.12 need not import any native 3DGS CUDA extensions.
    atomic_json(experiment_root/'environment.json',env_report)
    stackenv['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+stackenv.get('PYTHONPATH','')
    bridge_py=Path(__file__).resolve().parents[1]/'pipeline/stage_04_5_cams_gs/longsplat_research.py'
    # Module invocation required for relative imports in geometry/audit phases.
    bridge_mod='pipeline.stage_04_5_cams_gs.longsplat_research'
    base=[sys.executable,'-m',bridge_mod,'--video',str(recorder_local/'raw_capture.mp4'),
          '--longsplat-root',str(nvidia_root),'--workspace',str(workspace),
          '--python',str(env_root/'bin/python'), '--fps',str(profile['fps']),
          '--width',str(profile['width']),'--post-iter',str(profile['post_iter']),
          '--prune-ratio',str(profile['prune_ratio']),'--rotation-cw','90',
          '--accept-research-license']
    if getattr(args,'approve_new_training',False):
        base.append('--approve-new-training')
    phases=('prepare','train','render','export','geometry','audit')
    if args.stop_after:
        phases=phases[:phases.index(args.stop_after)+1]
    runstatus={'status':'STARTED','capture_sha256':captured_sha,'selected_profile':key,
               'gpu':h.model,'stages':{},'source_video_duration_s':native['full_video_duration_s'],
               'metric_survey_approval':False,'research_use_only':True}
    statusfile=experiment_root/'run_summary.json'
    if statusfile.is_file():
        old=json.loads(statusfile.read_text())
        if old.get('capture_sha256')!=captured_sha or old.get('selected_profile')!=key:
            raise RuntimeError('EXPERIMENT_RUN_IDENTITY_MISMATCH')
        runstatus['stages']=old.get('stages',{})
    try:
        for phase in phases:
            print('NEXT:',phase, 'GPU:',h.model, 'full original video',flush=True)
            cmd=base+['--phase',phase]
            invoke_cmd(cmd,experiment_root/'logs'/(phase+'.log'),env=stackenv,
                       cwd=Path(__file__).resolve().parents[1])
            runstatus['stages'][phase]={'status':'COMMAND_EXITED_ZERO_ARTIFACTS_VERIFIED_BY_BRIDGE',
                                        'timestamp_utc':time.strftime('%FT%TZ',time.gmtime())}
            atomic_json(statusfile,runstatus)
        # Execution success and independently validated visual/BIM quality
        # are different. Never claim 'perfect geometry' from 3DGS alone.
        qa_reports=list(workspace.rglob('novel_view_qa.json'))
        audit=[]
        for report in qa_reports:
            try:audit.append(json.loads(report.read_text()))
            except (OSError,ValueError):pass
        if 'audit' in phases and not audit:
            raise RuntimeError('QA_REPORT_MISSING: all phases cannot be certified')
        if 'audit' not in phases:
            runstatus['status']='PHASES_COMPLETE_NO_VISUAL_QA'
        elif audit and all(x.get('status')=='HELDOUT_VIEWS_MEASURED' for x in audit):
            runstatus['status']='PHASES_DONE_VISUAL_QA_MEASURED_NOT_BIM_METRIC'
        else:
            runstatus['status']='PHASES_DONE_VISUAL_QUALITY_UNVERIFIED'
        runstatus['bim_metric_validated']=False
        runstatus['unseen_view_correctness_certified']=False
        atomic_json(statusfile,runstatus)
    except Exception as exc:
        runstatus.update(status='REFUSED_OR_INTERRUPTED',error_type=type(exc).__name__,
                         error=str(exc),stack_trace=traceback.format_exc()[-3000:])
        atomic_json(statusfile,runstatus)
        raise
    return {'status':runstatus['status'],'report':str(statusfile),'profile':key,
            'artifact_workspace':str(workspace),'warning':'Novel view must be visually inspected. Metric BIM accuracy requires held-out surveyed anchors.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--drive-root',type=Path,required=True)
    p.add_argument('--local-root',type=Path,default=Path('/content/MyCON_LongSplat_Work'))
    p.add_argument('--nvidia-root',type=Path,help='Reuse an existing checkout; still verifies the pinned commit and submodules')
    p.add_argument('--env-root',type=Path,help='Reuse an existing isolated environment; still verifies its receipt and CUDA kernels')
    p.add_argument('--quality',choices=('quality','conservative'),default='quality')
    p.add_argument('--stop-after',choices=('prepare','train','render','export','geometry','audit'))
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--approve-new-training',action='store_true',help='Only set after explicit approval for the new training budget')
    a=p.parse_args()
    print(json.dumps(run(a),ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('MYCON_AUTO_RUN_STOPPED_SAFELY:',type(e).__name__,str(e),file=sys.stderr)
        # Keep a durable explanation even if the failure occurs before stage
        # processing (e.g. original source ZIP, GitLab or CUDA package server).
        try:
            d=Path('/content/drive/MyDrive/MyCon_R4/research_longsplat_auto/diagnostics')
            if d.parent.is_dir():
                atomic_json(d/'last_failure.json',{'status':'REFUSED_NO_FALSE_SUCCESS',
                    'error_type':type(e).__name__,'error':str(e),
                    'timestamp_utc':time.strftime('%FT%TZ',time.gmtime()),
                    'traceback':traceback.format_exc()[-5000:]})
        except Exception:pass
        raise SystemExit(2)
