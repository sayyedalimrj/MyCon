#!/usr/bin/env python3
"""Repack installed native extensions without rebuilding them or changing the environment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for package in ('simple_knn','diff_gaussian_rasterization','fused_ssim'):
        dist=importlib.metadata.distribution(package)
        with tempfile.TemporaryDirectory(prefix='mycon-wheel-repack-') as directory:
            staging=Path(directory)
            for item in dist.files or []:
                relative=Path(str(item))
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unexpected external distribution file: '+str(item))
                if relative.suffix=='.pyc':continue
                source=Path(dist.locate_file(item))
                if source.is_file():
                    target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
            subprocess.run([sys.executable,'-m','wheel','pack',str(staging),'-d',str(a.output)],check=True)
    rope=a.source/'submodules/mast3r/dust3r/croco/models/curope'
    for binary in rope.glob('*.so'):shutil.copy2(binary,a.output/binary.name)
    import torch
    report={'schema':'mycon.native_stack_cache.v1','method':'repack installed files; no kernel rebuild',
            'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
            'device_capability':torch.cuda.get_device_capability(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip(),
            'files':{x.name:{'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()}
                     for x in sorted(a.output.iterdir()) if x.is_file()},
            'compatibility':'Restore only with the recorded Python/Torch/CUDA ABI and re-run kernel tests',
            'training_executed':False}
    (a.output/'cache_receipt.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
