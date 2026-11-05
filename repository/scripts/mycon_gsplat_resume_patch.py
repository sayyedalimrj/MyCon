#!/usr/bin/env python3
"""Create an isolated resumable copy of official Apache-2.0 gsplat v1.5.3.

NEVER edits the original simple_trainer.py. It inserts small, auditable changes
that serialize optimizer, LR scheduler, Gaussian parameters and densification
state AFTER the completed training step. On restart it restores those states
and begins at the next step. This is optimization continuation, not an exact
bitwise DataLoader replay (shuffle-worker RNG ordering can differ).
"""
from __future__ import annotations
import ast
import hashlib
import os
from pathlib import Path

PATCH_VERSION = "mycon-resume-v1"
ENTRY_NAME = "mycon_resumable_trainer.py"


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError("OFFICIAL_GSPLAT_V153_PATCH_SIGNATURE_CHANGED: "+before[:80])
    return text.replace(before, after, 1)


def patch_source(text: str) -> str:
    text = replace_once(text, "import time\n", "import time\nimport random\n")
    before = "        trainloader = torch.utils.data.DataLoader(\n"
    after = '''        # MyCON patch: restore complete model+optimizer+strategy state after
        # their creation, before iterating the dataset. Never use --ckpt for
        # training resume: official gsplat --ckpt is EVAL-ONLY.
        mycon_recovery = os.path.join(cfg.result_dir, "recovery")
        os.makedirs(mycon_recovery, exist_ok=True)
        mycon_saved = sorted(
            Path(mycon_recovery).glob("resume_step_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1])
        )
        if mycon_saved:
            checkpoint = torch.load(
                mycon_saved[-1], map_location=self.device, weights_only=False
            )
            if (checkpoint.get("patch") != "mycon-resume-v1"
                or checkpoint.get("data_dir") != str(cfg.data_dir)
                or checkpoint.get("max_steps") != max_steps
                or checkpoint.get("world_size") != world_size):
                raise RuntimeError("MYCON_RESUME_IDENTITY_OR_STEPS_MISMATCH")
            if set(checkpoint["splats"]) != set(self.splats):
                raise RuntimeError("MYCON_RESUME_SPLAT_KEYS_MISMATCH")
            for name, state in checkpoint["splats"].items():
                param = torch.nn.Parameter(state.to(device=self.device))
                self.splats[name] = param
                optimizer = self.optimizers[name]
                optimizer.param_groups[0]["params"] = [param]
                optimizer.load_state_dict(checkpoint["optimizers"][name])
            self.strategy_state = checkpoint["strategy_state"]
            if len(schedulers) != len(checkpoint["schedulers"]):
                raise RuntimeError("MYCON_RESUME_SCHEDULER_COUNT_MISMATCH")
            for scheduler, state in zip(schedulers, checkpoint["schedulers"]):
                scheduler.load_state_dict(state)
            torch.set_rng_state(checkpoint["torch_rng"].cpu())
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            random.setstate(checkpoint["python_rng"])
            init_step = int(checkpoint["last_completed_step"]) + 1
            if init_step >= max_steps:
                raise RuntimeError(
                    "MYCON_FINAL_RECOVERY_CHECKPOINT_PRESENT_USE_EVAL_ONLY"
                )
            print("MYCON_RESUME_RESTORED_FULL_STATE_AFTER_STEP",init_step-1,
                  "NEXT_STEP",init_step,flush=True)

        trainloader = torch.utils.data.DataLoader(
'''
    text=replace_once(text,before,after)
    before2="            # eval the full set\n"
    after2='''            # MyCON patch: save AFTER optimizer/scheduler and Gaussian
            # densification updates, so the restored step is truly next.
            # Snapshots are immutable and saved atomically on Drive.
            mycon_interval = int(os.environ.get("MYCON_RESUME_EVERY", "1000"))
            if mycon_interval and ((step + 1) % mycon_interval == 0 or
                                   step == max_steps - 1):
                target = os.path.join(
                    mycon_recovery, f"resume_step_{step:06d}.pt"
                )
                if os.path.exists(target):
                    raise RuntimeError("MYCON_EXISTING_RECOVERY_CHECKPOINT_PRESERVED")
                partial = target + ".partial"
                if os.path.exists(partial):
                    raise RuntimeError("MYCON_PARTIAL_RECOVERY_CHECKPOINT_PRESERVED")
                snapshot = {
                    "patch": "mycon-resume-v1",
                    "data_dir": str(cfg.data_dir),
                    "max_steps": max_steps,
                    "world_size": world_size,
                    "last_completed_step": step,
                    "splats": self.splats.state_dict(),
                    "optimizers": {
                        n: opt.state_dict() for n, opt in self.optimizers.items()
                    },
                    "schedulers": [sch.state_dict() for sch in schedulers],
                    "strategy_state": self.strategy_state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate()
                }
                torch.save(snapshot, partial)
                os.replace(partial, target)
                print("MYCON_IMMUTABLE_RECOVERY_SAVED", target, flush=True)

            # eval the full set
'''
    text=replace_once(text,before2,after2)
    ast.parse(text)
    return text


def ensure_trainer(source: Path) -> Path:
    original = source/"examples/simple_trainer.py"
    target = source/"examples"/ENTRY_NAME
    if not original.is_file() or not (source/".git").exists():
        raise RuntimeError("OFFICIAL_GSPLAT_SOURCE_MISSING_FOR_RESUME_PATCH")
    result=patch_source(original.read_text(encoding="utf-8"))
    if target.exists():
        if target.read_text(encoding="utf-8") != result:
            raise RuntimeError("MYCON_PATCHED_TRAINER_CONFLICT_PRESERVE_EXISTING")
        return target
    partial=target.with_suffix(".partial.py")
    if partial.exists():
        raise RuntimeError("MYCON_PARTIAL_TRAINER_PATCH_PRESERVED")
    try:
        partial.write_text(result,encoding="utf-8")
        os.replace(partial,target)
    finally:
        if partial.exists():partial.unlink()
    return target


if __name__=="__main__":
    import sys
    target=ensure_trainer(Path(sys.argv[1]))
    print("MYCON_RESUMABLE_GSPLAT_TRAINER_READY:",target)
