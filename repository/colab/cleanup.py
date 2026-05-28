"""Aggressive memory & disk cleanup helpers for Colab between heavy stages.

We import torch lazily so the module is safe to call before/while the
``requirements-da3.txt`` group has been installed.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Iterable


def free_memory(verbose: bool = False) -> dict[str, str]:
    """Run gc + (optionally) torch.cuda.empty_cache and return a summary."""
    summary: dict[str, str] = {}
    collected = gc.collect()
    summary["gc_collected"] = str(collected)

    try:  # torch may or may not be installed yet.
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            summary["cuda"] = "empty_cache+ipc_collect"
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                summary["cuda_free_mb"] = f"{free_b / 1024 / 1024:.0f}"
                summary["cuda_total_mb"] = f"{total_b / 1024 / 1024:.0f}"
            except Exception:  # pragma: no cover
                pass
        else:
            summary["cuda"] = "unavailable"
    except Exception:
        summary["cuda"] = "torch_not_installed"

    if verbose:
        print("[cleanup]", summary)
    return summary


def remove_path(path: Path | str, missing_ok: bool = True) -> bool:
    """Delete a file or directory tree if it exists."""
    p = Path(path)
    if not p.exists():
        return missing_ok
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return True


def disk_usage_summary(roots: Iterable[Path | str]) -> dict[str, dict[str, int | str]]:
    """Best-effort directory size summary for the UI."""
    out: dict[str, dict[str, int | str]] = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            out[str(rp)] = {"exists": "no"}
            continue
        try:
            total = 0
            count = 0
            for p in rp.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        continue
                    count += 1
            out[str(rp)] = {
                "exists": "yes",
                "files": count,
                "bytes": total,
                "mb": int(round(total / 1024 / 1024)),
            }
        except Exception as exc:  # pragma: no cover
            out[str(rp)] = {"exists": "yes", "error": str(exc)}
    return out
