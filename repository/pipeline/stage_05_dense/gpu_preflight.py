"""CUDA/GPU preflight and adaptive profile helpers for Stage 5."""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from pipeline.stage_03_colmap.colmap_cli import ColmapExecutionError, ColmapRunner

from .config_access import cfg_bool, cfg_get, cfg_int


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int


@dataclass(frozen=True)
class DenseRuntimeProfile:
    cuda_build_detected: bool | None
    visible_gpus: list[GpuInfo]
    selected_gpu_index: str
    overrides: dict[str, Any]
    notes: list[str]


def _record_text(record: Any) -> str:
    return "\n".join(getattr(record, "stdout_tail", []) or [])


def probe_colmap_cuda_build(runner: ColmapRunner, logger: logging.Logger) -> bool | None:
    """Return True for CUDA build, False for explicit non-CUDA, None if unknown."""
    record = runner.run(["help"], name="colmap_cuda_preflight", check=False)
    text = _record_text(record).lower()
    if "without cuda" in text:
        return False
    if "with cuda" in text:
        return True
    # Some downstream packages omit the build suffix from `colmap help`. In that
    # case avoid claiming either way; patch_match_stereo will still be the final
    # authority, but the report will show that the build marker was unknown.
    logger.warning("Could not determine whether COLMAP was built with CUDA from `colmap help`.")
    return None


def _probe_gpus_with_nvidia_smi(logger: logging.Logger) -> list[GpuInfo]:
    """Probe visible GPUs through nvidia-smi when available inside the container."""
    if shutil.which("nvidia-smi") is None:
        logger.info("nvidia-smi is not available inside the container; trying PyTorch CUDA probe if installed.")
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=15)
    except Exception as exc:  # pragma: no cover - runtime defensive path.
        logger.warning("Could not run nvidia-smi for Stage 5 GPU preflight: %s", exc)
        return []
    if result.returncode != 0:
        logger.warning("nvidia-smi failed during Stage 5 GPU preflight: %s", result.stderr.strip() or result.stdout.strip())
        return []
    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GpuInfo(index=int(parts[0]), name=parts[1], memory_total_mb=int(float(parts[2]))))
        except ValueError:
            continue
    return gpus


def _probe_gpus_with_torch(logger: logging.Logger) -> list[GpuInfo]:
    """Fallback GPU probe using torch when nvidia-smi is absent.

    Some CUDA-capable runtime images expose the NVIDIA driver libraries but not
    the nvidia-smi executable. Stage 6 will likely use PyTorch for DA3, so this
    optional probe gives us a second, dependency-free-at-runtime path: it is used
    only if torch is already installed in the image.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on target image.
        logger.info("PyTorch CUDA probe is unavailable: %s", exc)
        return []
    try:
        if not torch.cuda.is_available():
            logger.warning("PyTorch is installed but torch.cuda.is_available() is false inside the container.")
            return []
        gpus: list[GpuInfo] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=str(props.name),
                    memory_total_mb=int(props.total_memory // (1024 * 1024)),
                )
            )
        return gpus
    except Exception as exc:  # pragma: no cover - runtime defensive path.
        logger.warning("PyTorch CUDA probe failed during Stage 5 GPU preflight: %s", exc)
        return []


def probe_visible_gpus(logger: logging.Logger) -> list[GpuInfo]:
    """Probe visible GPUs using nvidia-smi, then optional PyTorch fallback."""
    gpus = _probe_gpus_with_nvidia_smi(logger)
    if not gpus:
        gpus = _probe_gpus_with_torch(logger)
    if not gpus:
        logger.warning(
            "No visible GPUs detected by nvidia-smi or optional PyTorch probe; "
            "Stage 5 GPU auto-tuning will use YAML defaults."
        )
    return gpus


def _best_gpu_index(gpus: list[GpuInfo]) -> str:
    if not gpus:
        return "-1"
    return str(max(gpus, key=lambda item: item.memory_total_mb).index)


def _cap_profile_for_image_count(overrides: dict[str, Any], input_image_count: int, profile_name: str) -> tuple[dict[str, Any], str]:
    """Cap aggressive high-VRAM settings when the image graph is large.

    PatchMatch memory is driven not only by image resolution and VRAM, but also
    by the number of source views in the dense workspace. In long construction
    walkthroughs, many images can observe the same area; therefore high-end GPUs
    should still avoid very large dense image sizes for large registered sets.
    """
    capped = dict(overrides)
    if input_image_count >= 200:
        cap = 1400
        capped["dense.num_patch_match_src_images"] = min(int(capped.get("dense.num_patch_match_src_images", 15)), 12)
        profile_name += "+very-large-image-set-cap"
    elif input_image_count >= 120:
        cap = 1500
        capped["dense.num_patch_match_src_images"] = min(int(capped.get("dense.num_patch_match_src_images", 15)), 14)
        profile_name += "+large-image-set-cap"
    elif input_image_count >= 50:
        cap = 1600
        capped["dense.num_patch_match_src_images"] = min(int(capped.get("dense.num_patch_match_src_images", 20)), 15)
        profile_name += "+medium-image-set-cap"
    else:
        return capped, profile_name
    capped["dense.max_image_size"] = min(int(capped.get("dense.max_image_size", cap)), cap)
    capped["dense.patch_match_max_image_size"] = min(int(capped.get("dense.patch_match_max_image_size", cap)), cap)
    return capped, profile_name


def _profile_overrides_for_memory(max_mem_mb: int, input_image_count: int = 0) -> tuple[dict[str, Any], str]:
    """Return dense overrides for the largest visible GPU and image-set size.

    The memory class chooses an upper envelope, then the input-image count caps
    resolution and dense source views to avoid OOM on long video walks. Server
    users can still disable adaptive_gpu_profile or manually override values for
    controlled benchmarking.
    """
    if max_mem_mb >= 22000:  # A5000 / 3090 Ti / 4090 class
        overrides = {
            "dense.max_image_size": 2200,
            "dense.patch_match_max_image_size": 2200,
            "dense.patch_window_radius": 5,
            "dense.patch_num_samples": 15,
            "dense.patch_num_iterations": 5,
            "dense.patch_match_cache_size": 64,
            "dense.num_patch_match_src_images": 20,
            "dense.fusion_max_image_size": -1,
        }
        return _cap_profile_for_image_count(overrides, input_image_count, "24GB-class")
    if max_mem_mb >= 14000:
        overrides = {
            "dense.max_image_size": 1800,
            "dense.patch_match_max_image_size": 1800,
            "dense.patch_window_radius": 5,
            "dense.patch_num_samples": 15,
            "dense.patch_num_iterations": 5,
            "dense.patch_match_cache_size": 48,
            "dense.num_patch_match_src_images": 18,
            "dense.fusion_max_image_size": -1,
        }
        return _cap_profile_for_image_count(overrides, input_image_count, "16GB-class")
    if max_mem_mb >= 9000:
        overrides = {
            "dense.max_image_size": 1500,
            "dense.patch_match_max_image_size": 1500,
            "dense.patch_window_radius": 5,
            "dense.patch_num_samples": 12,
            "dense.patch_num_iterations": 5,
            "dense.patch_match_cache_size": 32,
            "dense.num_patch_match_src_images": 15,
            "dense.fusion_max_image_size": -1,
        }
        return _cap_profile_for_image_count(overrides, input_image_count, "10-12GB-class")
    if max_mem_mb >= 5500:
        overrides = {
            "dense.max_image_size": 1200,
            "dense.patch_match_max_image_size": 1200,
            "dense.patch_window_radius": 5,
            "dense.patch_num_samples": 10,
            "dense.patch_num_iterations": 4,
            "dense.patch_match_cache_size": 24,
            "dense.num_patch_match_src_images": 12,
            "dense.fusion_max_image_size": -1,
        }
        return _cap_profile_for_image_count(overrides, input_image_count, "6-8GB-class")
    overrides = {
        "dense.max_image_size": 1000,
        "dense.patch_match_max_image_size": 1000,
        "dense.patch_window_radius": 5,
        "dense.patch_num_samples": 8,
        "dense.patch_num_iterations": 4,
        "dense.patch_match_cache_size": 16,
        "dense.num_patch_match_src_images": 10,
        "dense.fusion_max_image_size": -1,
    }
    return _cap_profile_for_image_count(overrides, input_image_count, "low-memory")


def build_dense_runtime_profile(runner: ColmapRunner, cfg: Any, logger: logging.Logger, input_image_count: int = 0) -> DenseRuntimeProfile:
    cuda_build = probe_colmap_cuda_build(runner, logger) if cfg_bool(cfg, "dense.cuda_preflight", True) else None
    require_cuda = cfg_bool(cfg, "dense.require_cuda", True)
    if require_cuda and cuda_build is False:
        raise ColmapExecutionError(
            "Stage 5 requires a CUDA-enabled COLMAP build for patch_match_stereo, but this COLMAP reports `without CUDA`. "
            "Use a CUDA-enabled COLMAP image/binary on the target server, then rerun Stage 5."
        )

    gpus = probe_visible_gpus(logger) if cfg_bool(cfg, "dense.gpu_preflight", True) else []
    require_visible_gpu = cfg_bool(cfg, "dense.require_visible_gpu", False)
    if require_visible_gpu and not gpus:
        raise ColmapExecutionError(
            "Stage 5 requires a visible NVIDIA GPU, but nvidia-smi reported none inside the container. "
            "Run with the GPU compose override or Docker --gpus all."
        )

    configured_gpu = str(cfg_get(cfg, "dense.patch_match_gpu_index", "auto")).strip()
    selected_gpu = _best_gpu_index(gpus) if configured_gpu.lower() == "auto" else configured_gpu
    overrides: dict[str, Any] = {"dense.patch_match_gpu_index": selected_gpu}
    notes: list[str] = []

    if cfg_bool(cfg, "dense.adaptive_gpu_profile", True) and gpus:
        max_mem = max(item.memory_total_mb for item in gpus)
        profile_image_count = input_image_count if cfg_bool(cfg, "dense.adaptive_image_count_caps", True) else 0
        adaptive_overrides, profile_name = _profile_overrides_for_memory(max_mem, input_image_count=profile_image_count)
        overrides.update(adaptive_overrides)
        notes.append(f"adaptive_gpu_profile={profile_name};max_gpu_memory_mb={max_mem};input_image_count={input_image_count}")
        logger.info("Stage 5 adaptive GPU profile selected: %s (max memory=%s MB, input images=%s)", profile_name, max_mem, input_image_count)
    elif cfg_bool(cfg, "dense.adaptive_gpu_profile", True):
        notes.append("adaptive_gpu_profile_unavailable_no_visible_gpu")

    logger.info(
        "Stage 5 GPU preflight: cuda_build=%s visible_gpus=%s selected_gpu_index=%s",
        cuda_build,
        [gpu.__dict__ for gpu in gpus],
        selected_gpu,
    )
    return DenseRuntimeProfile(cuda_build, gpus, selected_gpu, overrides, notes)
