"""Hardware-aware local/offline model recommendations for YOLO and VLM deployment."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ServerProfile:
    name: str
    gpu: str
    gpu_vram_gb: float | None
    ram_gb: float | None
    recommended_yolo: str
    recommended_vlm_live: str
    recommended_vlm_offline: str
    recommended_provider: str
    ollama_model: str
    hf_model: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_profile(name: str, gpu: str, ram_gb: float | None = None, gpu_vram_gb: float | None = None) -> ServerProfile:
    gpu_l = gpu.lower()
    if "a5000" in gpu_l:
        vram = gpu_vram_gb or 24.0
        return ServerProfile(
            name,
            gpu,
            vram,
            ram_gb,
            "yolo11x-seg.pt",
            "qwen3-vl:8b-thinking",
            "qwen3-vl:30b",
            "ollama_local",
            "qwen3-vl:8b-thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
            [
                "Use CUDA-enabled COLMAP for Stage 5.",
                "Use Ollama qwen3-vl:8b-thinking for live offline QA.",
                "Use qwen3-vl:30b or Qwen/Qwen3-VL-30B-A3B-Thinking only for slower offline review.",
            ],
        )
    if "3090" in gpu_l:
        vram = gpu_vram_gb or 24.0
        return ServerProfile(
            name,
            gpu,
            vram,
            ram_gb,
            "yolo11x-seg.pt",
            "qwen3-vl:8b-thinking",
            "qwen3-vl:30b",
            "ollama_local",
            "qwen3-vl:8b-thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
            [
                "Best single-GPU target for dense stereo plus segmentation.",
                "Keep VLM evidence-based; it explains outputs and must not invent geometry.",
                "Use the selected 8B Thinking profile first; only evaluate larger models as an explicit server-side experiment.",
            ],
        )
    if "1080" in gpu_l:
        vram = gpu_vram_gb or 8.0
        return ServerProfile(
            name,
            gpu,
            vram,
            ram_gb,
            "yolo11m-seg.pt",
            "qwen3-vl:4b",
            "qwen3-vl:8b-thinking",
            "ollama_local",
            "qwen3-vl:4b",
            "Qwen/Qwen3-VL-4B-Instruct",
            [
                "Use smaller VLM models and lower rendered image resolution.",
                "Avoid heavy Stage 5 production runs on this machine.",
                "Can still host the API and run light visual QA offline.",
            ],
        )
    return ServerProfile(
        name,
        gpu or "none",
        gpu_vram_gb,
        ram_gb,
        "yolo11s-seg.pt",
        "mock_or_cpu_local",
        "none",
        "mock",
        "qwen3-vl:2b",
        "Qwen/Qwen3-VL-2B-Instruct",
        ["CPU-only or unknown GPU: use for tests, preprocessing, reports, or API without live VLM."],
    )
