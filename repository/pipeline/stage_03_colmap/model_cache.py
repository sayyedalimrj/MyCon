"""Persistent ONNX model cache for COLMAP learned front ends.

COLMAP can download these models automatically, but Docker ``run --rm`` containers
lose ``/root/.cache``. Caching them under ``data/sfm/model_cache`` makes Stage 3
repeatable and avoids repeated long downloads.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import cfg_bool, cfg_get, cfg_int, resolve_project_path


@dataclass(frozen=True, slots=True)
class OnnxModelSpec:
    name: str
    filename: str
    url: str
    sha256: str


DEFAULT_MODELS: dict[str, OnnxModelSpec] = {
    "aliked_n16rot": OnnxModelSpec(
        name="aliked_n16rot",
        filename="aliked-n16rot.onnx",
        url="https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
        sha256="39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    ),
    "aliked_n32": OnnxModelSpec(
        name="aliked_n32",
        filename="aliked-n32.onnx",
        url="https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n32.onnx",
        sha256="a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c",
    ),
    "aliked_lightglue": OnnxModelSpec(
        name="aliked_lightglue",
        filename="aliked-lightglue.onnx",
        url="https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
        sha256="b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    ),
    "sift_lightglue": OnnxModelSpec(
        name="sift_lightglue",
        filename="sift-lightglue.onnx",
        url="https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx",
        sha256="e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
    ),
}


def _model_spec_from_config(cfg: Any, key: str) -> OnnxModelSpec:
    prefix = f"colmap.onnx_models.{key}"
    default = DEFAULT_MODELS[key]
    filename = str(cfg_get(cfg, f"{prefix}.filename", default.filename))
    url = str(cfg_get(cfg, f"{prefix}.url", default.url))
    sha256 = str(cfg_get(cfg, f"{prefix}.sha256", default.sha256))
    return OnnxModelSpec(name=key, filename=filename, url=url, sha256=sha256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, dest: Path, timeout_sec: int, logger: logging.Logger) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent), suffix=".download") as tmp:
        tmp_path = Path(tmp.name)
    try:
        logger.info("Downloading COLMAP ONNX model: %s", url)
        with urllib.request.urlopen(url, timeout=timeout_sec) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_model(cfg: Any, key: str, logger: logging.Logger) -> Path:
    cache_dir = resolve_project_path(cfg, "colmap.model_cache_dir", "data/sfm/model_cache")
    timeout = cfg_int(cfg, "colmap.model_download_timeout_sec", 1800)
    spec = _model_spec_from_config(cfg, key)
    path = cache_dir / spec.filename
    if path.exists() and path.stat().st_size > 0:
        actual = sha256_file(path)
        if actual == spec.sha256:
            logger.info("Using cached ONNX model %s: %s", key, path)
            return path
        logger.warning("Cached ONNX model hash mismatch for %s; re-downloading", path)
        path.unlink()
    _download_file(spec.url, path, timeout, logger)
    actual = sha256_file(path)
    if actual != spec.sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model hash mismatch for {spec.name}: expected {spec.sha256}, got {actual}")
    logger.info("Cached ONNX model %s at %s", key, path)
    return path


def feature_model_options(cfg: Any, feature_type: str, logger: logging.Logger) -> list[str]:
    """Return only options accepted by COLMAP feature_extractor."""
    if not cfg_bool(cfg, "colmap.download_models", True):
        return []
    feature_upper = feature_type.upper()
    if feature_upper == "ALIKED_N16ROT":
        return ["--AlikedExtraction.n16rot_model_path", str(ensure_model(cfg, "aliked_n16rot", logger))]
    if feature_upper == "ALIKED_N32":
        return ["--AlikedExtraction.n32_model_path", str(ensure_model(cfg, "aliked_n32", logger))]
    return []


def matcher_model_options(cfg: Any, matcher_type: str, logger: logging.Logger) -> list[str]:
    """Return only options accepted by COLMAP matcher commands."""
    if not cfg_bool(cfg, "colmap.download_models", True):
        return []
    matcher_upper = matcher_type.upper()
    if matcher_upper == "ALIKED_LIGHTGLUE":
        return ["--AlikedMatching.lightglue_model_path", str(ensure_model(cfg, "aliked_lightglue", logger))]
    if matcher_upper == "SIFT_LIGHTGLUE":
        return ["--SiftMatching.lightglue_model_path", str(ensure_model(cfg, "sift_lightglue", logger))]
    return []


def model_options_for_attempt(cfg: Any, feature_type: str, matcher_type: str, logger: logging.Logger) -> list[str]:
    """Backward-compatible helper; prefer feature_model_options/matcher_model_options."""
    return feature_model_options(cfg, feature_type, logger) + matcher_model_options(cfg, matcher_type, logger)
