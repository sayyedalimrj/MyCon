"""Capability probe for Open3D robust ICP kernels.

This module decouples the *intent* to use a robust loss (Huber / Tukey) from the
ability of the running Open3D build to actually deliver one. The point-to-plane
estimator gained a ``kernel=`` constructor argument in Open3D 0.17. On older
builds the call must fall back to non-robust point-to-plane and emit a
structured warning so the run report records why.

Design notes
------------

- We deliberately avoid importing Open3D at module import time. Some test
  environments stub Open3D; a clean import means the tests can probe
  ``HAS_ROBUST_LOSS`` without forcing a full Open3D import chain.
- ``build_robust_kernel`` returns ``None`` when the requested kernel is "none"
  *or* when the running build cannot construct one. Callers should treat
  ``None`` as "use the bare estimator" rather than as an error.
- The kernel scale parameter ``k_m`` is in meters (matched to
  ``progress.deviation_threshold_m`` by convention). For Huber, ``k`` is the
  cutoff at which the loss switches from quadratic to linear. For Tukey, ``k``
  is the redescending cutoff beyond which residuals contribute zero gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_KERNELS: tuple[str, ...] = ("none", "huber", "tukey")


@dataclass(frozen=True)
class RobustKernelDecision:
    """Trace record for the kernel construction step."""

    requested: str
    applied: str
    k_m: float
    binding_supports_kernel: bool
    fallback_reason: str | None


def _import_o3d() -> Any | None:
    try:
        import open3d as o3d  # noqa: WPS433 (intentional in-function import)
        return o3d
    except Exception:
        return None


def binding_supports_robust_kernel() -> bool:
    """Return True if Open3D's point-to-plane estimator accepts ``kernel=``.

    The probe constructs a default ``HuberLoss`` and tries the kernel-keyword
    constructor of ``TransformationEstimationPointToPlane``. Both Open3D 0.17+
    behaviors are accepted (positional or keyword).
    """
    o3d = _import_o3d()
    if o3d is None:
        return False
    reg = getattr(o3d, "pipelines", None)
    if reg is None or not hasattr(reg, "registration"):
        return False
    reg = reg.registration
    huber_cls = getattr(reg, "HuberLoss", None)
    p2pl_cls = getattr(reg, "TransformationEstimationPointToPlane", None)
    if huber_cls is None or p2pl_cls is None:
        return False
    try:
        kernel = huber_cls(0.05)
    except Exception:
        return False
    try:
        p2pl_cls(kernel)
        return True
    except Exception:
        return False


def normalize_kernel_name(name: str | None) -> str:
    """Normalize a config-provided kernel name to one of SUPPORTED_KERNELS."""
    if not name:
        return "none"
    raw = str(name).strip().lower()
    if raw in {"", "none", "off", "disabled", "l2"}:
        return "none"
    if raw in {"huber"}:
        return "huber"
    if raw in {"tukey", "biweight"}:
        return "tukey"
    return "none"


def build_robust_kernel(name: str, k_m: float) -> tuple[Any | None, RobustKernelDecision]:
    """Construct a robust kernel object plus a structured trace record.

    Returns ``(kernel_or_none, decision)``. The decision lets callers attach
    "what we tried, what we got" to their JSON report without having to repeat
    the capability dance.
    """
    requested = normalize_kernel_name(name)
    k_m = float(k_m) if k_m is not None else 0.05
    if k_m <= 0:
        return None, RobustKernelDecision(
            requested=requested,
            applied="none",
            k_m=k_m,
            binding_supports_kernel=binding_supports_robust_kernel(),
            fallback_reason=f"non_positive_kernel_scale:{k_m}",
        )

    if requested == "none":
        return None, RobustKernelDecision(
            requested="none",
            applied="none",
            k_m=k_m,
            binding_supports_kernel=binding_supports_robust_kernel(),
            fallback_reason=None,
        )

    o3d = _import_o3d()
    supports = binding_supports_robust_kernel()
    if o3d is None or not supports:
        return None, RobustKernelDecision(
            requested=requested,
            applied="none",
            k_m=k_m,
            binding_supports_kernel=supports,
            fallback_reason="open3d_binding_does_not_expose_robust_kernel",
        )

    reg = o3d.pipelines.registration
    cls_name = "HuberLoss" if requested == "huber" else "TukeyLoss"
    cls = getattr(reg, cls_name, None)
    if cls is None:
        return None, RobustKernelDecision(
            requested=requested,
            applied="none",
            k_m=k_m,
            binding_supports_kernel=supports,
            fallback_reason=f"missing_kernel_class:{cls_name}",
        )

    try:
        kernel = cls(k_m)
    except Exception as exc:
        return None, RobustKernelDecision(
            requested=requested,
            applied="none",
            k_m=k_m,
            binding_supports_kernel=supports,
            fallback_reason=f"kernel_construction_failed:{exc}",
        )

    return kernel, RobustKernelDecision(
        requested=requested,
        applied=requested,
        k_m=k_m,
        binding_supports_kernel=True,
        fallback_reason=None,
    )


__all__ = [
    "SUPPORTED_KERNELS",
    "RobustKernelDecision",
    "binding_supports_robust_kernel",
    "normalize_kernel_name",
    "build_robust_kernel",
]
