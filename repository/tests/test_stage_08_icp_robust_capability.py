"""Tests for the Stage 8 ICP robust-kernel capability probe.

This test file MUST run successfully whether the environment has Open3D ≥ 0.17
(robust kernels available), older Open3D (no robust kernels), or no Open3D at
all. It does that by gating the "kernel built" assertions on the runtime
capability probe rather than on Open3D directly.
"""

from __future__ import annotations

from pipeline.stage_08_bim_eval.icp_robust_capability import (
    SUPPORTED_KERNELS,
    RobustKernelDecision,
    binding_supports_robust_kernel,
    build_robust_kernel,
    normalize_kernel_name,
)


def test_supported_kernels_set() -> None:
    assert set(SUPPORTED_KERNELS) == {"none", "huber", "tukey"}


def test_normalize_kernel_name_canonicalizes() -> None:
    assert normalize_kernel_name(None) == "none"
    assert normalize_kernel_name("") == "none"
    assert normalize_kernel_name("L2") == "none"
    assert normalize_kernel_name("disabled") == "none"
    assert normalize_kernel_name("Huber") == "huber"
    assert normalize_kernel_name("TUKEY") == "tukey"
    assert normalize_kernel_name("biweight") == "tukey"
    # Unknown values fall back to "none" (we never want to silently apply the
    # wrong kernel because of a typo in YAML).
    assert normalize_kernel_name("welsch") == "none"


def test_build_robust_kernel_none_always_succeeds() -> None:
    kernel, decision = build_robust_kernel("none", k_m=0.05)
    assert kernel is None
    assert decision.requested == "none"
    assert decision.applied == "none"
    assert decision.fallback_reason is None


def test_build_robust_kernel_non_positive_k_falls_back() -> None:
    kernel, decision = build_robust_kernel("tukey", k_m=0.0)
    assert kernel is None
    assert decision.applied == "none"
    assert decision.fallback_reason is not None
    assert "non_positive" in decision.fallback_reason


def test_build_robust_kernel_unknown_name_normalized_to_none() -> None:
    _, decision = build_robust_kernel("garbage", k_m=0.05)
    assert decision.requested == "none"
    assert decision.applied == "none"


def test_build_robust_kernel_huber_when_supported() -> None:
    kernel, decision = build_robust_kernel("huber", k_m=0.05)
    if binding_supports_robust_kernel():
        assert kernel is not None
        assert decision.applied == "huber"
        assert decision.fallback_reason is None
    else:
        # Older Open3D / missing Open3D: fallback path must be honest.
        assert kernel is None
        assert decision.applied == "none"
        assert decision.fallback_reason is not None


def test_build_robust_kernel_tukey_when_supported() -> None:
    kernel, decision = build_robust_kernel("tukey", k_m=0.05)
    if binding_supports_robust_kernel():
        assert kernel is not None
        assert decision.applied == "tukey"
    else:
        assert kernel is None
        assert decision.applied == "none"


def test_decision_is_immutable_dataclass() -> None:
    _, decision = build_robust_kernel("none", k_m=0.05)
    assert isinstance(decision, RobustKernelDecision)
    # frozen=True → field assignment must raise.
    try:
        decision.k_m = 0.10  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("RobustKernelDecision should be frozen")
