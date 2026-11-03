# Colab V4 execution repair — 2026-11-03

This branch builds on PR #29 (e904847), then incorporates the newer Python controller and tests from the supplied V4 handoff. It does not merge into main or replace the original Recorder.

## Observed in the connected Colab

- Existing runtime: Tesla T4, 15360 MiB, driver 580.82.07; host toolkit 13.0.
- Existing NVIDIA checkout: 19750775a9d19f30aa05a8333c4c6c231b2d5f4a. All seven recursive submodules verified after the Git whitespace fix. No clone was repeated.
- Root cause: `stdout.strip()` removed the clean-prefix space on the first `git submodule status` line. Preserve leading whitespace, continue rejecting `+`, `-` and `U` states.
- cu121 PyTorch cannot build extensions with the host CUDA 13 compiler. Install matching toolkit 12.1 in an isolated prefix, including headers and libraries. Do not change the Colab host toolkit.
- Recorder archive actually contains 436645888-byte video, pose.csv, imu.csv, camera calibration and integrity metadata. QR events are empty, controls are empty, depth is disabled with sample_count=0. Do not infer surveyed metric scale from ARCore pose.
- Current LongSplat research directory had no .ply/.pth/.pt training artifact at inventory time. This does not assert that older independent reconstruction runs lack checkpoints.

## Safety and acceptance

Partial environments are preserved. New training requires `--approve-new-training`; completed phases may be reused without that flag. The pose recovery only computes motion QA from temporary, time-sorted copies, never injects corrected sensor poses as surveyed truth.

CUDA installation, actual CUDA kernels, MASt3R loading, training, held-out evaluation, novel-view inspection and independent metric validation have distinct acceptance states. A source verification or PLY is not a final reconstruction result.

The existing depth-fusion product comes from LongSplat renderer depths. It is a **dependent, nonmetric candidate**, not the independent geometry branch required for final BIM acceptance. Independent image/trajectory geometry and external controls remain necessary.

44 targeted offline tests passed at the first integration commit. Live CUDA outcome must be read from the timestamped Drive diagnostics; this document does not certify GPU kernels or training.

## Additional live evidence

769 full-duration upright frames were extracted at 8 FPS with per-frame SHA256 inventory, and a six-frame contact sheet was visually inspected. The official MASt3R checkpoint was cached on Drive: 2754910614 bytes, SHA256 e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2. Basic PyTorch CUDA allocation and arithmetic passed on the T4. This is not an extension-kernel or training certificate.

The first toolkit attempt exposed another actual failure: a CUDA 12.1 meta-package resolved nvcc 13.3. The installer now pins the NVIDIA version label, nvcc, and cuda-version, then inspects `nvcc --version` before building anything. PyTorch3D is pinned to the stable revision actually resolved by the upstream requirements (75ebeeaea0908c5527e7b1e305fbc7681382db47); builds run against installed Torch without isolated build dependencies.

The original 5346 pose rows contain 2853 positive unique timestamps, 2388 duplicate timestamp groups, and zero groups with conflicting pose/tracking values. The source files were not changed. A bounded, read-only prior-artifact inventory found 24 older model/result files; these remain candidates rather than approved global metric geometry.

45 offline regression tests now pass. A dedicated rasterizer CUDA forward/backward smoke is included; its result must still be read from actual Colab execution before declaring native-stack readiness.
