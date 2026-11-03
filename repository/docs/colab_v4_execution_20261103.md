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
