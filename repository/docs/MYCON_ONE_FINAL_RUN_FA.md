# MyCON — ONE final Colab test path (۱۴ آبان ۱۴۰۵)

> **Start only here:** [MyCON_FINAL_ACCEPTANCE.ipynb](../MyCON_FINAL_ACCEPTANCE.ipynb). The new pull request includes the previously separate LongSplat research and V4 repair branches; do not mix old notebooks, ad-hoc hotfix ZIPs, or host Python with this path.

## Objective and honest acceptance status

The target is **real** NVIDIA LongSplat training on the **full, integrity-verified ~96 s native Recorder capture**; actual new-view RGB/depth renders; a 3DGS PLY; a separately labeled depth-geometry candidate; and held-out view QA. No statement about as-built↔BIM metric accuracy is allowed without independent site/BIM control points.

The baseline evidence is NOT final training success:
- 769 upright SHA256-inventoried frames at 8 fps / 512 px were prepared.
- Previous T4/CUDA12.1 environment at `/content/MyCON_agent_cuda121_20261105` passed real native CUDA and rasterizer forward/backward smoke on 2026-11-05.
- A subsequent V4 launch reinstalled an isolated environment and failed at PyTorch3D dependency installation; the new path first tests whether the proven environment is actually reusable.
- The previous train invocation began, but no finished model or held-out visual result is certified.

## User procedure (ONLY two modes in the same notebook)

1. Select Colab **Runtime → GPU**. Open `repository/MyCON_FINAL_ACCEPTANCE.ipynb` from **this PR branch**.
2. Run the single main cell with `MODE = "preflight"`. It mounts Google Drive, obtains one pinned repository checkout, validates original Recorder SHA256 and 90–110 s duration, reuses only GPU-verified CUDA environment or installs an isolated new prefix, verifies upstream NVIDIA checkout and MASt3R weight SHA, prepares full frames, **runs real MASt3R two-view inference on the actual prepared capture**, and **stops before training**. The full CUDA native smoke is required by the controller. It is NOT a certification of the model or real BIM.
3. If no ZIP is found unambiguously, set `CAPTURE_ZIP` to the existing **full original Recorder ZIP** in Drive. Never use the 10-second clip. The notebook will refuse to guess.
4. The status cell must report **`PREFLIGHT_PASSED_CUDA_AND_MAST3R_NO_TRAINING`** and the `mast3r_2view_smoke` stage must have passed. If either is missing, **do not launch training**. Only after this passes, set `MODE = "all"` in the same cell to explicitly authorize a **new** GPU training run, renderer, export, geometry candidate and view audit. Training may take significant GPU time and may still encounter native upstream model/data errors; no hidden retry/reset.
5. Run the status cell. Inspect the `run_summary.json` and stage logs under:
   `MyDrive/MyCon_R4/research_longsplat_auto/<capture_sha16>/<profile>/attempts/stabilization_20261105/`.
6. If training stops halfway without a supported checkpoint, **do not delete or restart over the same folder**. Preserve the folder and use a new explicit attempt name after diagnosing the error. The upstream adapter does not support a proven mid-training resume.

The Colab interpreter is called using `python` (not Windows `py`). The new NVIDIA environment remains separate from the user's main R4 environment. The official source and trained models are never silently rewritten.

## Stage gates and verdicts

| Gate | Pass evidence | Not proven by that gate |
|---|---|---|
| Source integrity | Original ZIP + sidecar SHA/CRC, complete ~96 s video, source checkout and submodules | Visual accuracy |
| GPU/native | Matching Torch/CUDA/SM and executable native forward/backward CUDA kernels | Successful LongSplat training |
| Real-model preflight | NVIDIA's MASt3R infers finite 3D points from two verified frames; separate receipt | Optimization, full reconstruction or metric BIM |
| Preparation | Full upright frame inventory; parameter identity; separate attempt directory | 3D model |
| Training | Authentic LongSplat checkpoint/iteration and successful train phase | Novel-view correctness |
| Render/export | Actual novel-view RGB/depth and 3DGS PLY | Metric surfaces or BIM |
| Appearance QA | Measured held-out source views (PSNR etc.) with visible inspected outputs | Arbitrary never-seen views |
| BIM acceptance | Independent survey/BIM anchors including held-out points and actual Stage 8/9 evaluation | Nothing until site geometry is supplied |

The training subprocess now sends rate-limited progress markers to Colab and stores full original logs on Drive, including source error tails when upstream exits nonzero.

The default 8fps × 512 px × 20,000 iterations is a T4 resource profile, **not** a promise of accuracy. Upstream NVIDIA LongSplat is noncommercial research/evaluation only.

## Safety invariants

- Never delete or edit any original Recorder archive, camera/pose/IMU data, pretrained weight cache, existing checkpoints or training folders.
- The `--attempt-id` guard accepts only strict safe names; the old experiment root remains unchanged.
- Existing CUDA environments are only adopted after **live GPU kernel tests**, never solely by a receipt or import.
- Incomplete isolated envs fail closed; select a different prefix rather than overwriting/reinstalling in place.
- New GPU training requires `--approve-new-training` (automatically passed by the **explicit** notebook `all` or `train` mode only).
- A process exit code of 0 means that stage commands exited successfully, **not** a peer-reviewed thesis, verified metric BIM, or universally correct new-view geometry.

## Offline tests and live acceptance

Run offline inside repo:
```bash
cd repository
python -m pytest -q tests/test_mycon_thesis_one_run.py tests/test_mycon_colab_gpu_auto.py
```

Offline tests cover input ambiguity, mode separation, checkpoint attempt isolation, pathname guards, live-verified env selection and fail-closed behavior. Actual Colab T4 execution is a separate **mandatory** test before PR merge or public claims of end-to-end success.

## Provenance

- Original NVIDIA code: https://github.com/NVlabs/LongSplat — pinned source commit and original research licence verified by existing adapter.
- LongSplat research integration: `docs/longsplat_thesis_integration.md` in this branch.
- V4 repairs: `docs/colab_v4_execution_20261103.md` in this branch.
- Earlier upstream PRs: #29 (new-view and geometry), #30 (V4 preflight repairs). This new PR is intentionally opened against `main` as the **single consolidated review target**, without deleting or rewriting old PRs.


## Why we use NVIDIA scripts rather than an alleged official Colab notebook

The [official NVlabs/LongSplat repository](https://github.com/NVlabs/LongSplat) documents its ready-made **shell scripts** `scripts/train_custom.sh`, `train.py`, `render.py`, and `metrics.py`, plus an **optional CroCo RoPE CUDA build**. It does not provide a verified official LongSplat Colab notebook in the reviewed source tree. This MyCON notebook is our controlled wrapper around the **pinned original NVIDIA code**, not an official NVIDIA notebook.

The core official custom-video path is followed (source frames under `data/<scene>/images`, pose estimation, training, rendering). We explicitly use `-r 1` because frame preparation has already resized to 512px, whereas NVIDIA's example `-r 4` targets not-yet-resized original images. Conversion to general 3DGS is a separate NVIDIA command. The independent held-out appearance audit is maintained as a separate research gate; neither this check nor NVIDIA's training exit establishes BIM metric accuracy.
