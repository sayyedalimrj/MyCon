# Current Project Status

This document is the short final status snapshot for the laptop-prepared codebase before server execution.

## Current readiness

| Area | Status | Notes |
|---|---:|---|
| Repository hygiene | Ready | Runtime/generated paths are ignored and cache artifacts are excluded. |
| Requirements/build contract | Ready | Root and mirrored requirements files are present. |
| Lightweight pytest framework | Ready | Tests are split and laptop-safe checks are available. |
| Stage 1-2 ingest/keyframes | Ready scaffold | Needs real video on server/project machine. |
| Stage 3-5 COLMAP/dense | Server-required | Wrappers exist; real execution is server/heavy. |
| Stage 6 DA3 | Optional/server-required | Skip-safe scaffold; real model not yet validated. |
| Stage 7 cleanup | Ready in Docker core | Real quality depends on dense cloud. |
| Stage 7.5 VLM QA | Mock/local-ready | Real Qwen endpoint must be validated on server. |
| Stage 7.6 viewer export | Package-ready | Real Potree/Cesium conversion remains optional. |
| Stage 7.7 CAMS-GS evidence | Prepared-only | 3DGS training is optional and server-only. |
| Stage 8 metric alignment | Hardened scaffold | Includes quality gates, RANSAC helper, anchor validation, visual anchor workflow. |
| Stage 8 registration | Partial/server-required | Needs real IFC, real scan, and metric anchors/visual observations. |
| Stage 9 progress | Hardened scaffold | Includes conservative decision gate and visibility-aware interpretation. |
| Stage 10 copilot | Hardened scaffold | Includes answer validator and audit persistence. |
| Server readiness | Ready | Non-strict laptop check and strict server gate are available. |

## What is intentionally not claimed yet

- Real Qwen/Qwen3-VL-8B-Thinking inference is not validated on laptop.
- Real DA3 inference is not validated.
- Real 3DGS/CAMS-GS training is not validated.
- Real end-to-end COLMAP dense + BIM registration + progress on project data is server work.
- Stage 9 does not claim `not_built` from absence alone; it distinguishes `not_evidenced`, `not_observed_in_visible_area`, and `uncertain_low_registration`.

## Server blockers expected on laptop

The laptop should normally fail strict server mode for GPU checks:

- `tool.nvidia-smi`
- `gpu.nvidia_smi_runtime`

These are expected until the code is run inside a GPU-capable server/container.

## Server-first validation order

1. Clean clone/build.
2. Run lightweight pytest.
3. Run `scripts/server_readiness_strict_gate.py --strict`.
4. Prepare/download model cache on server only.
5. Run Stage 1-5 on a small real mini-case.
6. Validate Stage 8 metric alignment using anchors or visual observations.
7. Run Stage 8 registration.
8. Run Stage 9 progress.
9. Run Stage 10 real Qwen copilot and inspect audit JSON.


## Important packaging note

The working tree is the source of truth. Server handoff ZIPs must be generated with `scripts/export_server_handoff_zip.py`. Manually created ZIPs are not considered valid unless they include the requirements files, mirrored requirements directory, and `env/server.env.example`.


## Handoff ZIP verification

After creating a server handoff ZIP, verify it before upload:

```bash
python3 scripts/export_server_handoff_zip.py --output dist/construction-progress-ai-bim_server_handoff.zip
python3 scripts/verify_server_handoff_zip.py dist/construction-progress-ai-bim_server_handoff.zip
```

The verifier fails if requirements files, mirrored requirements, or `env/server.env.example` are missing, or if generated runtime paths such as `data/`, `runs/`, `exports/`, or `model_cache/` are included.
