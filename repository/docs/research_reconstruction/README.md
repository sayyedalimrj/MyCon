# MyCON — Reconstruction R / Round 2 (2026-11-13)

**Scope:** R-only independent, create-only multi-view acquisition and non-destructive registration. PR #49 targets `research/mycon-integration-20261112` in Draft; **not** `main`.

## Native implementation and tests

The GitHub branch now includes the complete local Round-2 source, verified byte-for-byte with local Git blob SHA for the original `workflow.py`, 10 original tests and 9 new tests. Another six isolated safety-gate tests verify SHA/frame/session/pose compatibility. Local source tests: `PYTHONPATH=. pytest -q tests/test_research_reconstruction_*.py` → **19 passed on synthetic CPU fixtures**, no real building evidence. Native branch-specialized CI requires its own exact-HEAD report and is authoritative for PR acceptance.

- `reconstruction/workflow.py`: read-only video frame selection, SHA-pinned manifest, blur/exposure/dHash and temporal coverage, fresh train-only COLMAP mapping, **then** separate test image localization (no historical pose substitution), per-view score, 6999 conservative comparison, receipts.
- `reconstruction/coverage.py`: registered-camera angular dispersion and missing views; **does not** imply wall/ceiling geometry visibility.
- `reconstruction/gs_experiment.py`: create-only train-only gsplat dataset with frozen COLMAP/model hashes; **does not train a new model**.
- `reconstruction/visual_review.py`: create-only side-by-side video and comparison strips, only for SHA-verified **real supplied renders** and camera/view match.
- `reconstruction/preflight.py`: inspect package/CUDA/COLMAP availability without installing or modifying sources.
- `reconstruction/safety_gates.py`: independent contract guard for hashes, capture-session leakage and camera record identity.

## How to operate

Read [Round-2 commands and source isolation](round2_operations.md) and [verified papers, code pins and notebook audit](evidence_review.md). Install the isolated CPU extras using `pip install -r requirements-research-reconstruction.txt`; do not alter global Colab configuration or owner A/B/C files.

All outputs go to a unique `recon-*` directory under a separate workstream Drive path; sources, historical COLMAP, `MyCon_R4`, `ckpt_6999_rank0.pt`, `resume_step_006999.pt` and existing PLYs remain read-only. Use `--execute-colmap` only for a **new** run; no `--force` or automatic GS training exists.

## Scientific and civil engineering gates

Not yet executed on real captures: new COLMAP camera validation, GPU/gsplat training, checkpoint/PLY production, independent held-out PSNR/SSIM/LPIPS, source-video panoramic QA, geometry extracted from dense MVS, measured wall/floor/ceiling/door/corner accuracy, or fair paired 6999 comparison. All geometry remains `reconstruction_units_unscaled`; `metric_scale_verified=false`; no BIM/quantity/progress claim. Historical 6999 reported log metrics are unverified, not comparable by themselves.
