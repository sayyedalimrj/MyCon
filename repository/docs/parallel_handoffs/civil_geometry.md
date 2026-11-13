# A / Civil Geometry — independent handoff (2026-11-13)

## Ownership / target

Project: `sayyedalimrj/MyCon`, topic branch `research/mycon-civil-geometry-20261113`, proposed PR target `research/mycon-integration-20261112` (never `main`). The earlier A/B/C PRs were already merged to Integration. **No attempt to rewrite their history, merge, rebase, or update main**. The unit of work is new A-owned `stage_07_cleanup/structural_review.py`, corresponding `test_research_geometry_*.py`, `docs/research_geometry/structural_review.md`, and this file; no B/C/common code touched.

## Immutable source and provenance

Inputs are a Stage-7 comparison-ready run's existing SHA-256-validated clean cloud and finite plane/edge/corner records, which are **read only**. Source `MyCon_R4` video, checkpoints `ckpt_6999_rank0.pt`, `resume_step_006999.pt`, historical COLMAP and the original reconstruction remain unaltered. No from-scratch training, forced writes or destructive processing. Independent `geom-civil-*` run IDs and create-only output paths.

## Code, actual tests and evidence

Implementation source and synthetic tests: see GitHub branch / PR and `repository/docs/research_geometry/structural_review.md`. Local standalone CPU test command: `cd repository && PYTHONPATH=. pytest -q tests/test_research_geometry_structural_review.py`: **7 passed, 0 failed, exit 0**; Python version and SHA256 of the final source should be recorded from the actual final checked-in commit. This does **not** assert complete repository checkout compatibility, GPU validation, real field geometry, or independently measured accuracy.

Inputs: artificial trihedral planes used solely in an isolated offline fixture (no train/test leakage inference). Outputs: generated `clean_cloud.ply/.npy`, support and confidence-gated `planes/edges/corners.json`, `planes_colored.ply`, `geometry_qa.json`, `viewer_overlay.json`, interactive `geometry_review.html`, 3 Shape3 PNGs, `components.csv`, `manual_review_template.csv`, and `civil_geometry_manifest.json` with SHA256. A sample synthetic artifact bundle is available separately in the user session; no source room/frame/site data used.

## Open integration and scientific blockers

- Full real checked-out repo tests and Stage3–8 regression: **not executed in this offline prototype**.
- Real video/COLMAP dense reconstruction, camera poses/source-frame overlay: **blocked_missing_accessible_real_reconstruction**. Historical large source files must be processed on the source-side Drive/Colab/server; never substitute Gaussian PLY centres as a civil surface.
- Gravity up / floor-vs-ceiling, metric scale, real construction corners, uncertainty calibration, surveyed GT and photo-based visibility: **unverified**, therefore labels and physical progress stay unknown.
- Re-detection under noise, occlusion, densities, GT-annotated corner precision/recall: **not executed in this companion**, and should be audited by independent validation C. The current noise/occlusion tests only refit fixed existing planar support.
- Viewer approval and study/report acceptance require a human-review CSV plus real source-image evidence and independent survey. No fabricated Accuracy/Completeness/F1 numbers or geometry superiority claim.

## Downstream interface to coordinator

Additive, versioned, read-only `mycon.civil_geometry_review.v1`; no existing Stage 8/9/11 fields changed. Consumers must validate SHA256, schema version, path, input-kind, units, coordinate frame, and explicitly false `metric_scale_verified` before use. They must not translate `observed_supported` into construction completion or into field-certified dimensions. Cross-team consumer changes require coordinator interface PR; none were attempted here.
## Exact implementation revisions and final-source verification

- Implementation + test + documentation commit before handoff: `3b6b65224088a28466a8f1324001ec65d37a5b5a`.
- `structural_review.py` Git blob SHA: `0c445bb64d349ffc7cd321ddef5b71201921c721` (retrieved from the new branch, 23620 Unicode characters).
- `test_research_geometry_structural_review.py` Git blob SHA: `8b2805909c99aa956fdd82afd065c646dcdce838` (6999 Unicode characters).
- Last handoff-only commit occurs after this entry; record the PR head SHA from GitHub when reviewing. No claim that the local isolated tests are GitHub Actions runs.

---
## Round 2 handoff — 2026-11-13 (PR #42; do not merge)

Governing Integration contract: `repository/docs/integration/parallel_6_streams_approval_20261113.md` at read-only Integration ref. Coordinator comment PR #42 and Issue #48 followed. New research review **already committed** `repository/docs/research_geometry/evidence_review.md`, commit `6aecc7f32cd9d7594665d2d06768a8f84e3dc20a`.

**Code/test exact Git blob SHA verified against local tested source**:
- `repository/pipeline/stage_07_cleanup/research_dense_candidates.py`: `6e7c41a0752fc57b1cbbcd49fecb554fce7bce7a` (content commit `7e34c35746b8dbb12d75dc42db3057158a6d89ba`)
- `repository/pipeline/stage_07_cleanup/structural_review.py`: `766046accfb73b8846d963e5dc055a3fe7f8eb0b` (commit `a9d0cf46c5bedbdeb5989abe6a68583a972d9578`)
- `repository/tests/test_research_geometry_structural_review.py`: `88350fc348c145b2383169bd7fe292a9c2ca72fd` (commit `62d825daf23d63c3670b19325640d24adcbe2d80`)
- Additive docs: `repository/docs/research_geometry/structural_review.md`, `repository/docs/research_geometry/real_depth_execution_evidence_20261113.json`, this `repository/docs/parallel_handoffs/civil_geometry.md`. Original Stage7 code and all R/V/D/C/B files unchanged.

**Local CPU execution (2026-11-13, Linux, Python 3.13.5, original A-owned code matched final Git blob SHA)**: `cd repository && PYTHONPATH=. pytest -q tests/test_research_geometry_structural_review.py` → **14 passed, 0 failed, exit 0**; same source-tested tree imported `research_dense_candidates` & `structural_review`. Existing 7 tests preserved and 7 new cases added (noise, density, tilted wall, artificial occlusion, distant false intersections, camera index SHA validation, synthetic fake-real gate). This is **not** a full cloned repository checkout. Upstream Open3D/2DGS notebook not executed (Open3D not installed, no network/CUDA/training permission).

**Read-only real-derived experiment**: Drive ID `1qQLBz4MKud_BCUfMKfeT568ihPmv96Tk`, source `depth_fused_candidate_nonmetric.ply`, 104161 points, bytes 1562666, SHA256 `c1f706ec951a1120eb748122e54dd75cee5b3fac80135efcb5cdee844e1ef3ee`; receipt SHA `7e47dffb9fafb0ae1076f45549693a50373db11a1f414d879c175f3ce5840ee1`. Units `arbitrary_reconstruction_units`; frame `colmap_world_unverified`; `metric_scale_verified=false`, `registration_verified=false`, `field_validated=false`.

- Output run IDs: `geom-ready-real-depth-20261113-002` and `geom-civil-real-depth-20261113-final1` in isolated workdir; create-only.
- Producer CLI executed exit 0 with `--threshold .09 --max-planes 9 --trials 100 --max-sample 14000 --min-points 220`; review CLI `--input-kind real_reconstruction` exit 0. Source PLY and original Drive unchanged.
- Candidate counts: 15 finite planes, 49 local edges, 7 extrapolated corner candidates, **0 surveyed/confirmed or observed-supported corners**, 57,850 unassigned points. 0 matched source-image calibrated poses/projections; no field floor/wall/ceiling or metric room dimensions.
- `comparison_ready.json` SHA256 `1d736aac85f1c327c6d775b38c329401aae2dab026fc5622cf74d005f83a9d67`; `civil_geometry_manifest.json` SHA256 `d9f5cc124a0b176b11473dd59407a44ac1cae862d2af1db252a003ccffc719b0`. Source and output hashes for each JSON/PLY/PNG/HTML/CSV in `repository/docs/research_geometry/real_depth_execution_evidence_20261113.json`; machine-readable source manifest also includes full output paths and SHA256.
- Engineering visual QA: detected features are visibly fragmented; **not accepted** as real as-built walls/room layout. Requires higher-quality dense reconstruction, calibrated frame-indexed source imagery, and engineer-approved feature matches before field certification. No synthetic results used as field accuracy.

**New interface**: unchanged old comparison-ready v1; opt-in add-on `mycon.civil_geometry_review.v1` with `planes/edges/corners.json`, `components.csv`, `stability_report.csv`, `geometry_qa.json`, `planes_colored.ply`, three camera-angle plots, SHA-provenant `viewer_overlay.json` and clickable HTML. Optional `mycon.camera_evidence_index.v1` input is strict SHA/frame/units/K-intrinsics/world-to-camera gated; no consumer modifications; request coordinator to align owner V's `mycon.viewer_bundle.v1` adapter. Human review template remains pending/unknown. No physical quantities inferred.

**Unrun / blockers**: full repository CI on exact latest GitHub PR head (check workflow independently), GPU model validation, true production COLMAP dense/Mesh QA, source-camera photographic comparison on this real cloud, independent survey/metric GT, held-out session split, independent benchmark C, false-positive rate/error in metres, reviewer decisions. Keep PR **DRAFT and unmerged**. No force-push/rebase/main edits.

---

## ROUND 3 HANDOFF — 2026-11-14 | COLMAP real source images / Depth Fusion co-registration

**Status: G1 implementation/isolated CPU PASS; partial G2 real-source diagnostic PASS; geometry/image registration G2 blocked; G3–G5 pending.** This Handoff *adds* to and does not replace round-2 evidence. PR #42 only, DRAFT, Integration target; no remote merge/rebase/force/main edits. Code/test source content is byte-for-byte identical to the local tested source:

- `repository/pipeline/stage_07_cleanup/camera_geometry_bridge.py` Git blob SHA **`e4f93303c89ce73f0035675d0f3054027532e9ec`**;
- `repository/tests/test_research_geometry_camera_bridge.py` Git blob SHA **`bb750a42156ae9bf72992fb50e40627b2502c206`**;
- `repository/docs/research_geometry/round3_camera_geometry.md` Git blob SHA **`d5717c6d767a8018844fd9c9fba0096cde3c30c6`**;
- `repository/docs/research_geometry/round3_execution_evidence_20261114.json` Git blob SHA **`2b18899819dc6a444c07afc512c40c671191e160`**.
- Additional A-owned `repository/docs/research_geometry/evidence_review.md` addendum updates both new DOI papers and COLMAP v4.2.1 BSD-3 tag.
- Source/test commit head verified at **`05d967e763bd63349c30acbce6164f5d191ef4f7`** before subsequent research/Handoff commits. Report the **actual final PR HEAD separately**, not falsely as this earlier source commit.

**Actual code/test commands executed in isolated Linux CPU checkout of A-owned source:**
`cd repository && PYTHONPATH=. pytest -q tests/test_research_geometry_camera_bridge.py tests/test_research_geometry_structural_review.py` → **22 passed, 0 failed, exit 0, 5.64 seconds**. 14 previous tests unchanged and 8 new fail-closed camera/registration tests. This is **NOT** whole-repository regression, browser/GPU test, signed engineer result or independent site accuracy. Run GitHub specialized CI against the exact final PR SHA after this documentation commit and record its actual result separately; do not confuse earlier CI with final-head CI.

**Real-source run** `geom-civil-camera-20261114-real-depth-002` executed with exit code **0** (new create-only isolated root): original immutable MyCON Depth Fusion cloud 104,161 points, 15 finite candidate planes, 49 candidate edges, 7 extrapolated corners, zero approved; 57,850 unclassified points. COLMAP `images.txt`/ `points3D.txt` SHA256 **`b5b6d85cc739ec29a853e9fa593dcc5b3ef2058e9c50a29636ac6593fa37cb94` / `a735bdbcb15c58e5e9ccb0357f76ad7fbc969d6345e83084cd9fceba0330fbed`**. 51,894 PLY↔SfM points match within 9.57e-7 arbitrary units. Five SHA-verified Drive photo bytes (`frame_000096/120/144/168/192.jpg`) matched registered frame names and 3D/2D reprojections: **0.674/0.780/0.762/0.745/0.713 px median** (each p95 below 2.45 px). This validates **COLMAP's own model** not the Depth Fusion coordinates.

**Registration outcome:** `UNREGISTERED`, `metric_scale_verified=false`, `feature_photo_projection_allowed=false`. 30k sampled Depth points unmatched KD-tree median distance to sparse ≈0.357 arbitrary units; shifted X+1 control ≈0.262, so no independent camera/depth geometric co-registration. Unapproved normals, semantic wall/floor/ceiling identification, corners and image visibility remain explicitly **UNKNOWN**; no numerical error in meters, no physical dimensions or quantities.

**Versioned outputs, all copied-input SHA receipts and exact file hashes:** `repository/docs/research_geometry/round3_execution_evidence_20261114.json`; interface `mycon.civil_camera_geometry_bridge.v1` and `mycon.civil_geometry_viewer_handoff.v1` under isolated run: `registration_qa.json`, `feature_evidence.json`, `feature_review.csv`, `unknown_regions.json`, `unknown_point_indices.npy`, `viewer_handoff.json`, `round3_shape3_photo_review.html`, 3 actual photographed COLMAP/sparse track vs **separate** depth geometry panel PNGs, 5 unchanged source-photograph copies. Existing R2 source `planes.json`, `edges.json`, `corners.json`, `components.csv`, `planes_colored.ply` and rotatable `geometry_review.html` are **preserved unmodified**; no artificial increase in feature count. Photo panel is context only and not source depth-feature projection. Viewer V should show two independent unregistered layers; do not edit V source, Stage 8, Integration, or main from A.

**Research**, dated DOI sources, Git revisions/licenses and notebook execution status: `evidence_review.md` including existing 5 papers/2 repositories and added 2016 COLMAP SfM/MVS + pinned official COLMAP 4.2.1 at `bd1fcf654d2dd8fefa1466999c190a246f83f4b9`, BSD 3-Clause. Official 2DGS rasterization Colab **not executed**; current original COLMAP producer version unknown and no rerun from scratch.

**Outstanding:** exact Depth Fusion source frame/pixel↔COLMAP image/pose and SHA ledger, provable cross-modal transform and independently held-out inlier/reprojection residual, source gravity and survey metric scale, actual three-dimensional occlusion/visibility evidence, operator-accepted feature↔image identities, independent validation C, and actual Viewer V browser/WebGL acceptance. Do not promote any physical room wall/corner from these diagnostics yet.
