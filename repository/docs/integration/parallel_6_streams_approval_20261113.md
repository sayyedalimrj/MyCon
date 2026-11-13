# MyCON — Coordinator-approved six-workstream ownership and research evidence amendment
Date: 2026-11-13 | Authority: user-designated Integration Lead | Status: ACTIVE
Integration target: `research/mycon-integration-20261112` ONLY. Do not merge or write to `main`.

## Precedence and backwards compatibility
This is a narrowly scoped **additive ownership amendment** to `parallel_research_contract_20261112.md`, `parallel_research_round2_contract_20261112.md`, and `model_agnostic_geometry_readiness_contract_20261112.md`. Existing A/B/C ownership and Stage8/9/11 frozen interfaces remain unchanged except for coordinator-reserved integration wiring. Where this amendment grants new isolated directories, those paths are exclusively owned by the listed new teams. The coordinator alone approves cross-owner edits and merges. No existing source/checkpoint/capture/data may be deleted, forced, overwritten or secretly retrained.

## Owner 1 — Reconstruction (R; Issue #41 approved)
Branch `research/mycon-reconstruction-20261113`. Exclusive ADDITIONS: `repository/reconstruction/**`, `repository/tests/test_research_reconstruction_*.py`, `repository/docs/research_reconstruction/**`, `repository/docs/parallel_handoffs/reconstruction.md`; optional new config/dependency paths `repository/configs/research_reconstruction/**` and `repository/requirements-research-reconstruction.txt`.
No edits to existing `pipeline/stage_01*` through `stage_08*`, A code, `colab/**`, or shared files. Emit versioned `mycon.reconstruction.dataset.v1`, camera poses, frame/pose/intrinsics, held-out scores and artifacts with explicit source frames, units, SHA256 and split/session provenance. Reuse source video / prior COLMAP/gsplat as read-only; new-colmap/new training only in new opt-in run directories. Do not claim old-v-new quality without a matched independent test set.

## Owner 2 — Civil Geometry (A; PR #42)
Existing ownership: `repository/pipeline/stage_07_cleanup/**`, `repository/tests/test_research_geometry_*.py`, `repository/docs/research_geometry/**`, `repository/docs/parallel_handoffs/civil_geometry.md`. Existing A stage 1–8 remains A-owned. Consumption is read-only `mycon.comparison_ready_geometry.v1`; optional output `mycon.civil_geometry_review.v1`. Reject 3DGS Gaussian centers alone as field-verified physical surfaces. Never certify floors, walls, corners or metric dimensions without independent evidence.

## Owner 3 — Civil Viewer (V; Issue #43 approved)
Branch `research/mycon-viewer-20261113`. Exclusive ADDITIONS: `repository/viewer_civil/**`, `repository/tests/test_research_viewer_*.py`, `repository/docs/research_viewer/**`, `repository/docs/parallel_handoffs/viewer.md`; optional `repository/requirements-research-viewer.txt`.
Read-only `mycon.viewer_bundle.v1` adapter consuming validated camera/image and geometry receipts. Canvas/WebGL fallbacks must not present synthetic overlays as real. Deviation color/metric distance is gated on metric scale, registration, independent surveyed controls and verified source meshes. No direct changes to `repository/colab/**`, `gui/**`, `pipeline/**` or other team files.

## Owner 4 — Metric + BIM (D; Issue #44 approved; PR #45)
Exclusive ADDITIONS: `repository/civil_metric_bim/**`, `repository/tests/test_civil_metric_bim_*.py`, `repository/docs/research_metric_bim/**`, `repository/docs/parallel_handoffs/metric_bim.md`.
Keep BIM-specific Stage8 A-owned. Output `mycon.metric_bim_alignment.v1` and conservative optional Sim(3)/scale estimates with named operator-reviewed real dimensions, uncertainty and independent holdout checks. Nominal brick/block/steel dimensions are not certified actual measurements. No stage8/9 automatic feed while `metric_scale_verified` or `registration_verified` are false.

## Owner 5 — Independent Validation (C; PR #46)
Exclusive existing `repository/benchmarks/mycon_2026/**`, `repository/tests/test_research_validation_*.py`, `repository/docs/research_validation/**`, `repository/docs/parallel_handoffs/validation.md` and additive evidence handoffs/configs. Benchmark real-vs-synthetic provenance, grouped capture-session splits, fair comparable held-out reference sets, PSNR/SSIM/LPIPS, geometry accuracy/completeness/F1, ablations, statistical uncertainty and error cases. `not_executed` is not a successful experiment.

## Owner 6 — Progress + Schedule (B; PR #47)
Existing B ownership for `repository/pipeline/stage_09_progress/**`, `repository/pipeline/stage_11_schedule_variance/**`, `repository/tests/test_research_progress_*.py`, and B's progress docs/handoff. Physical quantities and BOQ/WBS weights require approved evidence; absence, obstruction and out-of-FOV must not become 0% or 'not built'. No unsigned arbitrary scale fed into Stage9. Respect `mycon.bim_geometry_evidence.v1`, IFC `global_id`, `mycon.scientific_progress.v2`.

## Coordinator-exclusive integration and ordering
The coordinator owns any new Integration orchestration or edits to `repository/colab/**`, global notebooks, shared configuration, Stage8 bridge, and `repository/docs/integration/**`. A may request colab changes by proposed patch only, not write them concurrently. Coordinator reviews/merges PRs sequentially after full relevant tests and updated base checks: R -> A geometry -> V -> D -> C -> B (adapters may require a different order but are tested in isolation first). No chat merges PRs. PR base always `research/mycon-integration-20261112`; keep unique workstream Drive folders and run IDs.

## Mandatory actual literature/code/notebook research, for EACH of six workstreams
Before the next production code revision, author `repository/docs/research_<team>/evidence_review.md` (or an equivalent owner-owned file) with:
1. **At least three relevant peer-reviewed papers**, bibliographic title/authors/year/venue + verified DOI URL or a clearly recorded 'no verified DOI'; method, relevant engineering outcome and limitations. Include recent methods and meaningful competing baselines.
2. **At least two genuine GitHub implementations**, exact repository URL + tag or commit SHA + license + compatibility/dependency inspection, and an applicability decision (adopt/adapt/reject with reason).
3. **At least one real executable notebook or demonstrable implementation** (official Colab, Jupyter or published reproducible example) with exact link, commit/revision, dataset/inputs, environment and execution steps. Execute a lightweight smoke test if locally possible; otherwise explicitly report 'not_executed' with blocker. Never call a notebook executed just because it exists.
4. Table 'claim -> scientific source -> code evidence -> experiment -> civil-engineering interpretation'; include 2+ alternatives, failure risk and reproducibility/licensing.
5. Document what was genuinely searched (keywords, date, sources) and which results were used/discarded. Cite primary sources and do not fabricate benchmarks/DOIs.
Not a license to copy other projects indiscriminately. Avoid unnecessary IT/UI embellishments before reliable field outcomes.

## Mandatory submission and acceptance gates
Commit source/tests/docs plus handoff `repository/docs/parallel_handoffs/<team>.md` with Git commit SHA, exact changed paths, original input hashes, output manifest schema version, coordinate frames, reconstruction vs metric units, links to real logs/screenshots, test commands+exit codes, unrun tests and scientific limitations. Submit DRAFT PR to Integration only; don't rebase or force-push without coordinator consent.
G0 safety/ownership, G1 end-to-end runnable API + CPU test, G2 real MyCON data smoke, G3 independently verified metric scale/ground truth when applicable, G4 thesis-grade comparisons, G5 coherent notebook/UI + progress evidence. Distinguish `implemented`, `cpu_tested`, `gpu_tested`, `real_data_tested`, `field_validated`.
Original `MyCon_R4`, recorder archives/video, `ckpt_6999_rank0.pt`, `resume_step_006999.pt`, historical COLMAP, and all baseline research artifacts are IMMUTABLE. New source experiments require new unique `run_id`, no force/delete/overwrite, no from-scratch training without explicit user approval. Every cross-team manifest must carry schema version, SHA256, source-relative paths, coordinate frame, units, provenance, evidence confidence/QA and strict unknown states. No synthetic result is field truth.
