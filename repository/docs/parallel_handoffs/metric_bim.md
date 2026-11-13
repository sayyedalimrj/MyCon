# Workstream D — Metric Scale / Civil BIM Alignment Handoff

Updated: 2026-11-14 (Iran timezone; GitHub UTC 2026-11-13). Branch: `research/mycon-metric-bim-20261113`. PR destination: `research/mycon-integration-20261112`, never `main`. **Draft, D ownership APPROVED by coordinator; no PR merge permitted.** Integration baseline SHA: `dceaf49114a11d79e17dd12495d4ab99cb901c38`.

## Commit and custody
- Last pre-handoff package commit: `4a0b8588bbd2f107ab835d13d838f83e27e80077`. Follow-up implementation commit (BIM-only candidate scale disclosed): `b66396539ee16c9f2d1c5b0cbcdea785cf8285dd` (exact SHA).
- Primary source hardening SHA: `6f1210828534e0ef53801192e949b7c0ea8de159`.
- Tests latest SHA: `640b8231b9978c89642fa13fce1a710471745cb3`.
- Ownership approved by Integration contract: `repository/docs/integration/parallel_6_streams_approval_20261113.md` (Issue #48 and previous Issue #44).
- Original capture/videos, `MyCon_R4`, `ckpt_6999_rank0.pt`, `resume_step_006999.pt`, historic COLMAP and all A/B/C files remain untouched. No force/merge/rebase or training.

## Changes (new paths only)
- `repository/civil_metric_bim/__init__.py`
- `repository/civil_metric_bim/core.py` — optional stand-alone NumPy producer, candidate metre-scale inference from independently reviewed manual actual lengths; separate noncoplanar BIM-feature Sim3, fit/holdout partition, strict no-data masking, SHA256 and create-only output.
- `repository/civil_metric_bim/schemas/length_references.v1.schema.json`
- `repository/civil_metric_bim/schemas/bim_feature_adapter.v1.schema.json`
- `repository/tests/test_civil_metric_bim_synthetic.py` — **19 synthetic pytest functions defined**, including disjoint heldout controls, nominal-vs-measured references, insufficient data, Sim3, degenerate frames, masked heatmap, immutable output and tamper guard.
- `repository/docs/research_metric_bim/adapter_contract.md` — CLI, schemas, uncertainties and limitations.
- This handoff; `repository/docs/research_metric_bim/evidence_review.md` and `repository/tests/test_civil_metric_bim_adversarial.py` added.

## Interface and sample output layout
- Read `mycon.comparison_ready_geometry.v1` manifest (optional), verified outputs + hash; read `mycon.length_references.v1` and/or `mycon.bim_feature_adapter.v1` with compatible reconstruction source frame.
- Inputs have explicit `reconstruction_units` source and BIM `m` target, approval, physical source group, fit/control role and reviewed observation status; real BIM asset SHA checked. Original evidence/read-only sources unchanged.
- Create new `runs/metric-bim-<id>/metric_bim/`: `scale_estimate.json`, `alignment_transform.json`, `alignment_report.csv`, `heldout_control_errors.csv`, `deviation_map.csv`, `deviation_heatmap.json`, `viewer_handoff.json`, `metric_bim_manifest.json`.
- Each manifest output has repository-project-relative `path`, `sha256` and byte count. JSONs include version, units/frame, pending gate; CSVs include version/frame/unit/run ID. Signed distance is normal-oriented. A no-data mask is false with **null error**, never accepted as unbuilt.
- **Illustrative pending-only shape, not an executed artifact or result:** `{"schema_version":"mycon.scale_estimate.v1","status":"insufficient_independent_length_references","scale_m_per_reconstruction_unit":null,"metric_validation":"pending"}`. All actual values and sample counts must come from a real run manifest.

## Tests — verified exact tested commit and fail/repair history

**Tested source+docs commit:** `3aed8899d54c0eb32aca06e0802e94ddbd8c7a15`, PR #45. Official [specialized CI](https://github.com/sayyedalimrj/MyCon/actions/runs/43504855123), job `workstream-tests`, checkout exact PR merge of head `3aed8899...` into coordinator Integration `c8196233...`. Triggered by `pull_request`. GitHub-hosted Ubuntu Python 3.11 NumPy 1.26.4 pytest 9.2.0; 2 discovered modules, **39 passed, 0 failed, 0.25 s, exit code 0**. Actual command in CI is `python -m pytest -q tests/test_civil_metric_bim_adversarial.py tests/test_civil_metric_bim_synthetic.py` (the 19-test original subset is included; not a claimed stand-alone CLI measurement).

**Original 19-test proof:** [CI run 43503959957](https://github.com/sayyedalimrj/MyCon/actions/runs/43503959957) on head `b4777723c83bd46efec09bbe2c096e613cebad3d`, **19 passed, 0 failed, 0.18 s, exit 0**. This checks the original requested module `tests/test_civil_metric_bim_synthetic.py` in the specialized runner.

**Repair history honestly recorded:** An intermediate draft at `0375d904...` failed 1/19 because `viewer_handoff.json` was added but the original expected file list hadn't been updated. Fixed in `7dd1b9d17b41e90e2de401f46778d95282f55bac`: specialized CI [43504305175](https://github.com/sayyedalimrj/MyCon/actions/runs/43504305175) **19/19 passed**. No failed intermediate run is represented as green.

**Added adversarial cases:** rejected wrong physical length/scales, fewer than 3 independent lengths, no approved holdout, nominal-only profile, source/target unit and frame mismatches, invalid finite BIM polygons/normals, absent IFC source with SHA, missing reference model, RANSAC outlier and degenerate fit, withheld confidence and Shape4 no-data mask. Synthetic constant scale=2, translation=(1,2,3), invented plane and 0.1 signed offset are only algorithm test fixtures; **NOT building measurements**.

**NOT executed:** raw room scan, calibrated real point cloud, field survey/BIM reference, true IFC geometry extraction, Open3D official Jupyter notebooks, external hardware/GPU, certified engineering tolerances, longitudinal repeatability, quantitative real-site/GT paper ablations. No actual source input SHA available for field run (all real originals preserved). Neither the prior GS logs nor green tests establish metric accuracy.

## Research, method and Shape4 interface

- Evidence review: `repository/docs/research_metric_bim/evidence_review.md` (three verified DOI peer-reviewed works: Umeyama 1991, Bosché et al. 2015, Besl–McKay 1992; Open3D pinned v0.19.0 commit and MIT license, IfcOpenShell pinned commit/LGPL; real Open3D ICP + global Jupyter notebooks inspected but not executed).
- Candidate methods: length-only independent scale (>=3 approved physical fit references plus holdout), proper Sim(3) from reviewed corners/plane points, optional RANSAC with **operator-set metre inlier tolerance**, holdout error summary and jackknife sensitivity explicitly NOT formal field uncertainty. Point-to-plane ICP is reviewed as possible later independent refinement; it is not disguised as run/executed.
- Viewer receipt `mycon.metric_bim_viewer_handoff.v1`, `viewer_handoff.json`: relative SHA256-linked scale/transform/heatmap/control paths, source/target frames, reconstruction units and metres, `metric_coloring_allowed=false`. Pending provisional distances must not be rendered as actual survey truth. Unknown/occluded/out of FOV has null deviation and invalid mask.
- No original input point cloud/OBJ/PLY/motion camera coordinates were changed. Transform is separate derived matrix only. Frozen Stage8 / Stage9 interface remains untouched.

## Git revisions

- Core source latest implementation commit: `800f429c1ae45ad3cf7faac7e91a4ed7819951e2`; original code remains in existing D path.
- Adversarial test source commit: `9291bd84145fed1a612ceecb39626ff8a3daee3c`; existing 19-test update `7dd1b9d17b41e90e2de401f46778d95282f55bac`.
- Versioned BIM adapter schema update `1e184edd09436fcc001cb2d18f93e1687fb21352`.
- Full tested PR head before this handoff revision: `3aed8899d54c0eb32aca06e0802e94ddbd8c7a15`. This handoff is a subsequent documentation commit; check its own CI status if the coordinator requires an exact-HEAD G1 seal.
- Model/source file hashes: none supplied from real room; actual derived-output file hashes are emitted at per-run creation, not invented in this handoff.

## Blocking conditions and integration asks
1. Coordinator-approved D namespace/ownership is in force; coordinator must review PR diff and future integration. D does not alter A-owned Stage 8, A-owned Stage 7, B/C or common code.
2. A/coordinator define a **separate additive Stage 8 consumer** only after output contract review; current candidate hardcodes `metric_validation=pending`, `metric_scale_verified=false`, `registration_verified=false`, `can_feed_stage8=false`. Existing Stage8 refuses candidate by design; never silently switch it on.
3. On actual project, supply independently measured actual block/steel/field lengths with uncertainties, several disjoint fit/heldout physical groups, approved IFC/BIM revision + hashed source asset, observed finite planes/corners and operator-approved matching. Do not substitute nominal catalogue geometry for measured as-built.
4. Exact-head (tested SHA above) synthetic/negative tests executed; for any later commit rerun CI. Run real source-side read-only pilot with independent surveyed test controls, camera/scale QA and visually inspect heatmaps; record genuine residuals and uncertainty before any `metric_validation` approval. No construction percent or `not_built` can be inferred from missing observation.

**Release status:** isolated candidate `implemented=true`, `cpu_tested=true` (synthetic 39/39 on named CI commit), `real_data_tested=false`, `field_validated=false`, `gpu_tested=false`, `publication_ready=false`, `metric_validation=pending`.

## Current PR
- Draft [PR #45](https://github.com/sayyedalimrj/MyCon/pull/45) targets Integration; merge is explicitly forbidden until coordinator review. No new test was executed as part of this handoff revision.
