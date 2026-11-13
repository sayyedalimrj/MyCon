# Workstream D — Metric Scale / Civil BIM Alignment Handoff

Date: 2026-11-13. Branch: `research/mycon-metric-bim-20261113`. PR destination: `research/mycon-integration-20261112`, never `main`. **Draft / coordinator approval required**. Integration baseline SHA: `dceaf49114a11d79e17dd12495d4ab99cb901c38`.

## Commit and custody
- Last implementation/packaging commit **before this handoff**: `4a0b8588bbd2f107ab835d13d838f83e27e80077` (exact repository commit).
- Primary source hardening SHA: `6f1210828534e0ef53801192e949b7c0ea8de159`.
- Tests latest SHA: `280cc44c968be1b5d131e1cd8a03e9c79cfd5e1e`.
- New ownership request: https://github.com/sayyedalimrj/MyCon/issues/44 (**not yet coordinator-approved**).
- Original capture/videos, `MyCon_R4`, `ckpt_6999_rank0.pt`, `resume_step_006999.pt`, historic COLMAP and all A/B/C files remain untouched. No force/merge/rebase or training.

## Changes (new paths only)
- `repository/civil_metric_bim/__init__.py`
- `repository/civil_metric_bim/core.py` — optional stand-alone NumPy producer, candidate metre-scale inference from independently reviewed manual actual lengths; separate noncoplanar BIM-feature Sim3, fit/holdout partition, strict no-data masking, SHA256 and create-only output.
- `repository/civil_metric_bim/schemas/length_references.v1.schema.json`
- `repository/civil_metric_bim/schemas/bim_feature_adapter.v1.schema.json`
- `repository/tests/test_civil_metric_bim_synthetic.py` — **18 synthetic pytest functions defined**, including disjoint heldout controls, nominal-vs-measured references, insufficient data, Sim3, degenerate frames, masked heatmap, immutable output and tamper guard.
- `repository/docs/research_metric_bim/adapter_contract.md` — CLI, schemas, uncertainties and limitations.
- This handoff.

## Interface and sample output layout
- Read `mycon.comparison_ready_geometry.v1` manifest (optional), verified outputs + hash; read `mycon.length_references.v1` and/or `mycon.bim_feature_adapter.v1` with compatible reconstruction source frame.
- Inputs have explicit `reconstruction_units` source and BIM `m` target, approval, physical source group, fit/control role and reviewed observation status; real BIM asset SHA checked. Original evidence/read-only sources unchanged.
- Create new `runs/metric-bim-<id>/metric_bim/`: `scale_estimate.json`, `alignment_transform.json`, `alignment_report.csv`, `heldout_control_errors.csv`, `deviation_map.csv`, `deviation_heatmap.json`, `metric_bim_manifest.json`.
- Each manifest output has repository-project-relative `path`, `sha256` and byte count. JSONs include version, units/frame, pending gate; CSVs include version/frame/unit/run ID. Signed distance is normal-oriented. A no-data mask is false with **null error**, never accepted as unbuilt.
- **Illustrative pending-only shape, not an executed artifact or result:** `{"schema_version":"mycon.scale_estimate.v1","status":"insufficient_independent_length_references","scale_m_per_reconstruction_unit":null,"metric_validation":"pending"}`. All actual values and sample counts must come from a real run manifest.

## Tests — honest provenance
- **Executed here on exact checked-out MyCON:** none. The connected GitHub writer can commit the source, but the local CPU runtime has no DNS access to GitHub and this branch does not own shared CI workflow files.
- **Defined but not executed in this session:** `cd repository && PYTHONPATH=. pytest -q tests/test_civil_metric_bim_synthetic.py` (18 functions). No claim of passing, CPU code timing, external ground truth, numerical benchmark, real site, IFC producer or GPU result.
- **Synthetic test fixture definitions** make an artificial scale 2, arbitrary translation [1,2,3], orthogonal points, a finite BIM polygon and occluded/no-map examples; **these are not project dimensions** and no result/error statistic has been observed in this handoff.
- **Not executed:** GitHub Actions exact-HEAD checks, full integration regression, complete BIM/IFC conversion, real field measured control set, geometry heatmap review, real PLY/cloud processing, peer review, GPU.

## Blocking conditions and integration asks
1. Coordinator approve namespace/ownership [Issue #44] and assess PR diff. D does not alter A-owned Stage 8, A-owned Stage 7, B/C or common code.
2. A/coordinator define a **separate additive Stage 8 consumer** only after output contract review; current candidate hardcodes `metric_validation=pending`, `metric_scale_verified=false`, `registration_verified=false`, `can_feed_stage8=false`. Existing Stage8 refuses candidate by design; never silently switch it on.
3. On actual project, supply independently measured actual block/steel/field lengths with uncertainties, several disjoint fit/heldout physical groups, approved IFC/BIM revision + hashed source asset, observed finite planes/corners and operator-approved matching. Do not substitute nominal catalogue geometry for measured as-built.
4. Run exact-head synthetic/negative tests and integration CI. Run real source-side read-only pilot with independent surveyed test controls, camera/scale QA and visually inspect heatmaps; record genuine residuals and uncertainty before any `metric_validation` approval. No construction percent or `not_built` can be inferred from missing observation.

**Release status:** implemented as isolated candidate; `unit_tested=not_executed`; `real_case_validated=not_executed`; `gpu_validated=not_executed`; `publication_ready=false`.
