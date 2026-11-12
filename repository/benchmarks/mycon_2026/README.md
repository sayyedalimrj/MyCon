# MyCON 2026 — independent scientific benchmark (Owner C)

No imports from A/B pipeline code and no mutation of recordings, PLY/IFC/checkpoints, or existing stage outputs. Developed against the locked `parallel_research_contract_20261112.md`.

## Run

From `repository/`:

```bash
python -m pytest -q tests/test_research_validation_metrics.py
python -m benchmarks.mycon_2026.runner \
    --manifest configs/research_validation/your_verified_manifest.json \
    --output-root /separate/bench_outputs --run-id bench-case01-20261112
python -m benchmarks.mycon_2026.plots \
    --result-dir /separate/bench_outputs/bench-case01-20261112 \
    --task reconstruction --metric fscore_at_tau
```

`--include-fixtures` tests the evaluator only; it **cannot** populate `scientific_results.json`, `paired_comparisons.json`, or scientific figures. Benchmark results are created with `mkdir(exist_ok=False)` and never overwrite an existing run. Do not place the output directory under the manifest's parent folder. `--seed` (default 42), `--tau-m` (default 0.05 metres) and `--max-points` (default 20,000 per cloud) are explicit reproducibility knobs.

## Input contract v1

A JSON file with `schema_version=mycon.validation.v1`, top-level `samples` list, optional `model_versions`, and each sample's **unique** `id`, `site_id`, `building_id`, `capture_session_id`, `split` (`train`, `validation`, `test`), and `source_sha256` (64 hex). Cross-split reuse of site, building, capture session or capture digest fails closed. `source_path` relative to the manifest directory is required for real science, and file bytes must match `source_sha256`. Real samples require `data_kind=real`, `gt_independently_verified=true`, and `gt_review={reviewer,review_date_utc,...}`. This gate is provenance bookkeeping; it does not independently certify the reviewer or annotation quality. The signed ground-truth audit is required externally; see protocol.

All file paths are relative to the manifest directory, cannot escape it, and are read-only. Place real data in a **separate private directory** and copy/adapt the manifest template; avoid committing capture data/checkpoints or private annotations. A baseline and new method must point at the same GT and test capture.

| Task | Independent GT key | Method output key | Scoring |
|---|---|---|---|
| reconstruction | `reconstruction_gt_ply` | `reconstruction_ply` | symmetric precision/recall/F-score at tau, Chamfer L1 and squared L2 |
| bim_geometry | `asbuilt_gt_ply` | `aligned_scan_ply` | same geometry metrics in BIM/world metres |
| registration | `survey_anchors_csv` | `registered_anchors_csv` | mean/RMSE/p95 TRE (m), pairwise scale drift (%) |
| ifc | `ifc_elements_csv` | `ifc_elements_csv` | IFC GlobalId inventory precision/recall, matched class accuracy |
| progress | `progress_csv` | `element_metrics_csv` | selective 3-class PRF, unknown rate, progress MAE, Brier/ECE |
| performance | none | `telemetry_json` | reported wall time, peak VRAM/RSS and run failure flag |

**Geometry:** `reconstruction_ply` must be scaled/aligned into the independent reference frame *before scoring*. Stage 8 `scan_aligned.ply` already uses BIM frame and can enter `bim_geometry` when independent as-built GT exists. Do NOT silently use `bim_reference.ply` (design geometry) as as-built GT. ASCII and little-endian binary PLY XYZ vertex formats are supported; the point sample is deterministic. Scipy KD-tree is required for large clouds; a small brute-force fallback has a strict 2-million-pair cap. Geometry scores are sensitive to scanning density, surface coverage and chosen tau; use common sampling/visibility masks.

**Anchors:** CSV columns `anchor_id,x_m,y_m,z_m`; survey coordinates and registered coordinates must share the same BIM/world frame. TRE is not ICP residual/fitness; scale ratio uses non-coincident anchor pairs. If fewer than two matched anchors, scale drift is unavailable, not zero.

**IFC:** CSV columns `global_id,ifc_class` for GT; prediction CSV or `bim_elements.jsonl` with `global_id`/`GlobalId` and `ifc_class`/`type`. Both inventories must be independently established, not a copy of one another. This validator checks identity, reported IFC entity class, and GUID lexical format; it does not parse IFC placement/solids, manifoldness or schema relationships. Those must be audited with an IFC library separately.

**Progress:** GT CSV columns `global_id,status,completion_fraction,weight`; status is independently labeled `built|not_built|partial`, fraction and positive weight may be measured separately. Method CSV uses existing Stage 9 `global_id`, `completion_state` or `status`, optional `physical_progress_estimate`, numeric `evidence_confidence` or `confidence` in [0,1], and optional `observability_ratio`. Stage 9 textual confidence (`high/medium/low`) is *not calibrated* into a numeric probability. `not_evidenced`, `uncertain_*`, `insufficient_coverage` and `partially_observed` are **unknown**; they never imply `not_built`. Classification metrics apply to the *selectively evaluated* subset; the `unknown_rate` denominator is all GT elements. Progress MAE uses only paired **numeric** [0,1] estimates, with explicit pair count and weight coverage. Overall progress is bounded by assigning unknown-weight completion between 0 and 1; **these are not confidence intervals**. ECE/Brier are correctness calibration on classified elements with numeric confidence; no calibration is reported if none exists.

**Performance:** a pre-collected `telemetry_json` with `wall_time_sec` (or `elapsed_sec`) and optional `peak_vram_mb`, `peak_rss_mb`, `status`. Missing telemetry is `not_executed` rather than zero. GPU monitoring and training are NOT performed by this runner. Capture wall-time, peak memory and exceptions for every method under the same hardware and run conditions.

## Output contract

`<output-root>/<bench-run-id>/` contains `metadata.json`, `input_hashes.json` (SHA256 for all used GT/prediction inputs), `cases.json` (including per-task failure or `not_executed` reason), `scientific_results.json` (cluster bootstrap over buildings; empty if no verified real test metrics), `paired_comparisons.json` (strict per-metric matched sample intersections), `failure_summary.json`, `comparison.csv`, `thesis_table.md`, and `thesis_table.tex`. Optional `figure_*.png` + `.svg` are made from verified scientific summaries only. Each table includes the sample/building denominators. Pairwise deltas are descriptive, never automatic superiority claims. CI absent when fewer than 2 buildings.

Read `repository/docs/research_validation/protocol.md` for the prespecified test protocol, ablation comparisons, and the paper-methodology crosswalk. No real MyCON accuracy claim is supported until a properly reviewed, independent test set and comparable baseline outputs have been evaluated.
