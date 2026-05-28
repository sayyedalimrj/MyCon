# Scientific Upgrade Plan

This document specifies the upgrades being applied to the MyCon construction-progress
AI/BIM pipeline to make it defensible as a scientific contribution. Each item lists:

- **What** we are changing,
- **Why** (with a peer-reviewed or canonical reference),
- **How** (the concrete file/function touched),
- **Compat** (whether existing JSON/CSV contracts change).

The companion file `references.md` lists full citations.

---

## 1. Bug fixes (audit-trail correctness)

These are not "scientific" per se; they restore the integrity of the existing
quality gates and decision policies. Without them, the audit trail produced by
the pipeline is partially fictional.

| ID | File | Defect | Fix |
|---|---|---|---|
| B1 | `pipeline/stage_10_copilot/answer_validator.py` | `unverified_confidence_with_low_quality_risks` is asserted by three consecutive identical `elif` branches; the second and third are dead code. | Collapse to a single branch. |
| B2 | `pipeline/stage_09_progress/run_progress.py` | When a BIM element bbox-cropped target has fewer than 10 BIM points, the code falls back to the *whole BIM cloud*, producing artificially inflated coverage for small/sliver elements (railings, mullions, mullion-stops). | Mark such elements as `not_evidenced` with an explicit `notes` reason instead of evaluating against the whole BIM. |
| B3 | `pipeline/stage_06_da3_assist/run_da3_assist.py` | When DA3 is judged required by `assess_dense_coverage` but no provider is available, the stage still prints `STAGE_06_DA3_OK` and returns 0. Orchestrators see green even though the upstream signal is structurally missing. | Print a distinct marker (`STAGE_06_DA3_REQUIRED_BUT_UNAVAILABLE`) and add `report["status_classification"] = "skip_unsafe"` so downstream automation can detect it. The exit code is still controlled by `da3.fail_if_required_but_unavailable` to preserve laptop-baseline behavior. |
| B4 | `scripts/run_stage.py` | `_STAGE_MODULES` knows only `stage_01_ingest` and `stage_02_keyframes`; calling for stages 3–10 raises an argparse error. | Extend to all stage entry points. |
| B5 | Hardcoded literal seeds (`42`, `9`, `75`) in `stage_07_5_vlm_qa/rendering.py`, `stage_08_bim_eval/registration_quality.py`, `stage_09_progress/run_progress.py` ignore `project.random_seed`. | Route them through `pipeline.common.determinism.derived_seed(...)`. |
| B6 | `pipeline/stage_08_bim_eval/coarse_registration.py` Open3D FPFH RANSAC is not seeded; Stage 8b coarse step is non-deterministic. | Pass `seed=` to Open3D RANSAC when the binding accepts it (Open3D ≥ 0.17), with a graceful fallback for older builds. |
| B7 | `pipeline/stage_08_bim_eval/metric_alignment.py` has duplicate `_rewrite_metric_alignment_report_output` definitions. | Confirmed via re-read; the second silently shadows the first. Keep one. |
| B8 | `pipeline/stage_09_progress/run_progress.py::_status_from_metrics` has a hardcoded default `partial_threshold=0.20` that drifts from the configured `progress.partial_observed_threshold`. | Remove the default and require an explicit value at the call site. |

---

## 2. Robust-loss option for Stage 8b ICP (`O1`)

**What.** Add a `bim.icp_robust_loss` config knob with values `none` (current
behavior, default), `huber`, and `tukey`. When enabled and Open3D ≥ 0.17 is
available, the point-to-plane ICP stage uses
`open3d.pipelines.registration.TransformationEstimationPointToPlane(kernel=...)`
with the requested kernel and a configurable scale `bim.icp_robust_loss_k_m`
(meters; default `0.05`, matched to `progress.deviation_threshold_m`).

**Why.** Construction scans contain systematic outliers from temporary objects,
scaffolding, transient workers/equipment, and partial occlusion. Squared-residual
ICP is sensitive to these. A bounded influence function (Huber) or a redescending
M-estimator (Tukey) caps their contribution; Tukey is preferred when the outlier
fraction is bounded (~<30 %), Huber when it is not.

**Reference.** Zhang & Singh, *A Field Analysis on Degeneracy-aware Point Cloud
Registration in the Wild* (arXiv 2408.11809, 2024); Zhang et al., *Fast and Robust
ICP* (arXiv 2007.07627). See `references.md` for full citations.

**How.**
- `pipeline/stage_08_bim_eval/refine_icp.py::_estimation()` becomes
  `_estimation(method, robust_loss, robust_loss_k)` and returns a kernel-wrapped
  estimation when supported.
- `pipeline/stage_08_bim_eval/refine_icp.py` records the requested kernel
  (`method`, `robust_loss`, `robust_loss_k`) in the result so the decision is
  audit-traceable.
- A capability probe (`pipeline/stage_08_bim_eval/icp_robust_capability.py`)
  reports whether the running Open3D exposes robust kernels; if not, the stage
  logs a warning and falls back to non-robust point-to-plane.

**Compat.** Default `bim.icp_robust_loss=none` keeps existing runs identical.
Robust losses are opt-in.

---

## 3. Bidirectional metrics + per-element uncertainty for Stage 9 (`O2`)

**What.** Replace the one-sided "coverage = mean(distances ≤ τ)" metric with two
paired metrics plus an F-score, computed per BIM element:

- **Accuracy (scan → BIM, τ):** fraction of element-relevant scan points whose
  nearest BIM-point distance is ≤ τ. This is the existing
  `in_tolerance_ratio` recast as a precision-style metric.
- **Completeness (BIM → scan, τ):** fraction of BIM-element samples whose
  nearest scan-point distance is ≤ τ. This is **new**. It distinguishes
  *"the scan covers the BIM element"* from *"the scan agrees with the BIM where
  it covers it"*.
- **F-score @ τ:** harmonic mean of accuracy and completeness — a single defensible
  progress score per element. τ matches `progress.deviation_threshold_m` (default
  0.05 m).

For each metric we report a **confidence interval**:

- Ratios (accuracy, completeness, F) → **Wilson score interval** (better small-
  sample behavior than normal-approximation; standard in epidemiology and
  classification metrics).
- Distance summaries (mean, median, p95) → **percentile bootstrap CI** with
  `B=1000` resamples (operator-configurable via
  `progress.bootstrap_iterations`). Bootstrap RNG is seeded from
  `pipeline.common.determinism.derived_seed("stage_09_uncertainty", element_id)`.

**Why.** The single-direction proximity metric cannot distinguish *"not built"*
from *"not observed"* and cannot distinguish *"built incorrectly"* from
*"out of view"*. Splitting accuracy from completeness is standard MVS evaluation
practice (Tanks-and-Temples F-score, Knapitsch et al. SIGGRAPH 2017; Chamfer
Distance forward/backward decomposition, e.g. arXiv 2505.14218, 2024). For
construction progress specifically, completeness ≈ "have we built it" and
accuracy ≈ "have we built it correctly" — answering both requires the bidirectional
form. CIs convert point estimates into defensible interval claims, which the
current label-bin approach cannot.

**Reference.** Knapitsch et al., *Tanks and Temples* (SIGGRAPH 2017);
Wilson, *Probable Inference, the Law of Succession, and Statistical Inference*
(JASA 1927); Efron, *The Jackknife, the Bootstrap, and Other Resampling Plans*
(SIAM 1982). See `references.md`.

**How.**
- New module `pipeline/stage_09_progress/bidirectional_metrics.py`:
  `compute_bidirectional(scan_points, bim_target_points, tau_m) -> dict`
  returning `accuracy`, `completeness`, `f_score`, plus per-side counts and
  matched-distance arrays.
- New module `pipeline/stage_09_progress/uncertainty.py`:
  `wilson_interval(successes, n, alpha)` and
  `bootstrap_ci(values, statistic, B, alpha, rng)`.
- `pipeline/stage_09_progress/run_progress.py` writes additional columns to
  `element_metrics.csv` and additional fields to `progress_summary.json`. Old
  columns are preserved and unchanged. Behavior is gated by
  `progress.bidirectional_metrics_enabled` (default `true`); operators can set
  it to `false` to reproduce historical runs exactly.

**Compat.** Existing CSV/JSON keys are preserved. New keys are additive:
`completeness`, `f_score`, `accuracy_wilson_lo`, `accuracy_wilson_hi`,
`completeness_wilson_lo`, `completeness_wilson_hi`, `f_score_wilson_lo`,
`f_score_wilson_hi`, `mean_deviation_ci_lo`, `mean_deviation_ci_hi`,
`median_deviation_ci_lo`, `median_deviation_ci_hi`. The `notes` column gains
provenance markers (`bidirectional_v1`, `bootstrap_B=N`).

---

## 4. Centralized determinism (`O3`)

**What.** A new module `pipeline/common/determinism.py` exposing:

- `derived_seed(label: str, *parts) -> int` — deterministic 32-bit seed derived
  from `project.random_seed` + a stable hash of `(label, *parts)`. This lets each
  stage produce reproducible-but-distinct streams without colliding.
- `seeded_rng(label: str, *parts) -> numpy.random.Generator` — convenience.
- `set_global_determinism_envs(seed)` — sets `PYTHONHASHSEED`, `OMP_NUM_THREADS=1`
  (opt-in via `project.deterministic_threads=true`), and Python `random.seed`.

Callers are: Stage 6 alignment (already config-driven, will switch to
`derived_seed("stage_06_alignment", image_idx)`), Stage 7.5 rendering, Stage 8b
coarse RANSAC, Stage 9 NN sampling, Stage 9 bootstrap.

**Why.** Reproducibility is a hard prerequisite for any metric claim. The current
mix of literal seeds (`42`, `9`, `75`) means changing `project.random_seed`
*partially* re-randomizes the run. A centralized utility is the standard fix.

**Compat.** Seeds change for the three sites that previously used `42`, `9`, `75`
(B5). This is intentional — those literal-seed runs are not reproducible from the
config seed in the first place.

---

## 5. Run-level metrics aggregator (`O4`)

**What.** A new script `scripts/aggregate_run_metrics.py` that walks
`runs/<run_id>/reports/*.json` plus `data/bim/metrics/<project>/*.csv|json` and
produces a single `runs/<run_id>/reports/run_metrics.json` with a flat schema:

```
run_id, stage, status, elapsed_sec,
  fitness?, inlier_rmse?, points?, registered_images?,
  accuracy?, completeness?, f_score?, registration_confidence_label?
```

It also emits `runs/<run_id>/reports/run_metrics.csv` for cross-run comparison.

**Why.** The current pipeline writes a JSON report per stage but provides no
single-run summary, which makes it impossible to compare two runs without manual
collation. This is the minimum infrastructure for an ablation study.

**Compat.** Read-only with respect to existing artifacts. Pure additive.

---

## 6. Ablation harness (`O5`)

**What.** A new module `pipeline/common/ablation.py` with:

- `AblationGrid(name, axes: dict[str, list], base_config: Path)` — Cartesian
  grid generator that emits one overlay YAML per cell.
- `apply_overlay(base_cfg, overlay) -> dict` — pure functional config merge.

A new script `scripts/ablation_run.py` accepts `--config base.yaml --grid
grid.yaml --stage stage_08_bim_eval.run_registration` and produces one
`runs/<run_id>/_ablation/<cell_name>/` directory per cell, then runs
`scripts/aggregate_run_metrics.py` per cell, and finally writes
`runs/<run_id>/_ablation/ablation_summary.csv`.

The shipped example grid `configs/ablation_examples/icp_robust.yaml` toggles:

```yaml
axes:
  bim.icp_robust_loss: [none, huber, tukey]
  bim.icp_max_corr_distance_m: [0.05, 0.08, 0.12]
  project.random_seed: [42, 43, 44]
```

**Why.** Without an ablation harness, every new knob (robust loss, voxel size,
threshold) is a parameter of an N=1 study. Even a Cartesian grid runner is enough
to support a defensible "we evaluated A vs B over K seeds" claim.

**Compat.** Pure additive. Heavy stages like Stage 5 dense MVS are *not*
re-executed per cell unless the grid touches a Stage-5 axis; the harness reads
`pipeline_plan.json` to determine the minimum recompute set.

---

## 7. References doc (`O6`)

A new `docs/references.md` file lists the citations referenced here, plus the
canonical references for each stage (COLMAP, Open3D, IfcOpenShell, Tanks-and-
Temples, Depth-Anything V2, 3DGS, 2DGS, FPFH, ICP).

---

## 8. Explicitly out of scope this turn

These are large-scope items that would distract from the stated goals
("debug the main pipeline, add some optimizer, make it worth as scientific work").
They are listed so reviewers can see what is *deferred* and why.

| Item | Why deferred |
|---|---|
| In-process Depth-Anything V2 inference for Stage 6. | Requires (1) torch model cache plumbing already documented in `docs/server_model_cache_and_vlm_setup.md`, (2) GPU container build, and (3) DA-V2 license review. The existing Stage 6 abstraction (per-image manifest, RANSAC scale alignment, edge-aware fusion) is *already correct*; only the depth oracle is external. We add `da3.provider="depth_anything_v2"` as a recognized provider name with a clear `not_wired` status, mirroring the PixSfM hook pattern. The pipeline becomes "ready for DA-V2", not "running DA-V2". |
| Real 3DGS / Splatfacto training for Stage 4.5 / 7.7. | The existing stance — `is_metric_truth: false`, `readiness: prepared_stub_only` — is *correct* per the literature: 3DGS optimization is photometric, not geometric, so its surface is visually plausible but metrically unreliable without explicit geometry constraints (2DGS, arXiv 2403.17888; GS2Mesh). For a *progress* pipeline whose claims must be metric, GS-derived geometry should remain decoration, not evidence. We document this stance with a reference rather than wire training. |
| Snakemake/Prefect DAG orchestrator. | The file-contract DAG is already documented (`docs/data_contracts.md`). Promoting it to a workflow engine is a separate refactor and would change runtime semantics. We instead extend `scripts/run_stage.py` to launch all stages, which closes the worst gap (B4) without changing semantics. |
| Real Open3D off-screen renderer for Stage 10. | Stage 10's "evidence card placeholder" is honest about what it is. Replacing it would require a headless GL stack and Open3D `OffscreenRenderer`, which is non-trivial in CI/sandbox environments and is orthogonal to the metric-defensibility goal. |

---

## 9. Acceptance check

After all items are merged, the following must hold:

1. Existing test suite remains green (current baseline: 189 passed, 1 skipped).
2. New tests cover: `determinism.derived_seed`, `bidirectional_metrics`,
   `uncertainty.wilson_interval`, `uncertainty.bootstrap_ci`, robust-ICP capability
   probe (without requiring Open3D ≥ 0.17 at test time), and
   `aggregate_run_metrics`.
3. `scripts/run_stage.py stage_08_bim_eval` resolves correctly (was a hard error).
4. With `progress.bidirectional_metrics_enabled=false`, `element_metrics.csv` is
   byte-identical to a pre-upgrade run on the same input and seed.
5. With the default config plus `bim.icp_robust_loss=none`, Stage 8b output is
   bit-stable across seeds (modulo Open3D version) — meaning we have closed B6.
