# MyCON-B: reliable physical progress, observability, and uncertainty (v2)

**Scope:** Stage 9 + Stage 11 research extension, 2026-11-12.
**Safety:** No Stage 8, C, GPU training, raw capture or model checkpoints changed.
**Schema:** `mycon.element_progress.v2` (additional Stage 9 CSV columns) and
`mycon.scientific_progress.v2` (independent Stage 11 JSON output).

## 1. Diagnosed baseline defects

1. Stage 9 uses a BIM-to-scan nearest-neighbour *proximity* fraction as `coverage`;
   this is not construction quantity, visibility or a sufficient absence test.
2. Expanded element AABBs overlap. Previously each matching point could be counted
   as evidence for several unrelated IFC elements. A sliver's crop could contain
   only neighbouring wall points. Empty/tiny BIM targets must not fall back to
   the whole BIM cloud (legacy fallback already removed).
3. Stage 11 v1 maps `not_evidenced => 0` and `uncertain => 0.5`
   and calls the weighted mean `actual_percent_complete`. These are *subjective
   acceptance proxies*, not measured executed construction quantities.
4. Stage 11 v1 takes a Wilson score interval of the **unweighted count of completed
   elements** but places it around a **weighted mean of 0/0.5/1 status scores**.
   The point estimate and interval target different estimands. Besides, scan
   points within the same element are strongly spatially correlated and the
   iid Bernoulli assumption of a standard Wilson interval is not justified.
5. Schedule `percent_complete` usually denotes planner-reported *actual* progress,
   not an approved time-phased baseline. Linear interpolation between dates is
   only a provisional time-based proxy, not an earned-value baseline.

Bosché (2010) and Braun et al. (2015) distinguish object recognition,
as-built geometry and the role of occlusion. Tuttas et al. (2014) describe
explicit free/occupied/unknown spatial reasoning rather than claiming unseen
building parts were not constructed.

## 2. Three separate measurable concepts

### `physical_progress_estimate`

For an IFC element `e`, a physical fraction is defined only if planned
`Q_e > 0`, executed `0 <= q_e <= Q_e`, units match and the site quantity
is approved. Define `p_e = q_e / Q_e`. A metric match or BIM recognition
**never** automatically implies `q_e = Q_e`. Synthetic input, status
`acceptable`, low coverage, occlusion, or absence of scan points cannot
substitute for an approval record.

Required additive Stage 9 fields:

- `planned_quantity`, `executed_quantity`, `quantity_unit`,
  `quantity_verified` = true, `quantity_source` in
  `approved_measurement`, `signed_site_log`, `survey_qa`.
- No valid quantity evidence: blank CSV cell and `physical_progress_status=unknown`.
- Element value: fraction [0,1]. Activity/project value: percent [0,100].
- Quantity approval is still source-provided, not cryptographically authenticated:
  independent site audit is necessary for real-world acceptance.

For activity `a` with IFC mapping `E_a` and **approved comparable mapping
weights** `w_e>0` (explicit `weight_basis` from
`approved_boq|approved_quantity|approved_cost`):

`P_a = 100 * (sum_e w_e p_e) / (sum_e w_e)` **only if every
weighted element has a verified `p_e`.** Missing evidence means
`P_a = null`, not zero or 50%.

For missing p_e, *identification bounds*, conditional on valid mapping
weights and p_e in [0,1], are:

`L_a = 100 * sum_known(w_e p_e) / sum_all(w_e)`

`U_a = 100 * (sum_known(w_e p_e) + sum_unknown(w_e)) / sum_all(w_e)`

These are **NOT 95% CIs**, and do not model measurement error in
even approved quantities. If mapping weights are absent, defaulted,
non-finite, non-positive, duplicated or mixed in basis, both bounds
and estimate remain `null`. Mixed measurement units across elements
are allowed only after their contribution weights have a common
approved BOQ/cost basis.

### `observability_ratio`

The fraction of the expected BIM surface known to have been *visible
and inspectable* from registered camera viewpoints during capture.
Use independently audited surface/ray visibility, e.g. calibrated
camera frusta plus z-buffer, ray casting or manual occlusion audit.
Allowed provenance: `calibrated_camera_frustum`,
`raycast_zbuffer`, `manual_occlusion_audit`. It is **not** the
BIM-to-scan NN ratio `coverage`, accuracy, F-score, number of
points, or construction fraction. Unknown when no validated
visibility provenance exists.

The activity ratio is weight-averaged only if every weighted
element has an audited ratio; otherwise `null`.

### `evidence_confidence`

A distinct *quality/calibration score* in [0,1], only passed through
if the input claims a held-out/audited calibration
(`evidence_confidence_source=heldout_calibration|audited_quality_model`)
and registration is high/medium and non-synthetic. It is **not** a
probability of construction completion; even a source label must be
independently checked. Without a calibrated input it is null. Low
registration blocks geometric confidence, not independently signed
executed quantities. Per-activity aggregation requires complete
calibrated element coverage.

Do not conflate this value with `visibility_confidence`, which is
a legacy heuristic, or with Wilson sampling confidence intervals.

## 3. Point attribution and Stage 8 interface

`point_assignment.assign_exclusive_points` assigns an aligned scan
(or BIM reference) point only if exactly one element's expanded AABB
contains it. A point in multiple AABBs becomes `owner=-2`,
not N separate pieces of evidence. A point in none becomes `owner=-1`.
Stage 9 nearest-neighbour distances and bidirectional summaries use
only points that have exclusive owner attribution; no scan point from
an adjacent element is allowed to satisfy another element's metric.
Retain `ambiguous_scan_point_count`, `ambiguous_bim_point_count`,
`assigned_scan_point_count`; zero matching points are unknown evidence.

This is a **conservative safety fix**; exclusive AABB membership is
not a proof of element identity, and overlapping elements become
unresolved. To improve recall, owner A/Stage 8 must provide, in the
same aligned metric frame:

- BIM surface samples or triangle meshes with authoritative
  `global_id` per surface sample/triangle (not just AABB);
- per-view camera poses, intrinsics and depth/occlusion evidence
  (for raycast visibility), with capture-session IDs;
- registration transform, residual/QA, units in metres, and
  frame/geometry version or content hash;
- a documented optional element-observation interface and reproducible
  unassigned/overlap handling.

**Do not change Stage 8 without lead approval.** B only supplies
an interface proposal.

## 4. Schedule semantics

The v2 report uses optional *approved* baseline CSV additions:
`approved_planned_percent` [0,100],
`baseline_source=approved_baseline`, nonempty `baseline_id`.
Without them, actual-vs-plan `schedule_variance_percent=null` and
`schedule_variance_status=unknown_schedule_baseline`, even when
physical progress is known. It reports
`provisional_linear_planned_percent` separately for visualization
only; `percent_complete` remains planner-reported actual (not a baseline).

`schedule_variance_percent = physical_progress_percent - approved_baseline_percent`
only when **both** are independently valid. Unknown physical quantity
returns `unknown_evidence` with possible deterministic variance
bounds if an approved baseline exists. `on_schedule_band_pct=5` is
a configurable reporting tolerance, **not** an uncertainty bound.
The project-level rollup requires independent approved comparable
activity weights (`activity_weight`, `activity_weight_basis`);
IFC element counts are never used as surrogate budget weights.

## 5. Compatibility and execution

Stage 9 appends CSV columns without deleting old fields. Stage 11
v1 `activity_progress`, `schedule_variance` and `dashboard_summary`
keep existing schema versions/keys for compatibility; a nested
`scientific_progress_v2` block is appended to each. The **new v2**
report (not the v1 KPI) is the scientifically interpretable output:

```bash
python -m pipeline.stage_11_schedule_variance.run_schedule_variance \
  --schedule-csv configs/research_progress/synthetic_sample/schedule.csv \
  --mapping-csv configs/research_progress/synthetic_sample/mapping.csv \
  --element-metrics-csv configs/research_progress/synthetic_sample/elements.csv \
  --activity-progress-json /tmp/mycon-B/activity_progress.v1.json \
  --schedule-variance-json /tmp/mycon-B/schedule_variance.v1.json \
  --dashboard-summary-json /tmp/mycon-B/dashboard.v1.json \
  --data-date-utc 2026-01-16 \
  --research-run-id progress-synthetic-demo
```

Unique `progress-*` run IDs; the standalone output is
`/tmp/mycon-B/progress-synthetic-demo/scientific_progress.v2.json`
and cannot be overwritten on retry. Inputs are SHA-256 recorded.
The sample files are fabricated minimal **fixtures**, never real
MyCON scan measurements.

## 6. Evidence experiments / validation plan

Separate controls, isolated building/capture-session splits:

| Case | Ground truth | v2 expected |
|---|---|---|
| Occlusion | BIM surface blocked in all registered views | obs=0 (if audited), physical unknown without certified quantity |
| Never viewed | no audited camera/coverage geometry | obs=null, physical unknown |
| Misregistration | pose/ICP out of metric tolerance | geometry state uncertain, confidence null |
| Partial construction | signed executed quantity 4/10 | physical=0.4 element; geometry may disagree |
| Geometric deviation | visible built element offset beyond tau | geometry nonconformance, not automatically 0% executed |
| Overlap | two IFC AABBs covering same points | ambiguous point never double-counted |
| Small element | 1-9 genuine BIM points | no whole-cloud fallback, no made-up completion |
| Incomplete BOQ | one unknown quantity | point estimate null; identification bounds |
| Non-independent scan points | clustered near-neighbours | no Wilson 95% CI on v2 weighted progress |

Required real-case evaluation before scientific claims: signed quantity
ground truth, approved project schedule baseline, audited mapping weights,
a held-out building/session, per-class precision/recall and calibration
error, dataset counts, environment/seed, failure cases, and input hashes.
No real-case metrics have been asserted by this work.

## 7. Research references

- Bosché, F. (2010), *Advanced Engineering Informatics* 24(1),
  107-118, DOI: 10.1016/j.aei.2009.08.006.
- Tuttas, S., Braun, A., Borrmann, A., Stilla, U. (2014), *ISPRS Archives*
  XL-3, 341-345, DOI: 10.5194/isprsarchives-XL-3-341-2014.
- Braun, A., Tuttas, S., Borrmann, A., Stilla, U. (2015),
  ISARC 2015, DOI: 10.22260/ISARC2015/0034.
- Wilson, E. B. (1927), *JASA* 22(158), score interval for a binomial
  parameter. Its assumptions/estimand do **not** match the legacy
  weighted non-Bernoulli construction-progress proxy.
