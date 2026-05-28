# Data Contracts

This document records the file-contract boundaries for the construction progress monitoring pipeline.

## Rules

1. Every stage reads only declared inputs.
2. Every stage writes only declared outputs.
3. All paths are configured in YAML.
4. Intermediate files are written atomically when feasible.
5. Optional modules must not block the baseline pipeline.
6. Generated data, raw videos, normalized videos, keyframes, dense workspaces, model weights, runs, and exports are not committed to Git.

## Stage 1: ingest and normalization

### Inputs

| Key | Path |
|---|---|
| `inputs.video` | `data/raw/site01.mp4` |

### Outputs

| Key | Path |
|---|---|
| `paths.normalized_video` | `data/normalized/site01_normalized.mp4` |
| `paths.metadata_json` | `data/normalized/site01_metadata.json` |
| `paths.quality_csv` | `data/normalized/site01_frame_quality.csv` |
| report | `runs/<run_id>/reports/stage_01_ingest_report.json` |

### Quality CSV columns

Contract columns:

```text
frame_index
timestamp_sec
sharpness_laplacian
exposure_mean
exposure_std
exposure_jump
motion_score
duplicate_similarity
novelty_score
quality_score
reject_reason
```

Hardened diagnostics:

```text
histogram_similarity
feature_count
feature_density_score
adaptive_blur_threshold
rolling_shutter_score
jitter_score
sampling_method
warning_reason
scoring_width
scoring_height
```

## Stage 2: adaptive keyframes

### Inputs

| Key | Path |
|---|---|
| `paths.normalized_video` | `data/normalized/site01_normalized.mp4` |
| `paths.quality_csv` | `data/normalized/site01_frame_quality.csv` |

### Outputs

| Key | Path |
|---|---|
| `paths.keyframes_dir` | `data/frames/key/site01/*.jpg` |
| `paths.manifest_csv` | `data/frames/key/site01_manifest.csv` |
| `paths.contact_sheet` | `data/frames/key/site01_contact_sheet.jpg` |
| report | `runs/<run_id>/reports/keyframe_summary.json` |

### Manifest columns

Required columns:

```text
keyframe_id
source_frame_index
timestamp_sec
image_path
segment_id
sharpness_laplacian
exposure_mean
motion_score
novelty_score
quality_score
keep_sparse
keep_dense
selection_reason
```

Additional diagnostics when available:

```text
exposure_jump
duplicate_similarity
reject_reason
warning_reason
feature_count
feature_density_score
histogram_similarity
rolling_shutter_score
jitter_score
selection_score
```

### Stage 2 engineering notes

- Stage 2 does not recompute Stage 1 quality metrics.
- Selection is segment-aware and preserves temporal coverage for downstream ordered-image matching.
- `keyframes.random_seek_extraction` defaults to `false`; sequential extraction is safer for mobile Long-GOP video.
- Stage 2 validates that `frame_index` values fit within the normalized video frame count.
- Stage 2 can verify `timestamp_sec ≈ frame_index / fps` to catch accidental VFR/CFR drift before keyframes are extracted.
- Fallback selection is controlled by config and records `selection_reason` when relaxed or emergency criteria are used.
- Emergency fallback is only a safety net to avoid an empty manifest on very poor videos; the JSON report flags this condition for review.
- The contact sheet uses bounded two-pass thumbnail rendering.
- The manifest records both `keep_sparse` and `keep_dense` so later stages can consume the same keyframe folder with different policies.

## Stage 3 handoff

Stage 3 should consume:

```text
data/frames/key/site01/*.jpg
data/frames/key/site01_manifest.csv
```

Stage 3 should not run until Stage 2 keyframes, manifest, contact sheet, and summary report are verified.
