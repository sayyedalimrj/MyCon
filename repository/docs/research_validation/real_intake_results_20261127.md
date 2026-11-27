# Independent C real-source intake — 2026-11-27

**Run:** `validation-c-real-source-a-20261127-003`

**Material:** A-owned immutable read-only `MyCON_Civil_Geometry_Round3_Camera_Registration_20261114.zip`, sourced from A's authorized Drive folder; source PR #42 (unmerged). ZIP SHA256: `902e11df91b7c159f787b5730cd946ba220def57ed631882fd341ad9a26f996c`.

**Independent program:** C-owned `benchmarks/mycon_2026/real_intake_audit.py`, not imported from the A pipeline. Recomputes COLMAP SIMPLE_RADIAL world-to-camera sparse reprojection using original zipped `cameras.txt`, `images.txt`, `points3D.txt`; validates A's output asset receipts and five original photo SHA256s by bytes, all without extracting or writing to originals.

**Observed real-source *internal fitting diagnostics* (NOT scientific field-performance numbers):**

| Diagnostic | Recomputed value | Proper interpretation |
|---|---:|---|
| COLMAP registered image records | 192 | In-sample COLMAP solved views, not independent test coverage |
| Original sparse point IDs | 51,894 | COLMAP sparse 3D point IDs, not surveyed reference surfaces |
| Projected 2D–3D observations | 283,059 | Training/BA sparse matches, not a held-out test denominator |
| SHA256-correct verified packaged source/output receipts | 25 | Internal package receipt consistency; historical full upstream source chain not included |
| Median of per-image median reprojection errors | 0.744059 px | Internal COLMAP fit, not depth–camera alignment error |
| Median of per-image p95 reprojection errors | 2.088479 px | Internal COLMAP fit, not field structural accuracy |

All 5 reviewed-photo summary values recomputed from raw sparse points agreed with A receipt within 1e-4 px. Upstream explicitly says `UNREGISTERED`; the depth fusion frame `colmap_world_unverified` is **not proven** to equal the COLMAP camera `COLMAP_sfm_world_unscaled` frame. `metric_scale_verified=false`, `depth_camera_registration_verified=false`. Ground Truth, held-out tests and independent survey missing. NVS PSNR/SSIM/LPIPS, metric geometric accuracy/completeness/F1, scale, ICP/TRE, 6999/new, >192 and physical progress: **not_evaluated**. No number from this diagnostic enters primary thesis comparison tables.

**Verified paths & receipts:** `benchmarks/mycon_2026/evidence/real_source_a_20261127/real_input_audit.json`, `colmap_internal_reprojection.csv` (192 real source rows flagged `scientific_eligible=false`), `source_asset_hashes.csv` (25 actual SHA256s); new unique run; `Thesis_Evidence` includes this as `real_internal_only/` alongside separate pending-real/ and synthetic-only/ compartments.

**Reproduce on the same immutable input package:**

```bash
cd repository
python -m benchmarks.mycon_2026.real_intake_audit \
  --package /READ_ONLY/MyCON_Civil_Geometry_Round3_Camera_Registration_20261114.zip \
  --expected-package-sha256 902e11df91b7c159f787b5730cd946ba220def57ed631882fd341ad9a26f996c \
  --output-root /NEW/validation-c-results \
  --run-id validation-c-real-source-a-20261127-003
```

Do not reuse run ID or overwrite output. Real source reproducibility is conditional on availability of this exact A ZIP; it is not committed to GitHub because it contains original photos and geometry and belongs to A. Authenticity/rights beyond manifest SHA and Drive context not independently certified.