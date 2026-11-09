# 🔒 FROZEN BASELINE — MYCON-GSPLAT-7000-20261106

**This branch marks the successfully executed MyCON gsplat baseline, not a new training run.**

| Item | Frozen value |
| --- | --- |
| Successful baseline CPU/GPU source pin | `abc97337e5268d82e49cec4623c0cf5730386b19` |
| Existing experiment | `MyCon_R4/research_colmap_gsplat/811c5a1323e45e2a/mycon_abi_audited_v6_20261106` |
| Gaussian model | 1,477,725 splats at training step 6999 (7,000 steps) |
| COLMAP | 192 registered frames, 51,894 sparse points |
| Heldout PSNR / SSIM / LPIPS | 20.7359 dB / 0.6176 / 0.5219 |
| Model frame | **COLMAP arbitrary units, NOT metric** |
| Phase 2 | 11/12 completed, experimental mesh blocked |
| LongSplat | Separate data prepared, **no verified trained LongSplat model** |

**Do not amend, rebase, force-push or delete this baseline branch or alter the matching Drive files.** The commit SHA and SHA256s are recorded in [`BASELINE_GSPLAT_7000_20261106.manifest.json`](BASELINE_GSPLAT_7000_20261106.manifest.json). Artifact bytes are too large to store in this Git branch; the manifest points to the original Google Drive files and records their hashes.

- [Frozen PLY](https://drive.google.com/file/d/1ebbWKdP76PcxhXCfAtO8A-3DqgypPpXe/view)
- [Frozen checkpoint](https://drive.google.com/file/d/1kUHTWhpjj5WCmAWTDNSbJpwKOeZgUBEx/view)
- [Camera trajectory video](https://drive.google.com/file/d/1uEaMlMby0uEcfLp249zIw72Ia3YFztO7/view)
- [Evaluation receipt](https://drive.google.com/file/d/1I8Bi7im8xRy31Bk_-zrXaVJotO69TKHR/view)
- [Phase 2 output folder](https://drive.google.com/drive/folders/1_pGh0pvNgeVOJQbCWWbZD4aoEUT45yG5)

## Scientific status

These recorded metrics are the baseline for future comparisons. This branch does **not** assert valid novel-view coverage, metric geometry, scene-wide camera scale, IFC-guided comparison, or BIM progress. The real T4 run also applied **runtime-only hotfixes**; the pinned repository commit alone is not a byte-for-byte reconstruction of the Python process that produced the outputs.

For the next attempt, use a separate branch, run ID, and destination. Never use the baseline Drive experiment as an output directory.
