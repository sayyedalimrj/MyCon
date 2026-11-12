# C — Round-2 execution evidence and publication readiness audit

**Date:** 2026-11-12. **Owner:** C; **PR:** #36; target **Integration**, not main. The frozen coordinator Round-2 contract was read from `research/mycon-integration-20261112` before implementation. **This document distinguishes actual execution from code-only implementation; no real-site performance claim is supported.**

## Source integrity and execution environment

- Actual workstream source commit immediately before evidence commit: `155c9b14d03ea2f203b7b53dcba6c3166f2fb52a`. Source snapshot was assembled from the exact committed C files and verified using Git blob SHA-1, not an unverified retyped script. A complete full-repository checkout was not available: outbound GitHub DNS/network from local container failed. GitHub connector was used for versioned commits; it cannot itself execute code.
- Actual local Python: `3.13.5`, Linux x86_64, 5 CPU cores enumerated, GPU unmeasured; pytest `9.0.2`, NumPy `2.3.5`, SciPy `1.17.0`, Pillow `12.3.0`, scikit-image `0.26.0`, matplotlib `3.10.8`. Test harness fixed seed 42; A/B model seeds and hardware have not been independently supplied.
- Exact source Git blobs (examples): `metrics.py` `908ec96cb45856146e46bc6f53383a389de35fe5`; `runner.py` `ae5a910ad707923c180f696faccfeb42b5523577`; `adapters.py` `873a338ed06b230d532401ab9fe6da897ae9cc01`; `tests/test_research_validation_round2_contract.py` `63bddbe03ec41266a7fdf3d7c2296df5aa66158d`.
- Machine-readable evidence: `repository/benchmarks/mycon_2026/evidence/round2_tool_smoke_20261112.json`. Original local full console logs, case-level results and manifests are provided separately in the downloadable Round-2 tool-test bundle. Local full evidence JSON SHA-256: `e7812f6f592c1e592f54f139f78788cb3f698ac3bd1b9520721cb64d952fbe46`.

## Commands actually executed

From the local source snapshot working directory `repository/`:

| Command / role | Exit code | Observed outcome |
|---|---:|---|
| `python -m pytest -q tests/test_research_validation_metrics.py tests/test_research_validation_round2_contract.py` | 0 | **31 passed, 0 failed**, isolated C lightweight tests |
| `python scripts/research_validation_run.py --help` | 0 | CLI documented arguments parse |
| `python scripts/research_validation_preflight.py --help` | 0 | A/B preflight CLI parses |
| `python scripts/research_validation_preflight.py --b-stage9-csv <fixture>/stage9.csv --b-stage11-v2-json <fixture>/v2.json` | 0 | Synthetic Stage9 quantity provenance and B-v2 unit contracts pass |
| `python -m benchmarks.mycon_2026.runner --manifest <fixture>/manifest.json --output-root <new-external-output> --run-id bench-round2-fixture-final2 --include-fixtures` | 0 | 9 synthetic task records; 5 evaluated for **tool correctness only**, 4 missing input = not_executed |
| `python -m benchmarks.mycon_2026.figures --result-dir <fixture-run>` | 0 | Every scientific figure reported `not_executed`, no real-data plot manufactured |

Actual synthetic source-manifest SHA-256 `afd1b9f77830d21e09ce7b5aa5c0e813e509e6d3c41d1eceb8eab6026fc726c5`. Exactly **one fictional site, one fictional building/session**, **zero real sites**. Scientific output `scientific_results.json` contains `[]`, SHA-256 `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`. Fixture `cases.json` SHA-256 `ea595a3d2d62172150b2eddf3daa4d62dcf12ffab08bd2f65f7be25d24231ad0`; `metadata.json` SHA-256 `c91768bba5de3b9316b74651c588958176ab2c0c398e4db638401b688cd2429f`; `figures_status.json` SHA-256 `c705634b064cc5c8c4268abb489c68fd5712d4edbc7b398bcf1973a83f851a16`.

**Scope limitation:** This is actual execution of committed C module bytes in a local lightweight source snapshot. It is **not** evidence that full MyCON repository regression, A/B integration paths, real scanner files, checkpoint restoration, CUDA training/rendering, Open3D registration, or GitHub Actions CI passed. None of those checks were performed in this round.

## Implemented vs actually evaluated

| Evidence component | Code status | Actual experiment status |
|---|---|---|
| Independent numeric vs categorical GT; fractions [0,1], percent [0,100], unknown and abstention denominators | Implemented and unit-tested | 31 suite tests, synthetic only |
| Stage9 additive scientific CSV and Stage11 `mycon.scientific_progress.v2` adapters | Implemented, fail-closed | Local synthetic CLI passed; **real site not_executed** |
| Optional A `mycon.bim_geometry_evidence.v1` manifest and verified independent survey hash | Implemented and synthetic contract tested | A real producer artifact not supplied: **not_executed** |
| COLMAP/VGGT/DUSt3R/GS geometry, multimethod metric reconstruction/registration, IFC identity | Tools implemented (some methods need external producer) | **not_executed** for actual real methods |
| Held-out photometric PSNR/SSIM and optional LPIPS | Implemented; synthetic PSNR/SSIM test passed | Real heldout image test and LPIPS evaluator **not_executed** |
| Approved independent BOQ progress and schedule project/activity error | Implemented and tested with fixture | Independent approved quantities/baseline **not_executed** |
| Paired baseline + 5-seed ablation and site-level bootstrap | Registered; synthetic math tests | No real paired sites or approved GT: **not_executed** |
| Real performance (wall time/GPU VRAM/RSS)/failure-rate and uncertainty | Readers and figure templates implemented | Real GPU, NVS, registration and end-to-end: **not_executed** |
| Full historical lightweight pytest regression & A+B+C integration tests | Not claimed | **not_executed**, full source checkout unavailable |

## Publication readiness — exact remaining evidence

1. Independently surveyed same-date metric as-built GT with instrument accuracy, coordinate frame, original signed source and SHA-256; separate registration-fit and **held-out** TRE anchors; IFC GlobalId/LoD correspondence approval.
2. Approved as-planned and as-built measured quantities per element, common units, approved BOQ and activity/project weights, reviewer signoff; independent schedule baseline and project data date.
3. Site-disjoint locked train/validation/test set, capture frames/RAW hashes, duplicate checks, held-out NVS images and signed GT manifests, registered sample/exclusion counts.
4. Paired real outputs from A/B plus COLMAP and optional VGGT/DUSt3R/GS surface extraction on *identical* test cases, checkpoint/model/config SHA, hardware, runtime/VRAM, A geometry evidence v1 and B scientific progress v2.
5. Execution of frozen full-repository lightweight suite plus Stage8/9/11 cross-module tests and practical end-to-end case, with independent check-point error and failure taxonomy. If GPU unavailable, mark GPU tests not_executed.
6. Complete preregistered baselines/ablations, site-level cluster CI (>=2 sites minimum for illustrative interval; more for reliable inferential claims), effect size, error by lighting/occlusion/view/class/density, and final blinded reviewer audit.

**Acceptance statement:** `implemented` and `unit_tested` apply to C tools; `gpu_validated`, `real_case_validated`, and `publication_ready` do **not**. Neither Duran's external ZFD/LSAA results nor synthetic toy scores establish any MyCON method superiority.
