# MyCON — independent critical integration audit & research-paper phase comparison
Date (Iran local): 2026-11-13  
Scope: \`research/mycon-integration-20261112\`, following A #38, B #37 and C #36 merging.  
Reference: Ayça Duran et al., *Automated facade solar retrofits using feed-forward 3D reconstruction and generative image editing*, *Automation in Construction*, Volume 193 (January 2027 issue), Article 107294; published online 2026-10-13, DOI: https://doi.org/10.1016/j.autcon.2026.107294 . Official implementation: https://github.com/ycdrn/image2bipv .

## 1. Review verdict and scope
**Code-integration candidate, not field-validated measurement software and not publication-ready.**
The paper's aim is street-view façade geometry, image rectification, BIPV generative design, and solar estimates. MyCON aims at site-video reconstruction, reference BIM comparison, verifiable construction quantities, and activity/schedule progress. Code overlaps are methodological only; published paper error numbers cannot be transferred to MyCON.

The user explicitly suspended site-specific 14021208.MAX registration. This review does not perform any MAX-specific fit, start fresh large-model training, merge to main, overwrite original video/PLY/checkpoints, or fabricate field results.

## 2. Critical code defects actually fixed in Integration
| Severity | Fault in committed code | Specific change / regression |
|---|---|---|
| High — geometry false support | Original \`extract_edges\` took global minimum/maximum of near-line points from each plane, creating a **single fictitious line over unobserved gaps** | Split near-line evidence into finite locally sampled spans; require support on **both** surfaces for each overlap; cap acceptable intersample gap (conservative false-negative preference); synthetic island/sparse-endpoint tests |
| High — edge/corner correspondence | Original \`edge_by_plane\` dict kept only the **last** segment per plane pair, causing unsupported/overwritten corner association | Keep all finite segments for every pair and choose actual locally containing segment around proposed corner; synthetic trihedral/false distant edge test |
| Medium — numeric stability | Integer-valued plane normals triggered an in-place NumPy float-division casting exception | Convert and normalize normals and offsets to finite float64; reject degenerate intersections |
| High — misleading metric readiness | Stage 8 metric alignment CLI printed \`STAGE_08_METRIC_ALIGNMENT_OK\` and returned process exit **0** even when \`can_feed_stage8=false\` | Return **exit 2** and \`BLOCKED\` label on invalid/missing surveyed alignment or failed quality gate; positive/negative CLI tests |
| High — misleading model provenance | Stage 7.5 accepted arbitrary VLM provider names but only generated deterministic/mock observations | Non-mock requests now fail **before file mutation**; evidence includes \`vlm_inference_executed=false\` and deterministic QA origin; two regression tests |
| Regression surface | A native \`outputs\` vs B/C fixture schema/feature mismatch already fixed during earlier Integration merge | Existing real-producer A→B→C contract; SHA tamper and false metric assertions blocked; additional synthetic A-produced-PLY→C metric execution path test, explicitly *not* real evidence |

**Residual algorithm limitations:** Convex hulls may still bridge scan holes, visually convincing Gaussian centers are not necessarily physical surfaces, RANSAC thresholds depend on observation density, two-density repeatability is not metrological Ground Truth, camera FOV projection cannot certify occlusion, and no calibrated/reviewed corner result exists for the real site.

## 3. Paper method vs MyCON implementation — no false equivalence

| Duran et al. actual step | MyCON code and execution evidence | Status |
|---|---|---|
| Feed-forward 3D from DUSt3R/VGGT | \`stage_04_5_cams_gs/research_backends.py\` only exposes **optional** VGGT demo and an external DUSt3R export adapter; **not executed** on original MyCON site. Historical COLMAP + official 3DGS checkpoint from separate pipeline is a different algorithm, not a replication of the paper | Partial interface only |
| Facade/element segmentation mask | Stage 3/5 mask hooks and Stage 7 \`semantic_context.py\`; no verified facade-specific per-pixel benchmark or mask-to-reconstruction paper equivalence | Partial |
| RANSAC dominant finite plane, proportional geometry | Stage 7 \`plane_extraction.py\`, \`comparison_ready.py\`: deterministic local planes, finite patches, candidate edges/corners, RGB provenance, tests | Implemented CPU / synthetic only |
| GeoCalib-estimated gravity | Stage 8 frame/metric tools and optional \`gravity_up\` argument; **no automated real GeoCalib model result or independent gravity verification** | Interface/diagnostic only |
| Gravity-aligned facade orthorectification | A corner-to-camera FOV projection is present; a **complete rectified facade image / inverse pixel resampling pipeline** as in the paper is **not validated/implemented end-to-end** | Missing |
| VLM-based facade height (metric cue), width = AR × height | General-purpose Stage 10 grounded local VLM client/answer controls exist; **no measured, validated article-style height/width inference pipeline**. For MyCON BIM, an independently surveyed metric anchor is preferable | Not implemented/validated |
| BIPV suitability mask, panel layout, diffusion/Gemini editing | Not a MyCON construction-progress objective | Out of scope |
| Edited image reprojected to street-view perspective | Geometric corner FOV projection is **not** generative-image inverse reprojection | Not implemented/out of scope |
| Façade solar yield estimation from irradiation | Not a MyCON objective | Out of scope |
| Paper’s real ZFD/LSAA evaluation | C benchmark supports synthetic Chamfer, accuracy/completeness, F-score, structural features, and VLM grounding; **no real ZFD/LSAA experiment or independent MyCON building GT** | Software implemented, scientific execution absent |

Article reported **ZFD DUSt3R multi aspect-ratio normalized MAE 14.90%** and **LSAA DUSt3R single 8.81%** (Table 5), and about **0.5 floors** for floor-count MAE. These are **published article-only results**, not MyCON accuracy. No apples-to-apples runtime, dataset, coordinate frame, or error definition comparison has been run. For MyCON, there is **no defensible real-site reconstruction accuracy/completeness percentage**.

## 4. Native MyCON implementation stage-by-stage
| Stage | Code status | Actual measured project evidence |
|---|---|---|
| 1–2 Video ingest and keyframes | Wrappers/tests available | Historical selected/captured frames documented; current integrated real run **not executed** |
| 3–4 COLMAP sparse, bundle refinement | Code and historical COLMAP reconstruction available | Full current real case/end-to-end quality not rerun |
| 4.5 CAMS-GS / LongSplat / 3DGS | Research adapters and provenance auditor; earlier official gsplat train reached reported 6999 checkpoint | Artifact existence/historical logs are **not** surface or held-out metric verification; LongSplat not confirmed |
| 5 Dense geometry | COLMAP dense pipeline exists | No current verified real dense → clean handoff |
| 6 DA3 assist | Depth alignment/fusion and optional model entry points | Real DA3 inference + metric comparison not verified |
| 7 Cleanup and comparison-ready features | Executable real PLY reader, finite planes, gap-safe candidate edges/corners, hash manifest, opt-in Stage7 reuse | Real 348,744,578-byte \`point_cloud_6999.ply\` was found in Drive but connector blocks >268,435,456-byte download; **site result unavailable** |
| 7.5 VLM QA | Deterministic geometric QA + preview **only** (now clearly labeled and fail-closed for real provider requests) | No real VLM inference |
| 8 Metric alignment / BIM registration | Survey anchors, robust Sim3/ICP, quality and IFC evidence contracts, fail-closed CLI | No independently surveyed metric anchors + reviewed real BIM correspondence |
| 9 Element progress | Geometric and quantity evidence adapters, ambiguity/visibility gates, \`unknown\` semantics | No independently approved site execution quantities; **physical progress unknown**, never fabricated 0% |
| 10 Copilot | Local VLM client and grounding/audit guard | No real endpoint-grounded site VLM run validated |
| 11 Schedule / BOQ | Scientific v2 quantity/weight schedule model plus legacy compatibility | No approved real schedule baseline and element-linked BOQ in end-to-end site run |
| C Scientific benchmark | CLI, synthetic score/protocol, independent GT hashes & split guards, plotting definitions | Real multi-site baselines/ablations and GT remain \`not_executed\` |

## 5. Evidence of tests and packaging
GitHub Actions:
- CPU integration workflow: https://github.com/sayyedalimrj/MyCon/actions/workflows/mycon-research-integration.yml . Tests on **actual repository checkout**, with representative synthetic A-generated outputs consumed by B and C. This tests interoperable code, not field error.
- Extended lightweight regression: https://github.com/sayyedalimrj/MyCon/actions/workflows/mycon-extended-regression.yml . Open3D/IfcOpenShell installed; GPU/server/slow tests deselected as indicated in job logs.
- Official server handoff: the CPU workflow executes \`scripts/export_server_handoff_zip.py\` followed by \`python -m scripts.verify_server_handoff_zip\` and uploads its verified ZIP as \`MyCON_Integration_Server_Handoff_RC\` (subject to successful CI execution).

All assertions of passing tests must use the **same exact commit as the final package**, including the SHA from that run. Do not confuse previous local scratch tests or older successful workflows with the final HEAD.

## 6. Release classification and remaining non-negotiable scientific gates

- **CPU research-code Integration:** pass iff exact-HEAD cross-workstream and extended lightweight CI both pass.
- **Real reconstruction acceptance:** blocked pending accessible real COLMAP dense/3DGS-derived structural PLY and camera/source evidence, with blind visual review of plane/edge/corner outputs. Raw Gaussians cannot substitute for physical surface GT without independent evidence.
- **Metric BIM geometry acceptance:** blocked pending independent real-world scale, gravity, IFC version/IDs, mutually consistent registration and held-out surveyed anchors; record target-frame residuals.
- **Physical progress acceptance:** blocked pending independently approved element quantities, BOQ weights, observation/visibility evidence, and accepted baseline; do not map geometric presence directly to percentage completion.
- **Article reproduction / publishable quantitative comparison:** blocked pending site-disjoint datasets, independently surveyed GT, frozen capture hashes, shared method baselines, uncertainty/ablations, and tested GPU implementations. The paper's BIPV editing/energy pathway is outside MyCON scope.
- **\`main\` branch:** unchanged. This research release candidate remains on Integration; preserve checkpoints, raw files, and branch history.
