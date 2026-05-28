# Literature Map — Q1 Civil/Construction Engineering & Adjacent Venues, 2024–2026

This file is the authoritative literature reference for Phase 4 of the
project *"Development of an Integrated AI- and BIM-Based Framework for
Automated Monitoring of Construction Project Progress"*. Every novelty
introduced in Phase 4 cites entries from this file. Entries flagged
**[CORE]** are the strongest direct competitors / baselines for our work.

The scope is deliberately narrow: only papers that are either (a) directly
about **AI-/BIM-based automated construction progress monitoring**, or
(b) provide **scientific machinery** we will adapt (uncertainty
calibration, evidential multi-view fusion, VLM hallucination grounding,
HITL active learning).

Content was rephrased for compliance with licensing restrictions.

## 1. Target venues (priority order for our submission strategy)

| # | Venue | 2024 IF / Q | Why we target it |
|---|---|---|---|
| 1 | **Automation in Construction** (Elsevier) | ≈10 / Q1 | Single best fit: AiC has historically published all the directly comparable scan-vs-BIM and progress-monitoring work. |
| 2 | **Computer-Aided Civil and Infrastructure Engineering** (Wiley) | ≈10 / Q1 | Strong fit for the algorithmic/uncertainty contribution. |
| 3 | **Advanced Engineering Informatics** (Elsevier) | ≈8 / Q1 | Strong fit for VLM + BIM IFC retrieval contribution. |
| 4 | **Journal of Computing in Civil Engineering** (ASCE) | Q1 | Strong fit for the deterministic geometric-evidence pipeline. |
| 5 | **Journal of Construction Engineering and Management** (ASCE) | Q1 | Strong fit for the management / HITL angle. |
| 6 | **ITcon** (Q1 since 2024) | Q1 | Open-access, fast turnaround; strong for the framework paper. |
| 7 | **Construction Robotics** (Springer) | emerging | Strong if we add the SLAM-aware / BIM-2-RDT angle. |
| 8 | **Journal of Building Engineering** (Elsevier) | Q1 | Adjacent fit. |

## 2. CORE direct competitors (2024–2026, exact-topic)

### 2.1 [CORE-1] Ersoz & Bosché, *Automation in Construction*, Apr 2025

[Evaluating confidence in geometric matching between 3D point clouds and
BIM models by integrating coverage, distance, and distribution metrics](https://www.x-mol.com/paper/1984847047594516480) — CyberBuild Lab,
University of Edinburgh.

- **Problem.** Trustworthy quantitative confidence for scan-vs-BIM
  matching, which is exactly the gate at the centre of our Stage 8/9.
- **Method.** Combines three indices — coverage, distance, distribution —
  into a single confidence score. ISPRS-Annals 2024 precursor:
  [Integrating surface-related indicators of coverage, distance and
  distribution for quantifying scan-to-BIM confidence
  level](https://isprs-annals.copernicus.org/articles/X-4-2024/223/2024/index.html).
- **Limitation we exploit.** Their confidence is a **point estimate** with
  no calibration or interval. There is no per-element bidirectional
  accuracy/completeness (the F-score @ τ that our Stage 9 already
  reports), no temporal/multi-view fusion, and no replay against
  human review.
- **Our positioning.** We *adopt* the coverage+distance+distribution
  decomposition as one of our deterministic baselines, then **extend** it
  with (a) Wilson + bootstrap CIs (already in `stage_09_progress/uncertainty.py`),
  (b) Dempster-Shafer multi-view fusion with explicit conflict mass, and
  (c) post-hoc calibration against HITL reviews.

### 2.2 [CORE-2] Ersoz et al., *Automation in Construction*, Nov 2024

[UAV-based automated earthwork progress monitoring using deep learning
with image inpainting](http://www.ahmetersoz.com/) — also CyberBuild
(Edinburgh).

- **Problem.** Removes occluding construction machinery from UAV imagery
  before progress estimation.
- **Method.** Image inpainting (segmentation + diffusion-based filling)
  used as a **pre-processing** step.
- **Limitation we exploit.** Inpainting "hides" occlusion rather than
  modelling it as evidence uncertainty.
- **Our positioning.** We treat occluded views as *low-confidence
  evidence* via the per-view weight schedule in our multi-view
  evidential fusion, not as data to be fabricated. This is more honest
  for thesis defense.

### 2.3 [CORE-3] Wang et al., *Automation in Construction*, Feb 2025

[Neural radiance fields for construction site scene representation and
progress evaluation with BIM](https://www.x-mol.com/paper/1891393082711695360).

- **Problem.** Same end-to-end task we tackle: as-built capture, BIM
  registration, progress evaluation.
- **Method.** NeRF reconstruction + BIM alignment.
- **Limitation we exploit.** Heavy compute, no per-element uncertainty,
  no evidence-linked decisions.
- **Our positioning.** We keep COLMAP/MVS as the primary baseline (well
  understood, deterministic) but expose a NeRF/3DGS evidence track
  through the existing Stage 7.7 CAMS-GS scaffold so we can compare
  against this paper on identical metrics.

### 2.4 [CORE-4] Mahami et al., *Automation in Construction*, 2024

[Towards accurate correspondence between BIM and construction using
high-dimensional point cloud feature
tensor](https://www.x-mol.com/paper/1776102936491077632).

- **Problem.** Reducing BIM–as-built correspondence error.
- **Reported numbers.** Overall accuracy 93.8 %–99.9 %, error reduced
  from ~16 cm to ~3 cm, four-phase progress monitoring on 38 317
  instances with 1 % monitoring error.
- **Limitation we exploit.** No calibrated confidence; correspondence
  error is reported as a single value. No HITL recalibration.
- **Our positioning.** We can reuse their per-element error reduction
  number as the headline AiC-2024 baseline for the "accuracy of
  BIM-aligned progress" metric.

### 2.5 [CORE-5] Pfitzner, Hu, Braun, Borrmann, Fang (TUM), recent

[Monitoring concrete pouring progress using knowledge graph-enhanced
computer vision](https://www.cee.ed.tum.de/cms/team/fabian-pfitzner/) —
TUM Computing in Civil and Building Engineering, supervised by
Borrmann (a long-running thread that begins with Braun, Tuttas,
Borrmann & Stilla 2015 and continues into 2025).

- **Problem.** Activity-level progress (concrete pouring) inferred from
  imagery and constrained by a knowledge graph derived from BIM.
- **Limitation we exploit.** KG enforces consistency *within an
  activity* but not across multi-view temporal evidence; no calibration.
- **Our positioning.** Our per-element evidence index can be interpreted
  as a lightweight, ground-truth-traceable cousin of their KG.

### 2.6 [CORE-6] Ersoz, Dec 2024 (arXiv 2412.16108)

[Demystifying the Potential of ChatGPT-4 Vision for Construction Progress
Monitoring](https://arxiv.org/abs/2412.16108).

- **Problem.** First systematic evaluation of GPT-4V on real construction
  progress imagery.
- **Limitation we exploit.** No grounding guardrails: VLM outputs are
  taken at face value with no claim-level verification against the
  evidence package. No recalibration loop.
- **Our positioning.** Our Stage 10 grounding guardrails (claim
  decomposition + numeric tolerance check against the evidence package)
  are designed precisely to address this gap.

### 2.7 [CORE-7] Wang et al., Jan 2026 (arXiv 2601.10835)

[Can Vision-Language Models Understand Construction Workers? An
Exploratory Study](https://arxiv.org/abs/2601.10835) — GPT-4o reported at
F1 = 0.756 / accuracy 0.799 for action recognition and F1 = 0.712 /
accuracy 0.773 for emotion recognition.

- **Limitation we exploit.** No reliability diagram, no ECE, no HITL
  replay. The reported confidences are not calibrated.
- **Our positioning.** Our calibration report is the missing piece.

### 2.8 [CORE-8] BIM Informed Visual SLAM for Construction Monitoring (arXiv 2509.13972, Sept 2025) and BIM2RDT (arXiv 2509.20705)

These argue that **the BIM should be an active prior** in the SLAM /
robotic mapping loop, not just a reference file.

- **Our positioning.** This validates our Phase 1 plugin architecture
  where the BIM is consumed as a typed prior at multiple stages
  (Stage 8 metric alignment, Stage 9 visibility policy, Stage 10
  grounding). We can cite these as independent, recent confirmations of
  the design.

## 3. Direct competitors — earlier (Q1, foundational)

These are the canonical baselines that any AiC reviewer expects to see:

- Bosché, *Automated recognition of 3D CAD model objects in laser scans
  and calculation of as-built dimensions*, **AESM 2010**.
  Foundational scan-vs-BIM acceptance review semantics.
- Tuttas, Braun, Borrmann, Stilla, *Acquisition and Consecutive
  Registration of Photogrammetric Point Clouds for Construction Progress
  Monitoring*, **PFG 2017**. Multi-view aggregation as construction
  progress monitoring's "honest" historical baseline.
- Han et al., *Building a Visual Recognition Pipeline for Tracking
  As-built BIM Object Types from Images*, **JCCEE5 2015**. Object-type
  detection precursor.
- Kavaliauskas et al., *Automation of Construction Progress Monitoring by
  Integrating 3D Point Cloud Data with an IFC-Based BIM Model*,
  **Buildings 2022**. End-to-end pipeline that maps cleanly to ours.
- Vassena et al., *Construction Progress Monitoring through the
  Integration of 4D BIM and SLAM-Based Mapping Devices*, **Buildings
  2023**.
- Pal et al., *Construction Photo Localization in 3D Reality Models for
  Vision-Based Automated Daily Project Monitoring*, **JCCEE5 2024**.
- Mostafa et al., *Automated Detection and Segmentation of Mechanical,
  Electrical, and Plumbing Components ... YOLACT++*, **JCEMD4 2024**.

## 4. Methodological building blocks (foundation literature we adapt)

### 4.1 Calibration / reliability

- Naeini, Cooper & Hauskrecht, *Obtaining Well Calibrated Probabilities
  Using Bayesian Binning*, **AAAI 2015** — ECE.
- Brier, *Verification of Forecasts Expressed in Terms of Probability*,
  **MWR 1950** — Brier score.
- Roelofs et al., *Mitigating bias in calibration error estimation*,
  **AISTATS 2022** — equal-mass binning.
- Błasiok & Nakkiran, *Smooth ECE: Principled Reliability Diagrams via
  Kernel Smoothing*, **ICLR 2024** ([arXiv 2309.12236](https://arxiv.org/abs/2309.12236))
  — kernel-smoothed ECE, used as a small-sample-friendly alternative.

### 4.2 Evidential multi-view fusion

- Han, Zhang, Fu & Zhou, *Trusted Multi-View Classification*, **ICLR 2021**
  ([arXiv 2102.02051](https://arxiv.org/abs/2102.02051))
- Sensoy, Kaplan, Kandemir, *Evidential Deep Learning to Quantify
  Classification Uncertainty*, **NeurIPS 2018**
  ([arXiv 1806.01768](https://arxiv.org/abs/1806.01768))
- *Trusted Multi-View Evidential Fusion Framework for Commonsense
  Reasoning*, **LREC-COLING 2024** ([aclanthology.org](https://aclanthology.org/2024.lrec-main.152))
- *Evidential Deep Partial Multi-View Classification With Discount
  Fusion* ([arXiv 2408.13123](https://arxiv.org/abs/2408.13123), 2024)
- *Fairness-Aware Multi-view Evidential Learning with Adaptive Prior*
  ([arXiv 2508.12997](https://arxiv.org/abs/2508.12997), 2025)
- Dempster, *Upper and Lower Probabilities Induced by a Multivalued
  Mapping*, **AMS 1967**.

### 4.3 Conformal prediction (alternative uncertainty channel)

- *Conformal Prediction* tutorial ([arXiv 2410.06494](https://arxiv.org/abs/2410.06494), 2024)
- *Adaptive Uncertainty Quantification for Generative AI*
  ([arXiv 2408.08990](https://arxiv.org/abs/2408.08990), 2024) — split conformal wrapper
- *Conformal Prediction for Language Models* ([arXiv 2604.08885](https://arxiv.org/abs/2604.08885), 2026) — for the VLM answer track

### 4.4 VLM hallucination & grounding

- *Multi-Modal Hallucination Control by Visual Information Grounding*
  ([arXiv 2403.14003](https://arxiv.org/abs/2403.14003), 2024)
- *Pelican: Correcting Hallucination in Vision-LLMs via Claim
  Decomposition and Program of Thought Verification*
  ([arXiv 2407.02352](https://arxiv.org/abs/2407.02352), 2024)
- *CoRGI: Verified Chain-of-Thought Reasoning with Post-hoc Visual
  Grounding* ([arXiv 2508.00378](https://arxiv.org/abs/2508.00378), 2025)
- *Contextual Embeddings for Robust Hallucination Detection &
  Grounding in VLMs* ([arXiv 2411.19187](https://arxiv.org/abs/2411.19187), 2024)
- *Visual Attention Reasoning via Hierarchical Search and
  Self-Verification* ([arXiv 2510.18619](https://arxiv.org/abs/2510.18619), 2025)
- *An Attributable Benchmark for Diagnosing Object Hallucination in
  Vision-Language Models* ([arXiv 2604.22822](https://arxiv.org/abs/2604.22822), 2026)

### 4.5 HITL / active learning

- Beck et al., *Beyond Active Learning: Leveraging the Full Potential of
  Human Interaction*, **WACV 2024** ([CVF link](https://openaccess.thecvf.com/content/WACV2024/html/Beck_Beyond_Active_Learning_Leveraging_the_Full_Potential_of_Human_Interaction_WACV_2024_paper.html))
- Rožanec et al., *Human in the AI Loop via xAI and Active Learning for
  Visual Inspection*, **arXiv 2307.05508**, 2023.

### 4.6 Knapitsch / classical metric backbone

- Knapitsch et al., *Tanks and Temples: Benchmarking Large-Scale Scene
  Reconstruction*, **SIGGRAPH 2017** — F-score @ τ at the centre of our
  bidirectional metric.

## 5. Map: which competitor each Phase 4 contribution is positioned against

| Phase 4 contribution | Direct competitor(s) | What we add |
|---|---|---|
| Calibration & reliability report | Ersoz & Bosché 2025 (point-estimate confidence); Ersoz arXiv 2412.16108 (un-calibrated GPT-4V) | Naeini ECE + smooth-ECE + Brier on the discrete confidence labels, validated against HITL replay. |
| Trusted multi-view temporal fusion | Tuttas et al. 2017; Vassena et al. 2023 (implicit aggregation) | Han et al. 2021 / TMC 2024 evidential fusion with explicit Dempster conflict mass at the per-element level. |
| HITL corrections + replay | Beck WACV 2024; Rožanec 2023 | First-class structured HITL log scoped to the construction progress decisions (element acceptance, activity completion, VLM answer); deterministic replay → calibration. |
| VLM grounding guardrails | Ersoz arXiv 2412.16108; Wang arXiv 2601.10835; Pelican 2024 | Numeric-tolerance claim verification against the deterministic Stage 9 evidence package; integrates with the existing Stage 10 `answer_validator`. |
| Per-element evidence index | Pfitzner/TUM KG-enhanced CV; Ersoz & Bosché 2025 | Lightweight, deterministic per-IFC-GlobalId index that becomes the shared input for fusion / HITL / VLM grounding. |
| Literature comparison & baseline framework | n/a (paper-writing tooling, not a research method) | Schema for "method × metric × CI" comparison tables that can be exported to LaTeX directly for AiC submission. |

## 6. White-space we are claiming

After this scan, the following white-space remains:

1. **Calibrated, evidence-linked, HITL-replayable progress decisions**
   — none of the direct competitors close the loop from prediction →
   review → recalibration with auditable artefacts.
2. **Explicit Dempster conflict mass at the per-BIM-element level** —
   Trusted-MVC 2021 / TMC 2024 use the technique in classification, but
   not in scan-vs-BIM construction progress monitoring. We are the
   first.
3. **VLM claim-grounding against a deterministic evidence package** with
   numeric-tolerance verification — concurrent VLM-on-construction work
   (Ersoz 2024, Wang 2026) does *not* do this.

These three white-space areas are exactly what Phase 4 is implementing.
