# MyCon — Construction Progress AI+BIM Pipeline

> Development of an Integrated AI- and BIM-Based Framework for
> Automated Monitoring of Construction Project Progress.

A research-grade framework that takes a captured site (mobile video or
laser scan) and produces a calibrated, evidence-linked answer to the
question "is this project on schedule?" It joins:

- **Geometry** — COLMAP / Open3D / IfcOpenShell scan-vs-BIM (Stages 1–9);
- **Schedule** — canonical CSV imported from Microsoft Project /
  Primavera P6 / vendor exports (Stage 11);
- **Reasoning** — local VLM with deterministic claim grounding
  (Stage 10 + grounding guard);
- **Trust** — Wilson 95 % intervals, HITL corrections, ECE / smooth-ECE
  calibration (Phase 4);
- **Dashboard** — typed React UI with KPI strip, activities table,
  reliability diagram, HITL submit form, calibration replay button
  (Phase 3 + 5).

## Run on Google Colab

Open [`MyCon_Colab_Pipeline.ipynb`](MyCon_Colab_Pipeline.ipynb) in
Colab (GPU runtime recommended). The notebook mounts Google Drive,
installs system + Python deps, and launches a Gradio UI that drives
the pipeline stage-by-stage with live logs, memory cleanup between
heavy stages, and direct downloads. Helper code lives in
[`colab/`](colab/) and a user-facing walkthrough lives at
[`docs/colab_workflow.md`](docs/colab_workflow.md). All Colab outputs
are written directly to Drive so a Colab disconnect does not lose work.

## 30-second tour

```bash
# 1. Reproduce the synthetic walkthrough end-to-end (no GPU, no VLM,
#    no Open3D required; runs in under a second).
python3 scripts/run_end_to_end_walkthrough.py \
    --output-dir runs/example_walkthrough/ \
    --data-date-utc 2026-04-16
# -> writes activity_progress.json, schedule_variance.json,
#    dashboard_summary.json, calibration_report.json,
#    grounding_guard_demo.json, walkthrough_summary.json.
# Headline: 1 activity on schedule, 1 behind, ECE on 6 reviewer
# corrections measured, 3 VLM grounding-guard failure modes
# demonstrated.

# 2. Run the full lightweight test suite (~5 s, no heavy deps).
PYENV_VERSION=3.11.15 pytest -m lightweight \
    --ignore=tests/test_service_api.py \
    --ignore=tests/test_service_websocket.py
# -> 580+ tests pass.

# 3. Build and serve the dashboard.
cd gui && npm install && npm run build
# -> Schedule Compare page available at /schedule.
```

## Quick start

### A. Set up the Python environment

The pipeline targets **Python 3.11**. A laptop test set runs without
any geometry libraries; the full geometry stack (Open3D, OpenCV,
IfcOpenShell, COLMAP) is only needed for Stages 3–9.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Lightweight (Stages 11 + Phase 4 modules + dashboard backend):
pip install -r requirements-core.txt

# Full geometry stack (Stages 3-9):
pip install -r requirements-server.txt   # see docs/server_handoff_checklist.md
```

If `requirements-core.txt` is missing in your fork, the minimum
dependencies for the laptop test set are `numpy`, `pyyaml`, and
`pytest`.

### B. Reproduce the walkthrough

The committed synthetic site under
[`examples/end_to_end/`](examples/end_to_end/) exercises every
Phase 4 module deterministically:

```bash
python3 scripts/run_end_to_end_walkthrough.py \
    --output-dir runs/example_walkthrough/ \
    --data-date-utc 2026-04-16
```

Outputs (with [`schema_version`](docs/end_to_end_finishing_plan.md)
fields locked):

| File | Schema | What it carries |
|---|---|---|
| `activity_progress.json` | `activity_progress.v1` | Per-activity rollup of element-level acceptance |
| `schedule_variance.json` | `schedule_variance.v1` | Planned vs actual % at the data date, with Wilson 95 % CIs |
| `dashboard_summary.json` | `dashboard_summary.v1` | Exactly the JSON the GUI consumes |
| `calibration_report.json` | `calibration_report.v1` | ECE, MCE, Brier, smooth-ECE on the HITL log |
| `grounding_guard_demo.json` | `grounding_guard_demo.v1` | The three VLM hallucination failure modes |
| `walkthrough_summary.json` | `walkthrough_summary.v1` | Index of every output by SHA-256 |

### C. Bring your own project

Three input files plus the existing geometry pipeline are all you need:

1. **Schedule CSV** in canonical form
   (see [`docs/schedule_format.md`](docs/schedule_format.md)). If your
   schedule lives in MS Project or Primavera P6, run one of:
   ```bash
   python3 scripts/import_schedule_msp_xml.py --input my.xml --output configs/schedule.csv
   python3 scripts/import_schedule_p6_xer.py  --input my.xer --output configs/schedule.csv
   python3 scripts/import_schedule_generic_csv.py --input vendor.csv --output configs/schedule.csv \
       --activity-id-column "Task ID" --activity-name-column "Task Name" \
       --planned-start-column "Start Date" --planned-finish-column "Finish Date"
   ```
2. **BIM ↔ schedule mapping CSV** with `(activity_id, ifc_global_id, weight)` rows.
3. **Stage 9 `element_metrics.csv`** (already produced by the geometry
   pipeline). On a laptop you can synthesise this manually for
   experiments.

Then run Stage 11:

```bash
python3 -m pipeline.stage_11_schedule_variance.run_schedule_variance \
    --schedule-csv configs/schedule.csv \
    --mapping-csv  configs/bim_schedule_mapping.csv \
    --element-metrics-csv runs/<run_id>/reports/element_metrics.csv \
    --activity-progress-json runs/<run_id>/reports/activity_progress.json \
    --schedule-variance-json runs/<run_id>/reports/schedule_variance.json \
    --dashboard-summary-json runs/<run_id>/reports/dashboard_summary.json \
    --data-date-utc 2026-04-16
```

### D. Boot the dashboard

```bash
cd gui
npm install
npm run dev    # http://localhost:5173/schedule
# or
npm run build  # production bundle in gui/dist/
```

The page consumes `/api/v1/schedule/dashboard`,
`/api/v1/schedule/activities/{id}`, `/api/v1/calibration/report`,
`/api/v1/calibration/run`, `/api/v1/hitl/corrections`, and
`/api/v1/elements/{id}` — all served by `pipeline.service.app` (Phase 2)
plus the schedule / HITL / calibration routers (Phase 5).

## Repository layout

| Path | Purpose |
|---|---|
| `pipeline/common/` | Phase 1 typed config, registry, provenance, plugins; Phase 4 calibration, HITL, schedule I/O, BIM↔schedule mapping, method comparison |
| `pipeline/stage_*/` | One directory per pipeline stage; canonical I/O contracts and CLI runners |
| `pipeline/stage_11_schedule_variance/` | Phase 4 schedule-variance stage (laptop-runnable, no Open3D) |
| `pipeline/service/` | Phase 2 + 5 backend: pipeline / artefact / run-control / schedule / HITL / calibration endpoints |
| `pipeline/stage_10_copilot/grounding_guard.py` | VLM claim verification with imperial unit support and a plug-in `ClaimExtractor` Protocol |
| `gui/` | Phase 3 + 5 React + Vite + Tailwind dashboard with the Schedule Compare page |
| `examples/end_to_end/` | Synthetic walkthrough fixture and runner |
| `scripts/` | Side-car CLIs (schedule importers, walkthrough runner, calibration report, LaTeX comparison renderer) |
| `docs/` | Reviewer-facing reference (see "Documentation index" below) |
| `tests/` | Lightweight Python tests (`pytest -m lightweight`) |
| `configs/` | Sample YAML configs used by the geometry stages |
| `docker/` | Server-grade container build for the heavy geometry stack |

## Documentation index

| Document | What it covers |
|---|---|
| [`docs/end_to_end_finishing_plan.md`](docs/end_to_end_finishing_plan.md) | The architectural target: capture → schedule comparison → dashboard |
| [`docs/literature_q1_2024_2026.md`](docs/literature_q1_2024_2026.md) | Q1 2024–2026 literature map (AiC, CACAIE, AEI, JCCEE5, JCEMD4, ITcon, Construction Robotics, J. Building Eng.) |
| [`docs/schedule_format.md`](docs/schedule_format.md) | Canonical schedule CSV reference + side-car importers |
| [`docs/hitl_workflow.md`](docs/hitl_workflow.md) | HITL corrections workflow (Beck WACV'24-style) |
| [`docs/calibration_workflow.md`](docs/calibration_workflow.md) | Reliability / ECE / smooth-ECE workflow (Naeini AAAI'15, Błasiok-Nakkiran ICLR'24) |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Bit-for-bit reproducibility guide (this file is created in Phase 5 task 8) |
| [`docs/phase_4_summary.md`](docs/phase_4_summary.md) | Phase 4 algorithmic novelty (Trusted-MVC, evidential fusion) |
| [`docs/legacy_stage_reference.md`](docs/legacy_stage_reference.md) | Stage 1–10 inputs/outputs/CLI reference (predates Phase 1–5) |

## Running the tests

The test suite has two tiers, declared in `pytest.ini`:

- **`lightweight`** (default for laptop / CI without GPU):

  ```bash
  pytest -m lightweight \
      --ignore=tests/test_service_api.py \
      --ignore=tests/test_service_websocket.py
  ```

  580+ tests, ~5 s, no Open3D / OpenCV / IfcOpenShell required.

- **`geometry`**, **`server`**, **`vlm`**, **`colmap`**, **`slow`** — run
  inside the `docker/Dockerfile.core-dev` image with the heavy stack
  installed. See [`docs/legacy_stage_reference.md`](docs/legacy_stage_reference.md)
  §"PowerShell commands" / §"WSL commands".

The GUI suite uses [Vitest](https://vitest.dev/) + [MSW](https://mswjs.io/):

```bash
cd gui
npm install
npm run lint   # tsc -b --noEmit
npm test       # 60+ tests
npm run build  # vite production bundle
```

## Citation

If you use this repository in academic work, please cite the literature
map at [`docs/literature_q1_2024_2026.md`](docs/literature_q1_2024_2026.md)
together with the upstream work it positions this project against.

## License

See [`LICENSE`](LICENSE) at the repository root (if present); otherwise
contact the project authors for terms.

---

The pre-Phase-1 stage-by-stage reference (Stage 1 + 2 ingest commands,
Docker Compose snippets, server handoff ZIP recipes, etc.) is preserved
verbatim in [`docs/legacy_stage_reference.md`](docs/legacy_stage_reference.md)
for continuity.
