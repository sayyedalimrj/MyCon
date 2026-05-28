# Stage 7.5 VLM QA

Stage 7.5 runs after Stage 7 cleanup and before Stage 8 BIM registration.

Its purpose is to create deterministic visual/geometric evidence for the cleaned reconstruction output.

## Inputs

- `data/clean/site01/cleaned_cloud.ply`
- `data/clean/site01/mesh.ply`
- `data/clean/site01/planes.json`
- `runs/<run_id>/reports/cleanup_summary.json`

## Outputs

- `data/vlm_qa/site01/renders/clean_cloud_view.png`
- `data/vlm_qa/site01/renders/mesh_view.png`
- `data/vlm_qa/site01/renders/plane_overlay_view.png`
- `data/vlm_qa/site01/renders/qa_overview.png`
- `data/vlm_qa/site01/vlm_qa_evidence.json`
- `runs/<run_id>/reports/vlm_qa_summary.json`

## Interpretation

This stage does not prove BIM progress. It only checks whether the cleaned reconstruction output is visually/geometrically suitable to continue toward BIM registration.

The initial implementation uses deterministic heuristic QA and mock observations. A local Qwen/Ollama VLM provider can be connected later.
