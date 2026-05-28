# Stage 9 Scientific Metric Interpretation

Stage 9 deterministic metrics are necessary but not sufficient for research-grade progress claims.

The conservative interpretation layer adds:

- `observed_surface_ratio`
- `visibility_confidence`
- `completion_state`
- `evidence_state`
- `metric_truth_source`
- `interpretation_notes`

## Key rule

`not_evidenced` does not mean `not_built`.

If an element is not sufficiently observed, the pipeline must not claim it is incomplete. It should say the evidence is insufficient and recommend additional capture.

## Authority rule

Stage 8 registration confidence gates Stage 9 interpretation.

If registration confidence is low, even high apparent coverage must become:

```text
completion_state = uncertain_low_registration
```

## Command

```powershell
docker compose -f docker\docker-compose.yml run --rm core `
  python3 scripts/upgrade_stage09_progress_schema.py
```

This command is laptop-safe and post-processes CSV/JSON outputs only.
