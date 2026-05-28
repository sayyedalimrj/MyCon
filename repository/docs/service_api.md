# Service API (Phase 2)

The optional service layer exposes the foundation modules added in Phase 1
(`pipeline.common.schema`, `registry`, `provenance`, `plugins`) as a
read-mostly REST + WebSocket API. The future GUI consumes this contract.

## Installing

The service is opt-in. Operators who only run stages from the CLI do not
need it. To enable:

```bash
pip install -r requirements-service.txt
# Plus the dev dep used by the test client:
pip install -r requirements-dev.txt
```

## Running

```bash
uvicorn pipeline.service.app:make_default_app --factory --host 127.0.0.1 --port 8765
```

The default app:

- mounts the API under `/api`
- creates `runs/_service/{run_history.json, events/<run_id>.jsonl}` for
  persistence
- accepts CORS from `localhost:3000` and `localhost:5173` (dev frontend
  ports)

Open <http://127.0.0.1:8765/docs> for the auto-generated OpenAPI schema.

## Endpoints

### Health and discovery

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/api/health` | Liveness; reports broker/history counts |
| GET | `/api/registry/stages` | All `StageDescriptor.to_dict()` |
| GET | `/api/registry/stages/{name}` | One descriptor |
| GET | `/api/registry/vlm-backends` | VLM plugin entries |
| GET | `/api/registry/depth-providers` | Depth plugin entries |

### Configs

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/api/configs` | List YAML files under `configs/` |
| GET | `/api/configs/{name}` | Load + validate; returns `data` and `config_hash` |
| GET | `/api/configs/{name}/schemas/{stage}` | Hydrate the typed schema view for one stage |

The `{name}` segment accepts either `site01` or `site01.yaml`. Path
traversal is blocked.

### Runs

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST | `/api/runs` | Submit a new run; body `{config_path, requested_stages, force, label}` |
| GET | `/api/runs?limit=N` | List runs (newest first) |
| GET | `/api/runs/{run_id}` | Snapshot of one run |
| POST | `/api/runs/{run_id}/cancel` | Issue cancellation |
| GET | `/api/runs/{run_id}/events` | Replay persisted events |
| WS | `/api/runs/{run_id}/events/stream` | Live event stream |

### Artifacts

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/api/runs/{run_id}/artifacts` | One summary per stage that declares a `report_basename` |

## Run lifecycle

A run progresses through these states (mirrored on `RunRecord.status`):

```
queued → running → completed
                 ↘ failed
                 ↘ cancelled
```

For each stage the executor publishes:

```
stage.queued   → stage.started
              → stage.progress (one per stdout/stderr line, while running)
              → stage.finished | stage.failed | stage.cancelled
```

Plus run-level events:

```
run.queued → run.started
          → run.finished | run.failed | run.cancelled
```

Every event has a stable JSON shape:

```json
{
  "event_id": "uuid",
  "run_id": "20260528_120000_abc",
  "stage": "stage_03_colmap",
  "kind": "stage.progress",
  "timestamp_unix": 1779932305.5,
  "payload": {"stream": "stdout", "line": "..."}
}
```

## Cancellation semantics

`POST /api/runs/{run_id}/cancel` sends SIGTERM to the running stage's
process group, waits up to `terminate_grace_seconds`, then escalates to
SIGKILL. The run is marked `cancelled` and any not-yet-started stages are
marked `cancelled` as well.

**Pause / resume is intentionally not implemented.** Stage execution wraps
native Open3D / COLMAP code; pausing a long-running native call without
crashing it is a real research problem and not in Phase 2 scope. Use
cancellation instead.

## Concurrency

The default executor runs at most one stage at a time (`max_workers=1`).
This is the operationally correct setting because COLMAP, the GPU, and
disk I/O are global resources on a research workstation. Operators can
override via `RunExecutor(..., max_workers=N)` when wiring a custom app.

## Persistence

Run records live in `runs/_service/run_history.json`. Per-run event logs
live in `runs/_service/events/<run_id>.jsonl`. Both files are
human-readable JSON for easy backup. The store survives an API restart;
finished runs continue to be queryable.

## Architecture

```
pipeline/service/
  __init__.py        # lazy entrypoint; importing this does NOT pull in FastAPI
  events.py          # RunEvent + EventBroker (no FastAPI dep)
  run_history.py     # JSON persistence (no FastAPI dep)
  executor.py        # subprocess executor + cancellation (no FastAPI dep)
  artifacts.py       # report discovery + provenance parsing (no FastAPI dep)
  api.py             # FastAPI router (only file that imports FastAPI)
  app.py             # create_app() factory + uvicorn entrypoint
```

Tests under `tests/test_service_*.py` exercise each layer independently.
The first four modules can be exercised without `fastapi` installed.

## Relationship to other phases

- **Phase 1** (PR #3) is the prerequisite. The service layer reads from
  `STAGE_REGISTRY`, `compute_config_hash`, `VLM_REGISTRY`, etc.
- **Phase 3** (a future PR) builds the Next.js GUI against this API. The
  shape of every JSON response and WebSocket frame is the contract the
  frontend consumes.
- **Phase 4** is the algorithmic-upgrades phase. The service does not
  block it; algorithmic work continues to land in the foundations layer.
