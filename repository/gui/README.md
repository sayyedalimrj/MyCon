# MyCon GUI (Phase 3)

A research-grade single-page application that drives the MyCon backend
service. It is a strict client of the Phase 2 FastAPI surface; the GUI
contains **no pipeline logic of its own**.

## Stack

- **React 18** + **TypeScript 5** + **Vite 5**
- **TanStack Query 5** for server state (configs, registry, runs, artifacts)
- **React Router 6** for the panel layout
- **Tailwind CSS 3** for styling, with a class-based dark-mode strategy
- **Recharts** for metrics visualizations
- **Vitest + React Testing Library + MSW** for tests

## Layout

```
gui/
├── index.html
├── src/
│   ├── main.tsx                 # bootstrap (router + query client + theme)
│   ├── api/                     # typed API client; one file per resource
│   ├── app/                     # shell: layout, sidebar, theme, error boundary
│   ├── panels/                  # one folder per panel (Pipeline, Runs, …)
│   ├── components/              # shared primitives (Card, Badge, Spinner, …)
│   ├── hooks/                   # cross-cutting hooks (theme, websocket)
│   ├── lib/                     # pure helpers (diff, format, schemaToControls)
│   └── test/                    # MSW handlers, render helpers, setup
└── tailwind.config.ts
```

## Running

```bash
# from gui/
npm install
npm run dev          # serves on http://127.0.0.1:5173, proxies /api → :8000
npm run build        # outputs to gui/dist
npm test             # vitest run (jsdom + MSW)
```

By default the dev server proxies `/api` to `http://127.0.0.1:8000`. Set
`MYCON_API_URL` in the environment to point elsewhere.

## Backend contract

Every panel calls the Phase 2 API and *only* the Phase 2 API:

| Panel                    | Endpoints used                                                   |
| ------------------------ | ---------------------------------------------------------------- |
| Pipeline overview        | `GET /api/registry/stages`, `GET /api/health`                    |
| Stage editor             | `GET /api/configs`, `GET /api/configs/{name}/schemas/{stage}`    |
| Input manager            | `GET /api/configs/{name}` (read-only listing of declared inputs) |
| Run control              | `POST /api/runs`, `GET /api/runs`, `WS /api/runs/{id}/events/stream`, `POST /api/runs/{id}/cancel` |
| Artifact browser         | `GET /api/runs/{id}/artifacts`, `GET /api/runs/{id}`             |
| Metrics dashboard        | `GET /api/runs?limit=...`, `GET /api/runs/{id}/artifacts`        |
| VLM panel                | `GET /api/registry/vlm-backends`                                  |
| BIM/3D viewer            | `GET /api/runs/{id}/artifacts` (path surfaces only)              |
| Config diff              | `GET /api/configs/{a}`, `GET /api/configs/{b}`                   |
| Report generator         | `GET /api/runs/{id}`, `GET /api/runs/{id}/artifacts`             |

Endpoints not yet on the backend (config write, file upload, VLM
execution, point-cloud streaming, PDF export) are exposed in the UI as
**explicitly disabled controls** with a one-line note that points at the
phase they will land in. The GUI does **not** fabricate data for them.

## Test strategy

- **Unit:** pure helpers in `src/lib/` (diff, format, schemaToControls).
- **Component:** rendered through a `renderWithProviders` helper that
  mounts the QueryClient and the dark-mode provider.
- **Integration:** MSW intercepts the full `/api/...` surface and serves
  fixtures shaped exactly like the live backend's responses (validated
  via the live `make_default_app()` probe in PR #5).

Run all tests:

```bash
npm test
```
