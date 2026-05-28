import { http, HttpResponse } from "msw";

import {
  fixtureArtifacts,
  fixtureConfigList,
  fixtureDashboardSummary,
  fixtureDefaultServerConfig,
  fixtureDepthProviders,
  fixtureHealth,
  fixtureRunList,
  fixtureRunSnapshot,
  fixtureScheduleActivitiesResponse,
  fixtureScheduleVariance,
  fixtureSite01Config,
  fixtureStageSchema,
  fixtureStages,
  fixtureVlmBackends,
} from "../fixtures";

const API = "*/api";

// Default MSW handlers shaped like the real Phase 2 surface. Tests can
// override individual routes via `server.use(...)` for negative cases.

export const handlers = [
  http.get(`${API}/health`, () => HttpResponse.json(fixtureHealth)),
  http.get(`${API}/registry/stages`, () => HttpResponse.json(fixtureStages)),
  http.get(`${API}/registry/stages/:name`, ({ params }) => {
    const stage = fixtureStages.find((s) => s.name === params.name);
    return stage
      ? HttpResponse.json(stage)
      : HttpResponse.json({ detail: "not found" }, { status: 404 });
  }),
  http.get(`${API}/registry/vlm-backends`, () => HttpResponse.json(fixtureVlmBackends)),
  http.get(`${API}/registry/depth-providers`, () => HttpResponse.json(fixtureDepthProviders)),

  http.get(`${API}/configs`, () => HttpResponse.json(fixtureConfigList)),
  http.get(`${API}/configs/site01`, () => HttpResponse.json(fixtureSite01Config)),
  http.get(`${API}/configs/default_server_svc4`, () =>
    HttpResponse.json(fixtureDefaultServerConfig),
  ),
  http.get(`${API}/configs/:name/schemas/:stage`, () =>
    HttpResponse.json(fixtureStageSchema),
  ),

  http.get(`${API}/runs`, () => HttpResponse.json(fixtureRunList)),
  http.post(`${API}/runs`, async ({ request }) => {
    const body = (await request.json()) as { config_path: string; requested_stages: string[] };
    if (!body.config_path) {
      return HttpResponse.json({ detail: "config_path is required" }, { status: 400 });
    }
    const newId = `run-${String(Date.now()).slice(-6)}`;
    return HttpResponse.json(
      {
        run_id: newId,
        snapshot: { ...fixtureRunSnapshot, run_id: newId, status: "queued" },
      },
      { status: 201 },
    );
  }),
  http.get(`${API}/runs/:id`, ({ params }) =>
    HttpResponse.json({ ...fixtureRunSnapshot, run_id: String(params.id) }),
  ),
  http.post(`${API}/runs/:id/cancel`, () =>
    HttpResponse.json({ cancel_requested: true }),
  ),
  http.get(`${API}/runs/:id/events`, () => HttpResponse.json([])),
  http.get(`${API}/runs/:id/artifacts`, () => HttpResponse.json(fixtureArtifacts)),

  // Phase 4 schedule-comparison endpoints.
  http.get(`${API}/v1/schedule/dashboard`, () => HttpResponse.json(fixtureDashboardSummary)),
  http.get(`${API}/v1/schedule/variance`, () => HttpResponse.json(fixtureScheduleVariance)),
  http.get(`${API}/v1/schedule/activities`, () =>
    HttpResponse.json(fixtureScheduleActivitiesResponse),
  ),
  http.get(`${API}/v1/schedule/activities/:activity_id`, ({ params }) => {
    const aid = String(params.activity_id);
    const row = fixtureDashboardSummary.activities.find((a) => a.activity_id === aid);
    if (!row) {
      return HttpResponse.json(
        { error: { code: "not_found", message: "activity not found", details: { activity_id: aid } } },
        { status: 404 },
      );
    }
    return HttpResponse.json({
      schema_version: "activity_detail_response.v1",
      activity_id: row.activity_id,
      activity_name: row.activity_name,
      wbs_code: "",
      trade: "",
      location: "",
      predecessors: [],
      planned_start_iso: "2026-04-01T00:00:00+00:00",
      planned_finish_iso: "2026-05-01T00:00:00+00:00",
      planned_percent_complete: row.planned_percent_complete,
      data_date_utc: fixtureDashboardSummary.data_date_utc,
      actual: row,
      mapped_elements: [
        { ifc_global_id: `${aid}-elem-1`, weight: 1.0 },
        { ifc_global_id: `${aid}-elem-2`, weight: 1.0 },
      ],
    });
  }),
  http.get(`${API}/v1/calibration/report`, () =>
    HttpResponse.json({
      error: { code: "not_found", message: "no calibration report yet", details: {} },
    }, { status: 404 }),
  ),
];
