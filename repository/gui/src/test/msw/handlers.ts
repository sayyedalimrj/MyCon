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
  http.post(`${API}/v1/calibration/run`, () =>
    HttpResponse.json({
      schema_version: "calibration_run_response.v1",
      stored_path: "/runs/test/reports/calibration_report.json",
      n_replayed_records: 6,
      n_effective_records: 6,
      n_conflicts: 0,
      report: {
        schema_version: "calibration_report.v1",
        n_samples: 6,
        binning_strategy: "equal_mass",
        n_bins: 5,
        label_probability_mapping: { high: 0.85, medium: 0.65, low: 0.30, unverified: 0.5 },
        metrics: {
          expected_calibration_error: 0.083,
          maximum_calibration_error: 0.180,
          brier_score: 0.130,
          smooth_ece: 0.075,
        },
        reliability_table: [
          {
            bin_index: 0,
            lower_edge: 0.0,
            upper_edge: 0.5,
            count: 1,
            mean_confidence: 0.30,
            empirical_accuracy: 0.0,
            gap: 0.30,
          },
          {
            bin_index: 1,
            lower_edge: 0.5,
            upper_edge: 1.0,
            count: 5,
            mean_confidence: 0.79,
            empirical_accuracy: 0.80,
            gap: 0.01,
          },
        ],
        notes: ["replayed via /v1/calibration/run"],
      },
    }),
  ),

  // Phase 5 HITL endpoints
  http.post(`${API}/v1/hitl/corrections`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>;
    if (!body || typeof body !== "object") {
      return HttpResponse.json(
        { error: { code: "invalid_input", message: "payload must be a JSON object" } },
        { status: 400 },
      );
    }
    if (!body["reviewer_id"]) {
      return HttpResponse.json(
        { error: { code: "invalid_input", message: "missing reviewer_id" } },
        { status: 400 },
      );
    }
    return HttpResponse.json({
      schema_version: "hitl_submit_response.v1",
      stored_path: "/runs/test/reports/hitl_corrections.jsonl",
      correction: {
        schema_version: "hitl_correction.v1",
        target_kind: body["target_kind"] ?? "element_acceptance",
        target_id: body["target_id"] ?? "",
        predicted_value: body["predicted_value"] ?? "accept",
        predicted_confidence: body["predicted_confidence"] ?? "high",
        corrected_value: body["corrected_value"] ?? "reject",
        reviewer_id: body["reviewer_id"],
        timestamp_utc: "2026-05-28T12:00:00Z",
        rationale: body["rationale"] ?? "",
        evidence_refs: body["evidence_refs"] ?? [],
        run_id: body["run_id"] ?? "",
        record_id: "abc1234567ab",
      },
    });
  }),
  http.get(`${API}/v1/hitl/corrections`, () =>
    HttpResponse.json({
      schema_version: "hitl_list_response.v1",
      schema_version_record: "hitl_correction.v1",
      stored_path: "/runs/test/reports/hitl_corrections.jsonl",
      n_total_records: 0,
      n_effective: 0,
      n_conflicts: 0,
      effective: [],
      conflicts: [],
    }),
  ),
];
