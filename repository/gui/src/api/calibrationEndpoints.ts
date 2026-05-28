// Typed calibration API client.
//
// Mirrors the Python contract in pipeline/service/calibration_api.py:
//   POST /api/v1/calibration/run     -> calibration_run_response.v1
//   GET  /api/v1/calibration/report  -> calibration_report.v1
//
// Sits next to scheduleEndpoints.ts and hitlEndpoints.ts and uses the
// same ApiClient fetch wrapper so the GUI keeps a single network base.

import { ApiClient, defaultApiClient } from "./client";
import type { CalibrationReportPayload } from "../components/ReliabilityCard";

export interface CalibrationRunRequest {
  n_bins?: number;
  strategy?: "equal_mass" | "equal_width";
  target_kinds?: string[] | null;
}

export interface CalibrationRunResponse {
  schema_version: "calibration_run_response.v1";
  stored_path: string;
  n_replayed_records: number;
  n_effective_records: number;
  n_conflicts: number;
  report: CalibrationReportPayload;
}

export interface CalibrationEndpoints {
  runReplay(
    body?: CalibrationRunRequest,
    opts?: { runId?: string },
  ): Promise<CalibrationRunResponse>;
  getLatest(opts?: { runId?: string }): Promise<CalibrationReportPayload>;
}

function buildQuery(parts: Record<string, string | undefined>): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(parts)) {
    if (v !== undefined && v !== "") usp.set(k, v);
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

export function createCalibrationEndpoints(
  client: ApiClient = defaultApiClient,
): CalibrationEndpoints {
  return {
    runReplay: (body, { runId } = {}) =>
      client.post<CalibrationRunResponse>(
        `/v1/calibration/run${buildQuery({ run_id: runId })}`,
        body ?? {},
      ),
    getLatest: ({ runId } = {}) =>
      client.get<CalibrationReportPayload>(
        `/v1/calibration/report${buildQuery({ run_id: runId })}`,
      ),
  };
}

export const calibrationEndpoints = createCalibrationEndpoints();
