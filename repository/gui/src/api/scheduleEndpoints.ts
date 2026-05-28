// Typed schedule-comparison API client.
//
// Sits next to the existing endpoints.ts and uses the same ApiClient
// fetch wrapper so the GUI keeps a single network base path.

import { ApiClient, defaultApiClient } from "./client";
import type {
  ActivityDetailResponse,
  DashboardSummary,
  ElementStatusResponse,
  ScheduleActivitiesResponse,
  ScheduleVarianceReport,
} from "./scheduleTypes";

export interface ScheduleEndpoints {
  listActivities(opts?: {
    runId?: string;
    dataDateIso?: string;
  }): Promise<ScheduleActivitiesResponse>;
  getActivityDetail(
    activityId: string,
    opts?: { runId?: string; dataDateIso?: string },
  ): Promise<ActivityDetailResponse>;
  getVariance(opts?: { runId?: string }): Promise<ScheduleVarianceReport>;
  getDashboardSummary(opts?: { runId?: string }): Promise<DashboardSummary>;
  getElementStatus(
    ifcGlobalId: string,
    opts?: { runId?: string },
  ): Promise<ElementStatusResponse>;
}

function buildQuery(parts: Record<string, string | undefined>): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(parts)) {
    if (v !== undefined && v !== "") usp.set(k, v);
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

export function createScheduleEndpoints(
  client: ApiClient = defaultApiClient,
): ScheduleEndpoints {
  return {
    listActivities: ({ runId, dataDateIso } = {}) =>
      client.get<ScheduleActivitiesResponse>(
        `/v1/schedule/activities${buildQuery({
          run_id: runId,
          data_date_iso: dataDateIso,
        })}`,
      ),
    getActivityDetail: (activityId, { runId, dataDateIso } = {}) =>
      client.get<ActivityDetailResponse>(
        `/v1/schedule/activities/${encodeURIComponent(activityId)}${buildQuery({
          run_id: runId,
          data_date_iso: dataDateIso,
        })}`,
      ),
    getVariance: ({ runId } = {}) =>
      client.get<ScheduleVarianceReport>(
        `/v1/schedule/variance${buildQuery({ run_id: runId })}`,
      ),
    getDashboardSummary: ({ runId } = {}) =>
      client.get<DashboardSummary>(
        `/v1/schedule/dashboard${buildQuery({ run_id: runId })}`,
      ),
    getElementStatus: (ifcGlobalId, { runId } = {}) =>
      client.get<ElementStatusResponse>(
        `/v1/elements/${encodeURIComponent(ifcGlobalId)}${buildQuery({
          run_id: runId,
        })}`,
      ),
  };
}

export const scheduleEndpoints = createScheduleEndpoints();
