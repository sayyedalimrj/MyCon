// Typed HITL corrections API client.
//
// Mirrors the Python contract in pipeline/service/hitl_api.py:
//   POST /api/v1/hitl/corrections        -> hitl_submit_response.v1
//   GET  /api/v1/hitl/corrections        -> hitl_list_response.v1
//
// Sits next to scheduleEndpoints.ts and uses the same ApiClient fetch
// wrapper so the GUI keeps a single network base path.

import { ApiClient, defaultApiClient } from "./client";

export type HitlTargetKind =
  | "element_acceptance"
  | "activity_completion"
  | "vlm_answer"
  | "anchor_validation"
  | "registration_quality";

export type HitlDecisionValue = "accept" | "reject" | "uncertain" | "rework";

export type HitlPredictedConfidence =
  | "high"
  | "medium"
  | "low_to_medium"
  | "low"
  | "unverified";

export interface HitlCorrectionPayload {
  target_kind: HitlTargetKind;
  target_id: string;
  predicted_value: HitlDecisionValue;
  predicted_confidence: HitlPredictedConfidence;
  corrected_value: HitlDecisionValue;
  reviewer_id: string;
  rationale: string;
  evidence_refs?: string[];
  run_id?: string;
}

export interface HitlSubmitResponse {
  schema_version: "hitl_submit_response.v1";
  stored_path: string;
  correction: {
    schema_version: "hitl_correction.v1";
    target_kind: HitlTargetKind;
    target_id: string;
    predicted_value: HitlDecisionValue;
    predicted_confidence: HitlPredictedConfidence;
    corrected_value: HitlDecisionValue;
    reviewer_id: string;
    timestamp_utc: string;
    rationale: string;
    evidence_refs: string[];
    run_id: string;
    record_id: string;
  };
}

export interface HitlListResponse {
  schema_version: "hitl_list_response.v1";
  schema_version_record: "hitl_correction.v1";
  stored_path: string;
  n_total_records: number;
  n_effective: number;
  n_conflicts: number;
  effective: HitlSubmitResponse["correction"][];
  conflicts: Array<Record<string, unknown>>;
}

export interface HitlEndpoints {
  submitCorrection(
    payload: HitlCorrectionPayload,
    opts?: { runId?: string },
  ): Promise<HitlSubmitResponse>;
  listCorrections(opts?: {
    runId?: string;
    targetKinds?: HitlTargetKind[];
  }): Promise<HitlListResponse>;
}

function buildQuery(parts: Record<string, string | undefined>): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(parts)) {
    if (v !== undefined && v !== "") usp.set(k, v);
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

export function createHitlEndpoints(
  client: ApiClient = defaultApiClient,
): HitlEndpoints {
  return {
    submitCorrection: (payload, { runId } = {}) =>
      client.post<HitlSubmitResponse>(
        `/v1/hitl/corrections${buildQuery({ run_id: runId })}`,
        payload,
      ),
    listCorrections: ({ runId, targetKinds } = {}) =>
      client.get<HitlListResponse>(
        `/v1/hitl/corrections${buildQuery({
          run_id: runId,
          target_kinds: targetKinds && targetKinds.length > 0 ? targetKinds.join(",") : undefined,
        })}`,
      ),
  };
}

export const hitlEndpoints = createHitlEndpoints();
