// Typed contract for the Phase 4 schedule-comparison endpoints.
//
// Mirrors the JSON shapes produced by:
//   - pipeline.stage_11_schedule_variance.run_schedule_variance (CLI)
//   - pipeline.service.schedule_api (read endpoints)
//
// Schema versions are pinned as string literal types so a backend bump
// surfaces as a TypeScript compile error rather than a silent runtime
// surprise.

export type ScheduleActivitiesResponseSchema = "schedule_activities_response.v1";
export type ActivityDetailResponseSchema = "activity_detail_response.v1";
export type ScheduleVarianceSchema = "schedule_variance.v1";
export type DashboardSummarySchema = "dashboard_summary.v1";
export type ElementStatusResponseSchema = "element_status_response.v1";

export type ActivityVarianceStatus =
  | "on_schedule"
  | "ahead"
  | "behind"
  | "unknown_evidence";

export type ConfidenceLabel = "high" | "medium" | "low";

export interface PlannedActivity {
  activity_id: string;
  activity_name: string;
  wbs_code: string;
  trade: string;
  location: string;
  predecessors: string[];
  planned_start_iso: string;
  planned_finish_iso: string;
  planned_percent_complete: number;
}

export interface ScheduleProvenanceJson {
  source_path: string;
  source_sha256: string;
  source_bytes: number;
  schema_version: string;
  n_rows_total: number;
  n_rows_kept: number;
  n_rows_skipped: number;
  skip_reasons: Array<[string, number]>;
}

export interface ScheduleActivitiesResponse {
  schema_version: ScheduleActivitiesResponseSchema;
  data_date_utc: string;
  n_activities: number;
  activities: PlannedActivity[];
  schedule_provenance: ScheduleProvenanceJson;
}

export interface ActivityVarianceRow {
  activity_id: string;
  activity_name: string;
  planned_percent_complete: number;
  actual_percent_complete: number;
  actual_percent_complete_lower_95: number;
  actual_percent_complete_upper_95: number;
  schedule_variance_percent: number;
  status: ActivityVarianceStatus;
  confidence: ConfidenceLabel;
  n_evaluated_elements: number;
  n_mapped_elements: number;
  risks: string[];
}

export interface ActivityDetailResponse extends PlannedActivity {
  schema_version: ActivityDetailResponseSchema;
  data_date_utc: string;
  actual: ActivityVarianceRow | null;
  mapped_elements: Array<{ ifc_global_id: string; weight: number }>;
}

export interface ScheduleVarianceReport {
  schema_version: ScheduleVarianceSchema;
  data_date_utc: string;
  on_schedule_band_pct: number;
  n_activities: number;
  n_on_schedule: number;
  n_ahead: number;
  n_behind: number;
  n_unknown_evidence: number;
  overall_planned_percent_complete: number;
  overall_actual_percent_complete: number;
  overall_actual_lower_95: number;
  overall_actual_upper_95: number;
  overall_schedule_variance_percent: number;
  activities: ActivityVarianceRow[];
  provenance?: Record<string, unknown>;
}

export interface DashboardSummaryKpi {
  planned_percent: number;
  actual_percent: number;
  actual_lower_95: number;
  actual_upper_95: number;
  variance_percent: number;
  n_activities: number;
  n_on_schedule: number;
  n_behind: number;
  n_ahead: number;
  n_unknown_evidence: number;
}

export interface DashboardSummary {
  schema_version: DashboardSummarySchema;
  data_date_utc: string;
  kpi: DashboardSummaryKpi;
  activities: ActivityVarianceRow[];
  provenance?: Record<string, unknown>;
}

export interface ElementStatusResponse {
  schema_version: ElementStatusResponseSchema;
  ifc_global_id: string;
  element_metrics_row: Record<string, string>;
  mapped_to_activities: Array<{ activity_id: string; weight: number }>;
}
