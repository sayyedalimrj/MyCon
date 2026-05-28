// Typed mirrors of the Phase 2 API responses.
//
// These types are intentionally permissive at the boundary: the backend
// is the source of truth, and we'd rather render an "unknown field" than
// crash if a future stage descriptor adds a key. Anything in here should
// match what `make_default_app()` actually returns; the integration
// tests in src/test pin the contract.

export interface Health {
  status: "ok";
  subscriber_count: number;
  tracked_run_ids: string[];
  history_run_count: number;
  stage_count: number;
}

export type StageCapability = "heavy" | "optional" | "server_required" | "stub_or_partial" | string;

export interface StageDescriptor {
  name: string;
  order: number;
  title: string;
  description: string;
  cli_module: string;
  callable_name: string;
  dependencies: string[];
  inputs: string[];
  outputs: string[];
  required_config_keys: string[];
  report_basename: string | null;
  capabilities: StageCapability[];
}

export interface PluginInfo {
  name: string;
  description: string;
  capabilities: string[];
}

export interface ConfigListEntry {
  name: string;
  path: string;
  size_bytes: number;
  modified_at_unix: number;
}

export interface ConfigDocument {
  name: string;
  path: string;
  config_hash: string;
  data: Record<string, unknown>;
}

export interface StageSchemaResponse {
  config_name: string;
  stage: string;
  required_config_keys: string[];
  schema_class: string;
  schema: Record<string, unknown>;
}

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | string;
export type StageStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "skipped" | string;

export interface StageRuntimeView {
  name: string;
  status: StageStatus;
  started_at_unix: number | null;
  finished_at_unix: number | null;
  return_code: number | null;
}

export interface RunSnapshot {
  run_id: string;
  submission: {
    config_path: string;
    requested_stages: string[];
    force: boolean;
  };
  status: RunStatus;
  stages: StageRuntimeView[];
  config_hash: string;
  project_name: string;
  submitted_at_unix: number | null;
  started_at_unix: number | null;
  finished_at_unix: number | null;
  cancel_requested: boolean;
}

export interface RunListEntry {
  run_id: string;
  project_name: string;
  config_path: string;
  config_hash: string;
  status: RunStatus;
  requested_stages: string[];
  stage_statuses: Record<string, StageStatus>;
  submitted_at_unix: number | null;
  started_at_unix: number | null;
  finished_at_unix: number | null;
}

export type RunEventKind =
  | "run.queued"
  | "run.started"
  | "run.finished"
  | "run.failed"
  | "run.cancelled"
  | "stage.queued"
  | "stage.started"
  | "stage.progress"
  | "stage.finished"
  | "stage.failed"
  | "stage.cancelled"
  | "broker.backpressure_drop";

export interface RunEvent {
  event_id: string;
  run_id: string;
  stage: string | null;
  kind: RunEventKind;
  timestamp_unix: number;
  payload: Record<string, unknown>;
}

export interface ArtifactProvenance {
  schema_version?: string;
  config_hash?: string;
  git_sha?: string | null;
  git_dirty?: boolean | null;
  generated_at_unix?: number;
  stage?: string;
  artifact_type?: string;
  random_seed?: number | null;
  inputs?: Record<string, string>;
  environment?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface ArtifactSummary {
  stage: string;
  artifact_path: string;
  artifact_basename: string;
  exists: boolean;
  size_bytes: number;
  modified_at_unix: number | null;
  status: string | null;
  provenance: ArtifactProvenance | null;
  preview: Record<string, unknown>;
  parse_error: string | null;
}

export interface RunSubmissionRequest {
  config_path: string;
  requested_stages: string[];
  force?: boolean;
  label?: string | null;
}

export interface CancelResponse {
  cancel_requested: boolean;
  reason?: string;
  status?: string;
}

export interface ApiErrorBody {
  detail: string;
}

export class ApiError extends Error {
  public readonly status: number;
  public readonly body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}
