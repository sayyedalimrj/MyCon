// Fixtures shaped exactly like the live backend's responses.
//
// These were sampled from the running `make_default_app()` against the
// real `configs/site01.yaml` and the canonical 15-stage STAGE_REGISTRY.
// They are deliberately small so tests stay fast.

import type {
  ArtifactSummary,
  ConfigDocument,
  ConfigListEntry,
  Health,
  PluginInfo,
  RunListEntry,
  RunSnapshot,
  StageDescriptor,
  StageSchemaResponse,
} from "../api/types";

export const fixtureHealth: Health = {
  status: "ok",
  subscriber_count: 0,
  tracked_run_ids: [],
  history_run_count: 2,
  stage_count: 3,
};

export const fixtureStages: StageDescriptor[] = [
  {
    name: "stage_01_ingest",
    order: 10,
    title: "Video ingest and normalization",
    description:
      "Decode the raw video, normalize FPS, extract per-frame quality metrics.",
    cli_module: "pipeline.stage_01_ingest.run_ingest",
    callable_name: "run_ingest",
    dependencies: [],
    inputs: ["inputs.video"],
    outputs: ["paths.normalized_video", "paths.metadata_json", "paths.quality_csv"],
    required_config_keys: ["project.name", "project.run_id", "inputs.video"],
    report_basename: "stage_01_ingest_report.json",
    capabilities: ["heavy"],
  },
  {
    name: "stage_02_keyframes",
    order: 20,
    title: "Adaptive keyframe selection",
    description: "Segment the normalized video and select keyframes.",
    cli_module: "pipeline.stage_02_keyframes.select_keyframes",
    callable_name: "run_keyframe_selection",
    dependencies: ["stage_01_ingest"],
    inputs: ["paths.normalized_video", "paths.quality_csv"],
    outputs: ["paths.keyframes_dir", "paths.manifest_csv", "paths.contact_sheet"],
    required_config_keys: [
      "project.name",
      "keyframes.min_time_gap_sec",
      "keyframes.max_frames_first_run",
    ],
    report_basename: "keyframe_summary.json",
    capabilities: [],
  },
  {
    name: "stage_03_colmap",
    order: 30,
    title: "COLMAP sparse reconstruction",
    description: "Run COLMAP feature extraction, matching, and sparse SfM.",
    cli_module: "pipeline.stage_03_colmap.run_sparse",
    callable_name: "run_sparse",
    dependencies: ["stage_02_keyframes"],
    inputs: ["paths.keyframes_dir", "paths.manifest_csv"],
    outputs: ["paths.colmap_db", "paths.sparse_dir"],
    required_config_keys: ["project.name", "paths.colmap_db"],
    report_basename: "stage_03_colmap_report.json",
    capabilities: ["heavy", "server_required"],
  },
];

export const fixtureVlmBackends: PluginInfo[] = [
  {
    name: "mock",
    description: "In-process echo backend for tests.",
    capabilities: ["deterministic", "offline"],
  },
  {
    name: "ollama_local",
    description: "Local Ollama VLM server.",
    capabilities: ["local_only", "requires_endpoint"],
  },
];

export const fixtureDepthProviders: PluginInfo[] = [
  {
    name: "precomputed",
    description: "Reads depth maps already on disk.",
    capabilities: ["offline"],
  },
];

export const fixtureConfigList: ConfigListEntry[] = [
  {
    name: "site01",
    path: "/workspace/configs/site01.yaml",
    size_bytes: 4096,
    modified_at_unix: 1_700_000_000,
  },
  {
    name: "default_server_svc4",
    path: "/workspace/configs/default_server_svc4.yaml",
    size_bytes: 3120,
    modified_at_unix: 1_700_000_500,
  },
];

export const fixtureSite01Config: ConfigDocument = {
  name: "site01",
  path: "/workspace/configs/site01.yaml",
  config_hash: "18b982e41b6ffeed0011223344556677",
  data: {
    project: { name: "site01", run_id: "site01-baseline", root: "/workspace", random_seed: 42 },
    inputs: { video: "data/raw/site01.mp4", ifc: "data/bim/site01.ifc", schedule: "" },
    paths: {
      normalized_video: "data/normalized/site01.mp4",
      metadata_json: "data/normalized/site01.json",
      quality_csv: "data/normalized/site01.csv",
    },
    keyframes: { min_time_gap_sec: 0.5, max_frames_first_run: 800 },
  },
};

export const fixtureDefaultServerConfig: ConfigDocument = {
  name: "default_server_svc4",
  path: "/workspace/configs/default_server_svc4.yaml",
  config_hash: "aabbccddeeff00112233445566778899",
  data: {
    project: { name: "site01", run_id: "site01-baseline", root: "/workspace", random_seed: 99 },
    inputs: { video: "data/raw/site01.mp4", ifc: "data/bim/site01.ifc", schedule: "schedule.csv" },
    paths: {
      normalized_video: "data/normalized/site01.mp4",
      metadata_json: "data/normalized/site01.json",
      quality_csv: "data/normalized/site01.csv",
    },
    keyframes: { min_time_gap_sec: 1.0, max_frames_first_run: 600 },
  },
};

export const fixtureStageSchema: StageSchemaResponse = {
  config_name: "site01",
  stage: "stage_02_keyframes",
  required_config_keys: ["keyframes.min_time_gap_sec", "keyframes.max_frames_first_run"],
  schema_class: "Stage02KeyframesSchema",
  schema: {
    project: { name: "site01", run_id: "site01-baseline", root: "/workspace", random_seed: 42 },
    keyframes: {
      min_time_gap_sec: 0.5,
      max_frames_first_run: 800,
      selection_quality_weight: 0.4,
      selection_novelty_weight: 0.4,
      selection_feature_weight: 0.2,
    },
  },
};

export const fixtureRunList: RunListEntry[] = [
  {
    run_id: "run-001",
    project_name: "site01",
    config_path: "/workspace/configs/site01.yaml",
    config_hash: "18b982e41b6ffeed0011223344556677",
    status: "completed",
    requested_stages: ["stage_01_ingest", "stage_02_keyframes"],
    stage_statuses: { stage_01_ingest: "completed", stage_02_keyframes: "completed" },
    submitted_at_unix: 1_700_001_000,
    started_at_unix: 1_700_001_010,
    finished_at_unix: 1_700_001_300,
  },
  {
    run_id: "run-002",
    project_name: "site01",
    config_path: "/workspace/configs/site01.yaml",
    config_hash: "18b982e41b6ffeed0011223344556677",
    status: "running",
    requested_stages: ["stage_01_ingest"],
    stage_statuses: { stage_01_ingest: "running" },
    submitted_at_unix: 1_700_002_000,
    started_at_unix: 1_700_002_002,
    finished_at_unix: null,
  },
];

export const fixtureRunSnapshot: RunSnapshot = {
  run_id: "run-001",
  submission: {
    config_path: "/workspace/configs/site01.yaml",
    requested_stages: ["stage_01_ingest", "stage_02_keyframes"],
    force: false,
  },
  status: "completed",
  stages: [
    {
      name: "stage_01_ingest",
      status: "completed",
      started_at_unix: 1_700_001_010,
      finished_at_unix: 1_700_001_120,
      return_code: 0,
    },
    {
      name: "stage_02_keyframes",
      status: "completed",
      started_at_unix: 1_700_001_120,
      finished_at_unix: 1_700_001_300,
      return_code: 0,
    },
  ],
  config_hash: "18b982e41b6ffeed0011223344556677",
  project_name: "site01",
  submitted_at_unix: 1_700_001_000,
  started_at_unix: 1_700_001_010,
  finished_at_unix: 1_700_001_300,
  cancel_requested: false,
};

export const fixtureArtifacts: ArtifactSummary[] = [
  {
    stage: "stage_01_ingest",
    artifact_path: "/workspace/runs/run-001/reports/stage_01_ingest_report.json",
    artifact_basename: "stage_01_ingest_report.json",
    exists: true,
    size_bytes: 2048,
    modified_at_unix: 1_700_001_120,
    status: "completed",
    provenance: {
      schema_version: "1.0",
      config_hash: "18b982e41b6ffeed0011223344556677",
      git_sha: "84d01e0aabbccddeeff",
      git_dirty: false,
      generated_at_unix: 1_700_001_120,
      stage: "stage_01_ingest",
    },
    preview: { frames_total: 1234, fps: 30 },
    parse_error: null,
  },
  {
    stage: "stage_02_keyframes",
    artifact_path: "/workspace/runs/run-001/reports/keyframe_summary.json",
    artifact_basename: "keyframe_summary.json",
    exists: true,
    size_bytes: 1024,
    modified_at_unix: 1_700_001_300,
    status: "completed",
    provenance: {
      schema_version: "1.0",
      config_hash: "18b982e41b6ffeed0011223344556677",
      git_sha: "84d01e0aabbccddeeff",
      git_dirty: false,
      generated_at_unix: 1_700_001_300,
      stage: "stage_02_keyframes",
    },
    preview: { keyframes_selected: 412 },
    parse_error: null,
  },
  {
    stage: "stage_03_colmap",
    artifact_path: "/workspace/runs/run-001/reports/stage_03_colmap_report.json",
    artifact_basename: "stage_03_colmap_report.json",
    exists: false,
    size_bytes: 0,
    modified_at_unix: null,
    status: null,
    provenance: null,
    preview: {},
    parse_error: null,
  },
];



// ---------------------------------------------------------------------------
// Phase 4 schedule-comparison fixtures
// ---------------------------------------------------------------------------

export const fixtureScheduleActivitiesResponse = {
  schema_version: "schedule_activities_response.v1",
  data_date_utc: "2026-04-16T00:00:00+00:00",
  n_activities: 2,
  activities: [
    {
      activity_id: "A0001",
      activity_name: "Foundations",
      wbs_code: "1.1",
      trade: "structural",
      location: "Site",
      predecessors: [],
      planned_start_iso: "2026-03-01T00:00:00+00:00",
      planned_finish_iso: "2026-04-01T00:00:00+00:00",
      planned_percent_complete: 100,
    },
    {
      activity_id: "A0432",
      activity_name: "Floor 2 Zone B walls",
      wbs_code: "1.2.3",
      trade: "structural",
      location: "Floor 2 Zone B",
      predecessors: ["A0001"],
      planned_start_iso: "2026-04-01T00:00:00+00:00",
      planned_finish_iso: "2026-05-01T00:00:00+00:00",
      planned_percent_complete: 50,
    },
  ],
  schedule_provenance: {
    source_path: "/runs/r1/inputs/schedule.csv",
    source_sha256: "deadbeef".repeat(8),
    source_bytes: 1024,
    schema_version: "schedule.v1",
    n_rows_total: 2,
    n_rows_kept: 2,
    n_rows_skipped: 0,
    skip_reasons: [] as Array<[string, number]>,
  },
};

const fixtureActivityRows = [
  {
    activity_id: "A0001",
    activity_name: "Foundations",
    planned_percent_complete: 100,
    actual_percent_complete: 100,
    actual_percent_complete_lower_95: 80,
    actual_percent_complete_upper_95: 100,
    schedule_variance_percent: 0,
    status: "on_schedule" as const,
    confidence: "high" as const,
    n_evaluated_elements: 1,
    n_mapped_elements: 1,
    risks: [] as string[],
  },
  {
    activity_id: "A0432",
    activity_name: "Floor 2 Zone B walls",
    planned_percent_complete: 50,
    actual_percent_complete: 25,
    actual_percent_complete_lower_95: 10,
    actual_percent_complete_upper_95: 50,
    schedule_variance_percent: -25,
    status: "behind" as const,
    confidence: "medium" as const,
    n_evaluated_elements: 4,
    n_mapped_elements: 8,
    risks: ["schedule_behind", "low_confidence_actual_percent"],
  },
];

export const fixtureDashboardSummary = {
  schema_version: "dashboard_summary.v1",
  data_date_utc: "2026-04-16T00:00:00+00:00",
  kpi: {
    planned_percent: 75,
    actual_percent: 62.5,
    actual_lower_95: 45,
    actual_upper_95: 75,
    variance_percent: -12.5,
    n_activities: 2,
    n_on_schedule: 1,
    n_behind: 1,
    n_ahead: 0,
    n_unknown_evidence: 0,
  },
  activities: fixtureActivityRows,
};

export const fixtureScheduleVariance = {
  schema_version: "schedule_variance.v1",
  data_date_utc: "2026-04-16T00:00:00+00:00",
  on_schedule_band_pct: 5,
  n_activities: 2,
  n_on_schedule: 1,
  n_ahead: 0,
  n_behind: 1,
  n_unknown_evidence: 0,
  overall_planned_percent_complete: 75,
  overall_actual_percent_complete: 62.5,
  overall_actual_lower_95: 45,
  overall_actual_upper_95: 75,
  overall_schedule_variance_percent: -12.5,
  activities: fixtureActivityRows,
};
