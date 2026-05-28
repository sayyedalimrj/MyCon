// Centralized TanStack Query keys. Importing from here means a panel
// invalidating "runs" knows exactly which queries it touches.

export const queryKeys = {
  health: () => ["health"] as const,
  stages: () => ["registry", "stages"] as const,
  stage: (name: string) => ["registry", "stage", name] as const,
  vlmBackends: () => ["registry", "vlm"] as const,
  depthProviders: () => ["registry", "depth"] as const,
  configs: () => ["configs"] as const,
  config: (name: string) => ["configs", name] as const,
  stageSchema: (configName: string, stageName: string) =>
    ["configs", configName, "schema", stageName] as const,
  runs: (limit: number) => ["runs", { limit }] as const,
  run: (runId: string) => ["runs", runId] as const,
  runEvents: (runId: string) => ["runs", runId, "events"] as const,
  runArtifacts: (runId: string) => ["runs", runId, "artifacts"] as const,
  scheduleActivities: (runId?: string, dataDateIso?: string) =>
    ["schedule", "activities", { runId, dataDateIso }] as const,
  scheduleActivityDetail: (
    activityId: string,
    runId?: string,
    dataDateIso?: string,
  ) => ["schedule", "activity", activityId, { runId, dataDateIso }] as const,
  scheduleVariance: (runId?: string) => ["schedule", "variance", { runId }] as const,
  scheduleDashboard: (runId?: string) => ["schedule", "dashboard", { runId }] as const,
  elementStatus: (ifcGlobalId: string, runId?: string) =>
    ["elements", ifcGlobalId, { runId }] as const,
};
