// One named function per backend endpoint.
//
// Keeping this layer thin (no caching, no normalization) means TanStack
// Query stays in charge of state. Every panel uses query keys from
// `queryKeys.ts` so cache invalidation is centralized.

import { ApiClient, defaultApiClient } from "./client";
import type {
  ArtifactSummary,
  CancelResponse,
  ConfigDocument,
  ConfigListEntry,
  Health,
  PluginInfo,
  RunEvent,
  RunListEntry,
  RunSnapshot,
  RunSubmissionRequest,
  StageDescriptor,
  StageSchemaResponse,
} from "./types";

export interface MyConEndpoints {
  health(): Promise<Health>;
  listStages(): Promise<StageDescriptor[]>;
  getStage(name: string): Promise<StageDescriptor>;
  listVlmBackends(): Promise<PluginInfo[]>;
  listDepthProviders(): Promise<PluginInfo[]>;
  listConfigs(): Promise<ConfigListEntry[]>;
  getConfig(name: string): Promise<ConfigDocument>;
  getStageSchema(configName: string, stageName: string): Promise<StageSchemaResponse>;
  submitRun(body: RunSubmissionRequest): Promise<{ run_id: string; snapshot: RunSnapshot }>;
  listRuns(limit?: number): Promise<RunListEntry[]>;
  getRun(runId: string): Promise<RunSnapshot>;
  cancelRun(runId: string): Promise<CancelResponse>;
  replayEvents(runId: string): Promise<RunEvent[]>;
  listArtifacts(runId: string): Promise<ArtifactSummary[]>;
  websocketUrl(runId: string): string;
}

export function createEndpoints(client: ApiClient = defaultApiClient): MyConEndpoints {
  return {
    health: () => client.get<Health>("/health"),
    listStages: () => client.get<StageDescriptor[]>("/registry/stages"),
    getStage: (name) =>
      client.get<StageDescriptor>(`/registry/stages/${encodeURIComponent(name)}`),
    listVlmBackends: () => client.get<PluginInfo[]>("/registry/vlm-backends"),
    listDepthProviders: () => client.get<PluginInfo[]>("/registry/depth-providers"),
    listConfigs: () => client.get<ConfigListEntry[]>("/configs"),
    getConfig: (name) =>
      client.get<ConfigDocument>(`/configs/${encodeURIComponent(name)}`),
    getStageSchema: (configName, stageName) =>
      client.get<StageSchemaResponse>(
        `/configs/${encodeURIComponent(configName)}/schemas/${encodeURIComponent(stageName)}`,
      ),
    submitRun: (body) =>
      client.post<{ run_id: string; snapshot: RunSnapshot }>("/runs", body),
    listRuns: (limit = 100) =>
      client.get<RunListEntry[]>(`/runs?limit=${encodeURIComponent(String(limit))}`),
    getRun: (runId) => client.get<RunSnapshot>(`/runs/${encodeURIComponent(runId)}`),
    cancelRun: (runId) =>
      client.post<CancelResponse>(`/runs/${encodeURIComponent(runId)}/cancel`),
    replayEvents: (runId) =>
      client.get<RunEvent[]>(`/runs/${encodeURIComponent(runId)}/events`),
    listArtifacts: (runId) =>
      client.get<ArtifactSummary[]>(`/runs/${encodeURIComponent(runId)}/artifacts`),
    websocketUrl: (runId) =>
      client.websocketUrl(`/runs/${encodeURIComponent(runId)}/events/stream`),
  };
}

export const endpoints = createEndpoints();
