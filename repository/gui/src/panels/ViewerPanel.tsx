import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import {
  Badge,
  Card,
  CardHeader,
  Code,
  Empty,
  Notice,
  Spinner,
} from "../components/primitives";
import { PageHeader } from "./PageHeader";
import { formatBytes, formatUnixTimestamp, shortHash } from "../lib/format";

// BIM / 3D viewer.
//
// We expose every 3D-relevant artifact path the backend reports — point
// clouds (.ply), mesh outputs, BIM-aligned files, registration reports —
// so the operator knows exactly what files exist on disk for a chosen
// run. The actual three-dimensional rendering canvas is a Phase 4
// feature: it requires a backend `GET /api/files/...` endpoint to stream
// the (potentially gigabyte-scale) binary files, plus a WebGL viewer
// (three.js or similar). Phase 3 surfaces what exists and what is
// missing — without inventing fake geometry.

const VIEWER_STAGES = new Set([
  "stage_05_dense",
  "stage_06_da3_assist",
  "stage_07_cleanup",
  "stage_07_6_viewer_export",
  "stage_07_7_cams_gs_evidence",
  "stage_08_bim_eval",
  "stage_08_metric_alignment",
]);

export function ViewerPanel() {
  const runs = useQuery({
    queryKey: queryKeys.runs(50),
    queryFn: () => endpoints.listRuns(50),
    refetchInterval: 10_000,
  });
  const [runId, setRunId] = useState<string>("");
  useEffect(() => {
    if (!runId && runs.data && runs.data.length > 0) setRunId(runs.data[0].run_id);
  }, [runs.data, runId]);

  return (
    <div>
      <PageHeader
        title="3D / BIM viewer"
        subtitle="Inspect alignment, BIM evaluation, and dense reconstruction artifacts. Interactive 3D canvas lands in Phase 4."
        actions={
          <select
            data-testid="viewer-run-select"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            className="rounded-md border border-surface-border bg-surface-1 px-3 py-1.5 text-sm"
          >
            <option value="">— Choose a run —</option>
            {(runs.data ?? []).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id}
              </option>
            ))}
          </select>
        }
      />

      <div className="space-y-6 p-6">
        <Notice tone="info" title="Why is there no live 3D canvas yet?">
          Streaming dense point clouds, BIM-aligned IFC slices, and
          registration overlays into the browser needs a backend file route
          (<Code>GET /api/files/...</Code>) and a WebGL renderer
          (<Code>three.js</Code>). Both are scheduled for Phase 4. Phase 3
          surfaces every relevant file path so you can open it in CloudCompare,
          MeshLab, or BlenderBIM today.
        </Notice>

        {!runId ? (
          <Empty title="Pick a run to see its 3D-relevant artifacts" />
        ) : (
          <ViewerArtifacts runId={runId} />
        )}
      </div>
    </div>
  );
}

function ViewerArtifacts({ runId }: { runId: string }) {
  const q = useQuery({
    queryKey: queryKeys.runArtifacts(runId),
    queryFn: () => endpoints.listArtifacts(runId),
    refetchInterval: 5_000,
  });
  if (q.isLoading) {
    return (
      <div className="flex items-center gap-2 text-ink-muted">
        <Spinner /> Loading artifacts…
      </div>
    );
  }
  if (q.isError) {
    return (
      <Notice tone="err" title="Could not load artifacts">
        {(q.error as Error).message}
      </Notice>
    );
  }
  const items = (q.data ?? []).filter((a) => VIEWER_STAGES.has(a.stage));
  if (items.length === 0) {
    return (
      <Empty
        title="This run produced no 3D-relevant artifacts yet"
        hint="Run the dense, viewer-export, BIM-eval, or metric-alignment stages to populate this panel."
      />
    );
  }
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {items.map((a) => (
        <Card key={a.stage}>
          <CardHeader
            title={a.stage}
            subtitle={<Code>{a.artifact_basename}</Code>}
            right={
              <Badge tone={a.exists ? "ok" : "neutral"}>
                {a.exists ? "ready" : "not yet produced"}
              </Badge>
            }
          />
          <dl className="grid grid-cols-2 gap-y-1 text-xs text-ink-muted">
            <DescItem
              label="path"
              value={<Code className="break-all">{a.artifact_path}</Code>}
            />
            <DescItem label="size" value={formatBytes(a.size_bytes)} />
            <DescItem label="modified" value={formatUnixTimestamp(a.modified_at_unix)} />
            {a.provenance?.config_hash && (
              <DescItem
                label="config hash"
                value={
                  <span className="font-mono">
                    {shortHash(a.provenance.config_hash)}
                  </span>
                }
              />
            )}
          </dl>
          <Notice tone="neutral">
            Open the file in your favored 3D tool — the path above is what the
            stage actually wrote.
          </Notice>
        </Card>
      ))}
    </div>
  );
}

function DescItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <>
      <dt className="text-[10px] uppercase tracking-wider text-ink-subtle">{label}</dt>
      <dd className="text-ink">{value}</dd>
    </>
  );
}
