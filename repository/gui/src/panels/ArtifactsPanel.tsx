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
import { statusTone } from "../lib/status";
import type { ArtifactSummary } from "../api/types";

// Artifact browser.
//
// Lists every stage's report file for a chosen run, with the parsed
// `provenance` envelope (Phase 1) and a capped non-provenance preview.
// Download-as-JSON works against the file path the backend returns.
// Full content streaming is a Phase 4 backend concern.

export function ArtifactsPanel() {
  const runs = useQuery({
    queryKey: queryKeys.runs(50),
    queryFn: () => endpoints.listRuns(50),
    refetchInterval: 5_000,
  });
  const [runId, setRunId] = useState<string>("");

  useEffect(() => {
    if (!runId && runs.data && runs.data.length > 0) {
      setRunId(runs.data[0].run_id);
    }
  }, [runs.data, runId]);

  return (
    <div>
      <PageHeader
        title="Artifacts"
        subtitle="Per-stage reports, provenance, and previews for one run."
        actions={
          <select
            data-testid="artifact-run-select"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            className="rounded-md border border-surface-border bg-surface-1 px-3 py-1.5 text-sm"
          >
            <option value="">— Choose a run —</option>
            {(runs.data ?? []).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.status})
              </option>
            ))}
          </select>
        }
      />
      <div className="space-y-4 p-6">
        {!runId ? (
          <Empty title="Pick a run to inspect its artifacts" />
        ) : (
          <ArtifactList runId={runId} />
        )}
      </div>
    </div>
  );
}

function ArtifactList({ runId }: { runId: string }) {
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
  const items = q.data ?? [];
  if (items.length === 0) {
    return <Empty title="No artifacts declared by the registry" />;
  }
  return (
    <ul className="grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(360px,1fr))]">
      {items.map((a) => (
        <ArtifactCard key={a.stage} a={a} />
      ))}
    </ul>
  );
}

function ArtifactCard({ a }: { a: ArtifactSummary }) {
  return (
    <li>
      <Card>
        <CardHeader
          title={a.stage}
          subtitle={<Code>{a.artifact_basename}</Code>}
          right={
            <Badge tone={a.exists ? statusTone(a.status) : "neutral"}>
              {a.exists ? a.status ?? "present" : "missing"}
            </Badge>
          }
        />
        {a.parse_error && (
          <Notice tone="warn" title="Could not parse">
            {a.parse_error}
          </Notice>
        )}
        <dl className="grid grid-cols-2 gap-y-1 text-xs text-ink-muted">
          <DescItem
            label="path"
            value={<Code className="break-all">{a.artifact_path}</Code>}
          />
          <DescItem label="size" value={formatBytes(a.size_bytes)} />
          <DescItem label="modified" value={formatUnixTimestamp(a.modified_at_unix)} />
          {a.provenance?.config_hash && (
            <DescItem
              label="provenance hash"
              value={
                <span className="font-mono">{shortHash(a.provenance.config_hash)}</span>
              }
            />
          )}
          {a.provenance?.git_sha != null && (
            <DescItem
              label="git_sha"
              value={
                <span className="font-mono">
                  {shortHash(a.provenance.git_sha as string, 10)}
                </span>
              }
            />
          )}
          {a.provenance?.generated_at_unix != null && (
            <DescItem
              label="generated"
              value={formatUnixTimestamp(a.provenance.generated_at_unix as number)}
            />
          )}
        </dl>

        {Object.keys(a.preview).length > 0 && (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs text-ink-muted hover:text-ink">
              preview ({Object.keys(a.preview).length} keys)
            </summary>
            <pre className="mt-2 max-h-48 overflow-y-auto rounded-md border border-surface-border bg-surface-0 p-2 font-mono text-[11px]">
{JSON.stringify(a.preview, null, 2)}
            </pre>
          </details>
        )}
      </Card>
    </li>
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
