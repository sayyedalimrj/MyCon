import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import { Badge, Card, CardHeader, Empty, Notice, Spinner } from "../components/primitives";
import { PageHeader } from "./PageHeader";
import { layoutStages } from "../lib/dag";
import { Link } from "react-router-dom";
import type { StageDescriptor } from "../api/types";

// Pipeline overview dashboard.
//
// Drives off two endpoints: /api/registry/stages (the canonical 15-stage
// DAG) and /api/runs (so we can paint the latest known status onto each
// node). No other data is fabricated.

export function PipelinePanel() {
  const stagesQuery = useQuery({
    queryKey: queryKeys.stages(),
    queryFn: endpoints.listStages,
    staleTime: 60_000,
  });
  const runsQuery = useQuery({
    queryKey: queryKeys.runs(20),
    queryFn: () => endpoints.listRuns(20),
    refetchInterval: 5_000,
  });

  const lastRun = runsQuery.data?.[0];
  const stages = stagesQuery.data ?? [];

  const layout = useMemo(() => layoutStages(stages), [stages]);
  const levels = useMemo(() => {
    const grouped = new Map<number, typeof layout>();
    for (const e of layout) {
      if (!grouped.has(e.level)) grouped.set(e.level, []);
      grouped.get(e.level)!.push(e);
    }
    return Array.from(grouped.entries()).sort((a, b) => a[0] - b[0]);
  }, [layout]);

  return (
    <div>
      <PageHeader
        title="Pipeline overview"
        subtitle="Live DAG of every registered stage and the status of the most recent run."
        actions={
          <Link
            to="/runs"
            className="rounded-md border border-surface-border bg-surface-1 px-3 py-1.5 text-sm hover:bg-surface-2"
          >
            Run control →
          </Link>
        }
      />

      <div className="space-y-6 p-6">
        {stagesQuery.isLoading && (
          <div className="flex items-center gap-2 text-ink-muted">
            <Spinner /> Loading stage registry…
          </div>
        )}
        {stagesQuery.isError && (
          <Notice tone="err" title="Could not load stage registry">
            {(stagesQuery.error as Error).message}
          </Notice>
        )}

        {stages.length > 0 && (
          <Card>
            <CardHeader
              title={`DAG (${stages.length} stages)`}
              subtitle={
                lastRun
                  ? `Status painted from latest run · ${lastRun.run_id} · ${lastRun.status}`
                  : "No runs yet — submit one from the Run control panel"
              }
              right={
                lastRun ? (
                  <Link
                    to={`/runs/${encodeURIComponent(lastRun.run_id)}`}
                    className="text-xs text-accent hover:underline"
                  >
                    open latest run →
                  </Link>
                ) : null
              }
            />

            <div className="space-y-4 overflow-x-auto">
              {levels.map(([level, items]) => (
                <div key={level} className="flex items-stretch gap-3">
                  <div className="w-16 shrink-0 text-[10px] uppercase tracking-widest text-ink-subtle">
                    level {level}
                  </div>
                  <div className="grid min-w-0 flex-1 gap-2 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
                    {items.map(({ stage }) => {
                      const lastStatus = lastRun?.stage_statuses?.[stage.name];
                      return <StageNode key={stage.name} stage={stage} lastStatus={lastStatus} />;
                    })}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {!stagesQuery.isLoading && stages.length === 0 && (
          <Empty
            title="No stages registered"
            hint="The backend registry is empty. Check the API health badge above and confirm the service is reachable."
          />
        )}
      </div>
    </div>
  );
}

function StageNode({
  stage,
  lastStatus,
}: {
  stage: StageDescriptor;
  lastStatus: string | undefined;
}) {
  return (
    <article className="flex h-full flex-col rounded-lg border border-surface-border bg-surface-1 p-3 transition hover:border-accent/50">
      <div className="flex items-start justify-between gap-2">
        <h3 className="truncate text-sm font-semibold text-ink" title={stage.title}>
          {stage.title}
        </h3>
        <Badge tone={statusTone(lastStatus)} className="shrink-0">
          {lastStatus ?? "—"}
        </Badge>
      </div>
      <p
        className="mt-1 line-clamp-3 text-[11px] text-ink-muted"
        title={stage.description}
      >
        {stage.description}
      </p>
      <footer className="mt-auto flex flex-wrap items-center gap-1 pt-3 text-[10px] uppercase tracking-wider text-ink-subtle">
        <span className="font-mono normal-case">{stage.name}</span>
        {stage.capabilities?.map((c) => (
          <Badge key={c} tone="neutral">
            {c}
          </Badge>
        ))}
      </footer>
    </article>
  );
}

function statusTone(status: string | undefined) {
  if (!status) return "neutral" as const;
  if (status === "completed") return "ok" as const;
  if (status === "running" || status === "queued") return "info" as const;
  if (status === "failed") return "err" as const;
  if (status === "cancelled") return "warn" as const;
  return "neutral" as const;
}
