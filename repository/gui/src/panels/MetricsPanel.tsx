import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import { Badge, Card, CardHeader, Empty, Spinner } from "../components/primitives";
import { PageHeader } from "./PageHeader";
import type { ArtifactSummary, RunListEntry } from "../api/types";

// Metrics dashboard.
//
// Charts are derived from real data, never fabricated:
//   - "runs over time" uses the run history list,
//   - "stage status mix (latest run)" uses /api/runs/{id}/artifacts of the
//     newest run,
//   - "config-hash drift" highlights how many of the last N runs share a
//     hash with the latest one (a real-research signal: "did anyone change
//     the config?").
//
// When data is sparse the panel renders a clear empty state instead of
// padding with synthetic numbers.

const RECENT_RUNS_LIMIT = 30;

export function MetricsPanel() {
  const runs = useQuery({
    queryKey: queryKeys.runs(RECENT_RUNS_LIMIT),
    queryFn: () => endpoints.listRuns(RECENT_RUNS_LIMIT),
    refetchInterval: 10_000,
  });

  const latest = runs.data?.[0];
  const latestArtifacts = useQuery({
    queryKey: queryKeys.runArtifacts(latest?.run_id ?? ""),
    queryFn: () => endpoints.listArtifacts(latest!.run_id),
    enabled: !!latest,
  });

  return (
    <div>
      <PageHeader
        title="Metrics"
        subtitle="Trends derived from the actual run history and artifact reports — no fabricated values."
      />
      <div className="grid gap-6 p-6 lg:grid-cols-2">
        <RunTrendCard runs={runs.data ?? []} loading={runs.isLoading} />
        <HashDriftCard runs={runs.data ?? []} loading={runs.isLoading} />
        <StageStatusCard
          runId={latest?.run_id}
          artifacts={latestArtifacts.data ?? []}
          loading={latestArtifacts.isLoading}
        />
        <ProvenanceCoverageCard
          artifacts={latestArtifacts.data ?? []}
          loading={latestArtifacts.isLoading}
        />
      </div>
    </div>
  );
}

function RunTrendCard({ runs, loading }: { runs: RunListEntry[]; loading: boolean }) {
  const data = useMemo(
    () =>
      runs
        .slice()
        .reverse()
        .map((r, i) => ({
          idx: i + 1,
          duration: r.started_at_unix && r.finished_at_unix
            ? Math.max(0, r.finished_at_unix - r.started_at_unix)
            : 0,
          status: r.status,
        })),
    [runs],
  );
  return (
    <Card>
      <CardHeader
        title="Run duration trend"
        subtitle={`Last ${runs.length} runs, oldest → newest. Bars are colored by status.`}
      />
      {loading ? (
        <Spinner />
      ) : data.length === 0 ? (
        <Empty title="No runs yet" />
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer>
            <BarChart data={data}>
              <CartesianGrid stroke="rgb(var(--surface-border))" strokeDasharray="2 2" />
              <XAxis dataKey="idx" tick={{ fontSize: 10 }} stroke="rgb(var(--ink-subtle))" />
              <YAxis tick={{ fontSize: 10 }} stroke="rgb(var(--ink-subtle))" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--surface-1))",
                  border: "1px solid rgb(var(--surface-border))",
                  fontSize: 12,
                }}
                formatter={(value: number) => [`${value.toFixed(1)} s`, "duration"]}
              />
              <Bar dataKey="duration">
                {data.map((d) => (
                  <Cell key={d.idx} fill={statusColor(d.status)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

function HashDriftCard({ runs, loading }: { runs: RunListEntry[]; loading: boolean }) {
  const data = useMemo(() => {
    if (runs.length === 0) return [];
    const latest = runs[0];
    return runs.map((r, i) => ({
      idx: runs.length - i,
      same: r.config_hash === latest.config_hash ? 1 : 0,
    }));
  }, [runs]);

  return (
    <Card>
      <CardHeader
        title="Config-hash drift"
        subtitle="Each spike marks a run whose config_hash matches the latest. Gaps mean the config changed between runs."
      />
      {loading ? (
        <Spinner />
      ) : data.length === 0 ? (
        <Empty title="No runs yet" />
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer>
            <LineChart data={data}>
              <CartesianGrid stroke="rgb(var(--surface-border))" strokeDasharray="2 2" />
              <XAxis dataKey="idx" tick={{ fontSize: 10 }} stroke="rgb(var(--ink-subtle))" />
              <YAxis
                domain={[0, 1]}
                ticks={[0, 1]}
                tick={{ fontSize: 10 }}
                stroke="rgb(var(--ink-subtle))"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--surface-1))",
                  border: "1px solid rgb(var(--surface-border))",
                  fontSize: 12,
                }}
              />
              <Line
                type="stepAfter"
                dataKey="same"
                stroke="rgb(var(--accent))"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

function StageStatusCard({
  runId,
  artifacts,
  loading,
}: {
  runId: string | undefined;
  artifacts: ArtifactSummary[];
  loading: boolean;
}) {
  const counts = useMemo(() => {
    const m = new Map<string, number>();
    for (const a of artifacts) {
      const k = a.exists ? a.status ?? "present" : "missing";
      m.set(k, (m.get(k) ?? 0) + 1);
    }
    return Array.from(m.entries()).map(([status, n]) => ({ status, n }));
  }, [artifacts]);

  return (
    <Card>
      <CardHeader
        title="Latest run · stage status mix"
        subtitle={runId ? `From /api/runs/${runId}/artifacts` : "No run selected"}
      />
      {loading ? (
        <Spinner />
      ) : counts.length === 0 ? (
        <Empty title="No artifact data for the latest run" />
      ) : (
        <div className="h-56 w-full">
          <ResponsiveContainer>
            <BarChart data={counts}>
              <CartesianGrid stroke="rgb(var(--surface-border))" strokeDasharray="2 2" />
              <XAxis dataKey="status" tick={{ fontSize: 10 }} stroke="rgb(var(--ink-subtle))" />
              <YAxis tick={{ fontSize: 10 }} stroke="rgb(var(--ink-subtle))" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgb(var(--surface-1))",
                  border: "1px solid rgb(var(--surface-border))",
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="n" name="stages">
                {counts.map((d) => (
                  <Cell key={d.status} fill={statusColor(d.status)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}

function ProvenanceCoverageCard({
  artifacts,
  loading,
}: {
  artifacts: ArtifactSummary[];
  loading: boolean;
}) {
  const total = artifacts.length;
  const withProv = artifacts.filter((a) => a.provenance && a.provenance.config_hash).length;
  const ratio = total > 0 ? Math.round((withProv / total) * 100) : 0;

  return (
    <Card>
      <CardHeader
        title="Provenance coverage"
        subtitle="Fraction of latest-run artifacts carrying a Phase 1 provenance envelope (config_hash, git_sha, timestamps)."
      />
      {loading ? (
        <Spinner />
      ) : total === 0 ? (
        <Empty title="No artifacts available yet" />
      ) : (
        <div className="flex items-center gap-6">
          <div className="text-5xl font-bold tabular-nums text-ink">{ratio}%</div>
          <div className="space-y-1 text-sm">
            <div>
              <Badge tone="ok">{withProv}</Badge>{" "}
              <span className="text-ink-muted">artifacts with provenance</span>
            </div>
            <div>
              <Badge tone="neutral">{total - withProv}</Badge>{" "}
              <span className="text-ink-muted">missing provenance block</span>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

function statusColor(status: string): string {
  switch (status) {
    case "completed":
    case "present":
      return "rgb(var(--status-ok))";
    case "running":
    case "queued":
      return "rgb(var(--status-info))";
    case "failed":
      return "rgb(var(--status-err))";
    case "cancelled":
      return "rgb(var(--status-warn))";
    default:
      return "rgb(var(--ink-subtle))";
  }
}
