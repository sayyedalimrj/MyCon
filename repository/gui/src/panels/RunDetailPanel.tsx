import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";

import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  Code,
  Empty,
  Notice,
  Spinner,
} from "../components/primitives";
import { PageHeader } from "./PageHeader";
import {
  formatRunDuration,
  formatUnixTimestamp,
  shortHash,
} from "../lib/format";
import { isTerminal, statusTone } from "../lib/status";
import { useRunStream } from "../hooks/useRunStream";
import type { RunEvent } from "../api/types";

// Live view of one run.
//
// Combines:
//   - GET /api/runs/{id}        (snapshot, refetched every 2s)
//   - WS  /api/runs/{id}/events/stream  (live events)
// Gives the operator a per-stage progress strip, a streaming console,
// cancel, and a clear "rerun with same settings" jump.

export function RunDetailPanel() {
  const { runId } = useParams<{ runId: string }>();
  const qc = useQueryClient();

  const snapshot = useQuery({
    queryKey: queryKeys.run(runId ?? ""),
    queryFn: () => endpoints.getRun(runId!),
    enabled: !!runId,
    refetchInterval: (q) => (isTerminal(q.state.data?.status) ? false : 2_000),
  });

  const stream = useRunStream(runId);
  const cancel = useMutation({
    mutationFn: () => endpoints.cancelRun(runId!),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.run(runId ?? "") }),
  });

  const linesByStage = useMemo(() => buildConsoleByStage(stream.events), [stream.events]);

  if (!runId) return <Empty title="Missing run id" />;
  if (snapshot.isLoading) {
    return (
      <div className="p-6">
        <Spinner /> Loading run…
      </div>
    );
  }
  if (snapshot.isError) {
    return (
      <div className="p-6">
        <Notice tone="err" title="Could not load run">
          {(snapshot.error as Error).message}
        </Notice>
      </div>
    );
  }
  const data = snapshot.data!;

  return (
    <div>
      <PageHeader
        title={
          <>
            Run <span className="font-mono text-accent">{data.run_id}</span>
          </>
        }
        subtitle={
          <>
            <span className="mr-3">project: <Code>{data.project_name}</Code></span>
            <span className="mr-3">config: <Code>{data.submission.config_path}</Code></span>
            <span>hash: <Code>{shortHash(data.config_hash)}</Code></span>
          </>
        }
        actions={
          <div className="flex items-center gap-2">
            <Badge tone={statusTone(data.status)}>{data.status}</Badge>
            {!isTerminal(data.status) && (
              <Button
                variant="danger"
                onClick={() => cancel.mutate()}
                disabled={cancel.isPending || data.cancel_requested}
                data-testid="cancel-detail"
              >
                {data.cancel_requested ? "cancelling…" : "Cancel run"}
              </Button>
            )}
            <Link
              to="/runs"
              className="rounded-md border border-surface-border bg-surface-1 px-3 py-1.5 text-sm hover:bg-surface-2"
            >
              all runs
            </Link>
          </div>
        }
      />

      <div className="grid gap-6 p-6 lg:grid-cols-2">
        <Card>
          <CardHeader
            title={`Stages (${data.stages.length})`}
            subtitle={`Submitted ${formatUnixTimestamp(data.submitted_at_unix)} · running for ${formatRunDuration(
              data.started_at_unix,
              data.finished_at_unix,
            )}`}
            right={
              <Badge tone={stream.state === "open" ? "ok" : "neutral"}>
                stream {stream.state}
              </Badge>
            }
          />
          <ul className="space-y-2">
            {data.stages.map((s) => {
              const lines = linesByStage.get(s.name) ?? 0;
              return (
                <li
                  key={s.name}
                  className="flex items-center gap-3 rounded-md border border-surface-border bg-surface-1 px-3 py-2"
                >
                  <Badge tone={statusTone(s.status)}>{s.status}</Badge>
                  <span className="flex-1 truncate">
                    <span className="font-mono text-xs">{s.name}</span>
                  </span>
                  <span className="text-xs text-ink-muted">{lines} lines</span>
                  <span className="text-xs text-ink-muted">
                    {formatRunDuration(s.started_at_unix, s.finished_at_unix)}
                  </span>
                </li>
              );
            })}
          </ul>
        </Card>

        <Card>
          <CardHeader
            title="Live console"
            subtitle="Stdout/stderr from each stage subprocess, oldest first. Capped at 5 000 events."
            right={
              stream.state === "error" ? (
                <Badge tone="err">stream error</Badge>
              ) : (
                <Badge tone="neutral">{stream.events.length} events</Badge>
              )
            }
          />
          <Console events={stream.events} />
        </Card>
      </div>
    </div>
  );
}

function Console({ events }: { events: RunEvent[] }) {
  if (events.length === 0) {
    return (
      <Empty
        title="No events yet"
        hint="Once the executor publishes events they appear here in real time."
      />
    );
  }
  return (
    <div
      className="max-h-[480px] overflow-y-auto rounded-md border border-surface-border bg-surface-0 p-2 font-mono text-[11px] leading-tight"
      data-testid="run-console"
    >
      {events.map((ev) => (
        <ConsoleLine key={ev.event_id} ev={ev} />
      ))}
    </div>
  );
}

function ConsoleLine({ ev }: { ev: RunEvent }) {
  const ts = new Date(ev.timestamp_unix * 1000).toISOString().slice(11, 19);
  if (ev.kind === "stage.progress") {
    const stream = (ev.payload?.stream as string) ?? "stdout";
    const line = (ev.payload?.line as string) ?? "";
    const tone = stream === "stderr" ? "text-warn" : "text-ink";
    return (
      <div className={tone}>
        <span className="text-ink-subtle">{ts}</span>{" "}
        <span className="text-accent">{ev.stage}</span> · {line}
      </div>
    );
  }
  return (
    <div className="text-ink-muted">
      <span className="text-ink-subtle">{ts}</span> {ev.kind}
      {ev.stage ? <span className="text-accent"> · {ev.stage}</span> : null}
    </div>
  );
}

function buildConsoleByStage(events: RunEvent[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const ev of events) {
    if (ev.kind === "stage.progress" && ev.stage) {
      counts.set(ev.stage, (counts.get(ev.stage) ?? 0) + 1);
    }
  }
  return counts;
}
