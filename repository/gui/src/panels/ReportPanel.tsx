import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

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
import { statusTone } from "../lib/status";
import type { ArtifactSummary, RunSnapshot } from "../api/types";

// Report generator.
//
// Phase 3 produces a structured **JSON** report from real backend data
// (run snapshot + per-stage artifacts + provenance). PDF / DOCX export
// requires a server-side renderer that is part of Phase 4. The generated
// JSON is exactly what a future `POST /api/reports` endpoint will accept,
// so the contract stays stable.

export function ReportPanel() {
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
        title="Report generator"
        subtitle="Compose a thesis-grade summary from one run’s real snapshot, per-stage statuses, and provenance metadata."
        actions={
          <select
            data-testid="report-run-select"
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
        <Notice tone="info" title="PDF / DOCX export lands in Phase 4">
          The structured JSON report this panel produces is the same shape a
          server-side renderer will consume in Phase 4. Today you can copy it
          or download it as <Code>.json</Code>.
        </Notice>
        {runId ? <ReportBody runId={runId} /> : <Empty title="Pick a run to generate a report" />}
      </div>
    </div>
  );
}

function ReportBody({ runId }: { runId: string }) {
  const snap = useQuery({
    queryKey: queryKeys.run(runId),
    queryFn: () => endpoints.getRun(runId),
  });
  const artifacts = useQuery({
    queryKey: queryKeys.runArtifacts(runId),
    queryFn: () => endpoints.listArtifacts(runId),
  });

  const report = useMemo(
    () => (snap.data && artifacts.data ? composeReport(snap.data, artifacts.data) : null),
    [snap.data, artifacts.data],
  );

  if (snap.isLoading || artifacts.isLoading) {
    return (
      <div className="flex items-center gap-2 text-ink-muted">
        <Spinner /> Assembling report…
      </div>
    );
  }
  if (snap.isError || artifacts.isError) {
    return (
      <Notice tone="err" title="Could not assemble report">
        {((snap.error || artifacts.error) as Error)?.message}
      </Notice>
    );
  }
  if (!report) return null;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader title="Run summary" />
        <dl className="grid grid-cols-2 gap-y-1 text-sm">
          <DescItem label="run id" value={<Code>{report.run.run_id}</Code>} />
          <DescItem label="project" value={report.run.project_name} />
          <DescItem
            label="status"
            value={<Badge tone={statusTone(report.run.status)}>{report.run.status}</Badge>}
          />
          <DescItem
            label="config_hash"
            value={<Code>{shortHash(report.run.config_hash, 16)}</Code>}
          />
          <DescItem label="submitted" value={formatUnixTimestamp(report.run.submitted_at_unix)} />
          <DescItem
            label="duration"
            value={formatRunDuration(report.run.started_at_unix, report.run.finished_at_unix)}
          />
        </dl>
      </Card>

      <Card>
        <CardHeader
          title="Per-stage outcomes"
          subtitle={`${report.stages_total} requested · ${report.stages_completed} completed · ${report.stages_failed} failed`}
        />
        <ul className="space-y-1 text-sm">
          {report.stages.map((s) => (
            <li
              key={s.name}
              className="flex items-center gap-2 rounded-md border border-surface-border bg-surface-1 px-2 py-1"
            >
              <Badge tone={statusTone(s.status)}>{s.status}</Badge>
              <span className="font-mono text-xs">{s.name}</span>
              <span className="ml-auto text-xs text-ink-muted">
                {s.artifact_status ?? "—"}
              </span>
            </li>
          ))}
        </ul>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader
          title="Provenance roll-up"
          subtitle="Distinct provenance fingerprints across all artifacts of this run."
        />
        {report.provenance.length === 0 ? (
          <Empty title="No artifacts had a provenance envelope" />
        ) : (
          <ul className="space-y-2 text-sm">
            {report.provenance.map((p, i) => (
              <li
                key={i}
                className="flex flex-wrap items-center gap-2 rounded-md border border-surface-border bg-surface-1 px-2 py-1"
              >
                <Code>{shortHash(p.config_hash, 12)}</Code>
                <span className="text-xs text-ink-muted">git: {shortHash(p.git_sha, 10)}</span>
                <span className="ml-auto">
                  <Badge tone="neutral">{p.count} artifacts</Badge>
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader
          title="Export"
          subtitle="Download the structured report or copy it to clipboard."
          right={
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                onClick={() => navigator.clipboard?.writeText(JSON.stringify(report, null, 2))}
                data-testid="report-copy"
              >
                Copy JSON
              </Button>
              <Button
                variant="primary"
                onClick={() => downloadJson(`report_${runId}.json`, report)}
                data-testid="report-download"
              >
                Download .json
              </Button>
              <Button variant="ghost" disabled title="Phase 4">
                PDF (Phase 4)
              </Button>
            </div>
          }
        />
        <pre className="max-h-96 overflow-auto rounded-md border border-surface-border bg-surface-0 p-2 font-mono text-[11px]">
{JSON.stringify(report, null, 2)}
        </pre>
      </Card>
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

interface StageReportEntry {
  name: string;
  status: string;
  artifact_status: string | null;
  artifact_path: string | null;
  has_provenance: boolean;
}

interface ProvenanceRollup {
  config_hash: string | null;
  git_sha: string | null;
  count: number;
}

interface ComposedReport {
  generated_at_unix: number;
  run: RunSnapshot;
  stages: StageReportEntry[];
  stages_total: number;
  stages_completed: number;
  stages_failed: number;
  provenance: ProvenanceRollup[];
}

export function composeReport(snap: RunSnapshot, artifacts: ArtifactSummary[]): ComposedReport {
  const artifactByStage = new Map<string, ArtifactSummary>();
  for (const a of artifacts) artifactByStage.set(a.stage, a);

  const stages: StageReportEntry[] = snap.stages.map((s) => {
    const a = artifactByStage.get(s.name);
    return {
      name: s.name,
      status: s.status,
      artifact_status: a?.exists ? a.status ?? "present" : "missing",
      artifact_path: a?.exists ? a.artifact_path : null,
      has_provenance: !!a?.provenance?.config_hash,
    };
  });

  const provGroups = new Map<string, ProvenanceRollup>();
  for (const a of artifacts) {
    const ch = (a.provenance?.config_hash as string | undefined) ?? null;
    const gs = (a.provenance?.git_sha as string | null | undefined) ?? null;
    if (!ch && !gs) continue;
    const key = `${ch ?? "?"}::${gs ?? "?"}`;
    const cur = provGroups.get(key);
    if (cur) cur.count += 1;
    else provGroups.set(key, { config_hash: ch, git_sha: gs, count: 1 });
  }

  return {
    generated_at_unix: Date.now() / 1000,
    run: snap,
    stages,
    stages_total: stages.length,
    stages_completed: stages.filter((s) => s.status === "completed").length,
    stages_failed: stages.filter((s) => s.status === "failed").length,
    provenance: Array.from(provGroups.values()).sort((a, b) => b.count - a.count),
  };
}

function downloadJson(filename: string, data: unknown): void {
  if (typeof document === "undefined") return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}
