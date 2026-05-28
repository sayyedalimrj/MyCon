import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";

import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import { ApiError } from "../api/types";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  Empty,
  Notice,
  Spinner,
} from "../components/primitives";
import { PageHeader } from "./PageHeader";
import { formatRunDuration, formatUnixTimestamp, shortHash } from "../lib/format";
import { statusTone } from "../lib/status";

// Run Control Center.
//
// Combines submission, listing, and per-row actions (cancel, view).
// Live progress is rendered inside RunDetailPanel; this panel is the
// "fleet view" + the launcher.

export function RunsPanel() {
  return (
    <div>
      <PageHeader
        title="Run control"
        subtitle="Submit pipeline runs, watch them execute live, and inspect history."
      />
      <div className="space-y-6 p-6">
        <RunLauncher />
        <RunHistory />
      </div>
    </div>
  );
}

function RunLauncher() {
  const qc = useQueryClient();
  const stagesQuery = useQuery({
    queryKey: queryKeys.stages(),
    queryFn: endpoints.listStages,
    staleTime: 60_000,
  });
  const configsQuery = useQuery({
    queryKey: queryKeys.configs(),
    queryFn: endpoints.listConfigs,
    staleTime: 30_000,
  });

  const [configPath, setConfigPath] = useState<string>("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [force, setForce] = useState(false);
  const [label, setLabel] = useState("");

  const submit = useMutation({
    mutationFn: () =>
      endpoints.submitRun({
        config_path: configPath,
        requested_stages: Array.from(selected),
        force,
        label: label || null,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      setSelected(new Set());
    },
  });

  const stages = stagesQuery.data ?? [];
  const configs = configsQuery.data ?? [];

  const allChecked = stages.length > 0 && selected.size === stages.length;
  const toggleAll = () =>
    setSelected(allChecked ? new Set() : new Set(stages.map((s) => s.name)));

  const toggleStage = (name: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const canSubmit =
    !!configPath && selected.size > 0 && !submit.isPending && !configsQuery.isLoading;

  return (
    <Card>
      <CardHeader
        title="Launch a run"
        subtitle="Pick a config and one or more stages. The executor runs them in topological order."
      />

      <div className="grid gap-4 md:grid-cols-3">
        <label className="block text-sm">
          <span className="mb-1 block text-xs uppercase tracking-wider text-ink-muted">
            Config
          </span>
          <select
            data-testid="config-select"
            value={configPath}
            onChange={(e) => setConfigPath(e.target.value)}
            className="w-full rounded-md border border-surface-border bg-surface-1 px-3 py-2 text-sm"
          >
            <option value="">— Choose a config —</option>
            {configs.map((c) => (
              <option key={c.name} value={c.path}>
                {c.name}
              </option>
            ))}
          </select>
        </label>

        <label className="block text-sm">
          <span className="mb-1 block text-xs uppercase tracking-wider text-ink-muted">
            Optional run label
          </span>
          <input
            data-testid="run-label"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g. baseline-rerun"
            className="w-full rounded-md border border-surface-border bg-surface-1 px-3 py-2 text-sm"
          />
        </label>

        <label className="flex items-center gap-2 self-end pb-1 text-sm">
          <input
            type="checkbox"
            checked={force}
            onChange={(e) => setForce(e.target.checked)}
            className="size-4"
          />
          <span>
            <span className="font-medium text-ink">Force re-run</span>
            <span className="block text-xs text-ink-muted">
              Pass <code className="font-mono">--force</code> to each stage CLI.
            </span>
          </span>
        </label>
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between border-b border-surface-border pb-2">
          <span className="text-xs uppercase tracking-wider text-ink-muted">
            Stages ({selected.size}/{stages.length})
          </span>
          <Button variant="ghost" onClick={toggleAll} disabled={stages.length === 0}>
            {allChecked ? "Clear all" : "Select all"}
          </Button>
        </div>
        <ul className="mt-2 grid max-h-72 gap-1 overflow-y-auto pr-1 [grid-template-columns:repeat(auto-fill,minmax(280px,1fr))]">
          {stages.map((s) => (
            <li key={s.name}>
              <label className="flex cursor-pointer items-center gap-2 rounded-md border border-transparent px-2 py-1 text-sm hover:border-surface-border hover:bg-surface-2">
                <input
                  type="checkbox"
                  checked={selected.has(s.name)}
                  onChange={() => toggleStage(s.name)}
                  data-testid={`stage-checkbox-${s.name}`}
                  className="size-4"
                />
                <span className="truncate">
                  <span className="font-medium text-ink">{s.title}</span>
                  <span className="ml-2 font-mono text-[10px] text-ink-subtle">
                    {s.name}
                  </span>
                </span>
              </label>
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-ink-muted">
          {!configPath && "Choose a config to enable submission."}
          {configPath && selected.size === 0 && "Select at least one stage."}
        </div>
        <Button
          variant="primary"
          disabled={!canSubmit}
          onClick={() => submit.mutate()}
          data-testid="submit-run"
        >
          {submit.isPending ? <Spinner /> : null}
          Submit run
        </Button>
      </div>

      {submit.isError && (
        <Notice tone="err" title="Submission failed">
          {submit.error instanceof ApiError
            ? `${submit.error.status}: ${submit.error.message}`
            : (submit.error as Error).message}
        </Notice>
      )}
      {submit.isSuccess && submit.data && (
        <Notice tone="ok" title="Run accepted">
          <span className="font-mono">{submit.data.run_id}</span>{" "}
          <Link
            to={`/runs/${encodeURIComponent(submit.data.run_id)}`}
            className="underline"
          >
            open live view →
          </Link>
        </Notice>
      )}
    </Card>
  );
}

function RunHistory() {
  const qc = useQueryClient();
  const runsQuery = useQuery({
    queryKey: queryKeys.runs(100),
    queryFn: () => endpoints.listRuns(100),
    refetchInterval: 5_000,
  });

  const cancel = useMutation({
    mutationFn: (runId: string) => endpoints.cancelRun(runId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });

  if (runsQuery.isLoading) {
    return (
      <Card>
        <Spinner /> Loading run history…
      </Card>
    );
  }

  const runs = runsQuery.data ?? [];

  return (
    <Card>
      <CardHeader
        title={`Run history (${runs.length})`}
        subtitle="Newest first. Live runs are highlighted; click any row to open its live view."
      />
      {runs.length === 0 ? (
        <Empty
          title="No runs yet"
          hint="Submit your first run from the launcher above."
        />
      ) : (
        <div className="-mx-4 overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-surface-2 text-left text-[11px] uppercase tracking-wider text-ink-muted">
              <tr>
                <th className="px-4 py-2">Run</th>
                <th className="px-4 py-2">Project</th>
                <th className="px-4 py-2">Status</th>
                <th className="px-4 py-2">Stages</th>
                <th className="px-4 py-2">Submitted</th>
                <th className="px-4 py-2">Duration</th>
                <th className="px-4 py-2">Hash</th>
                <th className="px-4 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr
                  key={r.run_id}
                  className="border-t border-surface-border hover:bg-surface-2/40"
                >
                  <td className="px-4 py-2">
                    <Link
                      to={`/runs/${encodeURIComponent(r.run_id)}`}
                      className="font-mono text-xs text-accent hover:underline"
                    >
                      {r.run_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2">{r.project_name}</td>
                  <td className="px-4 py-2">
                    <Badge tone={statusTone(r.status)}>{r.status}</Badge>
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-muted">
                    {r.requested_stages.length}
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-muted">
                    {formatUnixTimestamp(r.submitted_at_unix)}
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-muted">
                    {formatRunDuration(r.started_at_unix, r.finished_at_unix)}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">
                    {shortHash(r.config_hash)}
                  </td>
                  <td className="px-4 py-2 text-right">
                    {(r.status === "running" || r.status === "queued") && (
                      <Button
                        variant="danger"
                        onClick={() => cancel.mutate(r.run_id)}
                        disabled={cancel.isPending}
                        data-testid={`cancel-${r.run_id}`}
                      >
                        cancel
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
