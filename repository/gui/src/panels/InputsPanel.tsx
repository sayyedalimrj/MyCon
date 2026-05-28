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

// Input manager.
//
// Surfaces every declared `inputs.*` and `paths.*` from the chosen config
// so the operator can see what files the pipeline expects on disk. Real
// upload requires a `POST /api/inputs` endpoint that does not exist
// yet — the panel makes that gap explicit rather than fabricating it.

export function InputsPanel() {
  const configs = useQuery({
    queryKey: queryKeys.configs(),
    queryFn: endpoints.listConfigs,
  });
  const [configName, setConfigName] = useState<string>("");

  useEffect(() => {
    if (!configName && configs.data && configs.data.length > 0) {
      setConfigName(configs.data[0].name);
    }
  }, [configs.data, configName]);

  const cfg = useQuery({
    queryKey: queryKeys.config(configName),
    queryFn: () => endpoints.getConfig(configName),
    enabled: !!configName,
  });

  return (
    <div>
      <PageHeader
        title="Inputs"
        subtitle="Inspect the input paths every stage expects on disk for the chosen project config."
        actions={
          <select
            data-testid="input-config-select"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            className="rounded-md border border-surface-border bg-surface-1 px-3 py-1.5 text-sm"
          >
            <option value="">— Choose a config —</option>
            {(configs.data ?? []).map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        }
      />

      <div className="space-y-6 p-6">
        <Notice tone="info" title="Upload lands in Phase 4">
          Browser uploads need a backend <Code>POST /api/inputs</Code> endpoint
          and persistent staging area. Phase 3 surfaces what is{" "}
          <em>declared</em> in the config; Phase 4 will wire <em>upload</em>.
        </Notice>

        {cfg.isLoading && (
          <div className="flex items-center gap-2 text-ink-muted">
            <Spinner /> Loading config inputs…
          </div>
        )}
        {cfg.isError && (
          <Notice tone="err" title="Could not load config">
            {(cfg.error as Error).message}
          </Notice>
        )}
        {cfg.data && <InputTables data={cfg.data.data} />}
        {!configName && <Empty title="Pick a config to see its declared inputs" />}
      </div>
    </div>
  );
}

interface InputEntry {
  key: string;
  value: string;
}

function InputTables({ data }: { data: Record<string, unknown> }) {
  const inputs = pickStringEntries(data, "inputs");
  const paths = pickStringEntries(data, "paths");

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card>
        <CardHeader
          title={`inputs (${inputs.length})`}
          subtitle="Source files the pipeline consumes (video, BIM/IFC, schedule)."
        />
        <InputTable rows={inputs} kindLabel="input" />
      </Card>
      <Card>
        <CardHeader
          title={`paths (${paths.length})`}
          subtitle="Per-stage outputs and intermediates declared in this config."
        />
        <InputTable rows={paths} kindLabel="path" />
      </Card>
    </div>
  );
}

function InputTable({ rows, kindLabel }: { rows: InputEntry[]; kindLabel: string }) {
  if (rows.length === 0) {
    return <Empty title={`No ${kindLabel} entries declared`} />;
  }
  return (
    <ul className="space-y-1 text-sm">
      {rows.map((r) => (
        <li
          key={r.key}
          className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-surface-border bg-surface-1 px-2 py-1"
        >
          <Badge tone="neutral">{r.key}</Badge>
          <Code className="break-all">{r.value}</Code>
        </li>
      ))}
    </ul>
  );
}

function pickStringEntries(data: Record<string, unknown>, head: string): InputEntry[] {
  const v = data[head];
  if (!v || typeof v !== "object") return [];
  const out: InputEntry[] = [];
  for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
    if (typeof val === "string") out.push({ key: `${head}.${k}`, value: val });
  }
  return out;
}
