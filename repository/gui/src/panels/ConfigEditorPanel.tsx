import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

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
import { flattenSchema, groupBySection } from "../lib/schemaToControls";
import type { FieldControl } from "../lib/schemaToControls";
import { shortHash } from "../lib/format";

// Stage editor.
//
// Picks one config + one stage and renders the typed schema as form
// controls. Edits are tracked locally; the "Save" action is intentionally
// disabled with a clear note that the matching backend PUT lands in
// Phase 4. Reset-to-server-values is fully functional today.

export function ConfigEditorPanel() {
  const { configName } = useParams<{ configName: string }>();
  return (
    <div>
      <PageHeader
        title={
          <>
            Editing config{" "}
            <span className="font-mono text-accent">{configName}</span>
          </>
        }
        subtitle="Pick a stage to expand its typed parameters. Save lands in Phase 4 — reset and inspect work today."
      />
      {configName && <ConfigEditorBody configName={configName} />}
    </div>
  );
}

function ConfigEditorBody({ configName }: { configName: string }) {
  const stages = useQuery({
    queryKey: queryKeys.stages(),
    queryFn: endpoints.listStages,
    staleTime: 60_000,
  });
  const cfg = useQuery({
    queryKey: queryKeys.config(configName),
    queryFn: () => endpoints.getConfig(configName),
  });

  const [stageName, setStageName] = useState<string>("");

  useEffect(() => {
    if (!stageName && stages.data && stages.data.length > 0) {
      setStageName(stages.data[0].name);
    }
  }, [stages.data, stageName]);

  return (
    <div className="space-y-6 p-6">
      <Card>
        <CardHeader
          title="Configuration metadata"
          subtitle="Server-validated. The hash is recomputed on every load and is what each run is recorded against."
        />
        {cfg.isLoading && (
          <div className="flex items-center gap-2 text-ink-muted">
            <Spinner /> Loading config…
          </div>
        )}
        {cfg.isError && (
          <Notice tone="err" title="Config invalid or missing">
            {(cfg.error as Error).message}
          </Notice>
        )}
        {cfg.data && (
          <dl className="grid grid-cols-2 gap-y-2 text-sm md:grid-cols-3">
            <DescItem label="Name" value={cfg.data.name} />
            <DescItem label="Path" value={<Code>{cfg.data.path}</Code>} />
            <DescItem
              label="Config hash"
              value={
                <span className="font-mono">
                  {shortHash(cfg.data.config_hash, 16)}
                </span>
              }
            />
          </dl>
        )}
      </Card>

      <Card>
        <CardHeader
          title="Stage editor"
          subtitle="Choose a stage; its typed schema view becomes editable form controls."
          right={
            <select
              data-testid="stage-select"
              value={stageName}
              onChange={(e) => setStageName(e.target.value)}
              className="rounded-md border border-surface-border bg-surface-1 px-3 py-1.5 text-sm"
            >
              {(stages.data ?? []).map((s) => (
                <option key={s.name} value={s.name}>
                  {s.title} ({s.name})
                </option>
              ))}
            </select>
          }
        />
        {stageName && (
          <StageSchemaForm configName={configName} stageName={stageName} />
        )}
        {!stageName && (
          <Empty
            title="Pick a stage to begin"
            hint="Each stage exposes its own subset of typed parameters."
          />
        )}
      </Card>
    </div>
  );
}

function DescItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wider text-ink-muted">{label}</dt>
      <dd className="mt-0.5 break-all text-ink">{value}</dd>
    </div>
  );
}

function StageSchemaForm({
  configName,
  stageName,
}: {
  configName: string;
  stageName: string;
}) {
  const q = useQuery({
    queryKey: queryKeys.stageSchema(configName, stageName),
    queryFn: () => endpoints.getStageSchema(configName, stageName),
  });

  const initialControls = useMemo<FieldControl[]>(
    () => (q.data ? flattenSchema(q.data.schema) : []),
    [q.data],
  );

  const [edits, setEdits] = useState<Record<string, unknown>>({});
  // Reset edits when we load a new stage.
  useEffect(() => setEdits({}), [stageName, configName, q.dataUpdatedAt]);

  if (q.isLoading) {
    return (
      <div className="flex items-center gap-2 text-ink-muted">
        <Spinner /> Loading schema…
      </div>
    );
  }
  if (q.isError) {
    return (
      <Notice tone="err" title="Schema validation failed">
        {(q.error as Error).message}
      </Notice>
    );
  }
  if (!q.data) return null;

  const sections = groupBySection(initialControls);
  const editedKeys = Object.keys(edits);

  return (
    <div className="space-y-5">
      <Notice tone="info" title="Editing is local">
        These controls are bound to the backend-validated schema view (
        <Code>{q.data.schema_class}</Code>). Persisting edits requires the
        Phase 4 <Code>PUT /api/configs/{`{name}`}</Code> endpoint, which is
        deliberately not exposed yet. <strong>Reset</strong> reverts the
        local form to the server's current values.
      </Notice>

      <div className="flex items-center gap-2 text-xs">
        <Badge tone="neutral">{initialControls.length} fields</Badge>
        <Badge tone={editedKeys.length > 0 ? "warn" : "neutral"}>
          {editedKeys.length} edited
        </Badge>
        <Button
          variant="ghost"
          onClick={() => setEdits({})}
          disabled={editedKeys.length === 0}
          data-testid="reset-edits"
        >
          Reset to server values
        </Button>
        <Button
          variant="primary"
          disabled
          title="Phase 4: backend PUT /api/configs/{name} is not yet wired."
        >
          Save (Phase 4)
        </Button>
      </div>

      {sections.map(({ section, controls }) => (
        <section
          key={section}
          className="rounded-lg border border-surface-border bg-surface-1 p-4"
        >
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-ink-muted">
            {section}
          </h3>
          <div className="grid gap-3 md:grid-cols-2">
            {controls.map((c) => (
              <FieldRow
                key={c.path}
                control={c}
                value={c.path in edits ? edits[c.path] : c.value}
                edited={c.path in edits}
                onChange={(v) =>
                  setEdits((prev) => {
                    const next = { ...prev };
                    next[c.path] = v;
                    return next;
                  })
                }
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function FieldRow({
  control,
  value,
  edited,
  onChange,
}: {
  control: FieldControl;
  value: unknown;
  edited: boolean;
  onChange: (v: unknown) => void;
}) {
  const id = `field-${control.path}`;
  return (
    <label
      htmlFor={id}
      className="flex flex-col gap-1 rounded-md border border-transparent p-2 hover:border-surface-border"
    >
      <span className="flex items-center justify-between text-xs">
        <span className="text-ink">{control.label}</span>
        <span className="font-mono text-[10px] text-ink-subtle">
          {control.path}
          {edited && <span className="ml-1 text-warn">·edited</span>}
          {control.isNull && <span className="ml-1 text-ink-subtle">·null</span>}
        </span>
      </span>
      {renderControl(control, value, onChange, id)}
    </label>
  );
}

function renderControl(
  c: FieldControl,
  value: unknown,
  onChange: (v: unknown) => void,
  id: string,
): React.ReactNode {
  if (c.type === "boolean") {
    return (
      <input
        id={id}
        type="checkbox"
        checked={Boolean(value)}
        onChange={(e) => onChange(e.target.checked)}
        className="size-4 self-start"
        data-testid={`field-${c.path}`}
      />
    );
  }
  if (c.type === "number") {
    return (
      <input
        id={id}
        type="number"
        value={String(value ?? "")}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") onChange(null);
          else {
            const n = Number(raw);
            onChange(Number.isNaN(n) ? raw : n);
          }
        }}
        className="rounded-md border border-surface-border bg-surface-1 px-2 py-1 font-mono text-sm"
        data-testid={`field-${c.path}`}
      />
    );
  }
  if (c.type === "json") {
    return (
      <textarea
        id={id}
        value={JSON.stringify(value, null, 2)}
        onChange={(e) => {
          try {
            onChange(JSON.parse(e.target.value));
          } catch {
            // Keep the literal text so the user can fix the JSON.
            onChange(e.target.value);
          }
        }}
        rows={Math.min(8, Math.max(2, JSON.stringify(value, null, 2).split("\n").length))}
        className="rounded-md border border-surface-border bg-surface-1 px-2 py-1 font-mono text-xs"
        data-testid={`field-${c.path}`}
      />
    );
  }
  return (
    <input
      id={id}
      type="text"
      value={String(value ?? "")}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-surface-border bg-surface-1 px-2 py-1 font-mono text-sm"
      data-testid={`field-${c.path}`}
    />
  );
}
