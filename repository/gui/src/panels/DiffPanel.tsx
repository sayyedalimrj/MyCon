import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

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
import { diffJson, summarizeDiff } from "../lib/diff";
import { shortHash } from "../lib/format";

// Config diff viewer.
//
// Loads two configs from /api/configs/{name} and produces a flat diff
// using `diffJson` (unit-tested). Rollback lands in Phase 4 alongside
// the config-write API.

export function DiffPanel() {
  const configs = useQuery({
    queryKey: queryKeys.configs(),
    queryFn: endpoints.listConfigs,
  });

  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");

  useEffect(() => {
    if (configs.data && configs.data.length >= 2) {
      if (!left) setLeft(configs.data[0].name);
      if (!right) setRight(configs.data[1].name);
    }
  }, [configs.data, left, right]);

  return (
    <div>
      <PageHeader
        title="Config diff"
        subtitle="Pick any two YAML configs; the panel walks the parsed JSON tree and shows every additive, removed, and changed key."
      />
      <div className="space-y-6 p-6">
        <div className="grid gap-4 md:grid-cols-2">
          <ConfigPicker
            label="left"
            value={left}
            onChange={setLeft}
            options={(configs.data ?? []).map((c) => c.name)}
            testId="diff-left"
          />
          <ConfigPicker
            label="right"
            value={right}
            onChange={setRight}
            options={(configs.data ?? []).map((c) => c.name)}
            testId="diff-right"
          />
        </div>
        {left && right && <DiffBody left={left} right={right} />}
        {!left || !right ? <Empty title="Pick two configs to diff" /> : null}
      </div>
    </div>
  );
}

function ConfigPicker({
  label,
  value,
  onChange,
  options,
  testId,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  testId: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wider text-ink-muted">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        data-testid={testId}
        className="w-full rounded-md border border-surface-border bg-surface-1 px-3 py-2 text-sm"
      >
        <option value="">— Choose a config —</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function DiffBody({ left, right }: { left: string; right: string }) {
  const a = useQuery({ queryKey: queryKeys.config(left), queryFn: () => endpoints.getConfig(left) });
  const b = useQuery({ queryKey: queryKeys.config(right), queryFn: () => endpoints.getConfig(right) });

  const entries = useMemo(() => {
    if (!a.data || !b.data) return [];
    return diffJson(a.data.data, b.data.data);
  }, [a.data, b.data]);
  const summary = useMemo(() => summarizeDiff(entries), [entries]);

  if (a.isLoading || b.isLoading) {
    return (
      <div className="flex items-center gap-2 text-ink-muted">
        <Spinner /> Loading configs…
      </div>
    );
  }
  if (a.isError) {
    return <Notice tone="err">Left side: {(a.error as Error).message}</Notice>;
  }
  if (b.isError) {
    return <Notice tone="err">Right side: {(b.error as Error).message}</Notice>;
  }

  return (
    <Card>
      <CardHeader
        title={`Diff · ${entries.length} change${entries.length === 1 ? "" : "s"}`}
        subtitle={
          <>
            <Code>{left}</Code> ({shortHash(a.data?.config_hash)}) &nbsp;↔&nbsp;{" "}
            <Code>{right}</Code> ({shortHash(b.data?.config_hash)})
          </>
        }
        right={
          <span className="flex items-center gap-2 text-xs">
            <Badge tone="ok">+{summary.added} added</Badge>
            <Badge tone="err">-{summary.removed} removed</Badge>
            <Badge tone="warn">~{summary.changed} changed</Badge>
          </span>
        }
      />

      {entries.length === 0 ? (
        <Empty title="No structural differences" hint="The two configs are identical." />
      ) : (
        <div className="-mx-4 overflow-x-auto" data-testid="diff-table">
          <table className="min-w-full text-sm">
            <thead className="bg-surface-2 text-left text-[11px] uppercase tracking-wider text-ink-muted">
              <tr>
                <th className="px-4 py-2">Path</th>
                <th className="px-4 py-2">Op</th>
                <th className="px-4 py-2">Left</th>
                <th className="px-4 py-2">Right</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.path} className="border-t border-surface-border align-top">
                  <td className="px-4 py-1 font-mono text-xs">{e.path}</td>
                  <td className="px-4 py-1">
                    <Badge
                      tone={
                        e.op === "added" ? "ok" : e.op === "removed" ? "err" : "warn"
                      }
                    >
                      {e.op}
                    </Badge>
                  </td>
                  <td className="px-4 py-1 font-mono text-xs">
                    {renderValue(e.left)}
                  </td>
                  <td className="px-4 py-1 font-mono text-xs">
                    {renderValue(e.right)}
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

function renderValue(v: unknown): string {
  if (v === undefined) return "—";
  if (v === null) return "null";
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
