import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

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

// VLM control panel.
//
// Reads the live registry of VLM backends so backend names, descriptions,
// and the `online` flag come from the actual registry — never invented.
// The prompt editor & sampler controls are local-only: there is no
// `POST /api/vlm/run` endpoint yet (Phase 4 will add it). Submitting
// today produces a JSON envelope the user can copy.

export function VlmPanel() {
  const backends = useQuery({
    queryKey: queryKeys.vlmBackends(),
    queryFn: endpoints.listVlmBackends,
    staleTime: 60_000,
  });
  const depth = useQuery({
    queryKey: queryKeys.depthProviders(),
    queryFn: endpoints.listDepthProviders,
    staleTime: 60_000,
  });

  const [backend, setBackend] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState<string>(
    "You are a construction-progress analyst inspecting BIM-aligned imagery. Cite evidence by frame id.",
  );
  const [userPrompt, setUserPrompt] = useState<string>("");
  const [temperature, setTemperature] = useState<number>(0.2);
  const [topP, setTopP] = useState<number>(0.9);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [retries, setRetries] = useState<number>(2);
  const [history, setHistory] = useState<string[]>([]);

  const buildEnvelope = () => ({
    backend,
    system_prompt: systemPrompt,
    user_prompt: userPrompt,
    parameters: {
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      retries,
    },
  });

  return (
    <div>
      <PageHeader
        title="VLM control"
        subtitle="Pick a backend, draft prompts, set sampling controls. Execution lands in Phase 4."
        actions={
          <Badge tone="info">
            {(backends.data?.length ?? 0)} backends · {(depth.data?.length ?? 0)} depth providers
          </Badge>
        }
      />

      <div className="grid gap-6 p-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader
            title="Backend selection"
            subtitle="From /api/registry/vlm-backends"
          />
          {backends.isLoading && <Spinner />}
          {backends.isError && (
            <Notice tone="err">{(backends.error as Error).message}</Notice>
          )}
          <ul className="space-y-2">
            {(backends.data ?? []).map((b) => (
              <li key={b.name}>
                <label
                  className="flex cursor-pointer items-start gap-2 rounded-md border border-surface-border bg-surface-1 p-2 hover:bg-surface-2"
                >
                  <input
                    type="radio"
                    name="vlm-backend"
                    value={b.name}
                    checked={backend === b.name}
                    onChange={() => setBackend(b.name)}
                    className="mt-1"
                    data-testid={`vlm-backend-${b.name}`}
                  />
                  <div className="flex-1">
                    <div className="flex flex-wrap items-center gap-1">
                      <span className="font-medium text-ink">{b.name}</span>
                      {b.capabilities.map((c) => (
                        <Badge key={c} tone="neutral">
                          {c}
                        </Badge>
                      ))}
                    </div>
                    <p className="text-xs text-ink-muted">{b.description}</p>
                  </div>
                </label>
              </li>
            ))}
            {!backends.isLoading && (backends.data?.length ?? 0) === 0 && (
              <Empty title="No VLM backends registered" />
            )}
          </ul>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader
            title="Prompt"
            subtitle="System and user prompts, plus sampling controls."
            right={
              <Badge tone="neutral">{backend || "no backend"}</Badge>
            }
          />
          <div className="space-y-3">
            <PromptArea
              label="System prompt"
              value={systemPrompt}
              onChange={setSystemPrompt}
              testId="vlm-system-prompt"
            />
            <PromptArea
              label="User prompt"
              value={userPrompt}
              onChange={setUserPrompt}
              testId="vlm-user-prompt"
              placeholder="e.g. 'Did the south stairwell handrail get installed between frames 2034 and 2210?'"
            />
            <div className="grid gap-3 md:grid-cols-4">
              <NumberInput label="temperature" value={temperature} onChange={setTemperature} step={0.05} min={0} max={2} testId="vlm-temp" />
              <NumberInput label="top_p" value={topP} onChange={setTopP} step={0.05} min={0} max={1} testId="vlm-topp" />
              <NumberInput label="max_tokens" value={maxTokens} onChange={setMaxTokens} step={32} min={1} max={8192} testId="vlm-maxtok" />
              <NumberInput label="retries" value={retries} onChange={setRetries} step={1} min={0} max={5} testId="vlm-retries" />
            </div>
          </div>

          <Notice tone="info" title="Execution lands in Phase 4">
            The Phase 2 backend exposes the registry but does not yet route a
            request to the backend chosen here. Pressing the button below adds
            the prompt envelope to the local history so you can copy it into
            other tools.
          </Notice>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              disabled={!backend || !userPrompt.trim()}
              onClick={() => {
                const envelope = buildEnvelope();
                setHistory((h) => [JSON.stringify(envelope, null, 2), ...h].slice(0, 20));
              }}
              data-testid="vlm-submit"
            >
              Save to local history
            </Button>
            <Button
              variant="ghost"
              onClick={() => setHistory([])}
              disabled={history.length === 0}
            >
              Clear history
            </Button>
          </div>

          {history.length > 0 && (
            <details className="mt-4">
              <summary className="cursor-pointer text-xs text-ink-muted hover:text-ink">
                Prompt history ({history.length})
              </summary>
              <ul className="mt-2 space-y-2">
                {history.map((h, i) => (
                  <li
                    key={i}
                    className="rounded-md border border-surface-border bg-surface-0 p-2"
                  >
                    <pre className="font-mono text-[11px]">{h}</pre>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </Card>
      </div>

      {depth.data && depth.data.length > 0 && (
        <div className="px-6 pb-8">
          <Card>
            <CardHeader
              title="Depth providers"
              subtitle="From /api/registry/depth-providers — paired with VLM at Stage 6."
            />
            <ul className="grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
              {depth.data.map((p) => (
                <li
                  key={p.name}
                  className="rounded-md border border-surface-border bg-surface-1 p-2"
                >
                  <div className="flex flex-wrap items-center gap-1">
                    <Code>{p.name}</Code>
                    {p.capabilities.map((c) => (
                      <Badge key={c} tone="neutral">
                        {c}
                      </Badge>
                    ))}
                  </div>
                  <p className="mt-1 text-xs text-ink-muted">{p.description}</p>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  );
}

function PromptArea({
  label,
  value,
  onChange,
  testId,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  testId: string;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wider text-ink-muted">
        {label}
      </span>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        placeholder={placeholder}
        className="w-full rounded-md border border-surface-border bg-surface-1 px-3 py-2 font-mono text-xs"
        data-testid={testId}
      />
    </label>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  step,
  min,
  max,
  testId,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step: number;
  min: number;
  max: number;
  testId: string;
}) {
  return (
    <label className="block text-sm">
      <span className="mb-1 block text-xs uppercase tracking-wider text-ink-muted">
        {label}
      </span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const n = Number(e.target.value);
          onChange(Number.isNaN(n) ? value : n);
        }}
        className="w-full rounded-md border border-surface-border bg-surface-1 px-2 py-1 font-mono text-sm"
        data-testid={testId}
      />
    </label>
  );
}
