// HITL correction submission form.
//
// Embedded in the Schedule Compare drilldown panel; lets a reviewer
// submit a correction directly without leaving the dashboard. POSTs
// to /api/v1/hitl/corrections via the typed hitlEndpoints client.
//
// Design notes:
// - The form locks ``target_kind = element_acceptance`` for now (this
//   is the dominant correction kind on the Schedule Compare page).
//   Future revisions can expose a target_kind selector.
// - ``target_id`` is supplied by the parent (the IFC GlobalId of the
//   element the reviewer clicked); the input is read-only so the
//   user can see which element they are correcting.
// - We use uncontrolled-style state hooks because the form is short-
//   lived and we don't need cross-field validation. On submit we POST
//   and let TanStack Query refresh the parent list.
// - Empty ``rationale`` is allowed but discouraged; the placeholder
//   tells the reviewer why it matters.

import { useMutation } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import clsx from "clsx";

import {
  hitlEndpoints,
  type HitlCorrectionPayload,
  type HitlDecisionValue,
  type HitlPredictedConfidence,
  type HitlSubmitResponse,
} from "../api/hitlEndpoints";

const DECISION_OPTIONS: HitlDecisionValue[] = [
  "accept",
  "reject",
  "uncertain",
  "rework",
];

const CONFIDENCE_OPTIONS: HitlPredictedConfidence[] = [
  "high",
  "medium",
  "low_to_medium",
  "low",
  "unverified",
];

export interface HitlCorrectionFormProps {
  /** IFC GlobalId of the element being corrected. */
  ifcGlobalId: string;
  /** Optional run id; forwarded as ``?run_id=`` query parameter. */
  runId?: string;
  /** Pre-fill of the original predicted_value if known. */
  predictedValue?: HitlDecisionValue;
  /** Pre-fill of the original predicted_confidence if known. */
  predictedConfidence?: HitlPredictedConfidence;
  /** Optional evidence refs (e.g. [`"runs/<run>/reports/element_metrics.csv"`]). */
  evidenceRefs?: string[];
  /** Called after the server confirms the submission. */
  onSubmitted?: (response: HitlSubmitResponse) => void;
}

export function HitlCorrectionForm({
  ifcGlobalId,
  runId,
  predictedValue = "accept",
  predictedConfidence = "high",
  evidenceRefs,
  onSubmitted,
}: HitlCorrectionFormProps) {
  const [reviewerId, setReviewerId] = useState("");
  const [rationale, setRationale] = useState("");
  const [predicted, setPredicted] = useState<HitlDecisionValue>(predictedValue);
  const [predictedConf, setPredictedConf] =
    useState<HitlPredictedConfidence>(predictedConfidence);
  const [corrected, setCorrected] = useState<HitlDecisionValue>("reject");
  const [lastResponse, setLastResponse] = useState<HitlSubmitResponse | null>(null);

  const mutation = useMutation({
    mutationFn: async (payload: HitlCorrectionPayload) =>
      hitlEndpoints.submitCorrection(payload, { runId }),
    onSuccess: (resp) => {
      setLastResponse(resp);
      onSubmitted?.(resp);
      // Reset only the volatile fields; reviewer_id is sticky for
      // multi-correction sessions.
      setRationale("");
      setCorrected("reject");
    },
  });

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!reviewerId.trim()) return;
    const payload: HitlCorrectionPayload = {
      target_kind: "element_acceptance",
      target_id: ifcGlobalId,
      predicted_value: predicted,
      predicted_confidence: predictedConf,
      corrected_value: corrected,
      reviewer_id: reviewerId.trim(),
      rationale: rationale.trim(),
      ...(evidenceRefs && evidenceRefs.length > 0
        ? { evidence_refs: evidenceRefs }
        : {}),
      ...(runId ? { run_id: runId } : {}),
    };
    mutation.mutate(payload);
  }

  const submitDisabled = mutation.isPending || !reviewerId.trim();

  return (
    <form
      onSubmit={handleSubmit}
      data-testid="hitl-correction-form"
      className="flex flex-col gap-2 rounded-md border border-surface-border bg-surface-1 p-3 text-sm"
    >
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-widest text-ink-subtle">
          Submit correction
        </h4>
        <span
          className="text-[10px] text-ink-subtle"
          data-testid="hitl-correction-target-id"
        >
          {ifcGlobalId}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <label className="flex flex-col gap-0.5 text-xs">
          Reviewer id
          <input
            type="text"
            data-testid="hitl-reviewer-id"
            value={reviewerId}
            onChange={(e) => setReviewerId(e.target.value)}
            placeholder="alice@example.com"
            className="rounded-sm border border-surface-border bg-surface-2 px-2 py-1 text-sm text-ink"
            required
          />
        </label>
        <label className="flex flex-col gap-0.5 text-xs">
          Predicted (was)
          <select
            data-testid="hitl-predicted-value"
            value={predicted}
            onChange={(e) => setPredicted(e.target.value as HitlDecisionValue)}
            className="rounded-sm border border-surface-border bg-surface-2 px-2 py-1 text-sm text-ink"
          >
            {DECISION_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-0.5 text-xs">
          Predicted confidence
          <select
            data-testid="hitl-predicted-confidence"
            value={predictedConf}
            onChange={(e) =>
              setPredictedConf(e.target.value as HitlPredictedConfidence)
            }
            className="rounded-sm border border-surface-border bg-surface-2 px-2 py-1 text-sm text-ink"
          >
            {CONFIDENCE_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-0.5 text-xs">
          Corrected (now)
          <select
            data-testid="hitl-corrected-value"
            value={corrected}
            onChange={(e) => setCorrected(e.target.value as HitlDecisionValue)}
            className="rounded-sm border border-surface-border bg-surface-2 px-2 py-1 text-sm text-ink"
          >
            {DECISION_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label className="flex flex-col gap-0.5 text-xs">
        Rationale
        <textarea
          data-testid="hitl-rationale"
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          rows={2}
          placeholder="Why are you overruling the pipeline? (cited in the audit log; cf. Beck WACV'24)"
          className="rounded-sm border border-surface-border bg-surface-2 px-2 py-1 text-sm text-ink"
        />
      </label>

      <div className="flex items-center justify-between">
        <button
          type="submit"
          data-testid="hitl-submit-button"
          disabled={submitDisabled}
          className={clsx(
            "rounded-md border px-3 py-1 text-xs font-medium",
            submitDisabled
              ? "cursor-not-allowed border-surface-border text-ink-subtle"
              : "border-emerald-500/60 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20",
          )}
        >
          {mutation.isPending ? "Submitting…" : "Submit correction"}
        </button>
        {mutation.isError && (
          <span
            data-testid="hitl-error"
            className="text-xs text-rose-300"
          >
            Failed: {(mutation.error as Error).message}
          </span>
        )}
        {lastResponse && (
          <span
            data-testid="hitl-success"
            className="text-xs text-emerald-200"
          >
            Stored: {lastResponse.correction.record_id}
          </span>
        )}
      </div>
    </form>
  );
}
