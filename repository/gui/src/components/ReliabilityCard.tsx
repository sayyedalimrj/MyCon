// Reliability summary card for the Schedule Compare page.
//
// Reads the latest calibration_report.v1 (Phase 4) and shows ECE / Brier /
// smooth-ECE so the reviewer sees at a glance how trustworthy the
// confidences in the table actually are.
//
// We *don't* fetch the calibration report inside this component; the
// parent passes the loaded payload as a prop so the page can decide
// whether/where to fetch it (the calibration endpoint is optional and may
// 404 on runs that have no HITL log yet).

import clsx from "clsx";

export interface CalibrationMetrics {
  expected_calibration_error: number;
  maximum_calibration_error: number;
  brier_score: number;
  smooth_ece: number;
}

export interface CalibrationReportPayload {
  schema_version: string;
  n_samples: number;
  metrics: CalibrationMetrics;
}

export interface ReliabilityCardProps {
  /** ``null`` when the calibration report is not yet available for the run. */
  report: CalibrationReportPayload | null;
}

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(3);
}

function eceVerdict(ece: number): { label: string; tone: "good" | "warn" | "bad" } {
  if (!Number.isFinite(ece)) return { label: "—", tone: "warn" };
  if (ece <= 0.05) return { label: "well calibrated", tone: "good" };
  if (ece <= 0.15) return { label: "mildly miscalibrated", tone: "warn" };
  return { label: "miscalibrated", tone: "bad" };
}

const TONE_PALETTE = {
  good: "border-emerald-500/40 text-emerald-200",
  warn: "border-amber-500/40 text-amber-200",
  bad: "border-rose-500/40 text-rose-200",
} as const;

export function ReliabilityCard({ report }: ReliabilityCardProps) {
  if (!report) {
    return (
      <section
        data-testid="reliability-card"
        className="flex flex-col gap-2 rounded-md border border-surface-border bg-surface-1 px-4 py-3"
      >
        <h3 className="text-sm font-semibold text-ink">Reliability</h3>
        <p className="text-xs text-ink-muted">
          No calibration report yet. Submit reviewer corrections via{" "}
          <code className="rounded bg-surface-2 px-1">pipeline.common.hitl</code> and run{" "}
          <code className="rounded bg-surface-2 px-1">scripts/run_calibration_report.py</code> to populate this card.
        </p>
      </section>
    );
  }

  const verdict = eceVerdict(report.metrics.expected_calibration_error);

  return (
    <section
      data-testid="reliability-card"
      className={clsx(
        "flex flex-col gap-2 rounded-md border bg-surface-1 px-4 py-3",
        TONE_PALETTE[verdict.tone],
      )}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Reliability</h3>
        <span className="text-xs uppercase tracking-widest">{verdict.label}</span>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink-muted sm:grid-cols-4">
        <div>
          <dt className="text-[10px] uppercase tracking-widest">ECE</dt>
          <dd className="text-base tabular-nums text-ink">{fmt(report.metrics.expected_calibration_error)}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-widest">MCE</dt>
          <dd className="text-base tabular-nums text-ink">{fmt(report.metrics.maximum_calibration_error)}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-widest">Brier</dt>
          <dd className="text-base tabular-nums text-ink">{fmt(report.metrics.brier_score)}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-widest">Smooth ECE</dt>
          <dd className="text-base tabular-nums text-ink">{fmt(report.metrics.smooth_ece)}</dd>
        </div>
      </dl>
      <p className="text-[11px] text-ink-subtle">
        Computed from {report.n_samples} reviewer corrections. ECE per Naeini et&nbsp;al. AAAI&nbsp;2015; Smooth ECE per Błasiok &amp; Nakkiran ICLR&nbsp;2024.
      </p>
    </section>
  );
}
