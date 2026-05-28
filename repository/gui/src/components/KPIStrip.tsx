// Glanceable four-card KPI strip for the Schedule Compare page.
//
// The four numbers are the headline of the entire pipeline:
// planned %, actual % with 95% interval, signed variance, and the
// activity-status mix (on / behind / ahead / unknown). All of them
// come straight from the dashboard_summary.v1 KPI block.

import clsx from "clsx";

import type { DashboardSummaryKpi } from "../api/scheduleTypes";

function formatPct(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return `${value.toFixed(1)}%`;
}

function formatSignedPct(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} pp`;
}

function variancePalette(variance: number): string {
  // Use the same band semantics the backend uses: |variance| <= 5 is
  // 'on schedule', otherwise the sign decides ahead vs behind.
  if (!Number.isFinite(variance)) return "border-surface-border";
  if (Math.abs(variance) <= 5) return "border-emerald-500/60";
  return variance > 0 ? "border-sky-500/60" : "border-amber-500/60";
}

interface CardProps {
  label: string;
  value: string;
  detail?: string;
  className?: string;
  testId: string;
}

function Card({ label, value, detail, className, testId }: CardProps) {
  return (
    <div
      data-testid={testId}
      className={clsx(
        "flex min-w-0 flex-1 flex-col gap-1 rounded-md border border-surface-border bg-surface-1 px-4 py-3",
        className,
      )}
    >
      <div className="text-[11px] uppercase tracking-widest text-ink-subtle">{label}</div>
      <div className="text-2xl font-semibold tabular-nums text-ink">{value}</div>
      {detail && <div className="text-xs text-ink-muted">{detail}</div>}
    </div>
  );
}

export interface KPIStripProps {
  kpi: DashboardSummaryKpi;
}

export function KPIStrip({ kpi }: KPIStripProps) {
  const variance = kpi.variance_percent;
  const intervalDetail = `95% CI: ${formatPct(kpi.actual_lower_95)} – ${formatPct(kpi.actual_upper_95)}`;
  const statusMix = `${kpi.n_on_schedule} on • ${kpi.n_behind} behind • ${kpi.n_ahead} ahead • ${kpi.n_unknown_evidence} unknown`;

  return (
    <div
      role="region"
      aria-label="Schedule KPI strip"
      className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4"
    >
      <Card testId="kpi-planned" label="Planned" value={formatPct(kpi.planned_percent)} />
      <Card
        testId="kpi-actual"
        label="Actual"
        value={formatPct(kpi.actual_percent)}
        detail={intervalDetail}
      />
      <Card
        testId="kpi-variance"
        label="Variance"
        value={formatSignedPct(variance)}
        detail={Math.abs(variance) <= 5 ? "Within ±5 pp band" : variance > 0 ? "Ahead of plan" : "Behind plan"}
        className={variancePalette(variance)}
      />
      <Card
        testId="kpi-activities"
        label="Activities"
        value={String(kpi.n_activities)}
        detail={statusMix}
      />
    </div>
  );
}
