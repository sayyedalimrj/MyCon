// Reliability summary card for the Schedule Compare page.
//
// Reads the latest calibration_report.v1 (Phase 4) and shows ECE / Brier /
// smooth-ECE so the reviewer sees at a glance how trustworthy the
// confidences in the table actually are. Phase 5 adds an inline SVG
// reliability diagram (per-bin mean confidence vs empirical accuracy)
// that mirrors the standard reliability diagram from Naeini et al.,
// AAAI 2015 (binned form) and is the conventional figure expected in
// every Q1 calibration paper.
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

/**
 * One bin of a reliability diagram, mirroring the row produced by
 * ``pipeline.common.calibration.ReliabilityBin.to_dict()``.
 */
export interface ReliabilityBinPayload {
  bin_index: number;
  lower_edge: number;
  upper_edge: number;
  count: number;
  mean_confidence: number;
  empirical_accuracy: number;
  gap: number;
}

export interface CalibrationReportPayload {
  schema_version: string;
  n_samples: number;
  metrics: CalibrationMetrics;
  /**
   * Per-bin reliability table from the calibration report. Optional so
   * existing call sites keep working unchanged; the chart only renders
   * when this is supplied.
   */
  reliability_table?: ReliabilityBinPayload[];
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

// ---------------------------------------------------------------------------
// Inline SVG reliability diagram.
//
// Why hand-rolled SVG instead of a chart library:
// - The dashboard already uses Recharts, but Recharts is heavy and is not
//   loaded for the Schedule Compare page; pulling it in just for this
//   small diagram would add ~100 KB to the bundle.
// - The reliability diagram has only two visual elements (per-bin
//   bar/dot pairs and a diagonal reference line); SVG fits in 30 lines.
// - Hand-rolled SVG lets us render an accessible chart with
//   ``role="img"``, an ``aria-label`` summary, and a non-graphical
//   table fallback (already provided by the metric grid above).
// - Tests can assert visible bins via ``data-testid`` attributes
//   without needing a canvas-aware renderer.
//
// The chart is dimensionless (uses a ``viewBox``) so it scales with
// its container. Padding leaves room for the axis labels.
// ---------------------------------------------------------------------------

const CHART_VB_W = 320;
const CHART_VB_H = 200;
const PADDING = { top: 8, right: 12, bottom: 28, left: 32 };
const PLOT_W = CHART_VB_W - PADDING.left - PADDING.right;
const PLOT_H = CHART_VB_H - PADDING.top - PADDING.bottom;

function _isPopulated(bin: ReliabilityBinPayload): boolean {
  return (
    bin.count > 0 &&
    Number.isFinite(bin.mean_confidence) &&
    Number.isFinite(bin.empirical_accuracy)
  );
}

function _xToPlot(x: number): number {
  return PADDING.left + Math.max(0, Math.min(1, x)) * PLOT_W;
}

function _yToPlot(y: number): number {
  // SVG y grows downward; flip so accuracy=1 sits at the top.
  return PADDING.top + (1 - Math.max(0, Math.min(1, y))) * PLOT_H;
}

interface ReliabilityChartProps {
  bins: ReliabilityBinPayload[];
}

function ReliabilityChart({ bins }: ReliabilityChartProps) {
  const populated = bins.filter(_isPopulated);
  if (populated.length === 0) {
    return (
      <p
        data-testid="reliability-chart-empty"
        className="text-[11px] text-ink-subtle"
      >
        Reliability diagram not available: every bin in the report is
        empty. Submit more reviewer corrections to populate the chart.
      </p>
    );
  }

  // Reference diagonal (perfect calibration).
  const diagX1 = _xToPlot(0);
  const diagY1 = _yToPlot(0);
  const diagX2 = _xToPlot(1);
  const diagY2 = _yToPlot(1);

  // Bar width per bin; a bin's *count* mass scales the dot radius.
  const totalCount = populated.reduce((acc, b) => acc + b.count, 0);
  const maxRadius = 5;

  return (
    <figure
      data-testid="reliability-chart"
      className="mt-1 flex flex-col gap-1"
    >
      <svg
        role="img"
        aria-label={`Reliability diagram with ${populated.length} populated bins from ${totalCount} corrections`}
        viewBox={`0 0 ${CHART_VB_W} ${CHART_VB_H}`}
        className="w-full max-w-md text-ink-subtle"
      >
        {/* Plot area background */}
        <rect
          x={PADDING.left}
          y={PADDING.top}
          width={PLOT_W}
          height={PLOT_H}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.25}
          strokeWidth={1}
        />
        {/* Reference diagonal */}
        <line
          x1={diagX1}
          y1={diagY1}
          x2={diagX2}
          y2={diagY2}
          stroke="currentColor"
          strokeOpacity={0.4}
          strokeDasharray="4 4"
          strokeWidth={1}
          data-testid="reliability-chart-diagonal"
        />
        {/* Per-bin gap bar (vertical line from y=mean_conf to y=acc on the x=mean_conf vertical) */}
        {populated.map((bin) => {
          const cx = _xToPlot(bin.mean_confidence);
          const yConf = _yToPlot(bin.mean_confidence);
          const yAcc = _yToPlot(bin.empirical_accuracy);
          const r = Math.max(
            2,
            Math.min(maxRadius, (bin.count / Math.max(1, totalCount)) * maxRadius * 4),
          );
          const isOver = bin.mean_confidence > bin.empirical_accuracy;
          return (
            <g
              key={bin.bin_index}
              data-testid={`reliability-bin-${bin.bin_index}`}
            >
              <line
                x1={cx}
                y1={yConf}
                x2={cx}
                y2={yAcc}
                stroke="currentColor"
                strokeOpacity={isOver ? 0.7 : 0.5}
                strokeWidth={1.5}
              />
              <circle
                cx={cx}
                cy={yAcc}
                r={r}
                className={clsx(
                  isOver ? "fill-rose-300" : "fill-emerald-300",
                )}
                stroke="currentColor"
                strokeOpacity={0.6}
                strokeWidth={0.5}
              />
            </g>
          );
        })}
        {/* X-axis tick labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text
            key={`xt-${v}`}
            x={_xToPlot(v)}
            y={CHART_VB_H - PADDING.bottom + 14}
            textAnchor="middle"
            className="fill-current text-[8px]"
          >
            {v.toFixed(2)}
          </text>
        ))}
        {/* Y-axis tick labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text
            key={`yt-${v}`}
            x={PADDING.left - 4}
            y={_yToPlot(v) + 3}
            textAnchor="end"
            className="fill-current text-[8px]"
          >
            {v.toFixed(2)}
          </text>
        ))}
        {/* Axis titles */}
        <text
          x={PADDING.left + PLOT_W / 2}
          y={CHART_VB_H - 4}
          textAnchor="middle"
          className="fill-current text-[9px]"
        >
          mean confidence
        </text>
        <text
          x={8}
          y={PADDING.top + PLOT_H / 2}
          textAnchor="middle"
          transform={`rotate(-90, 8, ${PADDING.top + PLOT_H / 2})`}
          className="fill-current text-[9px]"
        >
          empirical accuracy
        </text>
      </svg>
      <figcaption className="text-[10px] text-ink-subtle">
        Reliability diagram (Naeini et&nbsp;al. AAAI&nbsp;2015). Dashed
        diagonal is perfect calibration. Dot size scales with bin
        sample count. Red dots indicate over-confidence; green dots
        indicate well-calibrated or under-confidence bins.
      </figcaption>
    </figure>
  );
}

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
      {report.reliability_table && report.reliability_table.length > 0 && (
        <ReliabilityChart bins={report.reliability_table} />
      )}
      <p className="text-[11px] text-ink-subtle">
        Computed from {report.n_samples} reviewer corrections. ECE per Naeini et&nbsp;al. AAAI&nbsp;2015; Smooth ECE per Błasiok &amp; Nakkiran ICLR&nbsp;2024.
      </p>
    </section>
  );
}
