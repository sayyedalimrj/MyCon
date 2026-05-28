// Schedule Compare — the dashboard page that closes the end-to-end
// finishing target (see docs/end_to_end_finishing_plan.md Section 7).
//
// Layout (top to bottom):
//   1. KPI strip
//   2. Activities table (sortable, filterable)
//   3. Activity drilldown (mapped elements + risks + actions)
//   4. Reliability summary card (ECE / Brier from the latest calibration
//      report)
//   5. Comparison export button (downloads the dashboard JSON)
//
// We use TanStack Query for fetches; the schedule, variance, and
// calibration endpoints all exist behind /api/v1/.

import { useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { scheduleEndpoints } from "../api/scheduleEndpoints";
import { queryKeys } from "../api/queryKeys";
import type {
  ActivityVarianceRow,
  ActivityVarianceStatus,
} from "../api/scheduleTypes";
import { defaultApiClient } from "../api/client";
import { ApiError } from "../api/types";

import { KPIStrip } from "../components/KPIStrip";
import { ActivityTable } from "../components/ActivityTable";
import {
  ReliabilityCard,
  type CalibrationReportPayload,
} from "../components/ReliabilityCard";
import { HitlCorrectionForm } from "../components/HitlCorrectionForm";
import { PageHeader } from "../panels/PageHeader";

const STATUS_LABEL: Record<ActivityVarianceStatus, string> = {
  on_schedule: "On schedule",
  ahead: "Ahead",
  behind: "Behind",
  unknown_evidence: "No evidence",
};

interface DrilldownProps {
  row: ActivityVarianceRow | null;
  runId?: string;
}

function ActivityDrilldown({ row, runId }: DrilldownProps) {
  const detailQuery = useQuery({
    queryKey: queryKeys.scheduleActivityDetail(row?.activity_id ?? "", runId),
    queryFn: () =>
      scheduleEndpoints.getActivityDetail(row!.activity_id, { runId }),
    enabled: !!row,
  });
  const queryClient = useQueryClient();
  const [correctionTarget, setCorrectionTarget] = useState<string | null>(null);

  if (!row) {
    return (
      <section className="rounded-md border border-surface-border bg-surface-1 p-4 text-sm text-ink-muted">
        Select an activity in the table to inspect its mapped elements,
        risks, and recommended actions.
      </section>
    );
  }

  const detail = detailQuery.data;

  return (
    <section
      data-testid="activity-drilldown"
      className="flex flex-col gap-3 rounded-md border border-surface-border bg-surface-1 p-4"
    >
      <header className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-ink">
            {row.activity_id} — {row.activity_name}
          </h3>
          <p className="text-xs text-ink-subtle">
            {row.n_evaluated_elements} of {row.n_mapped_elements} mapped elements have evidence.
            Confidence: <span className="font-mono">{row.confidence}</span>. Status:{" "}
            <span className="font-mono">{STATUS_LABEL[row.status]}</span>.
          </p>
        </div>
        <div className="text-right text-xs tabular-nums">
          <div>
            Planned: <span className="font-semibold">{row.planned_percent_complete.toFixed(1)}%</span>
          </div>
          <div>
            Actual:{" "}
            <span className="font-semibold">{row.actual_percent_complete.toFixed(1)}%</span>{" "}
            <span className="text-ink-subtle">
              ({row.actual_percent_complete_lower_95.toFixed(1)}–
              {row.actual_percent_complete_upper_95.toFixed(1)}%)
            </span>
          </div>
          <div>
            Variance:{" "}
            <span className="font-semibold">
              {row.schedule_variance_percent >= 0 ? "+" : ""}
              {row.schedule_variance_percent.toFixed(1)} pp
            </span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div>
          <h4 className="mb-1 text-xs uppercase tracking-widest text-ink-subtle">Risks</h4>
          {row.risks.length === 0 ? (
            <p className="text-xs text-ink-muted">None reported.</p>
          ) : (
            <ul className="space-y-1 text-sm">
              {row.risks.map((r) => (
                <li key={r} className="font-mono text-amber-200">
                  • {r}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <h4 className="mb-1 text-xs uppercase tracking-widest text-ink-subtle">Mapped elements</h4>
          {detailQuery.isLoading ? (
            <p className="text-xs text-ink-muted">Loading…</p>
          ) : detail && detail.mapped_elements.length > 0 ? (
            <ul className="max-h-40 space-y-1 overflow-y-auto text-xs">
              {detail.mapped_elements.map((e) => (
                <li
                  key={e.ifc_global_id}
                  className="flex items-center justify-between rounded bg-surface-2 px-2 py-1 font-mono"
                >
                  <span>{e.ifc_global_id}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-ink-subtle">w={e.weight}</span>
                    <button
                      type="button"
                      data-testid={`hitl-open-form-${e.ifc_global_id}`}
                      onClick={() =>
                        setCorrectionTarget(
                          correctionTarget === e.ifc_global_id ? null : e.ifc_global_id,
                        )
                      }
                      className="rounded border border-surface-border px-1.5 py-0.5 text-[10px] text-ink-muted hover:bg-surface-1 hover:text-ink"
                    >
                      {correctionTarget === e.ifc_global_id ? "Cancel" : "Correct"}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-ink-muted">No mapping for this activity.</p>
          )}
        </div>
      </div>

      {correctionTarget && (
        <HitlCorrectionForm
          ifcGlobalId={correctionTarget}
          runId={runId}
          predictedValue={
            row.status === "behind" ? "uncertain" : row.status === "on_schedule" ? "accept" : "uncertain"
          }
          predictedConfidence={row.confidence}
          evidenceRefs={
            runId
              ? [`runs/${runId}/reports/element_metrics.csv`]
              : ["runs/latest/reports/element_metrics.csv"]
          }
          onSubmitted={() => {
            // Refresh the calibration card if it had loaded a 404
            // empty-state earlier; the dashboard will pull a new
            // report on the next replay run.
            queryClient.invalidateQueries({
              queryKey: ["calibration", "report"],
            });
            setCorrectionTarget(null);
          }}
        />
      )}
    </section>
  );
}

interface ScheduleCompareProps {
  /** Override for tests; in normal use the page binds to the latest run. */
  runId?: string;
}

export function ScheduleComparePage({ runId }: ScheduleCompareProps = {}) {
  const dashboardQuery = useQuery({
    queryKey: queryKeys.scheduleDashboard(runId),
    queryFn: () => scheduleEndpoints.getDashboardSummary({ runId }),
  });

  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [filter, setFilter] = useState("");

  const calibrationQuery = useQuery<CalibrationReportPayload | null>({
    queryKey: ["calibration", "report", { runId }],
    queryFn: async () => {
      try {
        return await defaultApiClient.get<CalibrationReportPayload>(
          "/v1/calibration/report" + (runId ? `?run_id=${encodeURIComponent(runId)}` : ""),
        );
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    retry: false,
  });

  const selectedRow = useMemo<ActivityVarianceRow | null>(() => {
    if (!dashboardQuery.data || !selectedId) return null;
    return (
      dashboardQuery.data.activities.find((a) => a.activity_id === selectedId) ?? null
    );
  }, [dashboardQuery.data, selectedId]);

  function handleExport() {
    if (!dashboardQuery.data) return;
    const blob = new Blob([JSON.stringify(dashboardQuery.data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `schedule_dashboard_${dashboardQuery.data.data_date_utc.slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex flex-col gap-6 px-6 py-6" data-testid="schedule-compare-page">
      <PageHeader
        title="Schedule Compare"
        subtitle="Planned vs as-built progress per activity, with calibrated confidence."
        actions={
          <button
            type="button"
            onClick={handleExport}
            disabled={!dashboardQuery.data}
            className="rounded-md border border-surface-border bg-surface-1 px-3 py-1 text-xs font-medium text-ink hover:bg-surface-2 disabled:opacity-50"
            data-testid="schedule-compare-export"
          >
            Export dashboard JSON
          </button>
        }
      />

      {dashboardQuery.isLoading && (
        <p className="text-sm text-ink-muted">Loading schedule dashboard…</p>
      )}
      {dashboardQuery.isError && (
        <p className="text-sm text-rose-300">
          Failed to load dashboard: {(dashboardQuery.error as Error).message}
        </p>
      )}

      {dashboardQuery.data && (
        <>
          <KPIStrip kpi={dashboardQuery.data.kpi} />
          <ActivityTable
            activities={dashboardQuery.data.activities}
            onSelect={setSelectedId}
            selectedActivityId={selectedId}
            filterText={filter}
            onFilterTextChange={setFilter}
          />
          <ActivityDrilldown row={selectedRow} runId={runId} />
          <ReliabilityCard
            report={
              calibrationQuery.data ?? null
            }
          />
        </>
      )}
    </div>
  );
}
