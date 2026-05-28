// Sortable, filterable activities table for Schedule Compare.
//
// Renders one row per activity row from dashboard_summary.activities,
// colour-codes the status / confidence cells, and lets the parent
// component react to row clicks (drilldown panel) via onSelect.

import { useMemo, useState } from "react";
import clsx from "clsx";

import type {
  ActivityVarianceRow,
  ActivityVarianceStatus,
  ConfidenceLabel,
} from "../api/scheduleTypes";

const STATUS_LABEL: Record<ActivityVarianceStatus, string> = {
  on_schedule: "On schedule",
  ahead: "Ahead",
  behind: "Behind",
  unknown_evidence: "No evidence",
};

const STATUS_PALETTE: Record<ActivityVarianceStatus, string> = {
  on_schedule: "bg-emerald-500/15 text-emerald-200 border-emerald-500/40",
  ahead: "bg-sky-500/15 text-sky-200 border-sky-500/40",
  behind: "bg-amber-500/15 text-amber-200 border-amber-500/40",
  unknown_evidence: "bg-surface-2 text-ink-subtle border-surface-border",
};

const CONFIDENCE_PALETTE: Record<ConfidenceLabel, string> = {
  high: "bg-emerald-500/15 text-emerald-200 border-emerald-500/40",
  medium: "bg-amber-500/15 text-amber-200 border-amber-500/40",
  low: "bg-rose-500/15 text-rose-200 border-rose-500/40",
};

type SortKey =
  | "activity_id"
  | "planned_percent_complete"
  | "actual_percent_complete"
  | "schedule_variance_percent"
  | "status"
  | "confidence";

interface SortState {
  key: SortKey;
  dir: "asc" | "desc";
}

export interface ActivityTableProps {
  activities: ActivityVarianceRow[];
  onSelect?: (activityId: string) => void;
  selectedActivityId?: string;
  filterText?: string;
  onFilterTextChange?: (value: string) => void;
}

function compare(a: number | string, b: number | string): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

export function ActivityTable({
  activities,
  onSelect,
  selectedActivityId,
  filterText,
  onFilterTextChange,
}: ActivityTableProps) {
  const [sort, setSort] = useState<SortState>({ key: "activity_id", dir: "asc" });
  const [internalFilter, setInternalFilter] = useState("");
  const filterValue = filterText ?? internalFilter;

  const filtered = useMemo(() => {
    if (!filterValue) return activities;
    const needle = filterValue.toLowerCase();
    return activities.filter((row) =>
      [row.activity_id, row.activity_name, row.status]
        .map((s) => String(s).toLowerCase())
        .some((s) => s.includes(needle)),
    );
  }, [activities, filterValue]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = a[sort.key];
      const vb = b[sort.key];
      const cmp = compare(va as number | string, vb as number | string);
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [filtered, sort]);

  function toggleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key ? { key, dir: prev.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" },
    );
  }

  function handleFilter(value: string) {
    if (onFilterTextChange) onFilterTextChange(value);
    else setInternalFilter(value);
  }

  return (
    <section className="flex flex-col gap-3" data-testid="activity-table">
      <div className="flex items-center gap-2">
        <input
          type="text"
          aria-label="Filter activities"
          placeholder="Filter by id, name, status…"
          value={filterValue}
          onChange={(e) => handleFilter(e.target.value)}
          className="w-full max-w-sm rounded-md border border-surface-border bg-surface-1 px-3 py-1 text-sm text-ink placeholder:text-ink-subtle focus:border-accent focus:outline-none"
        />
        <span className="text-xs text-ink-muted">
          {sorted.length} of {activities.length}
        </span>
      </div>
      <div className="overflow-x-auto rounded-md border border-surface-border">
        <table className="min-w-full text-sm">
          <thead className="bg-surface-1 text-left text-[11px] uppercase tracking-widest text-ink-subtle">
            <tr>
              {(
                [
                  ["activity_id", "Activity"],
                  ["planned_percent_complete", "Planned"],
                  ["actual_percent_complete", "Actual"],
                  ["schedule_variance_percent", "Variance"],
                  ["status", "Status"],
                  ["confidence", "Confidence"],
                ] as Array<[SortKey, string]>
              ).map(([key, label]) => (
                <th
                  key={key}
                  scope="col"
                  className="cursor-pointer px-3 py-2 hover:text-ink"
                  onClick={() => toggleSort(key)}
                  aria-sort={
                    sort.key === key
                      ? sort.dir === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  {label}
                  {sort.key === key && <span className="ml-1">{sort.dir === "asc" ? "▲" : "▼"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-6 text-center text-ink-subtle">
                  No activities match the current filter.
                </td>
              </tr>
            ) : (
              sorted.map((row) => {
                const isSelected = selectedActivityId === row.activity_id;
                return (
                  <tr
                    key={row.activity_id}
                    onClick={() => onSelect?.(row.activity_id)}
                    className={clsx(
                      "cursor-pointer border-t border-surface-border transition",
                      isSelected ? "bg-surface-2" : "hover:bg-surface-2/60",
                    )}
                    data-testid={`activity-row-${row.activity_id}`}
                  >
                    <td className="px-3 py-2 font-medium text-ink">
                      <div>{row.activity_id}</div>
                      <div className="text-[11px] text-ink-subtle">{row.activity_name}</div>
                    </td>
                    <td className="px-3 py-2 tabular-nums">{row.planned_percent_complete.toFixed(1)}%</td>
                    <td className="px-3 py-2 tabular-nums">
                      <div>{row.actual_percent_complete.toFixed(1)}%</div>
                      <div className="text-[11px] text-ink-subtle">
                        {row.actual_percent_complete_lower_95.toFixed(1)}–
                        {row.actual_percent_complete_upper_95.toFixed(1)}%
                      </div>
                    </td>
                    <td className={clsx("px-3 py-2 tabular-nums", row.schedule_variance_percent < -5 ? "text-amber-200" : row.schedule_variance_percent > 5 ? "text-sky-200" : "text-ink")}>
                      {row.schedule_variance_percent >= 0 ? "+" : ""}
                      {row.schedule_variance_percent.toFixed(1)} pp
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={clsx(
                          "rounded-full border px-2 py-0.5 text-xs",
                          STATUS_PALETTE[row.status],
                        )}
                      >
                        {STATUS_LABEL[row.status]}
                      </span>
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={clsx(
                          "rounded-full border px-2 py-0.5 text-xs",
                          CONFIDENCE_PALETTE[row.confidence],
                        )}
                      >
                        {row.confidence}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
