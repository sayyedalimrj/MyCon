// Smoke test for the Schedule Compare page.
//
// Exercises the typed schedule API client + the KPI strip + the
// activities table. Uses MSW handlers from gui/src/test/msw/handlers.ts.

import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithProviders } from "../test/render";
import { ScheduleComparePage } from "./ScheduleCompare";

beforeAll(() => {
  // jsdom doesn't implement URL.createObjectURL.
  // Avoid breaking the export button test.
  if (!URL.createObjectURL) {
    URL.createObjectURL = vi.fn(() => "blob:fake") as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn() as unknown as typeof URL.revokeObjectURL;
  }
});

describe("ScheduleComparePage", () => {
  it("renders the four KPI cards from the dashboard summary", async () => {
    renderWithProviders(<ScheduleComparePage />);
    await waitFor(() =>
      expect(screen.getByTestId("schedule-compare-page")).toBeInTheDocument(),
    );
    await waitFor(() => expect(screen.getByTestId("kpi-planned")).toBeInTheDocument());
    expect(screen.getByTestId("kpi-planned").textContent).toContain("75.0%");
    expect(screen.getByTestId("kpi-actual").textContent).toContain("62.5%");
    expect(screen.getByTestId("kpi-actual").textContent).toContain("45.0% – 75.0%");
    expect(screen.getByTestId("kpi-variance").textContent).toContain("-12.5 pp");
    expect(screen.getByTestId("kpi-activities").textContent).toContain("2");
  });

  it("renders one row per activity and shows the status badges", async () => {
    renderWithProviders(<ScheduleComparePage />);
    const a432 = await screen.findByTestId("activity-row-A0432");
    expect(a432).toBeInTheDocument();
    expect(within(a432).getByText(/Floor 2 Zone B walls/i)).toBeInTheDocument();
    expect(within(a432).getByText(/Behind/)).toBeInTheDocument();
    expect(within(a432).getByText(/medium/)).toBeInTheDocument();
  });

  it("filters the activities table when the user types in the filter box", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ScheduleComparePage />);
    await screen.findByTestId("activity-row-A0432");
    const filter = screen.getByLabelText("Filter activities");
    await user.type(filter, "Foundations");
    expect(screen.getByTestId("activity-row-A0001")).toBeInTheDocument();
    expect(screen.queryByTestId("activity-row-A0432")).not.toBeInTheDocument();
  });

  it("opens the drilldown panel with risks and mapped elements when a row is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ScheduleComparePage />);
    const a432 = await screen.findByTestId("activity-row-A0432");
    await user.click(a432);
    const drilldown = await screen.findByTestId("activity-drilldown");
    expect(within(drilldown).getByText(/A0432 — Floor 2 Zone B walls/)).toBeInTheDocument();
    expect(within(drilldown).getByText(/schedule_behind/)).toBeInTheDocument();
    await waitFor(() =>
      expect(within(drilldown).getByText(/A0432-elem-1/)).toBeInTheDocument(),
    );
  });

  it("shows the empty-state reliability card when no calibration report is available", async () => {
    renderWithProviders(<ScheduleComparePage />);
    const card = await screen.findByTestId("reliability-card");
    expect(card.textContent).toMatch(/No calibration report yet/i);
  });

  it("download button is enabled once the dashboard payload arrives", async () => {
    renderWithProviders(<ScheduleComparePage />);
    const btn = await screen.findByTestId("schedule-compare-export");
    await waitFor(() => expect(btn).not.toBeDisabled());
  });
});
