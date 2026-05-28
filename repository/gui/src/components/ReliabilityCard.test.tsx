// Unit tests for the Phase 5 reliability per-bin chart.
//
// We render the card directly (no MSW; the prop carries the report
// payload verbatim) and assert the three rendering states:
//   - report=null         -> empty-state guidance
//   - report without table -> headline metrics only (legacy path)
//   - report with table    -> SVG chart with one element per populated bin

import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";

import {
  CalibrationReportPayload,
  ReliabilityBinPayload,
  ReliabilityCard,
} from "./ReliabilityCard";

const _METRICS = {
  expected_calibration_error: 0.07,
  maximum_calibration_error: 0.18,
  brier_score: 0.16,
  smooth_ece: 0.06,
};

function _bin(
  bin_index: number,
  count: number,
  mean_confidence: number,
  empirical_accuracy: number,
): ReliabilityBinPayload {
  return {
    bin_index,
    lower_edge: bin_index / 5,
    upper_edge: (bin_index + 1) / 5,
    count,
    mean_confidence,
    empirical_accuracy,
    gap: Math.abs(mean_confidence - empirical_accuracy),
  };
}

describe("ReliabilityCard", () => {
  it("renders the empty-state guidance when report is null", () => {
    render(<ReliabilityCard report={null} />);
    const card = screen.getByTestId("reliability-card");
    expect(card.textContent).toMatch(/No calibration report yet/i);
    expect(screen.queryByTestId("reliability-chart")).toBeNull();
  });

  it("renders the four headline metrics when report has no table", () => {
    const report: CalibrationReportPayload = {
      schema_version: "calibration_report.v1",
      n_samples: 73,
      metrics: _METRICS,
    };
    render(<ReliabilityCard report={report} />);
    const card = screen.getByTestId("reliability-card");
    // Each metric label is present.
    expect(within(card).getByText(/^ECE$/)).toBeInTheDocument();
    expect(within(card).getByText("MCE")).toBeInTheDocument();
    expect(within(card).getByText("Brier")).toBeInTheDocument();
    expect(within(card).getByText("Smooth ECE")).toBeInTheDocument();
    // Verdict tone is "mildly miscalibrated" for ECE=0.07.
    expect(within(card).getByText(/mildly miscalibrated/i)).toBeInTheDocument();
    // No chart when reliability_table is absent.
    expect(screen.queryByTestId("reliability-chart")).toBeNull();
  });

  it("renders an SVG chart with one node per populated bin", () => {
    const report: CalibrationReportPayload = {
      schema_version: "calibration_report.v1",
      n_samples: 100,
      metrics: _METRICS,
      reliability_table: [
        _bin(0, 12, 0.20, 0.18),
        _bin(1, 25, 0.40, 0.45), // under-confident
        _bin(2, 30, 0.60, 0.50), // over-confident
        _bin(3, 0, NaN, NaN),    // empty bin -> filtered
        _bin(4, 33, 0.85, 0.84),
      ],
    };
    render(<ReliabilityCard report={report} />);
    const chart = screen.getByTestId("reliability-chart");
    expect(chart).toBeInTheDocument();
    // Diagonal reference line is rendered.
    expect(within(chart).getByTestId("reliability-chart-diagonal")).toBeInTheDocument();
    // Four populated bins are rendered (bin_index 0, 1, 2, 4).
    expect(within(chart).getByTestId("reliability-bin-0")).toBeInTheDocument();
    expect(within(chart).getByTestId("reliability-bin-1")).toBeInTheDocument();
    expect(within(chart).getByTestId("reliability-bin-2")).toBeInTheDocument();
    expect(within(chart).queryByTestId("reliability-bin-3")).toBeNull(); // empty
    expect(within(chart).getByTestId("reliability-bin-4")).toBeInTheDocument();
    // The aria-label summarises the populated bin and total counts.
    const svg = chart.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg!.getAttribute("aria-label")).toMatch(
      /4 populated bins from 100 corrections/,
    );
  });

  it("renders the empty-bins fallback message when every bin is empty", () => {
    const report: CalibrationReportPayload = {
      schema_version: "calibration_report.v1",
      n_samples: 0,
      metrics: _METRICS,
      reliability_table: [_bin(0, 0, NaN, NaN), _bin(1, 0, NaN, NaN)],
    };
    render(<ReliabilityCard report={report} />);
    const empty = screen.getByTestId("reliability-chart-empty");
    expect(empty.textContent).toMatch(/Reliability diagram not available/i);
    // No populated chart when every bin is empty.
    expect(screen.queryByTestId("reliability-chart")).toBeNull();
  });

  it("shows the well-calibrated tone when ECE is below 0.05", () => {
    const report: CalibrationReportPayload = {
      schema_version: "calibration_report.v1",
      n_samples: 50,
      metrics: { ..._METRICS, expected_calibration_error: 0.02 },
    };
    render(<ReliabilityCard report={report} />);
    expect(screen.getByText(/well calibrated/i)).toBeInTheDocument();
  });

  it("shows the miscalibrated tone when ECE exceeds 0.15", () => {
    const report: CalibrationReportPayload = {
      schema_version: "calibration_report.v1",
      n_samples: 50,
      metrics: { ..._METRICS, expected_calibration_error: 0.30 },
    };
    render(<ReliabilityCard report={report} />);
    expect(screen.getByText("miscalibrated")).toBeInTheDocument();
  });
});
