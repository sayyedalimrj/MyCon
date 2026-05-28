import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";

import { PipelinePanel } from "./PipelinePanel";
import { renderWithProviders } from "../test/render";

describe("<PipelinePanel />", () => {
  it("renders every stage from the registry", async () => {
    renderWithProviders(<PipelinePanel />);
    expect(await screen.findByText("Video ingest and normalization")).toBeInTheDocument();
    expect(screen.getByText("Adaptive keyframe selection")).toBeInTheDocument();
    expect(screen.getByText("COLMAP sparse reconstruction")).toBeInTheDocument();
  });

  it("paints stage status from the latest run", async () => {
    renderWithProviders(<PipelinePanel />);
    // Latest fixture run has stage_01 completed.
    await screen.findByText("Video ingest and normalization");
    // There should be at least one badge labelled "completed".
    const completedBadges = screen.getAllByText("completed");
    expect(completedBadges.length).toBeGreaterThan(0);
  });
});
