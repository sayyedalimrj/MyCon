import { describe, it, expect } from "vitest";
import { screen, waitFor } from "@testing-library/react";

import { DiffPanel } from "./DiffPanel";
import { renderWithProviders } from "../test/render";

describe("<DiffPanel />", () => {
  it("auto-picks the first two configs and renders a diff table", async () => {
    renderWithProviders(<DiffPanel />);
    await waitFor(() => {
      expect(screen.getByTestId("diff-table")).toBeInTheDocument();
    });
    // Site01 random_seed=42 vs default_server_svc4 random_seed=99 → changed.
    expect(screen.getByText("project.random_seed")).toBeInTheDocument();
    // schedule changed from "" to "schedule.csv" → changed.
    expect(screen.getByText("inputs.schedule")).toBeInTheDocument();
  });
});
