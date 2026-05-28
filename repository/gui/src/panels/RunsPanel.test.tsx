import { describe, it, expect } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { RunsPanel } from "./RunsPanel";
import { renderWithProviders } from "../test/render";

describe("<RunsPanel />", () => {
  it("disables submit until a config and a stage are chosen", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RunsPanel />);

    const submit = await screen.findByTestId("submit-run");
    expect(submit).toBeDisabled();

    // Wait for the option to appear before selecting; the configs query is async.
    const option = await screen.findByRole("option", { name: "site01" });
    const select = (await screen.findByTestId("config-select")) as HTMLSelectElement;
    await user.selectOptions(select, option);
    expect(submit).toBeDisabled(); // still no stages

    const stageBox = await screen.findByTestId("stage-checkbox-stage_01_ingest");
    await user.click(stageBox);

    expect(submit).not.toBeDisabled();
  });

  it("submits a run and surfaces the new run id", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RunsPanel />);

    // Wait for the config option to appear before trying to select it.
    const option = await screen.findByRole("option", { name: "site01" });
    const select = (await screen.findByTestId("config-select")) as HTMLSelectElement;
    await user.selectOptions(select, option);

    await user.click(await screen.findByTestId("stage-checkbox-stage_01_ingest"));
    await user.click(await screen.findByTestId("submit-run"));

    await waitFor(() => {
      expect(screen.getByText(/Run accepted/i)).toBeInTheDocument();
    });
  });

  it("shows a cancel button only for active runs in history", async () => {
    renderWithProviders(<RunsPanel />);
    await screen.findByText("run-001");
    // run-002 is "running" → cancel button present.
    expect(screen.getByTestId("cancel-run-002")).toBeInTheDocument();
    // run-001 is "completed" → no cancel button.
    expect(screen.queryByTestId("cancel-run-001")).toBeNull();
  });
});
