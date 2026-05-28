import { describe, it, expect } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ConfigEditorPanel } from "./ConfigEditorPanel";
import { renderWithProviders } from "../test/render";

describe("<ConfigEditorPanel />", () => {
  it("renders typed controls for every leaf in the schema", async () => {
    renderWithProviders(<ConfigEditorPanel />, {
      initialEntries: ["/configs/site01"],
      routePath: "/configs/:configName",
    });

    await screen.findByTestId("field-keyframes.min_time_gap_sec");
    expect(screen.getByTestId("field-keyframes.min_time_gap_sec")).toHaveValue(0.5);
    expect(screen.getByTestId("field-keyframes.max_frames_first_run")).toHaveValue(800);
    expect(screen.getByTestId("field-project.name")).toHaveValue("site01");
  });

  it("marks edits and resets to server values on demand", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ConfigEditorPanel />, {
      initialEntries: ["/configs/site01"],
      routePath: "/configs/:configName",
    });

    const field = await screen.findByTestId("field-keyframes.min_time_gap_sec");
    await user.clear(field);
    await user.type(field, "1.5");

    const reset = screen.getByTestId("reset-edits");
    await waitFor(() => expect(reset).not.toBeDisabled());
    await user.click(reset);

    expect(screen.getByTestId("field-keyframes.min_time_gap_sec")).toHaveValue(0.5);
  });
});
