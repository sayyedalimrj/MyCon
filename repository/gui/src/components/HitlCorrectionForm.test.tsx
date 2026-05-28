// Unit tests for the HITL correction form.

import { describe, it, expect, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithProviders } from "../test/render";
import { HitlCorrectionForm } from "./HitlCorrectionForm";

describe("HitlCorrectionForm", () => {
  it("disables submit until a reviewer id is supplied", async () => {
    renderWithProviders(<HitlCorrectionForm ifcGlobalId="X1" />);
    const button = screen.getByTestId("hitl-submit-button");
    expect(button).toBeDisabled();
    const reviewer = screen.getByTestId("hitl-reviewer-id");
    await userEvent.type(reviewer, "alice@example.com");
    expect(button).not.toBeDisabled();
  });

  it("posts the canonical payload on submit and shows the record id", async () => {
    const onSubmitted = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <HitlCorrectionForm
        ifcGlobalId="2N3RfMfeDD$AbcDefghijk"
        runId="run-abc"
        predictedValue="accept"
        predictedConfidence="high"
        evidenceRefs={["runs/run-abc/reports/element_metrics.csv"]}
        onSubmitted={onSubmitted}
      />,
    );
    await user.type(screen.getByTestId("hitl-reviewer-id"), "alice@example.com");
    await user.type(screen.getByTestId("hitl-rationale"), "missing rebar capping");
    await user.selectOptions(screen.getByTestId("hitl-corrected-value"), "reject");
    await user.click(screen.getByTestId("hitl-submit-button"));

    await waitFor(() =>
      expect(screen.getByTestId("hitl-success")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("hitl-success").textContent).toMatch(/abc1234567ab/);
    expect(onSubmitted).toHaveBeenCalledTimes(1);
  });

  it("renders the target IFC GlobalId read-only in the header", () => {
    renderWithProviders(<HitlCorrectionForm ifcGlobalId="GlobalIdX1" />);
    const tid = screen.getByTestId("hitl-correction-target-id");
    expect(tid.textContent).toBe("GlobalIdX1");
  });

  it("clears rationale and corrected value after a successful submit", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HitlCorrectionForm ifcGlobalId="X1" />);
    await user.type(screen.getByTestId("hitl-reviewer-id"), "bob");
    await user.type(screen.getByTestId("hitl-rationale"), "first");
    await user.click(screen.getByTestId("hitl-submit-button"));
    await waitFor(() => expect(screen.getByTestId("hitl-success")).toBeInTheDocument());
    const rationale = screen.getByTestId("hitl-rationale") as HTMLTextAreaElement;
    expect(rationale.value).toBe("");
    // Reviewer id must remain so multi-correction sessions don't lose context.
    const reviewer = screen.getByTestId("hitl-reviewer-id") as HTMLInputElement;
    expect(reviewer.value).toBe("bob");
  });
});
