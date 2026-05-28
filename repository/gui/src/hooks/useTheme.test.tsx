import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { useTheme } from "./useTheme";
import { renderWithProviders } from "../test/render";

function ThemeProbe() {
  const { theme, toggle } = useTheme();
  return (
    <button onClick={toggle} data-testid="probe">
      theme={theme}
    </button>
  );
}

describe("ThemeProvider", () => {
  it("defaults to dark and toggles to light", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ThemeProbe />);
    expect(screen.getByTestId("probe")).toHaveTextContent("theme=dark");
    await user.click(screen.getByTestId("probe"));
    expect(screen.getByTestId("probe")).toHaveTextContent("theme=light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });
});
