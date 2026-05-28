import { render, type RenderOptions } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactElement, ReactNode } from "react";

import { ThemeProvider } from "../hooks/useTheme";

interface ProviderOptions {
  initialEntries?: string[];
  /**
   * If provided, the rendered element is matched against this route pattern
   * so `useParams` resolves. When omitted the element is mounted at "/".
   */
  routePath?: string;
}

export function renderWithProviders(
  ui: ReactElement,
  {
    initialEntries = ["/"],
    routePath,
    ...options
  }: ProviderOptions & Omit<RenderOptions, "wrapper"> = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <MemoryRouter
            initialEntries={initialEntries}
            future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
          >
            {routePath ? (
              <Routes>
                <Route path={routePath} element={children} />
              </Routes>
            ) : (
              children
            )}
          </MemoryRouter>
        </ThemeProvider>
      </QueryClientProvider>
    );
  }
  return { ...render(ui, { wrapper: Wrapper, ...options }), queryClient };
}
