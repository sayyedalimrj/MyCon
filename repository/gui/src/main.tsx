import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { App } from "./app/App";
import { ThemeProvider } from "./hooks/useTheme";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // The pipeline is long-running and the operator usually has the
      // tab open all day. We refresh on focus and on reconnect so values
      // never look stale, but with a 30s background interval as a floor
      // for live screens.
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
      retry: 1,
      staleTime: 5_000,
    },
  },
});

const root = document.getElementById("root");
if (!root) {
  throw new Error("MyCon GUI: missing #root element in index.html");
}

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter
          future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
        >
          <App />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>,
);
