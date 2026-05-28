import { useQuery } from "@tanstack/react-query";
import { endpoints } from "../api/endpoints";
import { queryKeys } from "../api/queryKeys";
import { Badge, Button, Spinner } from "../components/primitives";
import { useTheme } from "../hooks/useTheme";

// Permanent status strip across the top of every panel. Reads /api/health
// every 5 seconds so an operator can see at a glance whether the
// backend is reachable.

export function TopBar() {
  const { theme, toggle } = useTheme();

  const health = useQuery({
    queryKey: queryKeys.health(),
    queryFn: endpoints.health,
    refetchInterval: 5_000,
    refetchIntervalInBackground: true,
  });

  let statusBadge: JSX.Element;
  if (health.isLoading && !health.data) {
    statusBadge = (
      <Badge tone="neutral">
        <Spinner /> connecting…
      </Badge>
    );
  } else if (health.isError) {
    statusBadge = <Badge tone="err">backend unreachable</Badge>;
  } else if (health.data) {
    statusBadge = (
      <Badge tone="ok" data-testid="health-badge-ok">
        backend ok · {health.data.stage_count} stages · {health.data.history_run_count} runs
      </Badge>
    );
  } else {
    statusBadge = <Badge tone="neutral">unknown</Badge>;
  }

  return (
    <header className="flex items-center gap-3 border-b border-surface-border bg-surface-1 px-4 py-2">
      <div className="flex items-center gap-2 text-sm text-ink-muted">
        <span className="hidden text-xs uppercase tracking-widest text-ink-subtle sm:inline">
          status
        </span>
        {statusBadge}
      </div>
      <div className="ml-auto flex items-center gap-2">
        <Button
          variant="ghost"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          onClick={toggle}
        >
          <span aria-hidden>{theme === "dark" ? "☾" : "☀"}</span>
          <span className="hidden sm:inline">
            {theme === "dark" ? "Dark" : "Light"}
          </span>
        </Button>
      </div>
    </header>
  );
}
