import { NavLink } from "react-router-dom";
import clsx from "clsx";

const NAV: Array<{ to: string; label: string; description: string }> = [
  { to: "/", label: "Pipeline", description: "Stage DAG and health" },
  { to: "/runs", label: "Runs", description: "Run control & history" },
  { to: "/configs", label: "Configs", description: "Editable parameters" },
  { to: "/inputs", label: "Inputs", description: "Sources & uploads" },
  { to: "/artifacts", label: "Artifacts", description: "Reports & previews" },
  { to: "/metrics", label: "Metrics", description: "Charts & trends" },
  { to: "/vlm", label: "VLM", description: "Multimodal QA" },
  { to: "/viewer", label: "3D Viewer", description: "BIM & alignment" },
  { to: "/diff", label: "Config Diff", description: "Compare versions" },
  { to: "/report", label: "Report", description: "Export & summarize" },
];

export function Sidebar() {
  return (
    <aside className="hidden h-full w-60 shrink-0 flex-col border-r border-surface-border bg-surface-1 lg:flex">
      <div className="flex items-center gap-2 px-4 py-4">
        <div
          aria-hidden
          className="grid size-7 place-items-center rounded-md bg-accent text-surface-0 font-bold"
        >
          M
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-ink">MyCon</span>
          <span className="text-[10px] uppercase tracking-widest text-ink-subtle">
            Construction&nbsp;Monitor
          </span>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 py-2">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              clsx(
                "group block rounded-md px-3 py-2 text-sm transition",
                isActive
                  ? "bg-surface-2 text-ink"
                  : "text-ink-muted hover:bg-surface-2 hover:text-ink",
              )
            }
          >
            <div className="font-medium">{item.label}</div>
            <div className="text-[11px] text-ink-subtle group-aria-[current=page]:text-ink-muted">
              {item.description}
            </div>
          </NavLink>
        ))}
      </nav>
      <footer className="border-t border-surface-border px-4 py-3 text-[10px] uppercase tracking-widest text-ink-subtle">
        v0.3 — Phase 3
      </footer>
    </aside>
  );
}
