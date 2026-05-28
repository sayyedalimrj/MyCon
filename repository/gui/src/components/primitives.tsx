// Small set of styled primitives used across panels.
//
// We deliberately avoid a heavyweight component library; the GUI is a
// research tool, not a marketing site, and a couple hundred lines of
// Tailwind primitives is easier to maintain than a Radix/MUI dependency.

import clsx from "clsx";
import type { HTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";

/* ----------------------------- Card -------------------------------- */

export function Card({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLDivElement> & { children: ReactNode }) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-surface-border bg-surface-1 p-4 shadow-sm",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  right,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <header className="mb-3 flex items-start justify-between gap-3">
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wider text-ink-muted">
          {title}
        </h2>
        {subtitle && <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>}
      </div>
      {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
    </header>
  );
}

/* ----------------------------- Badge -------------------------------- */

export type BadgeTone = "neutral" | "ok" | "warn" | "err" | "info" | "accent";

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: BadgeTone;
  className?: string;
  children: ReactNode;
}) {
  const tones: Record<BadgeTone, string> = {
    neutral: "bg-surface-2 text-ink-muted",
    ok: "bg-ok/15 text-ok",
    warn: "bg-warn/15 text-warn",
    err: "bg-err/15 text-err",
    info: "bg-info/15 text-info",
    accent: "bg-accent/15 text-accent",
  };
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* ----------------------------- Button -------------------------------- */

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  const variants: Record<ButtonVariant, string> = {
    primary:
      "bg-accent text-surface-0 hover:bg-accent-muted disabled:cursor-not-allowed disabled:opacity-50",
    secondary:
      "border border-surface-border bg-surface-1 hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50",
    ghost:
      "bg-transparent hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50",
    danger:
      "bg-err/90 text-white hover:bg-err disabled:cursor-not-allowed disabled:opacity-50",
  };
  return (
    <button
      type="button"
      className={clsx(
        "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition",
        variants[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/* ----------------------------- Spinner ------------------------------- */

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="loading"
      className={clsx(
        "inline-block size-4 animate-spin rounded-full border-2 border-surface-border border-t-accent",
        className,
      )}
    />
  );
}

/* ----------------------------- Empty --------------------------------- */

export function Empty({
  title,
  hint,
  icon,
}: {
  title: string;
  hint?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-surface-border bg-surface-2/40 px-6 py-10 text-center">
      {icon}
      <p className="text-sm font-medium text-ink">{title}</p>
      {hint && <p className="max-w-md text-xs text-ink-muted">{hint}</p>}
    </div>
  );
}

/* ----------------------------- Notice -------------------------------- */

export function Notice({
  tone = "info",
  title,
  children,
}: {
  tone?: BadgeTone;
  title?: ReactNode;
  children: ReactNode;
}) {
  const toneClass: Record<BadgeTone, string> = {
    neutral: "border-surface-border bg-surface-2/50 text-ink-muted",
    ok: "border-ok/40 bg-ok/10 text-ok",
    warn: "border-warn/40 bg-warn/10 text-warn",
    err: "border-err/40 bg-err/10 text-err",
    info: "border-info/40 bg-info/10 text-info",
    accent: "border-accent/40 bg-accent/10 text-accent",
  };
  return (
    <div className={clsx("rounded-lg border px-3 py-2 text-sm", toneClass[tone])}>
      {title && <div className="font-semibold">{title}</div>}
      <div>{children}</div>
    </div>
  );
}

/* ----------------------------- Code ---------------------------------- */

export function Code({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <code
      className={clsx(
        "rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-ink",
        className,
      )}
    >
      {children}
    </code>
  );
}
