// Status -> tone mapping. Used everywhere a status string becomes a Badge.

import type { BadgeTone } from "../components/primitives";
import type { RunStatus, StageStatus } from "../api/types";

export function statusTone(status: RunStatus | StageStatus | null | undefined): BadgeTone {
  switch (status) {
    case "completed":
      return "ok";
    case "running":
    case "queued":
      return "info";
    case "failed":
      return "err";
    case "cancelled":
      return "warn";
    case "skipped":
      return "neutral";
    default:
      return "neutral";
  }
}

export function isTerminal(status: RunStatus | null | undefined): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}
