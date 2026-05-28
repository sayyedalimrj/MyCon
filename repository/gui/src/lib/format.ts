// Small pure formatting helpers. They are unit-tested.

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || Number.isNaN(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`;
}

export function formatUnixTimestamp(unix: number | null | undefined): string {
  if (unix == null || Number.isNaN(unix)) return "—";
  const d = new Date(unix * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  // ISO without trailing milliseconds: easier to read in a research log.
  return d.toISOString().replace(/\.\d+Z$/, "Z");
}

export function formatDurationSeconds(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  const mm = m - h * 60;
  return `${h}h ${mm}m`;
}

export function formatRunDuration(
  startedAt: number | null | undefined,
  finishedAt: number | null | undefined,
): string {
  if (!startedAt) return "—";
  const end = finishedAt ?? Date.now() / 1000;
  return formatDurationSeconds(end - startedAt);
}

export function shortHash(hash: string | null | undefined, len = 12): string {
  if (!hash) return "—";
  return hash.length <= len ? hash : `${hash.slice(0, len)}…`;
}
