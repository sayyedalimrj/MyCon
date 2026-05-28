// Structural diff of two arbitrary JSON-shaped values, returning a flat
// list of changes the diff panel can render row-by-row.
//
// The output is purely descriptive (no patches); we don't need
// rfc6902/json-patch — we want a human-readable change list keyed by
// dotted path.

export type DiffOp = "added" | "removed" | "changed";

export interface DiffEntry {
  path: string;
  op: DiffOp;
  left: unknown;
  right: unknown;
}

export function diffJson(a: unknown, b: unknown, basePath = ""): DiffEntry[] {
  const out: DiffEntry[] = [];
  walk(a, b, basePath, out);
  // Stable order (dotted-path lexicographic) so test snapshots are deterministic.
  out.sort((x, y) => (x.path < y.path ? -1 : x.path > y.path ? 1 : 0));
  return out;
}

function walk(a: unknown, b: unknown, path: string, out: DiffEntry[]): void {
  if (a === undefined && b === undefined) return;
  if (deepEqual(a, b)) return;

  if (a === undefined) {
    out.push({ path, op: "added", left: undefined, right: b });
    return;
  }
  if (b === undefined) {
    out.push({ path, op: "removed", left: a, right: undefined });
    return;
  }

  // Primitives or mismatched types: report as a change.
  const aIsObj = isPlainObject(a);
  const bIsObj = isPlainObject(b);
  if (!aIsObj || !bIsObj) {
    out.push({ path, op: "changed", left: a, right: b });
    return;
  }

  const aObj = a as Record<string, unknown>;
  const bObj = b as Record<string, unknown>;
  const keys = new Set<string>([...Object.keys(aObj), ...Object.keys(bObj)]);
  for (const k of keys) {
    const next = path ? `${path}.${k}` : k;
    walk(aObj[k], bObj[k], next, out);
  }
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a === null || b === null) return a === b;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  if (typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a as object);
    const kb = Object.keys(b as object);
    if (ka.length !== kb.length) return false;
    for (const k of ka) {
      if (!deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])) {
        return false;
      }
    }
    return true;
  }
  return false;
}

export function summarizeDiff(entries: DiffEntry[]): { added: number; removed: number; changed: number } {
  return entries.reduce(
    (acc, e) => {
      if (e.op === "added") acc.added += 1;
      else if (e.op === "removed") acc.removed += 1;
      else acc.changed += 1;
      return acc;
    },
    { added: 0, removed: 0, changed: 0 },
  );
}
