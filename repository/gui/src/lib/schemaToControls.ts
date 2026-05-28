// Walk a hydrated schema dict (from `GET /configs/{name}/schemas/{stage}`)
// and produce a flat list of typed controls the StageEditor renders.
//
// The backend returns the schema as a plain JSON object — nested
// dataclasses become nested objects. Each leaf is either a primitive
// (string/number/bool/null) or a list. We map that directly to control
// types the editor knows how to render.
//
// We do NOT mutate the input. Walks are stable and deterministic so
// tests can pin the output order.

export type ControlType = "string" | "number" | "boolean" | "json";

export interface FieldControl {
  /** Dotted JSON path: "project.name", "keyframes.min_time_gap_sec". */
  path: string;
  label: string;
  type: ControlType;
  value: unknown;
  /** True when the original value was null/undefined (helps the editor render placeholders). */
  isNull: boolean;
  /** True for arrays / nested objects rendered as raw JSON. */
  isComplex: boolean;
}

export function flattenSchema(schema: unknown, basePath = ""): FieldControl[] {
  const out: FieldControl[] = [];
  walk(schema, basePath, out);
  return out;
}

function walk(value: unknown, path: string, out: FieldControl[]): void {
  if (value === null || value === undefined) {
    out.push({
      path,
      label: lastSegment(path),
      type: "string",
      value: value ?? "",
      isNull: true,
      isComplex: false,
    });
    return;
  }
  if (typeof value === "string") {
    out.push({ path, label: lastSegment(path), type: "string", value, isNull: false, isComplex: false });
    return;
  }
  if (typeof value === "number") {
    out.push({ path, label: lastSegment(path), type: "number", value, isNull: false, isComplex: false });
    return;
  }
  if (typeof value === "boolean") {
    out.push({ path, label: lastSegment(path), type: "boolean", value, isNull: false, isComplex: false });
    return;
  }
  if (Array.isArray(value)) {
    out.push({
      path,
      label: lastSegment(path),
      type: "json",
      value,
      isNull: false,
      isComplex: true,
    });
    return;
  }
  if (typeof value === "object") {
    // Nested object → recurse, building dotted paths.
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      out.push({
        path,
        label: lastSegment(path),
        type: "json",
        value: {},
        isNull: false,
        isComplex: true,
      });
      return;
    }
    for (const [k, v] of entries) {
      const nextPath = path ? `${path}.${k}` : k;
      walk(v, nextPath, out);
    }
    return;
  }
  // Anything else (functions, symbols) — render as JSON to be safe.
  out.push({
    path,
    label: lastSegment(path),
    type: "json",
    value: String(value),
    isNull: false,
    isComplex: true,
  });
}

function lastSegment(path: string): string {
  if (!path) return "(root)";
  const i = path.lastIndexOf(".");
  return i < 0 ? path : path.slice(i + 1);
}

/** Group controls by their first path segment for the editor's section headings. */
export function groupBySection(controls: FieldControl[]): Array<{
  section: string;
  controls: FieldControl[];
}> {
  const groups = new Map<string, FieldControl[]>();
  for (const c of controls) {
    const head = c.path.split(".")[0] || "(root)";
    if (!groups.has(head)) groups.set(head, []);
    groups.get(head)!.push(c);
  }
  return Array.from(groups.entries()).map(([section, controls]) => ({ section, controls }));
}
