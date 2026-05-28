import { describe, it, expect } from "vitest";
import { diffJson, summarizeDiff } from "./diff";

describe("diffJson", () => {
  it("reports no entries when objects are deeply equal", () => {
    expect(diffJson({ a: 1, b: { c: 2 } }, { a: 1, b: { c: 2 } })).toEqual([]);
  });

  it("reports added, removed, and changed leaves with dotted paths", () => {
    const a = { keyframes: { min_time_gap_sec: 0.5 }, project: { name: "site01" } };
    const b = {
      keyframes: { min_time_gap_sec: 1.0, new_field: 7 },
      project: { name: "site01" },
    };
    const out = diffJson(a, b);
    expect(out).toEqual([
      { path: "keyframes.min_time_gap_sec", op: "changed", left: 0.5, right: 1.0 },
      { path: "keyframes.new_field", op: "added", left: undefined, right: 7 },
    ]);
  });

  it("treats type mismatches as a single change at that path", () => {
    const out = diffJson({ a: 1 }, { a: { nested: true } });
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ path: "a", op: "changed" });
  });

  it("returns a stable lexicographic order across nested keys", () => {
    const a = { z: 1, a: { b: 1, a: 1 } };
    const b = { z: 2, a: { b: 2, a: 2 } };
    expect(diffJson(a, b).map((e) => e.path)).toEqual(["a.a", "a.b", "z"]);
  });

  it("summarizeDiff buckets correctly", () => {
    const entries = diffJson(
      { a: 1, b: 2, c: 3 },
      { a: 1, b: 999, d: 4 },
    );
    const sum = summarizeDiff(entries);
    expect(sum).toEqual({ added: 1, removed: 1, changed: 1 });
  });
});
