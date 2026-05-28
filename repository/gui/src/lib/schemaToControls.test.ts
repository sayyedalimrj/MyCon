import { describe, it, expect } from "vitest";
import { flattenSchema, groupBySection } from "./schemaToControls";

describe("flattenSchema", () => {
  it("emits one control per primitive leaf with dotted paths", () => {
    const out = flattenSchema({
      project: { name: "site01", random_seed: 42 },
      keyframes: { min_time_gap_sec: 0.5, enabled: true },
    });
    expect(out.map((c) => `${c.path}:${c.type}`)).toEqual([
      "project.name:string",
      "project.random_seed:number",
      "keyframes.min_time_gap_sec:number",
      "keyframes.enabled:boolean",
    ]);
  });

  it("treats null leaves as null-typed strings (so the form can render them)", () => {
    const out = flattenSchema({ a: null });
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ path: "a", type: "string", isNull: true });
  });

  it("represents arrays and empty objects as complex JSON controls", () => {
    const out = flattenSchema({ tags: ["a", "b"], extras: {} });
    expect(out.map((c) => `${c.path}:${c.type}:${c.isComplex}`)).toEqual([
      "tags:json:true",
      "extras:json:true",
    ]);
  });

  it("groupBySection buckets by first dotted segment", () => {
    const groups = groupBySection(
      flattenSchema({ project: { name: "x" }, keyframes: { gap: 1 } }),
    );
    expect(groups.map((g) => g.section)).toEqual(["project", "keyframes"]);
    expect(groups[0].controls.map((c) => c.path)).toEqual(["project.name"]);
  });
});
