import { describe, expect, it } from "vitest";
import { layoutStages } from "./dag";
import { fixtureStages } from "../test/fixtures";

describe("layoutStages", () => {
  it("places sources at level 0 and increments through dependencies", () => {
    const layout = layoutStages(fixtureStages);
    const byName = new Map(layout.map((e) => [e.stage.name, e.level]));
    expect(byName.get("stage_01_ingest")).toBe(0);
    expect(byName.get("stage_02_keyframes")).toBe(1);
    expect(byName.get("stage_03_colmap")).toBe(2);
  });

  it("preserves registry order within a level", () => {
    const a = { ...fixtureStages[0], name: "a", dependencies: [] };
    const b = { ...fixtureStages[0], name: "b", dependencies: [] };
    const out = layoutStages([a, b]);
    expect(out.map((e) => e.indexInLevel)).toEqual([0, 1]);
  });
});
