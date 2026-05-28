import { describe, it, expect } from "vitest";
import { endpoints } from "./endpoints";

// Pin the GUI side of the API contract end-to-end through MSW so a future
// backend shape drift surfaces here, not as a runtime "undefined.foo"
// crash inside a panel.
describe("endpoints contract", () => {
  it("listStages returns descriptors with the expected fields", async () => {
    const stages = await endpoints.listStages();
    expect(stages.length).toBeGreaterThan(0);
    for (const s of stages) {
      expect(typeof s.name).toBe("string");
      expect(typeof s.title).toBe("string");
      expect(Array.isArray(s.dependencies)).toBe(true);
      expect(Array.isArray(s.required_config_keys)).toBe(true);
      expect(Array.isArray(s.capabilities)).toBe(true);
    }
  });

  it("listVlmBackends returns capabilities[] not online/metadata", async () => {
    const backends = await endpoints.listVlmBackends();
    expect(backends.length).toBeGreaterThan(0);
    for (const b of backends) {
      expect(typeof b.name).toBe("string");
      expect(typeof b.description).toBe("string");
      expect(Array.isArray(b.capabilities)).toBe(true);
    }
  });

  it("getStageSchema returns a hydrated dataclass JSON view", async () => {
    const schema = await endpoints.getStageSchema("site01", "stage_02_keyframes");
    expect(schema.config_name).toBe("site01");
    expect(schema.stage).toBe("stage_02_keyframes");
    expect(typeof schema.schema).toBe("object");
    expect(Array.isArray(schema.required_config_keys)).toBe(true);
  });

  it("listRuns and listArtifacts shape match RunListEntry / ArtifactSummary", async () => {
    const runs = await endpoints.listRuns(50);
    for (const r of runs) {
      expect(typeof r.run_id).toBe("string");
      expect(Array.isArray(r.requested_stages)).toBe(true);
      expect(typeof r.stage_statuses).toBe("object");
    }
    const artifacts = await endpoints.listArtifacts("run-001");
    for (const a of artifacts) {
      expect(typeof a.stage).toBe("string");
      expect(typeof a.exists).toBe("boolean");
      expect(typeof a.preview).toBe("object");
    }
  });
});
