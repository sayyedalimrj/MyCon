import { describe, it, expect } from "vitest";
import { composeReport } from "./ReportPanel";
import { fixtureArtifacts, fixtureRunSnapshot } from "../test/fixtures";

describe("composeReport", () => {
  it("collapses provenance fingerprints and counts stage outcomes", () => {
    const report = composeReport(fixtureRunSnapshot, fixtureArtifacts);
    expect(report.stages_total).toBe(2);
    expect(report.stages_completed).toBe(2);
    expect(report.stages_failed).toBe(0);
    // Both artifact provenance blocks share one fingerprint.
    expect(report.provenance).toHaveLength(1);
    expect(report.provenance[0].count).toBe(2);
    expect(report.provenance[0].config_hash).toMatch(/^18b982e4/);
  });
});
