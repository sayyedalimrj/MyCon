import { describe, it, expect } from "vitest";
import {
  formatBytes,
  formatDurationSeconds,
  formatRunDuration,
  formatUnixTimestamp,
  shortHash,
} from "./format";

describe("format helpers", () => {
  it("formatBytes scales by powers of 1024 with sane precision", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.50 KB");
    expect(formatBytes(1024 * 1024 * 5)).toBe("5.00 MB");
  });

  it("formatBytes returns dash for nullish input", () => {
    expect(formatBytes(null)).toBe("—");
    expect(formatBytes(undefined)).toBe("—");
  });

  it("formatUnixTimestamp returns an ISO string without milliseconds", () => {
    const out = formatUnixTimestamp(1_700_000_000);
    expect(out).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
  });

  it("formatDurationSeconds picks a sensible unit", () => {
    expect(formatDurationSeconds(0.05)).toBe("50 ms");
    expect(formatDurationSeconds(45)).toBe("45.0 s");
    expect(formatDurationSeconds(125)).toBe("2m 5s");
    expect(formatDurationSeconds(3700)).toBe("1h 1m");
  });

  it("formatRunDuration uses now() when finished_at is null", () => {
    const startedAt = Date.now() / 1000 - 30;
    const out = formatRunDuration(startedAt, null);
    expect(out).toMatch(/s$/);
  });

  it("shortHash truncates correctly", () => {
    expect(shortHash("abcdefghijklmnop", 8)).toBe("abcdefgh…");
    expect(shortHash("short")).toBe("short");
    expect(shortHash(null)).toBe("—");
  });
});
