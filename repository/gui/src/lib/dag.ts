// Compute simple level-based positions for the stage DAG so the
// PipelinePanel can lay it out without bringing in a graph library.
//
// The registry is already topologically sorted (validated by the
// Phase 1 tests), so we just bucket each stage by "longest path from a
// root", which gives a tidy left-to-right layout.

import type { StageDescriptor } from "../api/types";

export interface StageLayoutEntry {
  stage: StageDescriptor;
  level: number;
  indexInLevel: number;
}

export function layoutStages(stages: StageDescriptor[]): StageLayoutEntry[] {
  const levelOf = new Map<string, number>();
  // Single linear pass: a stage's level is max(level(dep)) + 1.
  for (const s of stages) {
    const depLevels = s.dependencies.map((d) => levelOf.get(d) ?? 0);
    const lvl = depLevels.length === 0 ? 0 : Math.max(...depLevels) + 1;
    levelOf.set(s.name, lvl);
  }

  // Now assign indexInLevel preserving registry order within a level.
  const counters = new Map<number, number>();
  return stages.map((s) => {
    const level = levelOf.get(s.name) ?? 0;
    const indexInLevel = counters.get(level) ?? 0;
    counters.set(level, indexInLevel + 1);
    return { stage: s, level, indexInLevel };
  });
}
