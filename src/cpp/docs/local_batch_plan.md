> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Local Cond123 Preparation Plan

**Goal:** Explain and de-duplicate behavior panels without changing model math;
prepare all 96 exp123 subjects for sequential local fit/evaluate/plot with a
12-thread native task budget. Do not launch the full cohort in this task.

## Decisions

- Original repositories, existing results and raw data stay unchanged.
- The two blue panels mean autonomous true-category correctness versus received
  feedback credit. They are identical for deterministic binary feedback, not
  independent model trajectories. Cond3 half credit remains distinct.
- Suppress a binary feedback panel only when both observed values/masks and all
  plotted autonomous bands really duplicate correctness. Otherwise retain it.
- Keep trailing arithmetic means and straight line segments. Explain expanding
  warm-up denominators; do not silently introduce smoothing or change windows.
- Raise the Windows worker cap from 8 to the user's authorized 12. Light tests
  remain at 1-2 workers; do not change the scientific parameter space.
- Add full-trial local configs and a PowerShell orchestrator. Conditions and
  subjects run sequentially; native tasks get up to 12 threads, plotting one
  Python process. Fresh output and failure-stop behavior protect results.

## Tasks

- [x] Audit real saved bands and source feedback, and test rolling-mean examples.
- [x] Add red tests for duplicate-panel omission and incorrect saved feedback;
  implement minimal plot/validation changes, preserving all numerical arrays.
- [x] Add red worker-budget test; permit 12 workers on Windows and rebuild.
- [x] Add three full local configs, all-stage script, script/config tests and
  instructions. Explicit smoke/subset options must not leak into full defaults.
- [x] Run complete C++ and Python tests; run one tiny three-condition script
  sequence and inspect regenerated real-data PNGs without rerunning old fits.
- [x] Review changes, document results and leave full fits for the user's launch.

## Verification

Native 34 cases / 18,955 assertions and Python 28 tests / 13 subtests passed.
The real script completed all nine stages for S122/S206/S314, four trials each,
two workers; a separate tiny evaluation accepted jobs=12. A default dry plan
contains 96 subjects / 288 stages with no truncation. Fresh read-only review
found no actionable scoped defects. All 32 saved Cond1 bundles were re-plotted
without re-running scientific tasks. New full-cohort fitting remains unstarted.
