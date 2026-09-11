# Model 0826 condition 2 implementation plan

**Goal:** Fit PMH only for Fig1 representative subject 229 (1088 trials), retaining condition 1 work and deferring condition 3 and ablations.

**Architecture:** Extend the existing shared particle filter to explicit condition 2 with four fixed task labels and binary feedback. Retain the canonical four-category catalog, boundary geometry, memory, controller, beta formula and parameter support. Do not add label learning or use true category as a model input. Condition 3 remains rejected. Binary-only transmission diagnostics stay explicitly unavailable for condition 2.

**Tech stack:** Python, NumPy, joblib, YAML, pytest.

**Spec:** User instructions on 2026-09-09; current PMH configuration and model_0826.tex define inherited mechanisms. This is a task extension, not a new frozen manuscript model.

- [x] Add regression tests for real subject 229, both execution modes, deterministic seeds, normalized probabilities, causal prediction timing, binary feedback validation and deferred condition 3.
- [x] Generalize category-sized PF arrays and forward condition through dispatch and simulation. Keep binary orientation axes unchanged; reject unsupported diagnostics explicitly.
- [x] Run targeted tests and independent saved journal numerical references; smoke-test the simulation CLI on one job and a short sequence.
- [x] Create new condition 2 engine/run/search YAML with actual catalog and similarity provenance. Preserve full 1088 trials, search grids and particle/repeat budgets. Allocate 48 workers while condition 1 uses 16; cap numerical-library threads at 1.
- [x] Launch PMH search then fitted simulation in a new subject directory with logs, checkpoints, config/data/code provenance and live status; inspect actual worker progress.
- [ ] Produce individual fit diagnostics and trajectory outputs, verify each phase and report genuine completion status. Do not call a submitted fit complete.

No raw data or pre-existing numerical results are to be overwritten. All plots PNG. Search budget is an engineering choice inherited from the approved condition 1 representative run, not a recovered four-category parameter support claim.
