# Shared model implementation plan

**Goal:** Maintain the journal and dissertation model mechanisms once, in src/Bayesian_state.
**Architecture:** Journal configurations and entrypoints depend on the shared model. Existing journal Python paths forward to canonical modules for compatibility. Existing dataset resolution remains the adapter boundary; four-category/partial-feedback extensions are not part of this refactor.
**Tech stack:** Python, YAML, NumPy, pytest.
**Spec:** User-approved shared-core / separate-project configuration design, 2026-09-08.

## Constraints

Preserve data, historical outputs, seeds, model parameters, prediction timing, and legacy model support. No formal fitting. Use current workspace as requested; do not commit automatically.

## Steps

- [x] Capture frozen numeric regression cases from the pre-refactor journal implementation (four PF cases, two autonomous cases), with input/config hashes.
- [x] Replace duplicated journal numerical modules with explicit compatibility forwards; change active figure/config imports to src.Bayesian_state. Preserve journal-only parameter validation and CLI defaults as thin policy wrappers.
- [x] Move recovery orchestration to src/Bayesian_state/run_recovery.py; keep both existing entrypoints. Fingerprint canonical implementation files.
- [x] Add journal dataset-scope validation and document configuration, shared ownership, and reproducible publication freezes.
- [x] Run fixed-reference regressions, mechanism/recovery contracts, relevant legacy and figure tests, CLI help, and a small simulation in a fresh temporary output directory.

Validation commands: python -m pytest -q CategoryLearning_codes/Bayesian_model/tests CategoryLearning_codes/figures CategoryLearning_codes/tests tests/bayesian_state/test_model_0826_versioning.py tests/bayesian_state/evaluation/test_oral_alignment.py; canonical and journal CLI --help. Tests must compare actual numeric results with saved pre-refactor arrays, not two aliases of the same function.

Outcome: 202 targeted tests passed; saved pre-refactor arrays and 32-trial CLI outputs match. See CategoryLearning_codes/Bayesian_model/SHARED_CORE_VALIDATION.md.
