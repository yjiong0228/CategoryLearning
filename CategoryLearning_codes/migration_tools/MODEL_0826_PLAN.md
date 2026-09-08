# Model 0826 implementation migration plan

Goal: a self-contained CategoryLearning_codes.Bayesian_model package containing the
Model 0826 dependency graph, preserving the established responsibility layers.

- [x] Map manuscript mechanisms to configured implementations and identify live dependency roots.
- [x] Copy the transitive implementation dependencies, prune obsolete public exports, preserve
      module boundaries and exact numerical code. Record source hashes and excluded files.
- [x] Supply local 0826 engine/parameter configuration and a lightweight CLI configuration.
      Keep existing src package/configs/results untouched for compatibility and parity testing.
- [x] Test old/new PF and autonomous generation under fixed seeds for chi=0/1, plus
      migrated mechanism/recovery tests and import/resource independence.
- [x] Document scientific boundaries, retained shared infrastructure, discrepancies and entry points.

No four-category extension, parameter tuning, full recovery or cohort fit in this migration.
