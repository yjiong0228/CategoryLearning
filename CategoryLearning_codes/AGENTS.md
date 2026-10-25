# Journal code instructions

Root AGENTS.md applies. Read Bayesian_model/README.md for model work and
figures/README.md for figure work.

- Maintain mechanisms in src/Bayesian_state; this tree owns journal configuration,
  workflow policy, analysis and figures. Preserve existing compatibility imports.
- Journal inputs come from data/exp123/ only. Keep source-file task labels distinct
  from feedback-condition numbers; use prepared condition metadata for model semantics.
- Keep Fig1 behavior-only evidence separate from model-dependent interpretations.
- Retain the chosen oral encoder sigma=0.05 unless a scientific change is requested;
  oral reports are external validation, not fitted choices. Do not silently merge
  outputs produced using different encoding scales.
- Figure code goes in figures/figN; drafts in figures/outputs/figN/version.
  Update the current figure index when a version is selected.
- Do not delete source tables, data-correction audits or numeric regression fixtures
  as intermediate clutter. Outputs may be ignored by Git and have no recoverable copy.
- Root pytest does not collect these suites automatically. Run affected suites:
  python -m pytest -q CategoryLearning_codes/Bayesian_model/tests
  python -m pytest -q CategoryLearning_codes/figures CategoryLearning_codes/tests
