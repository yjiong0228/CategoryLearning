# AGENTS.md

## Scope and precedence

These instructions apply to the entire repository. A more deeply nested
`AGENTS.md` may add to or override them for its own subtree. Direct user
instructions always take precedence.

## Repository context

This repository contains behavioral category-learning models, experiment
configurations, analysis scripts, and generated scientific results.

- `src/Bayesian_state/` is the single actively maintained model implementation
  shared by the journal and dissertation. Do not duplicate its algorithms.
- `CategoryLearning_codes/Bayesian_model/` holds journal configs, policy wrappers,
  compatibility imports and regression tests. New model imports use the shared core.
- The journal uses only `data/exp123/`; the dissertation also uses `data/exp4/`,
  `data/exp5/`, and `data/meg/`. Configurable paths do not imply that Model 0826
  already supports every task's category/feedback semantics.
- `CategoryLearning_paper/` holds the journal manuscript; `src/Bayesian_state/docs/model_architecture/` holds
  model technical specifications, currently `model_0826.tex`.
- Figure code and draft outputs belong to `CategoryLearning_codes/figures/`,
  grouped by figure number. Only confirmed figures belong in
  `CategoryLearning_paper/figures/`. Default export is PNG.
- `src/Bayesian/` is a legacy/baseline implementation; preserve compatibility
  unless the task explicitly targets a migration.
- `src/Hybrid/`, `src/RNN_old/`, `src/RNN_new/`, `src/SUSTAIN/`, and
  `src/Cohen/` are separate model families. Do not change them merely to make
  an unrelated implementation uniform.
- `configs/{exp123,exp4,exp5,meg}/` organize experiment configurations;
  `configs/shared/` holds reusable definitions. MEG has no migrated YAML yet.
- `data/{exp123,exp4,exp5,meg}/` contains source or processed research data.
- `results/` (including report bundles) and `logs/` contain generated artifacts and may be
  expensive to reproduce.

For work on the active pipeline, read `src/Bayesian_state/README.md` before
changing its interfaces, configuration schema, or workflow.

## Working rules

1. Inspect `git status` and the relevant surrounding code before editing.
   Preserve all unrelated user changes in a dirty worktree.
2. Keep changes focused on the requested task. Prefer extending the current
   design over broad rewrites or cross-model refactors.
3. Search for all call sites and configuration references before changing a
   public function, CLI option, YAML key, output filename, or result schema.
4. Treat raw data as read-only. Do not delete, rename, rewrite, or normalize
   files under `data/` unless the user explicitly requests it.
5. Do not overwrite or delete existing research results, reports, checkpoints,
   numerical caches, or logs. Use a new output directory for exploratory runs.
   During an authorized repository cleanup, untracked `__pycache__` and
   `.pytest_cache` outside data/research-output directories may be removed.
   Do not treat `results/cache`, notebook checkpoints, or ignored files as
   disposable runtime caches. See `src/Bayesian_state/docs/history/repository_management_20260908.md`.
6. Do not launch full grid searches, repeated simulations, or other long and
   compute-intensive jobs unless the user explicitly asks for them. Start with
   a small or targeted validation when possible.
7. Never commit secrets, machine-specific absolute paths, large generated
   artifacts, or temporary files.
8. Unless the user explicitly requests another format, generate and retain
   plots as PNG only. Do not emit redundant PDF, SVG, or TIFF copies by
   default.

## Python and configuration conventions

- Run commands from the repository root so imports such as
  `src.Bayesian_state...` resolve consistently.
- Use `python -m ...` for package entry points and `python -m pytest ...` for
  tests.
- Follow the style of the module being edited. Prefer explicit imports,
  `pathlib.Path`, type hints on new public interfaces, and small functions with
  clear responsibilities.
- Keep model structure and experiment parameters in YAML when the existing
  workflow already exposes them there; avoid duplicating configuration as
  hard-coded Python constants.
- Preserve backward compatibility for existing config files unless a breaking
  change is explicitly required. If a key must change, update its loaders,
  validation, examples, and documentation together.
- Add comments for scientific intent, non-obvious numerical choices, and model
  assumptions—not for code that is already self-explanatory.

## Scientific correctness and reproducibility

- Do not silently change random seeds, default hyperparameters, objective
  ordering, trial filtering, prediction timing, or statistical definitions.
- Distinguish behavior-changing model work from refactoring in both code and
  the final summary.
- Preserve subject, condition, trial-order, and train/evaluation boundaries.
  Watch for accidental data leakage when adding analyses or predictors.
- For probability and trajectory code, check relevant invariants where
  practical: finite values, expected array shapes, valid masks, normalized
  probabilities, and deterministic behavior under a fixed seed.
- Record enough configuration and provenance in generated outputs for a run to
  be understood and reproduced later.

## Validation

Install the declared Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the smallest relevant validation first. Examples:

```bash
python -m pytest -q tests/bayesian_state/test_model_0806_framework.py
python -m pytest -q CategoryLearning_codes/Bayesian_model/tests
python -m pytest -q CategoryLearning_codes/figures CategoryLearning_codes/tests
```

The root pytest default collects `tests/` only, not all journal tests. Select
suites explicitly. For shared-core refactors, include the journal's saved
pre-refactor numeric references; comparing two aliases of the same code is not
an independent regression check. Do not regenerate references merely to pass a
failure. Explain and validate any scientific change first.

For CLI or configuration changes, also exercise the affected entry point with
a lightweight configuration or inspect its `--help` output. Do not represent a
long-running scientific pipeline as validated when only imports or unit tests
were checked.

If a test cannot be run because data, dependencies, hardware, or runtime are
unavailable, state that limitation explicitly rather than guessing at the
result.

## Documentation and handoff

- Update the nearest README or file-level documentation when changing a CLI,
  config schema, model assumption, workflow, or output format.
- In the final handoff, summarize the behavior changed, list the validation
  performed, and identify any unverified or long-running follow-up work.

## Repository maintenance

- Root README is the current entrypoint map and retention/cleanup policy. Keep them consistent with actual paths.
- Inspect references before moving old scripts/configs. Dated files, migration
  manifests, model specifications and audit reports are not automatically junk.
- Do not rewrite historical manifests to reflect new directory layouts; add a
  current navigation note instead. Preserve data-correction audit trails.
- For research-artifact cleanup, prepare a concrete path list with purpose,
  dependencies, regeneration cost and backup status before requesting a decision.
  Broad housekeeping authorization is sufficient for disposable runtime caches.
- Preserve unrelated pending changes. Do not stage the entire worktree or commit
  previous tasks while performing documentation/cleanup work unless requested.
- Freeze publication code by commit/tag plus separately archived data/results;
  never make a second editable model tree to preserve a paper version.

## Environment and compute

No canonical Conda environment, complete environment lock, or mandatory formatter/
linter is currently declared. Do not invent one in reports. requirements.txt is
not a complete development lock; PyYAML is needed for YAML and pytest for tests.
Record actual versions for reproducible runs. Do not reinstall the environment
merely to edit documentation.

For smoke validation, start with one subject, a short valid trial sequence and one
job; choose a new output directory. Existing recovery configs can request large
parallel jobs, so inspect them before running. Full fits, recovery and repeated
large simulations require explicit task authorization.

Current layout and historical-path mapping: src/Bayesian_state/docs/maintenance/LAYOUT_MIGRATION_20260908.md.
Keep performance scripts in src/Bayesian_state/workflows/benchmarks, historical docs in src/Bayesian_state/docs/history,
and migration/cleanup records in src/Bayesian_state/docs/maintenance. Use system temporary directories
for disposable caches; do not recreate the old top-level data/config directories.

Do not recreate root docs/, reports/, scripts/ or .github/ for ordinary work.
Model documentation belongs in src/Bayesian_state/docs; workflow tools belong in
src/Bayesian_state/workflows/{runs,analysis,reports,benchmarks}; generated reports
belong with their results. Repository audit bundles live in results/repository_maintenance.
Use AGENTS.md and current READMEs as the shared agent instructions; the old Copilot
intro prompt was removed because it described an obsolete architecture.
