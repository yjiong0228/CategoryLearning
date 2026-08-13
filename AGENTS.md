# AGENTS.md

## Scope and precedence

These instructions apply to the entire repository. A more deeply nested
`AGENTS.md` may add to or override them for its own subtree. Direct user
instructions always take precedence.

## Repository context

This repository contains behavioral category-learning models, experiment
configurations, analysis scripts, and generated scientific results.

- `src/Bayesian_state/` is the current, actively maintained model pipeline.
- `src/Bayesian/` is a legacy/baseline implementation; preserve compatibility
  unless the task explicitly targets a migration.
- `src/Hybrid/`, `src/RNN_old/`, `src/RNN_new/`, `src/SUSTAIN/`, and
  `src/Cohen/` are separate model families. Do not change them merely to make
  an unrelated implementation uniform.
- `configs/`, `configs_exp4/`, and `configs_exp5/` contain experiment-specific
  YAML configurations.
- `data*/` contains source or processed research data.
- `results/`, `reports/`, and `logs/` contain generated artifacts and may be
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
   files under `data*/` unless the user explicitly requests it.
5. Do not overwrite or delete existing results, reports, checkpoints, caches,
   or logs. Use a new, clearly named output directory for exploratory runs.
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
python -m pytest -q
```

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

## Project-specific details to add

This baseline intentionally leaves the following choices for the maintainers:

- Canonical Python/Conda environment name and supported Python version.
- Required formatter, linter, and type-check commands.
- Tests or smoke configurations required for each model family.
- Naming conventions for experiments, output directories, and checkpoints.
- Compute limits and rules for local, cluster, CPU, GPU, and parallel jobs.
- Canonical data provenance, privacy, and archival requirements.
- Maintainer-defined completion criteria for scientific analyses and figures.
