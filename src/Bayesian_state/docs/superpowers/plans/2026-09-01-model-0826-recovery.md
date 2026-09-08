# Model0826 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the reproducible full-trial Model0826 PF calibration, P/PM/PH/PMH module recovery, chi readout recovery, and multi-trajectory parameter recovery workflow.

**Architecture:** Reuse the public Model0826 engine, autonomous generator, particle filter, and Hyper-CD 2.0 rather than duplicating model equations. A focused Model0826 optimization module defines named parameter paths and architecture cells; a recovery module owns synthetic datasets, numerical calibration, fixed-parameter scoring, summaries, and PNG figures; one script orchestrates resumable phases.

**Tech Stack:** Python 3, NumPy, pandas, SciPy, Matplotlib, PyYAML, joblib, pytest, Bayesian_state simulation/inference/optimization APIs.

**Spec:** `docs/superpowers/specs/2026-09-01-model-0826-recovery-design.md`

## Global Constraints

- Use subjects 101/111/118 with exactly 320/320/256 valid condition-1 trials.
- Use real schedules and fixed perception parameters, but never observed Task2 choices for synthesis.
- Keep Model0826 P/M/H/beta/readout equations frozen.
- Average PF seed probabilities before computing NLL; never average generated trajectories.
- Module selection uses prefix-only parameter fitting and suffix-only frozen-parameter evaluation.
- Write only to the new `results/model_0826/recovery_v1/` tree and never overwrite existing results.
- Emit plots as PNG only.

---

### Task 1: General Model0826 parameter-space validation

**Files:**
- Modify: `src/Bayesian_state/optimization/parameter_space.py`
- Modify: `configs/specific_models/model_0826_cond1_parameter_space.yaml`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces: `load_model_parameter_space(path, expected_model_id=None)` and backward-compatible `load_parameter_space(path)`.
- Consumes: the existing Model0818/0826 YAML schemas.

- [ ] **Step 1: Write failing tests loading both frozen versions**

```python
def test_parameter_loader_accepts_0818_and_0826_without_cross_version_aliasing():
    old = load_model_parameter_space(PATH_0818, expected_model_id="model_0818")
    new = load_model_parameter_space(PATH_0826, expected_model_id="model_0826")
    assert old["provenance"]["model_id"] == "model_0818"
    assert new["provenance"]["model_id"] == "model_0826"
```

- [ ] **Step 2: Run the test and verify Model0826 is currently rejected**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k parameter_loader`

- [ ] **Step 3: Generalize validation while preserving the 0818 alias**

Parameter names, domains, architecture cells, fixed constants, subject ordering, and manuscript hash remain validated; only the accepted model id and required `event_history_excludes_latest_error` provenance differ.

- [ ] **Step 4: Add explicit fine supports to the 0826 YAML**

Use these sorted supports exactly:

```yaml
gamma: [0.0, 0.125, 0.25, 0.375, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.935, 0.97]
E_C: [0.02, 0.06, 0.10, 0.175, 0.25, 0.375, 0.50, 0.625, 0.75]
delta_E: [0.0, 0.125, 0.25, 0.36478654013094315, 0.4795730802618863, 0.6397865401309432, 0.80, 1.20, 1.60, 2.40, 3.20]
g_0: [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0]
c_A: [0.0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0, 1.50, 2.0, 3.0, 4.0, 5.0, 6.0]
c_G: [0.0, 0.05, 0.10, 0.175, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
beta_0: [0.50, 0.75, 1.0, 1.75, 2.50, 3.75, 5.0, 7.50, 10.0, 15.0, 20.0]
eta_plus: [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32]
eta_minus: [0.01, 0.02, 0.03, 0.05, 0.07, 0.11, 0.15, 0.225, 0.30, 0.45, 0.60, 0.80, 1.0]
```

The joint `(M,chi)` fine support is identical to its coarse support. Zero-spike parameters retain the literal `0.0` separately from positive values.

- [ ] **Step 5: Run 0818 and 0826 validation tests**

Run: `python -m pytest -q tests/bayesian_state/test_model_0818_parameter_space.py tests/bayesian_state/test_model_0826_versioning.py tests/bayesian_state/test_model_0826_recovery.py`

- [ ] **Step 6: Commit the task**

```bash
git add src/Bayesian_state/optimization/parameter_space.py configs/specific_models/model_0826_cond1_parameter_space.yaml tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): validate recovery parameter support"
```

### Task 2: Named parameters and four architecture cells

**Files:**
- Create: `src/Bayesian_state/optimization/model_0826.py`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces: `build_model_0826_cell_engine(base_engine, cell)`, `build_model_0826_hyper_config(analysis_config, parameter_space, cell, base_sim_config_path, output_dir, stage_budgets)`, `extract_model_0826_parameters(hyperparams)`.
- Consumes: `BayesianMemoryModule`, `DualMemoryModule`, Model0826 H config, profile-coordinate expansion.

- [ ] **Step 1: Write failing exact-cell tests**

```python
@pytest.mark.parametrize("cell,has_m,has_h", [
    ("P", False, False), ("PM", True, False),
    ("PH", False, True), ("PMH", True, True),
])
def test_cell_builder_changes_only_m_and_h(base_engine, cell, has_m, has_h):
    engine = build_model_0826_cell_engine(base_engine, cell)
    assert ("hypo_transitions_mod" in engine["modules"]) is has_h
    assert (engine["modules"]["memory_mod"]["class"].endswith("DualMemoryModule")) is has_m
    assert engine["modules"]["beta_mod"]["kwargs"]["update_scope"] == "active_hypotheses"
```

- [ ] **Step 2: Run the cell tests and verify missing implementation**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k cell_builder`

- [ ] **Step 3: Implement deep-copy cell assembly and named paths**

P/PM remove H from both `modules` and `agenda`; P/PH use `BayesianMemoryModule`; every cell retains readout power 1, zero lapse, and dynamic beta.

- [ ] **Step 4: Implement cell-specific coarse/fine Hyper-CD configurations**

Only the free parameters declared by `architecture_cells.<cell>.free_parameters` appear as coordinates. `(M,chi)`, `(E_C,delta_E)`, and `(g_0,c_G)` are packed mapping coordinates.

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py tests/bayesian_state/test_model_0826_versioning.py`

- [ ] **Step 6: Commit the task**

```bash
git add src/Bayesian_state/optimization/model_0826.py tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): build recovery architecture cells"
```

### Task 3: Frozen recovery configuration and synthetic datasets

**Files:**
- Create: `configs/specific_models/model_0826_recovery_v1.yaml`
- Create: `configs/simulation_cfg/model0826_cond1_recovery_base.yaml`
- Create: `src/Bayesian_state/evaluation/model_recovery.py`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces: `RecoveryDesign`, `load_recovery_design()`, `generate_synthetic_dataset()`, `synthetic_dataset_frame()`.
- Consumes: `run_autonomous_category_learning()` and real condition-1 schedule rows.

- [ ] **Step 1: Write failing design-count and trial-count tests**

```python
def test_design_has_exact_pre_registered_counts(design):
    assert design.subject_trial_counts == {101: 320, 111: 320, 118: 256}
    assert len(design.module_datasets) == 36
    assert len(design.parameter_datasets) == 40
    assert Counter(x.truth["chi"] for x in design.parameter_datasets) == {0: 20, 1: 20}
```

- [ ] **Step 2: Run tests and verify missing config/loader failures**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k design`

- [ ] **Step 3: Encode all truths, assignments, seeds, gates, and output paths in YAML**

Copy the exact 36/40 dataset design and six contrast profiles from the design spec. Set `max_trials: null`, `subjects: [101,111,118]`, and output root `results/model_0826/recovery_v1`.

- [ ] **Step 4: Implement autonomous generation and atomic NPZ/CSV artifacts**

Each dataset stores full stimulus, category, sampled choice, generated feedback, template subject, truth, generation seed, and schedule fingerprint. Reject an existing output unless its manifest fingerprint matches exactly.

- [ ] **Step 5: Test that replacing observed choice does not alter the generated-seed input**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k 'synthetic or observed_choice or trial_count'`

- [ ] **Step 6: Commit the task**

```bash
git add configs/specific_models/model_0826_recovery_v1.yaml configs/simulation_cfg/model0826_cond1_recovery_base.yaml src/Bayesian_state/evaluation/model_recovery.py tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): generate pre-registered recovery datasets"
```

### Task 4: PF calibration with paired and independent ensembles

**Files:**
- Modify: `src/Bayesian_state/evaluation/model_recovery.py`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces: `build_calibration_bank()`, `score_pf_bank()`, `summarize_pf_calibration()`, `freeze_smallest_passing_budget()`.
- Consumes: `run_state_model_particle_filter()` and seed-averaged probability scoring.

- [ ] **Step 1: Write failing bank, nesting, and gate tests**

```python
def test_calibration_bank_is_eight_fixed_candidates():
    bank = build_calibration_bank(ANCHOR)
    assert len(bank) == 8
    assert {(x["chi"], x["variant"]) for x in bank} == EXPECTED_BANK

def test_calibration_freezes_smallest_budget_passing_every_gate():
    summary = summarize_pf_calibration(PASSING_SYNTHETIC_SCORES)
    assert freeze_smallest_passing_budget(summary) == {"R": 64, "B": 8}
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k calibration`

- [ ] **Step 3: Implement probability, rank, winner, and MCSE diagnostics**

Use paired logical seeds for R16/R32/R64 ensemble A, prefixes B4/B8, and a disjoint ensemble B. Compute per-dataset candidate rank Spearman, winner agreement, probability RMSE, and trialwise MCSE quantiles.

- [ ] **Step 4: Implement escalation to R128/B16 and fail-closed status**

Only `freeze_smallest_passing_budget()` may write `frozen_budget.json`; it returns no budget when the maximum setting fails.

- [ ] **Step 5: Run calibration tests**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k calibration`

- [ ] **Step 6: Commit the task**

```bash
git add src/Bayesian_state/evaluation/model_recovery.py tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): calibrate particle-filter recovery budget"
```

### Task 5: Dataset-level Hyper-CD fitting and leakage-free held-out scoring

**Files:**
- Modify: `src/Bayesian_state/evaluation/model_recovery.py`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces: `fit_recovery_dataset()`, `score_frozen_candidate()`, `mean_probability_nll()`.
- Consumes: Hyper-CD 2.0, synthetic CSVs, frozen PF budget, prefix/suffix masks.

- [ ] **Step 1: Write failing aggregation and leakage tests**

```python
def test_nll_is_computed_after_probability_averaging():
    assert mean_probability_nll(PROBABILITY_RUNS, CHOICES, MASK) == pytest.approx(EXPECTED)

def test_module_fit_never_uses_suffix_during_parameter_selection(fake_optimizer):
    fit_recovery_dataset(MODULE_DATASET, fake_optimizer)
    assert fake_optimizer.score_roles == ["optimization"]
    assert frozen_scorer.last_role == "evaluation"
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k 'nll or leakage or heldout'`

- [ ] **Step 3: Implement per-dataset resolved YAML and Hyper-CD invocation**

Write immutable dataset simulation YAMLs under the result tree. Module fits set `train_fraction: 0.70`; parameter fits set evaluation mode `all`. Never pool datasets.

- [ ] **Step 4: Implement frozen suffix scoring with disjoint seeds**

After prefix final-rescore chooses one parameter point per cell, run full online filtering with new seeds and compute total NLL only on suffix indices 224:320 or 179:256.

- [ ] **Step 5: Run fit tests**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k 'fit or nll or heldout'`

- [ ] **Step 6: Commit the task**

```bash
git add src/Bayesian_state/evaluation/model_recovery.py tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): fit recovery datasets without suffix leakage"
```

### Task 6: Recovery summaries and PNG figures

**Files:**
- Modify: `src/Bayesian_state/evaluation/model_recovery.py`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces: `summarize_module_recovery()`, `summarize_parameter_recovery()`, `plot_module_recovery()`, `plot_parameter_recovery()`.
- Consumes: fit-score rows and pre-registered gates.

- [ ] **Step 1: Write failing confusion, zero-boundary, and near-best tests**

```python
def test_module_summary_uses_total_nll_and_delta_two_near_best():
    summary = summarize_module_recovery(MOCK_MODULE_SCORES)
    assert summary["overall_exact_recovery"] == pytest.approx(0.75)
    assert summary["true_cell_near_best_coverage"] == pytest.approx(1.0)

def test_parameter_summary_separates_zero_positive_classification():
    summary = summarize_parameter_recovery(MOCK_PARAMETER_ROWS)
    assert "zero_positive_balanced_accuracy" in summary["c_A"]
```

- [ ] **Step 2: Run tests and verify failures**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k summary`

- [ ] **Step 3: Implement statistics and gate decisions**

Report confusion matrices, Wilson intervals, bias/MAE/RMSE/Spearman, normalized MAE, zero/positive balanced accuracy, near-best coverage, and parameter-error correlations. Emit a per-parameter `supported` boolean using the exact spec thresholds.

- [ ] **Step 4: Implement source-backed figures**

Create `module_recovery_overview.png` and `parameter_recovery_overview.png`; every plotted number must also occur in a CSV. Do not emit PDF/SVG/TIFF.

- [ ] **Step 5: Run summary tests**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k 'summary or plot'`

- [ ] **Step 6: Commit the task**

```bash
git add src/Bayesian_state/evaluation/model_recovery.py tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): summarize module and parameter recovery"
```

### Task 7: Resumable CLI, manifests, and documentation

**Files:**
- Create: `scripts/run_model_0826_recovery.py`
- Modify: `src/Bayesian_state/optimization/README.md`
- Modify: `src/Bayesian_state/evaluation/README.md`
- Modify: `src/Bayesian_state/README.md`
- Test: `tests/bayesian_state/test_model_0826_recovery.py`

**Interfaces:**
- Produces CLI phases `smoke`, `generate`, `calibrate`, `module-fit`, `parameter-fit`, `summarize`, `all` and `--resume`.
- Consumes all earlier recovery interfaces.

- [ ] **Step 1: Write failing CLI and collision-policy tests**

```python
def test_recovery_cli_has_all_pre_registered_phases():
    parser = build_parser()
    assert parse_phase_choices(parser) == {
        "smoke", "generate", "calibrate", "module-fit",
        "parameter-fit", "summarize", "all"
    }

def test_existing_output_requires_resume_and_matching_manifest(tmp_path):
    with pytest.raises(FileExistsError):
        prepare_output(tmp_path / "recovery_v1", resume=False)
```

- [ ] **Step 2: Run CLI tests and verify failures**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py -k cli`

- [ ] **Step 3: Implement orchestration and atomic dataset completion records**

Each phase skips only artifacts with matching fingerprints and terminal success state. Calibration failure prevents both formal fit phases. `smoke` writes to `recovery_v1/smoke` and still uses complete subject sequences.

- [ ] **Step 4: Document commands and interpretation boundaries**

Document that PF is numerical integration, generated trajectories are separate observations, held-out scores select modules, and unsupported parameters must not be reported as stable individual differences.

- [ ] **Step 5: Run final code verification**

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py tests/bayesian_state/test_hyper_cd_v2.py tests/bayesian_state/test_model_0826_versioning.py tests/bayesian_state/test_model_0818_parameter_space.py`

Run: `python scripts/run_model_0826_recovery.py --help`

- [ ] **Step 6: Commit the task**

```bash
git add scripts/run_model_0826_recovery.py src/Bayesian_state/optimization/README.md src/Bayesian_state/evaluation/README.md src/Bayesian_state/README.md tests/bayesian_state/test_model_0826_recovery.py
git commit -m "feat(model0826): add resumable recovery workflow"
```

### Task 8: Execute smoke, calibration, formal recovery, and completion audit

**Files:**
- Create outputs only under: `results/model_0826/recovery_v1/`

**Interfaces:**
- Produces the complete result tree and final scientific conclusion.

- [ ] **Step 1: Run the full-trial smoke**

Run: `python scripts/run_model_0826_recovery.py --phase smoke`

Verify manifest trial counts are `{101: 320, 111: 320, 118: 256}` and all probability rows are finite and normalized.

- [ ] **Step 2: Run generation and PF calibration**

Run: `python scripts/run_model_0826_recovery.py --phase generate`

Run: `python scripts/run_model_0826_recovery.py --phase calibrate --resume`

Proceed only if `numerical_calibration/frozen_budget.json` has `status: passed`.

- [ ] **Step 3: Run module recovery**

Run: `python scripts/run_model_0826_recovery.py --phase module-fit --resume`

Verify 36 dataset IDs, 144 architecture fits, disjoint prefix/suffix scoring, and no failed dataset.

- [ ] **Step 4: Run parameter and readout recovery**

Run: `python scripts/run_model_0826_recovery.py --phase parameter-fit --resume`

Verify 40 independent datasets, chi truth balance 20/20, and one fitted point per trajectory.

- [ ] **Step 5: Summarize and audit every required artifact**

Run: `python scripts/run_model_0826_recovery.py --phase summarize --resume`

Run: `python -m pytest -q tests/bayesian_state/test_model_0826_recovery.py tests/bayesian_state/test_hyper_cd_v2.py`

Check that all CSV/PNG/JSON outputs named in the spec exist, the figures are non-empty PNG files, and `final_report.json` derives its claims from pre-registered gates.

- [ ] **Step 6: Commit only code/config/docs; retain generated results without adding large caches to git**

```bash
git status --short
git add scripts/run_model_0826_recovery.py src/Bayesian_state configs/specific_models/model_0826_recovery_v1.yaml configs/simulation_cfg/model0826_cond1_recovery_base.yaml tests/bayesian_state docs/superpowers
git commit -m "analysis(model0826): complete recovery workflow"
```
