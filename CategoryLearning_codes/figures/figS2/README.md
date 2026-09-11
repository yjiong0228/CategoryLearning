# Supplementary Fig2: five-part S129 review

Latest search-method check:
[joint search probe](../outputs/figS2/joint_search_probe_v2/README.md).
It includes 43 new fine-budget points, a real Start 4 coordinate-stationary
example, independent PF rescoring, and the C2/C4 comparison across two seed
families and their pooled 32 repeats. The old fine-budget trap does not persist
as a requirement for joint moves under the new scoring; C2 and C4 remain
numerically difficult to distinguish. This is a finite diagnostic probe, not a
full multistart refit or a global-optimum certificate.

```bash
MPLCONFIGDIR=/tmp/fig2_mpl python -m CategoryLearning_codes.figures.figS2.build_joint_search_probe \
  --probe results/model_0826/cond1/subject_129/joint_search_probe_20260910_v1 \
  --output CategoryLearning_codes/figures/outputs/figS2/joint_search_probe_new
```

The source bundle must include the completed four-point trap validation and
the pooled-old-candidate comparison. Plotting performs no new model execution.

The parameter-estimation audit now has two expanded PNGs:
[best advantage](../outputs/figS2/estimation_audit_v2/S2_1_best_advantage.png) and
[search coverage](../outputs/figS2/estimation_audit_v2/S2_1_search_coverage.png).
They add paired numerical uncertainty, the fine-to-final rank reversal, corrected
parameter extraction, and conditional grid coverage around C4. The other four
supplementary topics remain in `grouped_v1`.

```bash
MPLCONFIGDIR=/tmp/fig2_mpl python -m CategoryLearning_codes.figures.figS2.build_estimation_audit \
  --pipeline results/model_0826/cond1/subject_129/pipeline_20260908_v1 \
  --audit results/model_0826/cond1/subject_129/estimation_audit_20260910_v1 \
  --output CategoryLearning_codes/figures/outputs/figS2/estimation_audit_new
```

This command reads archived searches and the completed four-candidate replay;
it does not run the model. Use a fresh output directory. Its grid-distance plot
is descriptive and does not establish a connected near-optimal region.

Current: [grouped_v1 reading guide](../outputs/figS2/grouped_v1/README.md).
The five separate figures replace the crowded ten-panel layout for discussion:
parameter alternatives; behavioral errors; online states and numerical support;
oral diagnostic checks; autonomous learning. Full captions and source CSVs are
stored with the outputs. Main Fig2 is unchanged.

```bash
MPLCONFIGDIR=/tmp/fig2_mpl python -m CategoryLearning_codes.figures.figS2.build_grouped_diagnostics \
  --output CategoryLearning_codes/figures/outputs/figS2/grouped_v2 \
  --model-dir results/model_0826/cond1/subject_129/pipeline_20260908_v1/models/PMH \
  --case-sources CategoryLearning_codes/figures/outputs/fig2/subject129_sources_v1
```

This script reads existing outputs only and requires a new output directory.
Calibration, residual autocorrelation, seed quantiles, entropy and phase means
are descriptive postprocessing. Pending sensitivity/recovery/ablation analyses
are explicitly distinguished from supplied evidence. PNG only, width 183 mm,
450 dpi. No fitting or generation is launched.

## Historical ten-panel preview

# Supplementary Fig2: S129 diagnostic draft

Current PNG: [FigS2 v2](../outputs/figS2/figS2_v2/FigureS2_draft.png).
Ten panels use existing PMH outputs only; no refit, recovery or ablation is run.
Sources and detailed captions accompany the generated figure. This is a
single-participant diagnostic draft, not a completed group supplementary figure.

Reproduce from the repository root, always choosing a new output directory:

```bash
MPLCONFIGDIR=/tmp/fig2_mpl python -m CategoryLearning_codes.figures.figS2.build_subject_diagnostics \
  --output CategoryLearning_codes/figures/outputs/figS2/figS2_v3 \
  --model-dir results/model_0826/cond1/subject_129/pipeline_20260908_v1/models/PMH \
  --case-sources CategoryLearning_codes/figures/outputs/fig2/subject129_sources_v1
```

Panel a: fine-search paths; b: independent final candidate rescore; c–d:
observed-history accuracy and residual; e–f: active-rule probabilities and
online search probability; g: full-space oral overlap; h: terminal ancestry;
i–j: autonomous trajectories and sustained-mastery onset.

Export: PNG only, 183 × 250 mm, 450 dpi. Candidate scores and parameter values
are descriptive point estimates; no parameter uncertainty interval is claimed.
