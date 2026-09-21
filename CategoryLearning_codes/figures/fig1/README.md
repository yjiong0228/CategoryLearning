# Fig1 development

最新期刊版：[Fig1](../outputs/fig1/journal_20260921_v2/Figure1_journal.png) ·
[设计与图注](JOURNAL_FIGURE.md) · [整套Fig1–4/S2](../JOURNAL_SET_20260921.md)。
恢复原生任务/试次示意，以全96人的轨迹总览为主体；入口 `build_journal_figure.py`。

2026-09-21 较早审阅草稿：[摘要主线版 Fig1](../outputs/fig1/learning_story_20260921_v2/Figure1_learning_story.png)。
保留96人的完整行为基础，加入学习时点、行为形状、后期表现，并与最新 Fig3 的例子衔接。
入口为 `build_learning_story.py`；[设计](LEARNING_STORY_DESIGN.md)、[读图与复现](../fig2/VALIDATION_STORY.md)。
下文仍记录已确认的 v10，其正式副本没有替换。

Fig1 review bundle: [Fig. 1 v10](../outputs/fig1/fig1_v10/fig1_behavior_draft.png),
[non-oral candidates](../outputs/fig1/fig1_v10/figS4_nonoral_candidates.png),
[readout/legend](../outputs/fig1/fig1_v10/readout.md), [revision contract](REVISION_v10.md).

All outputs are grouped by figure number; see the [output index](../outputs/README.md).

## Directory policy

All code, tests, audit tables, source snapshots and unconfirmed figures are in
CategoryLearning_codes. CategoryLearning_paper/figures is reserved for confirmed
images; Figure1.png is the confirmed v10 copy. Previous outputs are preserved. The v1–v3 directory
migration is recorded in migration_20260907.json (its paths refer to the original figures root). The journal model workflow is in `../../Bayesian_model/`; its shared implementation is in `src/Bayesian_state/`.

## Reproduce

From the repository root, choose a fresh output directory:

```bash
MPLCONFIGDIR=/tmp/categorylearning-mpl python -m CategoryLearning_codes.figures.fig1.build_fig1 --output CategoryLearning_codes/figures/outputs/fig1/fig1_v11
python -m pytest -q CategoryLearning_codes/figures CategoryLearning_codes/tests
```

Existing output directories are refused. PNG only, 450 dpi, main figure 183×225 mm.
Outputs and snapshots are excluded from pytest collection and git tracking.

## Scientific scope

Task1 = condition1; Task2 = condition3 (hierarchical partial feedback); Task3 =
condition2. All 96 participants and all ambiguous stimuli are retained. Full
correctness means feedback=1. Categories are checked against the actual task rule
and raw STI/BHV tables, coarsening four leaf labels for the two-category task.
Recording duration is not asserted to be mastery time. Curves use trailing 32 trials;
session boundaries are marked. Nine illustrative participants remain fixed from v3.

Oral fields come exclusively from process_use and FEATURE_NAME_TO_PART reordering
in Preprocessor_B. F1–F4 task-colored short marks are explicit mentions; missing text is gray.
The mean report feature count uses reports containing at least one recognized
feature. Zero-coded missing text is never treated as an observed zero-feature report.

Panel d retains oral feature count and adds paired first/last median choice RT.
Both use disjoint 64-trial periods; S105's 64-trial record cannot form a pair.
The redundant oral coverage and irrelevant-feature graphs are removed from the
main image; their descriptive values remain available in source tables.

The separate non-oral sheet explores choice RT (including correct-choice sensitivity),
block-centered RT conditional on previous feedback, and accuracy versus objectively
defined nearest category boundary. No significance tests, causal feedback claims,
latent strategies or fitted models are used. Counts and trial-level values are saved.
Feedback-history descriptions retain sparse cells and extreme individual values;
stimulus difficulty and learning-stage confounds are documented in the readout.

## Modules

- behavior.py: chronological ordering, source audit, descriptive accuracy and oral metrics.
- schematics.py / render.py: task and procedure schematics, main figure and atlases.
- nonoral.py: RT, feedback-history and boundary-distance computations and candidate sheet.
- reporting.py: definitions, descriptive results and figure legend.
- build_fig1.py: entry point, independent coding checks, output bundle and provenance.
- config.json: task mapping, fixed examples, windows and export contract.

Source snapshots preserve old implementations for reproducibility. Use current code
for new work; historical files may describe earlier conventions and paths.
