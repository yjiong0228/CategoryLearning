# Fig. 1 v3 review and QA

## Delivered scope

One exploratory main figure and four supporting PNGs: three full-cohort task atlases
and one window/report sensitivity figure. Task schematics are illustrative, not screenshots;
no stimulus duration or verified stopping criterion is claimed. No model was fitted or
modified, and no old manuscript/reference file or data file was changed.

## Verification performed

- `python -m pytest -q CategoryLearning_paper/figures/test_behavior.py`: 6 passed.
- Complete build CLI executed successfully, and `--help` checked.
- All 62,720 source-row keys reconstructed and compared to the processed CSV.
- All 96 participants retained (32 in each condition); zero behavioral row exclusions.
- Processed choice/feedback checked against all 96 raw behavior files; raw stimulus
  joins use session + stimulus ID and validate unique mappings.
- Binary task requires coarsening four raw stimulus leaves: {1,2}->1, {3,4}->2.
  This mapping exactly agrees with all 10,048 binary-task behavior labels.
- Remaining category discrepancy: 139 rows in S319/session 5; 129 alter choice/label
  equality. Full feedback agrees with the task-level raw stimulus labels on every row.
- Every subject's maximum adjacent-window gain independently recomputed using
  convolution (16/32/64 windows), rather than the production cumulative-sum algorithm.
- Source, 192 raw files, and generated output hashes independently rechecked.
- PNG dimensions and embedded DPI checked: main 3242 × 3986 px (~183 × 225 mm),
  atlases 3242 × 4251 px (~183 × 240 mm), sensitivity 3242 × 2303 px (~183 × 130 mm),
  all 450 dpi. Supporting atlases are review sheets, not journal-size final panels.
- Visual inspection of all five figure designs. Revisions moved overlapping example
  labels outside axes, restored a clipped panel letter, and expanded the gain axis to
  show the negative-gain participant. No values were clipped to create a learning rise.

## Static figure preflight and policy exceptions

The skill validator reports 10 PASS, 3 WARN, 1 FAIL. It is **not** reported as an
unqualified passing submission preflight. Findings are adjudicated as follows:

- Missing SVG/PDF (FAIL) and TIFF (WARN): superseded by the user's repository-wide
  requirement to generate and retain PNG only. No redundant formats were generated.
- DPI not detected (WARN): DPI is read from config and passed to `savefig`; actual
  image metadata independently confirms 450 dpi.
- Width 4648.2 mm (WARN): the static parser misinterprets arithmetic in `183/25.4`.
  Actual exported pixel dimensions / DPI independently confirm 183 mm.

These are reviewed policy/static-parser exceptions, not evidence of final journal
submission readiness. No exact target journal has been chosen.

## Interpretation boundaries

Maximum gain depends on record length and smoothing. Examples illustrate a range,
not inferred learner classes. Actual record duration is not certified mastery time.
Ambiguous trials are retained, with a non-ambiguous sensitivity overlay. Rolling windows
can bridge session boundaries, which are shown with dotted lines. Task-specific chance
lines denote uniform choice and are not an adaptive-stimulus baseline.

Oral feature counts are lexical descriptions, not latent workspace size or validated
full strategies. The first/last panel excludes S105 only from paired oral comparisons
because its 64-trial record cannot supply disjoint 64-trial periods; S105 remains in
all cohort behavioral summaries and its task atlas. Other report denominators exclude
missing/unrecognized text and retain per-subject valid counts.

An exploratory observation worth pursuing: 30/32 Task-2 participants have a last-period
mean explicit feature count within 0.1 of two (25/32 for Task 3). This was examined after
plotting and is not a preregistered threshold or inferential result. It motivates checking
which features and relations are reported, not assuming a two-slot cognitive workspace.

## Pending scientific work

Verify contemporary experimental stopping/adaptive-sampling rules and detailed display
timings. Extend oral semantic validation beyond literal mentions. Assess learning-shape
evidence on trialwise responses while controlling sequence length and session changes.
Fig. 2 empirical panels await formal fits; four-category model support must be established
before extending the binary 0826 manuscript's claims to all tasks.
