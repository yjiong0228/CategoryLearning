# Fig. 1 — experimental design and the heterogeneity of learning

## Figure contract

**Claim.** Shared category-learning tasks produce different learning times and
trajectories, while trial-wise reports offer an observable description of the
rules participants express.

Schematic-led composite, Python/matplotlib, 183 × 215 mm, 450 dpi, PNG only.
The three aligned population heatmaps are the quantitative anchor. Experimental
schematics explain what was measured; individual examples connect the population
view to the later belief analyses; compact cohort summaries describe learning
without turning curve shapes into cognitive types. Panel text remains short;
definitions and limits belong in this legend.

| Panel | Evidence | Source / mapping |
|---|---|---|
| a | Continuously varying stimulus and the three category structures | Native Matplotlib stimulus/task drawings, structurally reused from the confirmed figure |
| b | Choice and verbal report precede feedback | Native screen-sequence schematic, restyled with existing animal and microphone primitives |
| c | All 96 individual learning records, 32 per task | `trial_source.csv`: subject × trial, trailing 32-trial accuracy; grey denotes unavailable cells |
| d | Three individual learning curves and explicitly mentioned features | Fixed S122, S206 and S215, also used in Fig. 3; no selection by oral agreement |
| e | Distribution of first criterion crossing | `subject_summary.csv`: `first_crossing64`; unobserved crossings drawn at record end with open triangles |
| f | First-to-last change in fully correct choices | Disjoint first and last 64 trials, n = 95 |
| g | First-to-last change in number of explicitly mentioned features | Same disjoint windows; at least one recognized report per window, n = 95 |

Task 1/2/3 correspond to stored conditions 1/3/2. All 96 participants and 62,720
trials remain in c; no raw data are changed. Quantitative results reuse the
audited behavior-only tables, with subject/trial order, choices and feedback
checked against the current processed input. The new code does not estimate
cognitive states or fit behavioral classes. Existing figures remain unchanged.

## Publication-style legend

**Fig. 1 | Experimental design and individual differences in category learning.**
**a,** Counterbalanced continuous stimulus features defined two categories in
Task 1 and four in Tasks 2–3. Task 2 additionally provided partial feedback for
the correct category pair. **b,** Participants categorized each stimulus and
reported their current rule before feedback; a two-category trial is illustrated.
**c,** All 96 participants (32 per task; 62,720 trials). Color shows fully correct
choices averaged over complete trailing 32-trial windows. Rows are ordered by
record length; grey indicates unavailable estimates. Lower strips show available
record counts. Horizontal ranges differ across tasks and retain actual trial
numbers. **d,** Three illustrative participants also examined in Fig. 3. Lines
show trailing 32-trial accuracy; dashed lines indicate chance and dotted lines
session boundaries. Feature strips show explicit mentions: colored, mentioned;
white, unmentioned; grey, missing text. Mentions do not establish correct rules
or decoded beliefs; examples do not define learner classes. **e,** First trial
reaching at least 90% accuracy in a complete trailing 64-trial window. Points
denote participants; bars show medians among observed crossings. Open triangles
mark unreached criteria at record end (3, 0 and 1 participants in Tasks 1–3).
**f,g,** First and last disjoint 64-trial periods: accuracy (f) and mean explicitly
named features per recognized report (g). Fine lines connect individual
participants; thick lines connect task means. Paired samples contain 31, 32 and
32 participants in Tasks 1–3; the sole 64-trial record remains in c and e but
is excluded here. No inferential statistics are shown. First crossing does not
establish persistent mastery; record length is not learning time.

## Methods notes

- Schematics align counterbalanced feature roles, with decision boundaries at
  0.5 on normalized dimensions. Stimulus and screen images are illustrations.
- Heatmap rows are sorted by record length and then participant identifier.
  Unavailable cells include the first 31 trials and trials beyond record end.
- The three examples are fixed S122, S206 and S215. Explicit mentions describe
  report content and do not encode correctness or uniquely identify a rule.
- S105 has only 64 recorded trials. Paired summaries require at least 128
  trials, and feature-count comparisons additionally require at least one
  recognized report in each period. First/last windows are therefore disjoint.
- All points and paired lines represent participants. There are no error bars,
  statistical tests or significance claims. Stopping and differing record
  lengths can influence the descriptive first/last comparisons.

## Reproduction and checks

Run from the repository root, with a new output directory:

```bash
python -m CategoryLearning_codes.figures.fig1.build_journal_figure \
  --output CategoryLearning_codes/figures/outputs/fig1/journal_20260921_v2
```

The output contains the PNG, exact source tables, an integrity/provenance
manifest, and runtime checks. The confirmed paper figure is never overwritten.
The canonical task palette, native schematic primitives, and feature-strip
definitions are reused; the composition and screen styling are newly built.
All artifact-generating work uses Python. Final visual QA checks alignment,
readability, clipping and legends at the specified physical size. The PNG-only
repository policy overrides the skill's default vector/export bundle.
