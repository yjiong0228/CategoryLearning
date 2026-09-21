# Fig2 journal revision QA

Final render: `../outputs/fig2/journal_20260921_v2/Figure2_journal.png`.
Native original Fig2a is reused directly, not raster-cropped or replaced by text boxes.
The seven-panel structure remains: framework, three parallel task examples, three cohort panels.

## Scientific checks

- 12 participants, four per task; all 7,936 saved trials remain in the source table.
- Formal scores use the previous common mask: 7,924 trials, 7,600 current reports.
- All 29/116 hypothesis rows are retained, in original order; no low-support rules are hidden.
- Belief rows normalize to one. Current oral validity matches the original archive.
- Recomputed full-rule compatibility for every displayed report matches the validated source table.
- Model heatmaps encode Q. Report heatmaps encode relative likelihood, O/max O, separately labelled.
  The report has been conditioned on its actual chosen category; no invented post-choice Q is used.
- Cases are illustrative. Cohort panels include S122/S314 and every other participant.
- Model fit is not held-out prediction; static/uniform references are not alternative cognitive models.
  The absent module-ablation analysis is not represented by made-up numbers or a labelled blank panel.

## Visual review

Both the authoring agent and an independent figure agent inspected the first render.
Two defects were corrected in v2: the task legend overlapped lower panel headings,
and a fixed y-limit clipped S118's changed-report value (0.175363).
The redundant task legend was removed (task colours are directly labelled above);
the last panel now derives its upper limit from all values, with headroom (0.20).
The last panel contains all 24 stable/changed endpoints.
The label “Choice calibration” identifies the full-sequence fit; “Report support”
and the relative-likelihood colourbar distinguish report encoding from model probability.
The final full-size render was inspected: labels and endpoints are visible, heatmaps
are aligned by trial, and there is no large report title or explanatory footer.

## Validation and export

The shared figure/journal analysis suites passed: **60 tests**.
This layout reuses the already-tested compatibility and interval-matching statistics;
there is no new model implementation, model fit, or simulation.
The new entry point was run to completion with actual data; syntax/import checks alone
are not represented as end-to-end validation.

Static figure preflight: 10 passes, 1 failure, 3 warnings. The failure asks for SVG/PDF;
repository instructions require PNG-only drafts. TIFF is omitted for the same reason.
450 dpi follows the existing project export contract, above the validator's 300-dpi floor.
The width warning mis-parses 183/25.4 as inches: the actual PNG is 3242×4074 pixels,
183×230 mm at approximately450 dpi. Sans-serif and editable-font settings are explicit.

Package versions, source hashes, original illustrative episode and code snapshot are
saved with the output. Existing drafts and confirmed manuscript figures remain unchanged.
