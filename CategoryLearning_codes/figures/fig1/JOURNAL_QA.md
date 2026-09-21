# Fig. 1 journal-layout QA

## Delivered draft

`outputs/fig1/journal_20260921_v2/Figure1_journal.png` (relative to the figures
directory): 3,242 × 3,809 pixels, 450.0118 dpi in PNG metadata, corresponding
to the requested 183 × 215 mm canvas. White background; Arial with DejaVu Sans
fallback; task colors and panel conventions shared with the revised figure set.

The complete image was opened and visually inspected after rendering. There
are no clipped labels, overlapping panels, detached legends or footer paragraphs.
An artist-level canvas check found no text outside the canvas. The shorter
Task 1 heatmap uses its full horizontal extent; all three x axes retain actual
trial numbers and explicitly show their different endpoints.

Version 2 updates only the external figure legend and provenance. Its PNG is
byte-identical to the visually inspected version 1; no graphical or numerical
changes were made. Additional technical definitions were retained in Methods
notes rather than omitted from the concise publication legend.

## Scientific validation

- Retained all 96 participants and 62,720 trials in the individual overview.
- Exact identity, session, block, trial, choice and feedback agreement with the
  current processed input; contiguous within-participant trial numbering.
- Recomputed every displayed rolling accuracy from `feedback == 1`; values
  agree with the audited tables, including the missing first 31 estimates.
- Each task has 32 records; criterion-unreached counts are 3, 0 and 1.
- Paired first/last summaries use disjoint 64-trial windows: n = 31, 32, 32.
  The single 64-trial participant remains in the population and criterion panels.
- `python -m pytest -q CategoryLearning_codes/figures/fig1`: **12 passed**.
- No cognitive fitting, simulations, numerical-result changes or source-data
  changes. The confirmed figure and all preceding drafts are preserved.

## Static preflight resolution

The source validator reports 10 passes, three warnings and one failure.
The failure requests SVG/PDF, and one warning requests TIFF; these are waived
because the repository explicitly requires PNG-only output unless requested
otherwise. The 450-dpi warning reflects the skill's 600-dpi default; the
requested 450-dpi export passes the tool's 300-dpi floor. The width warning
misparses the expression `183 / 25.4` as 183 inches; actual dimensions are
verified above. No unresolved source or rendering problem was identified.

## Interpretation limits

This is a behavior-only figure. Neither first criterion crossing nor final
performance proves stable mastery. First/last improvement is descriptive and
partly influenced by stopping and varying record lengths. The reported-feature
strips show words used, not uniquely decoded rules. S122, S206 and S215 are
fixed cross-figure illustrations, not representative estimates or cognitive
classes; the deliberately retained S122 example has imperfect oral/model
agreement in the later analyses. Dots and lines denote participants, not trials
or particle-filter replicates. No inferential p values are shown.
