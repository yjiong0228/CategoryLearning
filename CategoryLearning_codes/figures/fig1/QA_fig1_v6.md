# Fig. 1 revision QA

## Scope completed

- Development code and all draft outputs moved to `CategoryLearning_codes/figures`.
  The migration was hash-verified before changes; all earlier output hashes were
  checked again after migration. The paper figure directory is empty pending confirmation.
- Continuous feature ranges and aligned task contrasts redrawn from the old draft's
  scientific elements, with midpoint boundaries, neutral outlines and consistent colors.
- Trial procedure includes five overlapping display states, response keys, keyboard,
  ready cue, microphone, feedback and an illustrative verbal report. Task names, subtitles,
  microphone aspect and the speech-bubble tail were revised after rendered inspection.
- Previous duration/gain main panel removed. Example participant IDs are unchanged.
- Each example now has trial-aligned anatomical report strips; all 96 participants have
  companion behavior/oral atlases. Group oral summaries add required-feature coverage
  and task-irrelevant feature fraction to the first/last feature count.
- Data inputs incorporate the separately authorized S319 category correction. No data
  file was modified during this figure revision. No model source was moved or fitted.

## Verification

`python -m pytest -q CategoryLearning_codes/figures`: **9 passed**.

Tests include session-reset ordering, partial-feedback semantics, disjoint gain windows,
missing-report behavior, raw-leaf coarsening, participant-specific anatomy, current-stimulus
branch selection and rejecting a wrong category under the stated task rule.

Independent bundle checks confirmed:

- 62,720 rows, 96 subjects, 32 per condition; exact source-row/key alignment.
- Zero processed/stimulus/feedback category inconsistencies and zero task-rule mismatches.
- Every expected input/output SHA256 matches; migrated old outputs remain byte-identical.
- All oral strip entries are 0/1 or missing; missingness exactly follows unrecognized reports.
- Coverage and irrelevant fractions lie in [0,1]; paired summary means match independently
  sliced first/last trial source tables; one short record is ineligible only for oral pairs.
- Eight PNGs at 450 dpi. Main figure is 3242×3986 px (~183×225 mm). Behavior atlases
  are ~183×240 mm and oral atlases ~183×260 mm review sheets, not final journal panels.
- `git diff --check` passes. Current source imports and commands use the new package path;
  old snapshot paths remain historical provenance. Model source is unchanged.

## Static preflight

The figure skill validator reports 10 PASS, 3 WARN, 1 FAIL for the renderer. The FAIL
is missing SVG/PDF and one WARN is missing TIFF: the user's PNG-only policy overrides
those format expectations. DPI is provided from config and verified in PNG metadata.
The width warning misreads `183/25.4`; exported dimensions verify 183 mm. These exceptions
are explicitly reviewed, not represented as an unqualified submission-preflight pass.

## Scientific interpretation

The current report analysis measures literal mentions, not complete rule semantics.
White strips mean a feature was not explicitly named, not that it was unused mentally.
Known stimulus/task geometry defines the required path. The other branch's feature is
not globally irrelevant. Reduced irrelevant mentions and higher required-feature coverage
are descriptive observations; no inference of workspace capacity or causal mechanism is made.

Gain remains in exploratory checks and historical example-selection provenance only.
It is not interpreted as a learner class or formal changepoint. New main panel d contains
the expanded oral summaries (replacing the previous panel d/e arrangement).
