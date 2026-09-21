# Journal supplement QA

Reviewed output: `outputs/figS2/journal_20260921_v4` (relative to the figures root).
Main PNG: `FigureS2_model_resolution.png`; companion:
`FigureS2_belief_trajectories.png`. Both are review drafts, not confirmed paper assets.
Version 4 condenses the main legend to fewer than 300 words and the atlas legend
to fewer than 120, retaining complete definitions in Methods notes. Both PNGs
are byte-identical to the visually inspected version 3; only documentation and
source provenance changed.

## Scientific checks

- All 12 current fitted participants are included, four in each task. Task 2 is
  condition 3. There are 7,936 trials and 7,924 scored trials, with exactly the
  saved first-trial exclusion per person in score-based summaries.
- All 144 saved replays were checked for participant, candidate ID, 128-particle
  budget, choice/feedback identities, masks and finite normalized distributions.
- All 132 selected parameter values are extracted by the shared Model0826 helper
  and checked against the current cohort table. The nearby candidate changes 30
  displayed parameter cells. Full numeric values at both points are exported.
- Source integrity was checked against all 163 recorded input hashes after export.
  No source data, archived fits or prior result bundles were altered.
- Six focused tests pass: hand-calculated distances; equal weighting of distinct
  replay pairs; invariance to order; candidate comparison with unequal replay
  counts; rejection of invalid probabilities and incompatible dimensions.
- Saved oral scores and exact-gap matched report-change summaries are reused
  without encoder changes. Positive correspondence occurs for 10, 9 and 9 people
  at oral-change thresholds .25, .50 and .75, respectively; these are descriptive.
- Mean absolute change in participant-level oral compatibility between candidate
  points is .01053. This does not establish that complete latent trajectories are
  equally stable: repeat mean-pairwise belief TV ranges .065–.647, and is largest
  for S328. Candidate and replay ranges do not constitute parameter recovery.
- Panel f contains all four step-over-trend BIC>=6 participants, selected from
  behavior alone. All other panels and the full-trajectory atlas retain all 12.

## Visual and export checks

Native Python/matplotlib figures; no external raster assets, cropping of source
images, image synthesis or graphical alteration of data. Both rendered PNGs were
inspected, including matrix values, small panels, dense trajectories, legends,
range bars and highlighted low-compatibility cases. Numeric cells and shared
legends are legible, with no clipped labels. A leader distinguishes the two
low-compatibility participant labels.

- Main: 3242 × 3897 pixels, approximately 183 × 220 mm at 450 dpi.
- Atlas: 3242 × 3720 pixels, approximately 183 × 210 mm at 450 dpi.
- Figure text: sans-serif, minimum explicit 5.3 pt, mostly 6–7 pt, panel letters 8 pt.
- Whites, muted task colours, teal beliefs and grey nearby-candidate trajectories;
  line style/fill distinguishes parameter points as well as colour.
- Detailed captions and the scientific limits are outside the plot canvas.

The skill's static source preflight reports 10 passes, one failure and three
warnings. The failure asks for SVG/PDF and one warning asks for TIFF; repository
instructions explicitly require PNG-only drafts, so these formats are not added.
The 450-dpi warning reflects the generic 600-dpi default, not this export contract.
The width warning is a parser false positive: it reads `183 / 25.4` as 183 inches;
the physical widths were measured from PNG metadata and verified above.

## Interpretation limits

No new model fits, autonomous interventions, model comparisons or recovery jobs
were run. All current fits retain unresolved strict stopping diagnostics.
Historical recovery output failed numerical calibration and is not mixed into
this cohort. Selected-versus-nearby points are finite robustness checks, not a
confidence region, global-optimum certificate, ablation or posterior sample.
Probability distances for choices and full rules concern different state spaces;
their comparison describes numerical sensitivity at those two resolutions, not
an information-equivalence test. All intervals shown are particle-replay ranges,
not confidence intervals. Oral scores preserve report ambiguity and depend on
the existing catalogue. The atlas is observed-history filtering with parameters
estimated from the complete choice sequence, not held-out trajectory prediction.
