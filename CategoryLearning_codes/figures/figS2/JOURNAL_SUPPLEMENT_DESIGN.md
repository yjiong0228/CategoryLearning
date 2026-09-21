# FigS2: reliability and resolution of reconstructed beliefs

## Figure contract

**Conclusion.** Choice predictions can be more stable than decoded rule beliefs;
particle replays, nearby parameter points and report definitions therefore bound
the resolution of the interpretations in Figs2–4.

**Archetype.** A quantitative grid supporting the main model figure, accompanied
by a full-cohort trajectory atlas. The supplement answers reliability questions;
it does not repeat the main model schematic or present an optimization dashboard.

**Evidence chain.**

- a: all fitted resource, readout and search parameters, with nearby-candidate changes.
- b: within-parameter particle sensitivity, measured on full distributions.
- c: between-parameter sensitivity on the same probability-distance scale.
- d: whether current-report content correspondence survives the parameter check.
- e: whether report-change correspondence survives the change threshold.
- f: whether pre-improvement target belief survives both replay and window choices.
- Companion atlas: every fitted participant, showing complete target-belief
  trajectories at both parameter points and their particle-replay ranges.

The same 12 participants are used throughout, except panel f, which follows the
existing behavior-only criterion of step-over-trend BIC advantage at least 6
(S102, S307, S328 and S215). No participants are removed from cohort panels.

## Data and definitions

Read only the latest 12-participant state bundle and its derived tables, plus the
current-report validation tables. Reuse the existing sigma=0.05 oral encoder.
The selected parameter point has 8 independent 128-particle replays; one nearby
candidate has 4. Neither the candidate nor these repeats are independent people.
The alternate candidate is a search finalist, not a model ablation or posterior
draw. All parameters were fitted to the complete choice sequence.

For each participant, mean pairwise total-variation distance is computed over all
28 distinct pairs of selected-parameter replays, and over scored trials. Candidate
distance compares the mean distributions of the two parameter points. Total
variation is half the L1 distance, bounded by 0 and 1. Report metrics and event
definitions are reused exactly from their existing source tables.

The atlas shows unsmoothed pre-choice target-rule probabilities. Its bands and all
range bars are observed minima and maxima across particle replays, not confidence
intervals. The first trial is omitted only from scored summaries, following the
existing valid/score masks; the atlas retains the full trajectory.

## Compatibility audit and limits

Archived recovery bundles do not provide completed, compatible validation for
this cohort. In particular, the previous subject-first recovery manifest records
failed numerical calibration and failed priority-all execution. Do not turn this
into a parameter-recovery plot or infer recovery from agreement between replays.
Older S129 estimation probes concern other fitted outputs. No fair alternative
cognitive-model comparison or module ablation is available for these 12 people.
All current fits retain unresolved strict stopping diagnostics.

## Visual and export contract

Python/matplotlib exclusively, following the saved backend preference. Native
plots, restrained task colours, Q in teal, nearby candidate in neutral grey;
6–7 pt text and bold lower-case panel labels. Main figure 183 × 220 mm; atlas
183 × 210 mm; PNG only at 450 dpi as required by repository policy. Source data,
input hashes, definitions, exclusions and detailed captions accompany the images.
Build in a new versioned output directory and preserve every existing artifact.

The source is built anew; only established task-colour and typography conventions
are inherited. No legacy template statistics, recovery panels or simulated values
are carried into the new figure.
