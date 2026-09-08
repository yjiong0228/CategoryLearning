# Fig. 1 revision contract

The user requested: reserve the paper figure directory for confirmed images; improve
task/procedure illustrations by combining the old draft's scientific elements with the
new layout; explain gain, remove an uninformative duration/gain comparison, and expand
oral data alongside individual learning trajectories. This revision implements that
scope directly. Earlier outputs are migrated intact and kept as historical snapshots.

1. Move development files and all unconfirmed outputs to `CategoryLearning_codes/figures`;
   verify hashes; update current imports/CLI references. Do not migrate model code yet.
2. Test and implement task-relative oral measures, independently of the model catalogue.
   Validate the task rule against **all** corrected category labels. Anatomy maps are
   individual; no universal assumption that a particular body part is relevant.
3. Build reusable matplotlib schematics: continuous four-feature range illustration,
   aligned task trees with midpoint cuts and feedback differences, and overlapping
   trial screens with stimulus/choice, ready cue, microphone/report and feedback.
4. Keep all-cohort heatmaps and the same 9 example IDs. Under each example, show 4
   anatomical report strips sharing its exact trial axis. Display gain only in methods
   and exploratory checks, not as a claimed result or participant type.
5. Replace the old duration/gain main panel with oral evidence: first/last disjoint
   64-trial periods for feature count, required-path feature coverage and task-irrelevant
   feature fraction. Each participant is one paired line; missing report denominators
   remain explicit. This describes explicit text mentions, not internal capacity.
6. Build with corrected inputs into a new output directory; update audit wording so
   previously fixed label discrepancies are not described as current. Export source
   data for every new measure, full-cohort oral atlases, readout and source snapshots.
7. Run relevant tests, raw/derived invariants and visual QA before handoff. PNG only.

## Definitions

For condition 1, the required feature is feature1 and the task-irrelevant set is
feature2–4. For conditions 2 and 3, feature1 is required together with feature2 when
feature1 ≤ 0.5 or feature3 otherwise; feature4 is task-irrelevant. The unused branch
feature is not called irrelevant. Classifying required features uses the actual
stimulus, not the participant's choice or the model's belief. Category numbers are
generic C1–C4 in the task schematic; actual F/J keys are illustrated for Task 1.

Path coverage = fraction of currently required anatomical features explicitly named.
Irrelevant fraction = fraction of globally task-irrelevant anatomical features explicitly
named. Denominator = recognized reports with at least one explicit feature, not all
behavior trials. No-report and unrecognized text retain a missing marker in strips;
white means not explicitly named in a recognized report, not absence of internal use.

Gain retained only for reproducibility: maximum over splits t of mean correct[t:t+32]
minus mean correct[t-32:t]. It is a descriptive maximum, not learning slope, a formal
changepoint, or evidence for distinct learner classes. Example selection remains fixed
from v3 to avoid silently selecting more favorable oral patterns.
