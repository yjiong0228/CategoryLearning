# Fig. 3 — journal revision

## Figure contract

**Claim:** Performance gains and inferred support for the task rule can follow different time courses, including repeated losses of previously sustained support.

**Evidence chain:** a–c show the three existing behavior-selected examples, with observed performance above consideration, belief and execution of the target rule. d locates all 12 learners in the behavioral time/shape space. e shows every learner's sustained-support episodes and first behavioral criterion on the recorded trial axis.

**Archetype:** asymmetric longitudinal composite. The three paired trajectories form the dominant panel group. Cohort evidence is subordinate. No figure-wide title, question headings, narrative callouts or methodological footer are embedded in the image.

**Data:** all 12 fitted participants and all 7,936 recorded trials from the current analysis bundle. The three examples are the pre-existing behavior-only choices S122, S206 and S215; they are illustrations, not validated classes. Existing selected-parameter means over eight 128-particle replays are reused without refitting. The target is the task-defined rule H0 (Task 1) or H42 (Tasks 2/3).

**Mapping:** `correct_w32` and `predicted_w32` are trailing-32-trial performance summaries; `available`, `belief` and `executed` are unsmoothed pre-choice marginal states. Execution is defined only for persistent-rule readout. `criterion` is the first trailing-64 accuracy > .9. `delta_bic` is BIC(trend) − BIC(step); the ±6 region denotes weak separation between those two descriptive shapes. Sustained-support episodes use Q > .5 for at least 16 consecutive trials, with sessions separated. The whole qualifying episode is plotted; it is not a claim that all subsequent trials retain support.

**Review risks:** all parameters were fitted on the complete choice record. State trajectories are estimates, not observed mental states. S122 has weak verbal-report compatibility, so its latent interpretation is conditional on the fitted model. Fig. 2 and Fig. S2 expose validation discrepancies and parameter/particle sensitivity. Shape models do not adjust for stimulus difficulty or temporal dependence. No group test, clustering, subjective insight measurement or causal compensation claim is made.

**Backend/export:** Python/matplotlib, 183 × 192 mm, 450-dpi PNG only under repository instruction; 6–7 pt sans-serif text and 8-pt lowercase panel labels. Existing style is inherited; the layout is built anew. All original figures remain intact. Source data, code hashes, rendering versions and a standalone legend are saved beside the new image.
