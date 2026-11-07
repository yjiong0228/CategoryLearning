> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Conditional particle contents

Enable `evaluation.export_particles: true` for `evaluate`. No model equation,
random draw or fit budget changes. The separate `conditional_particle_contents.npz`
contains **all particles from the prespecified first PF seed**, not the autonomous
path, not a selected best seed, and not a smoother. The manifest records its seed,
shape/dtype inventory and SHA256. Other PF seeds still contribute to all ordinary
conditional summaries. Do not average particle slots across seeds or particles.

Snapshots are taken after `prepare_trial`/`decide` and current-choice importance
weighting, **before current-feedback learning and before resampling**. Thus they
describe real observed history, including this choice but excluding this feedback.
`particle_predictive_weights` excludes this choice as well. Oral reports never
enter model updates. A report about the post-feedback state must not be compared
as if it described the pre-feedback state.

| Array | Shape | Meaning |
| --- | --- | --- |
| `particle_weights`, `particle_predictive_weights` | T,P | Choice-filtered and predictive PF weights |
| `particle_active` | T,P,K | Valid slot mask |
| `particle_representation_kind` | T,P,K | 0 uncommitted, 1 template, 2 catalogue rule |
| `particle_semantic_kind` | T,P,K | 0 uncommitted, 1 example, 2 generalized template, 3 rule |
| `particle_representation_uid`, `particle_representation_parent`, `particle_born_trial` | T,P,K | Particle-local ancestry, not stable cross-particle IDs |
| `particle_catalogue_rule` | T,P,K | Stable rule ID for rule contents, otherwise -1 |
| `particle_belief`, `particle_beta` | T,P,K | Pre-feedback content belief and precision |
| `particle_choice_weights` | T,P,K | Realized execution indicator, zero for unexecuted slots |
| `particle_selection_probabilities` | T,P,K | Belief/inertia sampling probabilities before execution; distinct from realized one-hot execution |
| `particle_template_anchor`, `particle_template_mask` | T,P,K,4 | Template position and retained dimensions; mask false for non-templates |
| `particle_template_category_belief` | T,P,K,G,C | Template label distribution conditional on pairing |
| `particle_pairing` | T,P,K,G | Pairing belief; G=3 for Cond3, otherwise a singleton 1 |
| `particle_rule_choice_probabilities` | T,P,K,C | Pre-feedback response distribution for every active content, not just rules |
| `particle_feature_mask` | T,P,K | Bitmask of used dimensions: template mask or rule constraint support; uncommitted=0 |
| `particle_perceived_stimulus` | T,P,4 | Particle-specific noisy stimulus |

The file also contains physical stimuli, observed choices/feedback, valid/score
masks, trial/session IDs, truth and response mapping. Inactive float content fields
are NaN, IDs are -1, active/execution/mask entries are zero. Invalid trials have
zero active masks and NaN particle weights: exclude them, do not impute zeros.
Anchors and label distributions are only defined for templates. Catalogue geometry
is in `resources/catalogue_c{1,2}.json`; include it with the result handoff.

All feature coordinates follow `manifest.protocol.feature_order`, not a fixed
body-part order. For Cond3 use the **joint** pairing and conditional category
beliefs, not the product of independent averaged marginals.

Ordinary `evaluation_inputs.npz` additionally contains:

- `conditional_feature_set_probability[T,16]`: current-choice-filtered execution
  mass by retained-dimension bitmask, averaged across all PF seeds.
- `conditional_feature_probability[T,4]`: corresponding per-dimension marginal.
- Six `conditional_catalogue_*[T,H]` diagnostics; H0/H4/H6 are columns 0/4/6.

The 16-bin distribution supports dimension-set exact match and expected set
Jaccard; **these are not geometric region IoU**. The particle archive and catalogue
geometry provide the inputs for an independently specified geometric IoU analysis.
This change records that information; it does not invent an oral observation model
or claim the geometric IoU has already been computed.
