# Condition 3 implementation

Implement the approved Model 0826 §13 in the shared core. Keep the existing
condition 1/2 branches and parameter defaults unchanged. No full fitting runs.

- [x] Add a stateless three-pairing feedback kernel and explicit likelihood mode.
  Test event normalization, initial 0/.5 equivalence and coordinate relabelling.
- [x] Add joint rule/pairing fading memory, snapshot isolation and transport
  through the existing workspace transition. Test literal joint Bayes updates,
  gamma boundaries, survivor/newcomer cases and scalar-marginal agreement.
- [x] Connect engine lifecycle and validate the condition-specific module bundle.
- [x] Add chance-relative beta evidence and full-success search interpretation;
  retain raw feedback and the existing causal controller timing.
- [x] Preserve response keys and subject maps as metadata; implement ternary
  autonomous scoring and distinct species/family/reward summaries.
- [x] Enable PF condition 3 with pre/post pairing diagnostics and snapshot tests.
- [x] Add opt-in condition 3 model/smoke configs and concise support documentation.
- [x] Run focused tests, existing condition 1/2 and frozen-reference regressions,
  then a small real-subject condition 3 smoke run in a new output directory.

Interfaces: likelihood mode `hierarchical_pairing`; memory class
`HierarchicalPairingMemoryModule`, `joint` (rules × three pairings),
`feedback_evidence` (absolute pre-update event probabilities), and
`pairing_marginal()`; beta mode `hierarchical_feedback`; controller
`feedback_interpretation: full_success`. Pair order: 12|34, 13|24, 14|23.

Independent ownership: core memory/likelihood/assembly/configuration by primary
agent; controller/beta, data/autonomous, and PF/output integration by separate
agents. Existing uncommitted data, notebook and evaluation edits are preserved.

Validation: 214 focused core/regression tests and 26 evaluation tests passed.
Frozen numeric regression passed; the old manuscript-hash assertion still fails
because the already-edited specification differs from the frozen C1 provenance.
No numerical references were regenerated. Real-subject CLI and autonomous smoke
artifacts: results/model_0826/cond3/subject_301/implementation_smoke_20260915_v2.
No full fitting or recovery jobs were run.
