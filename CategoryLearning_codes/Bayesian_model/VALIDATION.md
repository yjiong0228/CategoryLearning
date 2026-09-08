# Migration validation

- 47 tests passed: 44 in the new package plus 3 original Model 0826 versioning tests.
- Fixed-seed PF parity: 16 real S101 trials, 4 particles, both chi values and both
  manuscript transport policies, with nonzero failure-history and search-range gains.
  Observable probabilities, state probabilities, latent summaries and resampling
  decisions match source exactly; predicted probabilities are finite and normalized.
- Autonomous parity: 24 real stimulus trials, both execution structures, fixed seed;
  generated choices/feedback, perceived stimuli, priors/posteriors and beta match exactly.
- All production modules import in a fresh subprocess that rejects src.Bayesian_state.
  Initially missed lazy-loaded group/residual metrics were found and included.
- Frozen engine configuration equals the original after class-namespace substitution.
  Manuscript and 29×29 similarity resource hashes match declared provenance.
  Both local recovery designs resolve to local engine, parameter and simulation configs.
- Equation tests cover lagged event history vs current range input, zero-gain boundaries,
  state restoration, slot-fraction transport and mass-preserving counterfactuals.
  Recovery-contract tests cover architecture coordinates, spike supports, parameter
  extraction, seed-averaged NLL, numerical-budget selection and recovery summaries.
- Actual new CLI smoke run completed: S101, 32 trials, 32 particles, one repeat,
  one job; artifacts are under outputs/smoke32_v1. The earlier 8-trial attempt correctly
  failed the existing 16-window metric requirement (minimum 17); its directory is retained.
- New optimization and recovery entry-point --help commands passed.
- Original src implementation, frozen configuration/manuscript and data were not
  edited. Only the old package README adds the new 0826 location.

Limits: short deterministic migration checks do not establish PF convergence at a
scientific budget, parameter recovery success, model fit quality, four-category
support, or validity of inferred psychological states. No full recovery/search or
cohort fitting was launched. Historical checkpoints are not resumed under new hashes.
