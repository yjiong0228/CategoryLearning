> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Results Handoff Plan

Package the completed 96-subject local run, without simulation or plotting and
without changing its source. Destination: a new collaboration directory under
`results/`. Keep all existing numerical results, fit caches/history, tables and
logs; exclude PNG images. Losslessly compress NPZ members and preserve their
exact NPY bytes. Record source/packaged hashes and update only copied manifests.

- [x] Test compression byte equality, invalid-file rejection and the exact
  predictive/filtered execution distinction on synthetic arrays.
- [x] Add a bounded-memory packager, per-subject Python-compatible fit/generation
  arrays, original subject CSV rows, metadata, indexes and integrity manifests.
- [x] Add an independent NumPy-only lazy content reader and field map.
- [x] Validate all 96 source bundles, package them using at most two workers,
  then verify all delivered file hashes and data-preservation checks.
- [x] Deliver the directory, size, compatibility instructions and limitations.

Do not fabricate an original Python multi-model pilot study, seed-wise latent
states, autonomous latent states for unsaved paths, or ESS arrays. Preserve
one-based Python responses separately from zero-based native responses. Keep
Cond3 conditional pairing/category beliefs and distinguish graded feedback from
choice correctness. Region IoU is not dimension-set Jaccard.

Validation completed: 46 targeted tests passed; the actual portable reader
validated all 96 subjects / 62,720 trials, including normalized particle weights
and Cond3 label axes. 22,848 fit-history source members are preserved. All 2,231
inventory files passed independent streaming SHA256/byte-count verification.
The directory has 2,232 files including its manifest, zero PNGs, about 19.073 GiB.
Original CategoryLearning worktree remains clean; no model run or upload occurred.
