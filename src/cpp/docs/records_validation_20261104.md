> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Result recording validation, 2026-11-04

Scientific model remains `1031-unified-v2`. No learning equation, selection rule,
random draw or optimization objective changes. Fitting does not allocate the new
particle archive. Original source data and the 96-subject local batch are unchanged.

## Added outputs

- All six stable per-rule conditional diagnostics, including H0/H4/H6.
- Current-choice-filtered, pre-feedback dimension-set distribution and marginals,
  aggregated across all PF repeats.
- Optional separate conditional content archive for all particles of the fixed
  first PF seed: template geometry/labels, pairing, rule ID, beta, prior belief,
  predictive/filtered PF weights, execution indicator and selection probabilities.
- Offline rule and feature CSV/PNG exporters. Missing oral text remains unscored;
  discrete dimension Jaccard is explicitly distinguished from geometric region IoU.
- Three increased-budget full-condition configs and a resource/deadline-gated
  Linux orchestration script. It does not launch automatically or claim an
  unmeasured four-hour completion time.

## Verification

- Windows/MSVC Release build passed.
- Native: 37 test cases, 342,274 assertions passed.
- Python: 270 passed, 2 non-applicable parameterized skips, 25 subtests passed.
- Cond1/S122, Cond2/S206 and Cond3/S314 actual 12-trial content archives passed
  independent weighted reconstruction of forecasts, current-choice Bayes weights,
  pre-feedback content mass, feature distributions and H0/H4/H6 execution mass.
- Native checks verify unchanged forecasts/posteriors with logging on/off,
  invariance to changing the current final feedback, skipped-trial masking and
  fixed-seed repeatability. Cond3 retains conditional label/pairing dimensions.
- Rule constraint supports match the frozen Python catalogue masks for all
  29/152 rules (independent read-only review).
- Existing behavior/state plots and the new rule/feature tables and PNGs were
  generated from an actual short evaluation output and visually checked.
- Review fixes cover missing oral reports and parameter-transfer acceptance of
  the optional boolean archive field. Original repository worktree remains clean.

## Remote limitation

Configured `ssh ZhanLab` resolves to user `zhuran`, not `ran`. The configured
identity works; hardcoded `ran@ZhanLab` was rejected. No SSH settings changed.
The isolated deployment target is
`~/ran/CategoryLearning1031cpp_records_20261104`.
On the server, `~/ran` is a symlink to `该用户的数据盘目录`.

The server has 128 logical CPUs, but the three-second utilization sample measured
127.933 busy CPU equivalents and only 0.0665 free. No logical CPU was below 25%
utilization. Load average was about 128.6, with other users' Python tasks active.
Memory and the data disk are sufficient, but CPU availability does not meet the
required 64 free threads. No full scientific job has been launched, no old job
was stopped, and four-hour completion and Linux runtime are not yet validated.

The source-only archive was uploaded and its SHA256 verified before extraction.
All participant data and old results were excluded. The environment requires
explicit approval before transferring the three local CSV inputs. Existing
server Task2 data also differ after newline-independent hashing, so they were
not silently substituted. CPU availability and data authorization both remain
prerequisites for the full rerun.

Full packet export uses substantially more disk than semantic means. Inspect
the per-subject packet schema and trial/feature-order alignment before geometric
oral IoU analysis; that analysis itself has not been implemented or validated here.
