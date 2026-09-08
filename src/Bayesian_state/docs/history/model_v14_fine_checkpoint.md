# Cond1 V14 fine checkpoint

- Coarse checkpoint commit: `677116f`
- Fine repeats: 256 per evaluated configuration
- Completed subjects: 8/8
- Exact coarse-to-fine selection retained: 2/8
- Controller family retained: 7/8
- State setting retained: 5/8
- Boundary probe required: 103, 112, 117, 127, 131

## Fine selections

- State: gain=0.35: 4, gain=0.50: 2, off: 2
- Controller family: choice_volatile_refresh: 4, early_explore_late_stable: 1, stable_dominant: 3
- Readout: map_hypothesis: 3, sharpened_expectation(p=2): 2, sharpened_expectation(p=4): 3

## Subject-level checkpoint

| Subject | Fine controller | State | Readout | gamma | w0 | Brier | CRPS | Coarse→fine changes |
|---:|---|---|---|---:|---:|---:|---:|---|
| 103 | early_explore_late_stable | gain=0.50 | sharpened_expectation(p=4) | 0.25 | 0.010 | 0.385567 | 0.074791 | readout |
| 105 | stable_dominant | gain=0.35 | map_hypothesis | 0.70 | 0.050 | 0.187326 | 0.124901 | gamma |
| 111 | choice_volatile_refresh | gain=0.35 | sharpened_expectation(p=2) | 0.85 | 0.050 | 0.358487 | 0.100437 | none |
| 112 | choice_volatile_refresh | off | map_hypothesis | 0.25 | 0.500 | 0.275582 | 0.069444 | state, readout |
| 117 | stable_dominant | gain=0.35 | map_hypothesis | 0.70 | 0.010 | 0.126047 | 0.088401 | state |
| 118 | choice_volatile_refresh | off | sharpened_expectation(p=4) | 0.85 | 0.100 | 0.371128 | 0.096805 | none |
| 127 | stable_dominant | gain=0.50 | sharpened_expectation(p=2) | 0.25 | 0.050 | 0.294089 | 0.057899 | family, state |
| 131 | choice_volatile_refresh | gain=0.35 | sharpened_expectation(p=4) | 0.50 | 0.500 | 0.393050 | 0.094589 | gamma |

## Interpretation

Fine scores are nominally lower than the frozen V13 reference by 0.0073 Brier and 0.0112 CRPS on average, with 7/8 subject-level wins for each metric. This is not an independent performance estimate because fine selected among candidate configurations and used a different Monte Carlo sample.

The structural result is more reliable than the exact parameter result: controller family was retained for 7/8 subjects, whereas only 2/8 retained the full controller/state/readout/memory tuple. Do not expand to the full sample before a targeted memory-boundary probe and frozen common-seed confirmation.

## Required next gate

1. Probe only the fine winners at current memory boundaries: gamma=0.10 below the current 0.25 floor, w0=0.005 below 0.01, and w0=0.75 above 0.50.
2. Freeze the resulting per-subject configurations.
3. Run an independent common-seed comparison of V13, the frozen V14 winner, and its matched state-off/state-on ablation.

## Validation

- subject_count: `8`
- all_fine_complete: `True`
- fine_metric_null_count: `0`
- coarse_history_retained: `True`
- fine_repeat_count: `256`
