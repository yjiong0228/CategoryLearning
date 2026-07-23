# Cond1 V14 frozen independent confirmation

- Frozen before this run: controller, state setting, readout, gamma, and w0
- Repeats: 1024 per configuration
- Candidate seed: 140017
- Common trajectory seeds within each subject: yes

## Frozen V14 versus frozen V13

- Mean Δ Brier: -0.004133 (95% subject bootstrap CI -0.012630, +0.004739); wins 7/8
- Mean Δ CRPS: -0.007994 (95% subject bootstrap CI -0.017856, -0.001066); wins 7/8

## State counterfactual

- Frozen state-on subjects: [103, 105, 111, 117, 127, 131]
- State-on frozen wins vs matched state-off: Brier 4/6, CRPS 3/6
- Frozen state-off subjects: [112, 118]
- State-off frozen wins vs matched gain=0.35: Brier 1/2, CRPS 2/2
- Mean frozen minus matched counterfactual: Brier -0.000803, CRPS -0.000933

## Subject-level results

| Subject | Frozen state | Δ Brier vs V13 | Δ CRPS vs V13 | Δ Brier vs state counterfactual | Δ CRPS vs state counterfactual |
|---:|---|---:|---:|---:|---:|
| 103 | on | +0.020409 | -0.013412 | -0.003236 | -0.004979 |
| 105 | on | -0.004977 | -0.001537 | +0.000547 | +0.000903 |
| 111 | on | -0.005879 | -0.002344 | -0.000210 | +0.002263 |
| 112 | off | -0.003387 | -0.001439 | +0.000282 | -0.000341 |
| 117 | on | -0.026664 | -0.038621 | +0.000434 | +0.000194 |
| 118 | off | -0.004987 | +0.004226 | -0.000560 | -0.001040 |
| 127 | on | -0.002164 | -0.002569 | -0.000987 | -0.000603 |
| 131 | on | -0.005419 | -0.008253 | -0.002690 | -0.003860 |

## Boundary check

For subject 127, frozen gamma=0.10 minus gamma=0.25: Δ Brier -0.000338, Δ CRPS -0.000944.

## Decision

- Proceed to a full-sample V14 evaluation with the search space and selection rule frozen before expansion.
- Treat the representative performance gate as passed: V14 wins 7/8 subjects on both metrics and the CRPS interval is entirely below zero. The Brier interval still crosses zero, largely because subject 103 trades worse Brier for better CRPS.
- Keep state-on and state-off as selectable alternatives. The state counterfactual has a small favorable mean but inconsistent subject-level wins, so these data do not justify enabling persistent state for everyone or attributing the overall V14 gain mainly to state.
- Keep subject 127 at gamma=0.10: its boundary improvement reproduced on both metrics under the independent seed.
- Do not tune these eight subjects again before full-sample evaluation; doing so would contaminate the controlling confirmation.

This run is the controlling performance check because no configuration was selected or changed using these confirmation outcomes.
