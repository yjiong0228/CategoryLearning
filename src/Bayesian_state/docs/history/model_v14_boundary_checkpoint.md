# Cond1 V14 memory-boundary checkpoint

- Design: paired common-random-number probe around fine memory boundaries
- Repeats: 512 per configuration
- Candidate seed: 140016
- Fine memory settings changed: 1/5

| Subject | Selected variant | Brier | Δ Brier vs fine boundary baseline | CRPS | Δ CRPS |
|---:|---|---:|---:|---:|---:|
| 103 | fine_boundary_baseline | 0.385559 | +0.000000 | 0.073485 | +0.000000 |
| 112 | fine_boundary_baseline | 0.278833 | +0.000000 | 0.070651 | +0.000000 |
| 117 | fine_boundary_baseline | 0.126474 | +0.000000 | 0.090775 | +0.000000 |
| 127 | gamma_0p1_w0_0p05 | 0.294047 | -0.000819 | 0.058416 | -0.000362 |
| 131 | fine_boundary_baseline | 0.397024 | +0.000000 | 0.096696 | +0.000000 |

The generated subjectwise configuration now contains the selected memory settings. These choices are still part of model selection; performance must be measured in the subsequent frozen independent common-seed confirmation.
