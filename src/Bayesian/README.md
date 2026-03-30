# Bayesian Module

`Bayesian` contains the earlier Bayesian model implementation centered on `StandardModel`.

## What This Module Contains

- `problems/model.py`: `BaseModel` / `StandardModel` and fitting logic
- `problems/partitions.py`: partition definitions and likelihood helpers
- `problems/modules/`: legacy module components (decision, memory, perception, transitions)
- `inference_engine/bayesian_engine.py`: Bayesian inference engine for the legacy flow
- `utils/optimizer.py`: legacy parameter optimization helper

## Current Status

- This module is important for baseline behavior and historical compatibility.
- The current official batch optimization entry points are in `src/Bayesian_state`, not here.

## When To Use This Module

- Reproducing earlier experiments built around `StandardModel`.
- Comparing state-based pipeline results against legacy Bayesian baselines.
- Inspecting historical model behavior before migration/refactor.

## Important Note

Some cross-module coupling still exists between `Bayesian` and `Bayesian_state`.
During refactor, this module should be gradually decoupled instead of removed directly.
