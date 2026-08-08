# Bayesian_state Problems Architecture

This package contains the model-side implementation of the Bayesian state model. It defines the hypothesis space, the `StateModel` assembly layer, and the modules that update perception, likelihood, memory, beta, and hypothesis transitions. The actual scheduler class is defined in `../inference_engine/`; `base_problem.py` re-exports it for compatibility.

If you are new to this code, read files in this order:

1. `model.py`: how a `StateModel` is built and how trials are fed into the engine.
2. `../inference_engine/README.md`: shared inference primitives and scheduler.
3. `partitions.py`: the main continuous hypothesis space and category likelihood geometry.
4. `discrete_partitions.py`: the compact discrete-rule alternative.
5. `modules/README.md`: detailed behavior of the pluggable engine modules.
6. `modules/*.py`: implementation details for each module.

## Core Objects

### `StateModel` in `model.py`

`StateModel` is the high-level model wrapper. It receives an `engine_config`, creates a `Partition`, builds the hypothesis set, initializes `BaseEngine`, and asks the engine to process trials one by one.

Main responsibilities:

- choose number of categories from `condition`
- create the partition hypothesis space
- expose subject and dataset context to modules through `engine`
- build modules according to `engine_config["modules"]`
- run `fit_step_by_step(data)` and store posterior/prior logs

A trial normally enters the model as:

```python
(stimulus, choice, feedback)
```

Some evaluation utilities may additionally use true category labels, e.g. for predicted accuracy diagnostics.

### `BaseEngine` in `../inference_engine/bayesian_engine.py`

`BaseEngine` is the shared state container and scheduler. It owns the current observation, prior, posterior, likelihood, active hypothesis mask, partition, and module instances. `base_problem.py` imports and re-exports this class so older imports continue to work.

The engine processes one trial by following `agenda`, for example:

```text
perception_mod -> hypo_transitions_mod -> likelihood_mod -> memory_mod -> beta_mod
```

Modules communicate through the engine rather than by calling each other directly. This is a blackboard-style design: each module reads and writes named engine fields.

### `Partition` in `partitions.py`

`Partition` defines the category-structure hypothesis space. A hypothesis is a particular way of dividing the feature space into categories.

The current public prototype representation is:

```python
partition.prototypes
```

with shape:

```text
[n_hypotheses, n_prototypes, n_categories, n_dimensions]
```

Currently `n_prototypes == 1`, so most code reads category centers as:

```python
partition.prototypes[hypo_idx, 0, cat_idx, :]
```

There is intentionally no separate `centers` or `prototypes_np` attribute now. `prototypes` is the single source of truth for prototype/category-center geometry.

## `partitions.py` Structure

`partitions.py` is split into two classes.

### `BasePartition`

`BasePartition` contains logic that is independent of the concrete split library:

- public likelihood API:
  - `calc_likelihood`
  - `calc_likelihood_entry`
  - `calc_trueprob_entry`
- distance mode dispatch:
  - `prototype`
  - `boundary`
- category-probability implementations:
  - `calc_category_probabilities_prototype`
  - `calc_category_probabilities_boundary`
- shared geometry helpers for boundary distance
- shared feedback-code mapping

The boundary probability implementation lives in `BasePartition` because it is generic once `self.regions` exists. It only needs category constraints of the form:

```python
{"A": A, "b": b}  # category region: A @ x <= b
```

### `Partition`

`Partition` provides the concrete hypothesis library:

- `get_all_splits()`: enumerates all supported split definitions
- `get_prototypes()`: builds numeric prototype arrays
- `build_regions()`: converts internal split definitions into boundary regions
- similarity helpers used by dynamic hypothesis transitions

This separation means:

```text
BasePartition = how likelihood is computed
Partition     = what hypotheses and regions exist
```

## Likelihood Distance Modes

The likelihood module can request either geometry through model configuration:

```yaml
modules:
  likelihood_mod:
    kwargs:
      distance_mode: prototype  # or boundary
```

### `prototype`

A stimulus is compared to category prototype centers. Distances are converted to category probabilities with softmax over negative distance.

### `boundary`

A stimulus is compared to each category region. Points inside a region have distance 0; outside points are projected to the nearest point in the region. Distances are again converted to category probabilities.

Both modes return the same shape:

```text
[n_categories, n_trials]
```

The shared feedback mapping then converts category probabilities into observed feedback likelihoods.

## Modules

The `modules/` folder contains pluggable inference steps. The usual PMH stack is:

- `perception.py`: maps raw stimulus to perceived stimulus
- `hypo_transition/`: the complete H module package, including the common two-step contract,
  subject-fixed strategies, discrete strategy-state dynamics, continuous controls, and versioned candidates
- `likelihood.py`: computes `p(data_t | h)` using `Partition`
- `memory.py`: integrates likelihood and prior/posterior memory
- `beta.py`: updates per-hypothesis inverse temperature
- `readout.py`: shared choice, output-noise, RT, and oral-report readouts

See `modules/README.md` for formulas and per-module details.

`discrete_partitions.py` defines `DiscreteRulePartition`, a small rule-table implementation with the same broad category-probability role. It is useful for compact tests or explicitly discrete model variants, but it is not the default PMH partition.

## Configuration Flow

Most runs start from config files outside this package:

```text
configs/model_struct/*.yaml
configs/hyper_grid_cfg/*.yaml
configs/hyper_cd_cfg/*.yaml
configs/simulation_cfg/*.yaml
```

Typical flow:

1. A hyper selector loads a `hyper_grid` or `hyper_cd` config.
2. The hyper config points to a base simulation config.
3. The simulation config points to a model-structure config.
4. `StateModel` receives the resolved `engine_config`.
5. `BaseEngine.build_modules()` instantiates module classes listed in `engine_config["modules"]`.
6. Per-trial inference follows `engine_config["agenda"]`.

## Reader Notes

A few implementation details are intentional:

- `partition.prototypes` is numeric, not a list of dictionaries.
- `partition.splits` stores internal split-spec objects, and `partition.regions` is generated from those split definitions for boundary distance.
- strategy-based transitions use `partition.similarity_matrix`, which is lazy-loaded and cached on disk under `problems/cache/`.
- `hypo_transition/_internal/strategy_policy.py` has a local `cached_dist`; that private cache is for center-to-center distances inside transition policies, not for likelihood distance caching.
- `cache/README.md` documents persisted similarity matrices and cache invalidation rules.
- `modules/hypo_transition/candidates/README.md` documents versioned strategy resources loaded by Hyper-CD/Grid configs.

## Suggested Improvements

From a reader-maintenance perspective, the next useful cleanups would be:

- Move the long split/prototype enumeration tables out of `partitions.py` into a dedicated builder module, such as `partition_builders.py`.
- Add unit tests for `Partition(4, 2)` and `Partition(4, 4)` shape, hypothesis count, prototype likelihood, and boundary likelihood.
