"""Autonomous continuation from a training-prefix particle distribution.

An isolated research worker observes model construction while the existing
particle filter runs unchanged. Final particle weights select starting states.
Each branch restores a selected state and receives independent future random
streams. Its responses and task feedback then drive the public model lifecycle.
The future stimulus schedule is fixed; observed future responses are not inputs.

This adapter supports the full-v3 expectation readout with output noise disabled.
The constructor observer is process-local and requires a single filter block.
Several branches from one filter share its numerical posterior approximation;
uncertainty comparisons must retain the filter seed as a simulation cluster.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def capture_prefix(**kwargs: Any) -> tuple[list[Any], Any]:
    """Run the existing filter and retain its final, post-feedback models."""
    import src.Bayesian_state.model as model_package
    from ..inference.backends.particle_filter import run_state_model_particle_filter
    if kwargs.get('particle_blocks', 1) != 1:
        raise ValueError('Prefix capture requires one particle block')
    if not kwargs.get('condition_on_observed_choice', True):
        raise ValueError('Prefix posterior requires conditioning on observed training choices')
    original = model_package.StateModel
    if not isinstance(original, type):
        raise ValueError('StateModel construction is already overridden')
    models = []

    def construct(*args, **named):
        model = original(*args, **named)
        models.append(model)
        return model

    model_package.StateModel = construct
    try:
        result = run_state_model_particle_filter(**kwargs)
    finally:
        model_package.StateModel = original
    trials = len(kwargs['choices'])
    if len(models) != kwargs['particle_count'] or any(
        model._pending_trial is not None or model._completed_trial_count != trials for model in models
    ):
        raise ValueError('Captured models do not represent the completed prefix')
    weights = np.asarray(result.final_weights, dtype=float)
    if weights.shape != (len(models),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Invalid terminal particle weights')
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0)
    return models, result


def generate_continuations(*, models: list[Any], weights: np.ndarray,
                           observed_choices: np.ndarray, observed_feedback: np.ndarray,
                           stimulus: np.ndarray, categories: np.ndarray,
                           branch_seeds: list[int], condition: int) -> dict[str, np.ndarray]:
    """Generate equal-weight branches from sampled prefix posterior states."""
    from ..metrics.task import category_learning_feedback
    from ..model.modules.base_module import ModuleRole
    from ..model.modules.readout.choice import (
        resolve_choice_readout_config, resolve_continuous_choice_weights, resolve_output_noise_config,
    )
    from ..utils.seeding import stable_seed
    weights = np.asarray(weights, dtype=float)
    prefix_choice = np.asarray(observed_choices, dtype=int)
    prefix_feedback = np.asarray(observed_feedback, dtype=float)
    x, labels = np.asarray(stimulus, dtype=float), np.asarray(categories, dtype=int)
    cut, length = len(prefix_choice), len(x)
    if weights.shape != (len(models),) or not len(models) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Need finite nonnegative weights for every terminal model')
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0)
    if not len(branch_seeds) or len(set(branch_seeds)) != len(branch_seeds):
        raise ValueError('Need distinct future branch seeds')
    if x.ndim != 2 or length == 0 or not np.isfinite(x).all() or labels.shape != (length,):
        raise ValueError('Need aligned future physical stimuli and categories')
    if prefix_feedback.shape != (cut,) or any(model._completed_trial_count != cut for model in models):
        raise ValueError('Observed prefix and filtered cognitive states differ')
    n_categories = models[0].n_cats
    if np.any((labels < 1) | (labels > n_categories)):
        raise ValueError('Future category outside the task response space')
    ancestors = np.array([np.random.default_rng(stable_seed(dict(role='prefix_terminal_particle', branch_seed=int(seed))))
        .choice(len(models), p=weights) for seed in branch_seeds], dtype=int)
    snapshots = {index: models[index].engine.state_dict() for index in set(ancestors)}
    results = {key: [] for key in ('choices', 'feedback', 'probabilities', 'search_probability', 'rule', 'contribution')}
    legacy = np.random.get_state()
    try:
        for ancestor, seed in zip(ancestors, branch_seeds):
            model = models[int(ancestor)]
            model.engine.load_state_dict(snapshots[int(ancestor)])
            model.engine.clear_module_logs()
            model._completed_trial_count = cut
            model._pending_trial = None
            for role, label in [(ModuleRole.HYPOTHESIS_TRANSITION, 'transition'), (ModuleRole.PERCEPTION, 'perception')]:
                module = model.engine.get_module(role)
                if module is not None and hasattr(module, 'reseed_future'):
                    module.reseed_future(stable_seed(dict(role='prefix_forecast_' + label, branch_seed=int(seed))))
            readout = resolve_choice_readout_config(None, model.engine_config)
            noise = resolve_output_noise_config(None, model.engine_config)
            if (readout['method'] != 'expectation' or readout['power'] != 1. or noise['enabled']
                    or readout.get('strategy_confidence_gain', 0.) != 0.
                    or readout.get('rule_commitment_confidence_gain', 0.) != 0.):
                raise ValueError('Continuation supports expectation readout and disabled output noise')
            choice_seed = stable_seed(dict(role='prefix_forecast_choice', branch_seed=int(seed)))
            rng = np.random.default_rng(choice_seed); np.random.seed(int(seed))
            choices = np.zeros(cut + length, dtype=int); choices[:cut] = prefix_choice
            feedback = np.full(cut + length, np.nan); feedback[:cut] = prefix_feedback
            transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            capacity = transition.capacity
            probabilities = np.empty((length, n_categories)); searches = np.empty(length)
            rules = np.full((length, capacity), -1, dtype=int); contributions = np.zeros((length, capacity))
            sticky = {}; lapse = 0.
            for index, (physical, category) in enumerate(zip(x, labels)):
                model.begin_trial(physical)
                prediction = model.predict_choice(choices=choices, feedback=feedback,
                    choice_readout_config=readout, output_noise_config=noise, rng=rng,
                    sticky_state=sticky, post_error_lapse_state=lapse,
                    latent_volatility_value=float(getattr(transition, 'latent_volatility_state', 0.)))
                probability = np.asarray(prediction.observed_probabilities)
                if not np.isfinite(probability).all() or np.any(probability < 0):
                    raise ValueError('Invalid forecast probability')
                np.testing.assert_allclose(probability.sum(), 1., atol=1e-12, rtol=0)
                choice_weights = resolve_continuous_choice_weights(model.engine)
                if choice_weights is None:
                    raise ValueError('Need the full-v3 decision rule mixture')
                active = transition.active
                count = len(active)
                rules[index, :count] = active
                contributions[index, :count] = choice_weights[active]
                probabilities[index] = probability
                searches[index] = transition.current_event_probability
                choice = int(rng.choice(n_categories, p=probability)) + 1
                outcome = category_learning_feedback(choice, int(category), condition=condition)
                model.complete_trial(choice, outcome)
                choices[cut + index], feedback[cut + index] = choice, outcome
                lapse = prediction.post_error_lapse_state
            for key, value in dict(choices=choices[cut:], feedback=feedback[cut:], probabilities=probabilities,
                search_probability=searches, rule=rules, contribution=contributions).items():
                results[key].append(value.copy())
    finally:
        # Return the captured cloud to its original endpoint so independent
        # repeated checks can reuse it without inheriting generated feedback.
        for index, snapshot in snapshots.items():
            models[index].engine.load_state_dict(snapshot)
            models[index]._completed_trial_count = cut
            models[index]._pending_trial = None
        np.random.set_state(legacy)
    arrays = {key: np.asarray(value) for key, value in results.items()}
    arrays.update(ancestors=ancestors, branch_seeds=np.asarray(branch_seeds), terminal_weights=weights.copy(),
                  categories=labels.copy())
    return arrays
