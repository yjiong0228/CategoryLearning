"""Autonomous public StateModel lifecycle driven by a blockwise task.

The task determines future stimuli from completed model choices. This adapter
uses begin_trial/predict_choice/complete_trial for every cognitive operation.
Choice RNG and persistent readout state match the fixed-schedule generator.
Only compact behavioral and H-event arrays are retained for long trajectories.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from ..metrics.task import category_learning_feedback
from ..model.config import ModelContext
from ..model.modules.base_module import ModuleRole
from ..model.modules.readout.choice import resolve_choice_readout_config, resolve_output_noise_config
from ..model.state_model import StateModel
from ..utils.seeding import inject_module_seed_from_trajectory, stable_seed
from .exp123_adaptive_task import AdaptiveTask


def run_adaptive_category_learning(*, engine_config: Mapping[str, Any], subject_id: int,
                                  task: AdaptiveTask, trajectory_seed: int,
                                  processed_data_dir=None, dataset_paths=None) -> dict:
    """Generate until the task passes its criterion or reaches its horizon.

The task receives sampled responses only. Its category labels enter the
feedback function after choice and the output record for later evaluation.
The cognitive state persists over block and session boundaries.
"""
    config = deepcopy(dict(engine_config))
    inject_module_seed_from_trajectory(config, int(trajectory_seed))
    model = StateModel(config, context=ModelContext(condition=task.condition, subject_id=subject_id,
        processed_data_dir=processed_data_dir, dataset_paths=dataset_paths))
    readout = resolve_choice_readout_config(None, config)
    noise = resolve_output_noise_config(None, config)
    choice_seed = stable_seed(dict(seed_role='state_model_autonomous_choice', trajectory_seed=int(trajectory_seed)))
    rng = np.random.default_rng(choice_seed)
    total = task.max_blocks * 64
    choices = np.zeros(total, dtype=int)
    feedback = np.full(total, np.nan)
    rows: dict[str, list] = {key: [] for key in ('stimulus', 'categories', 'ambiguous', 'probabilities',
        'search_probability', 'searched', 'active_count', 'capacity_drop', 'exit_count')}
    sticky: dict = {}
    lapse, index = 0., 0
    block = task.current
    legacy_state = np.random.get_state()
    np.random.seed(int(trajectory_seed))
    try:
        while block is not None:
            first = index
            for stimulus, category, ambiguous in zip(block.stimulus, block.categories, block.ambiguous):
                model.begin_trial(stimulus)
                transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
                prediction = model.predict_choice(choices=choices, feedback=feedback,
                    choice_readout_config=readout, output_noise_config=noise, rng=rng,
                    sticky_state=sticky, post_error_lapse_state=lapse,
                    latent_volatility_value=float(getattr(transition, 'latent_volatility_state', 0.)))
                lapse = prediction.post_error_lapse_state
                probability = np.asarray(prediction.observed_probabilities)
                if not np.isfinite(probability).all() or np.any(probability < 0) or not np.isclose(probability.sum(), 1., atol=1e-12, rtol=0):
                    raise ValueError('Invalid pre-choice probability')
                choice = int(rng.choice(model.n_cats, p=probability)) + 1
                outcome = category_learning_feedback(choice, int(category), condition=task.condition)
                model.complete_trial(choice, outcome)
                choices[index], feedback[index] = choice, outcome
                event = transition.transition_log[-1] if transition is not None and transition.transition_log else {}
                values = dict(stimulus=stimulus.copy(), categories=int(category), ambiguous=bool(ambiguous),
                    probabilities=probability.copy(), search_probability=float(event.get('search_probability', np.nan)),
                    searched=bool(event.get('search_event', False)), active_count=int(event.get('active_count', 0)),
                    capacity_drop=-1 if event.get('capacity_dropped') is None else int(event['capacity_dropped']),
                    exit_count=len(event.get('autonomous_exits', [])))
                for key, value in values.items():
                    rows[key].append(value)
                index += 1
            block = task.complete_block(choices[first:index])
    finally:
        np.random.set_state(legacy_state)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays.update(choices=choices[:index].copy(), feedback=feedback[:index].copy())
    return dict(arrays=arrays, blocks=task.history, passed=task.history[-1]['passed'],
                trajectory_seed=int(trajectory_seed), task_seed=task.seed, choice_seed=choice_seed)
