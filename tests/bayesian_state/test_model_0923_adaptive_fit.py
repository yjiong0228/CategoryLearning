"""Model-specific coverage, archived discoveries and unchanged shared scoring."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.optimization.adaptive_runtime import phase_seeds
from src.Bayesian_state.optimization.model_0923_adaptive import (
    CAPACITY_PATH, DEFAULT_CONFIG, Model0923Policy, load_config, make_contexts,
    parameter_support, run_fit, warm_candidates,
)
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, point_id


def test_capacity_coverage_and_stratified_starts_are_deterministic():
    config = load_config(DEFAULT_CONFIG)
    space, anchor, support = parameter_support(config)
    policy = Model0923Policy()
    points = policy.initial_points(space, anchor, 45, 84592)
    assert points == policy.initial_points(space, anchor, 45, 84592)
    assert len({point_id(p) for p in points}) == 45
    assert {p[CAPACITY_PATH] for p in points} == {2, 3, 4, 6, 8}
    assert len(space) == len(support['parameters']) == 13
    assert all(p[key] in values for p in points for key, values in space.items())


def test_natural_and_artificial_edges_are_distinguished():
    config = load_config(DEFAULT_CONFIG)
    space, anchor, support = parameter_support(config)
    point = deepcopy(anchor)
    point[support['parameters']['learning_rate']['path']] = 1.
    point[support['parameters']['failure_decay']['path']] = .99
    row = dict(candidate(point, 'test'), mean_nll=.4)
    report = Model0923Policy().boundary_report([row], support, .005, 2)
    kinds = {(h['parameter'], h['kind']) for h in report['hits']}
    assert ('learning_rate', 'mechanism_boundary') in kinds
    assert ('failure_decay', 'artificial_boundary') in kinds
    assert report['review_required']


def test_independent_model_namespace_and_seed_prefix():
    new = phase_seeds(2026101401, 307, 'cycle_0/independent_audit', 32, namespace='model0923_adaptive_full_v1')
    old = phase_seeds(2026101401, 307, 'cycle_0/independent_audit', 32)
    guide = phase_seeds(2026101401, 307, 'guide', 32, namespace='model0923_adaptive_full_v1')
    assert new[:16] == phase_seeds(2026101401, 307, 'cycle_0/independent_audit', 16, namespace='model0923_adaptive_full_v1')
    assert not set(new) & set(old)
    assert not set(new) & set(guide)


def test_meg_paths_uniform_perception_and_full_sequence():
    plan = run_fit(DEFAULT_CONFIG, None, dry_run=True)
    assert plan['subjects'][0]['subject'] == 335
    assert plan['subjects'][0]['trials'] == 748
    assert plan['subjects'][0]['scored_trials'] == 747
    config = load_config(DEFAULT_CONFIG)
    _, contexts, _ = make_contexts(config, None, False)
    context = contexts[335]
    assert context['engine']['modules']['perception_mod']['kwargs']['noise_mode'] == 'uniform'
    assert context['engine']['data']['trial_order_columns'] == ['iSession', 'iRun', 'iTrial']
    assert context['dataset_paths']['learning_data'].name == 'Task3b_processed.csv'
    assert context['dataset_paths']['perception_summary'].name == 'Task1b_errorsummary.csv'
    assert context['engine']['inference']['particle_blocks'] == 1


def test_old_307_finalists_are_imported_without_historical_scores():
    path = Path('configs/exp123/specific_models/model0923_adaptive_307.yaml').resolve()
    config = load_config(path)
    space, _, _ = parameter_support(config)
    _, contexts, _ = make_contexts(config, None, False)
    banks, files = warm_candidates(config, contexts, space)
    assert len(banks[307]) == 4
    assert all('mean_nll' not in row and 'historical_finalist' in row['sources'] for row in banks[307])
    assert any(p.name == 'final_rescore.jsonl' for p in files)
    assert len(contexts[307]['arrays'].choices) == 704


def test_config_rejects_unknown_scientific_options(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw['evaluation_protocol'] = {'mode': 'sequential_holdout'}
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='Unknown or missing'):
        load_config(path)


def test_invalid_expansion_violates_model_domain():
    config = load_config(DEFAULT_CONFIG)
    config['boundary']['extensions'] = {'learning_rate': [1.01]}
    with pytest.raises(ValueError, match='mathematical domain'):
        parameter_support(config)
