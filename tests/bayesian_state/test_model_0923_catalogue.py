"""Exercise the adopted 0923 catalogue through configuration and learning."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space import ContinuousPartition
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import (
    AXIS_PAIR_OVERLAP_EXTENSION,
    build_axis_pair_overlap_space,
    build_axis_pair_overlap_probe_space,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.assembly import build_partition
from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.oral_observation import predict_choice_oral
from src.Bayesian_state.model.oral_report_space import CatalogueReportSpace, ReportKernelParameters

ROOT = Path(__file__).resolve().parents[2]


def _config(condition, tmp_path):
    path = ROOT / f'configs/exp123/model_struct/model_0923_cond{condition}_B0.yaml'
    config = yaml.safe_load(path.read_text())
    config['partition']['kwargs'].update(similarity_n_samples=64, similarity_cache_dir=str(tmp_path))
    config['modules']['perception_mod']['kwargs'] = {
        'features': 4, 'mean': [0.] * 4, 'std': [0.] * 4,
    }
    # This fixed initialization draws [40, 92, 120] in the four-class space,
    # exercising an adopted rule through feedback rather than just counting it.
    config['modules']['hypo_transitions_mod']['kwargs']['module_seed'] = 0
    return config


@pytest.mark.parametrize('condition,expected', [(1, 29), (2, 128), (3, 128)])
def test_adopted_rules_participate_in_prior_search_and_feedback(condition, expected, tmp_path):
    config = _config(condition, tmp_path)
    model = StateModel(config, context=ModelContext(condition=condition, subject_id=314))
    transition = model.engine.modules['hypo_transitions_mod']
    assert model.partition_model.length == expected
    np.testing.assert_allclose(transition.base_prior, np.full(expected, 1 / expected))
    prepared = model.begin_trial([.8, .2, .4, .6])
    if condition != 1:
        assert 120 in prepared.log['active_indices']
        assert prepared.prior[120] > 0
    model.complete_trial(1, .5 if condition == 3 else 1.)
    posterior = model.engine.posterior
    assert posterior.shape == (expected,)
    assert np.isfinite(posterior).all() and np.all(posterior >= 0.)
    np.testing.assert_allclose(posterior.sum(), 1.)
    if condition != 1:
        assert posterior[120] > 0
        assert posterior[120] != pytest.approx(prepared.prior[120])
    inactive = np.setdiff1d(np.arange(expected), transition.active)
    proposal = transition._newcomer_proposal(posterior, inactive)
    assert proposal.shape == inactive.shape
    assert np.all(proposal > 0.)
    np.testing.assert_allclose(proposal.sum(), 1.)
    assert transition._local_kernel.shape == (expected, expected)
    if condition == 3:
        memory = model.engine.modules['memory_mod']
        assert memory.joint.shape == (128, 3)
        np.testing.assert_allclose(memory.joint.sum(axis=1), posterior)
        assert not np.allclose(memory.conditional_pairing()[120], np.full(3, 1 / 3))


def test_production_and_probe_use_the_same_geometry():
    adopted = build_axis_pair_overlap_space()
    probe = build_axis_pair_overlap_probe_space()
    assert adopted.signature == probe.signature
    assert [h.hyperplanes for h in adopted] == [h.hyperplanes for h in probe]
    assert all(h.label_permutation == (0, 1, 2, 3) for h in adopted)


@pytest.mark.parametrize('dimensions,categories', [(4, 2), (3, 4)])
def test_extension_cannot_silently_change_another_task(dimensions, categories):
    with pytest.raises(ValueError):
        ContinuousPartition(dimensions, categories, structural_extension=AXIS_PAIR_OVERLAP_EXTENSION)


def test_unknown_extension_rejected():
    with pytest.raises(ValueError, match='Unsupported structural_extension'):
        ContinuousPartition(4, 4, structural_extension='all_label_permutations')


@pytest.mark.parametrize('condition', [2, 3])
def test_saved_v1_configs_and_default_catalogue_still_have_116_rules(condition, tmp_path):
    legacy = _config(condition, tmp_path)
    legacy['provenance']['specification_version'] = '0923-B0-v1'
    del legacy['partition']['kwargs']['structural_extension']
    ModelConfig.from_mapping(legacy)
    assert build_partition(legacy, condition).length == 116
    assert ContinuousPartition(4, 4).length == 116


def test_v2_rejects_a_config_or_injected_partition_that_drops_the_extension(tmp_path):
    config = _config(2, tmp_path)
    mismatched = deepcopy(config)
    del mismatched['partition']['kwargs']['structural_extension']
    with pytest.raises(ValueError, match='structural_extension'):
        ModelConfig.from_mapping(mismatched)
    with pytest.raises(ValueError, match='adopted catalogue'):
        StateModel(config, context=ModelContext(condition=2, subject_id=206),
                   partition=ContinuousPartition(4, 4))


def test_similarity_is_version_isolated_and_matches_independent_region_assignments(tmp_path):
    old = ContinuousPartition(4, 4, similarity_n_samples=73, similarity_cache_dir=tmp_path)
    new = ContinuousPartition(4, 4, structural_extension=AXIS_PAIR_OVERLAP_EXTENSION,
                              similarity_n_samples=73, similarity_cache_dir=tmp_path)
    args = dict(distance_mode='boundary', n_samples=73, random_state=0, sample_distribution='uniform')
    old_key, new_key = old.similarity._cache_key(**args), new.similarity._cache_key(**args)
    assert old_key != new_key
    assert old.similarity._cache_path(old_key) != new.similarity._cache_path(new_key)
    # Even the default sampling settings cannot point the adopted space to the
    # bundled 116-rule resource (shape checking alone is not its provenance).
    resource = new.similarity._compatible_resource_path('boundary', 100000, 0)
    assert resource.name == '__no_compatible_similarity_resource__'
    assert old.similarity._compatible_resource_path('boundary', 100000, 0).is_file()
    np.save(old.similarity._cache_path(old_key), np.ones((116, 116)))
    matrix = new.get_similarity_matrix(distance_mode='boundary')
    points = np.random.default_rng(0).random((73, 4))
    assignments = []
    for rule in new.hypothesis_space:
        memberships = np.column_stack([
            np.logical_or.reduce([np.all(points @ part.A.T <= part.b, axis=1)
                                  for part in category.components])
            for category in rule.categories
        ])
        assert np.all(memberships.sum(axis=1) == 1)
        assignments.append(memberships.argmax(axis=1))
    labels = np.asarray(assignments)
    expected = np.mean(labels[:, None, :] == labels[None, :, :], axis=2)
    np.testing.assert_array_equal(matrix, expected)
    assert matrix.shape == (128, 128)
    np.testing.assert_array_equal(np.diag(matrix), np.ones(128))
    assert new.similarity._cache_path(new_key).is_file()


def test_report_interface_consumes_the_adopted_partition(tmp_path):
    partition = build_partition(_config(2, tmp_path), 2)
    reports = CatalogueReportSpace.from_catalogue(partition.hypothesis_space)
    kernel = reports.matrix(ReportKernelParameters(.8, .02, .5, .75, .1))
    assert kernel.shape == (128, 4, 506)
    choices = np.stack([partition.get_category_probabilities(
        h, (np.array([[.8, .2, .4, .6]]),), 5., distance_mode='boundary').reshape(4)
        for h in range(128)])
    predicted = predict_choice_oral(np.full(128, 1 / 128), choices, kernel)
    np.testing.assert_allclose(predicted.joint_probabilities.sum(axis=1), choices.mean(axis=0))
    np.testing.assert_allclose(predicted.joint_probabilities.sum(), 1.)
