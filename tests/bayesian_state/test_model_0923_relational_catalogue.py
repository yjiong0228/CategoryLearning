"""152-rule adoption: binary semantics, geometry, search and causal readout."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.optimize import linprog

from src.Bayesian_state.hypothesis_space import ContinuousPartition
from src.Bayesian_state.hypothesis_space.rule_programs import build_rule_edit_costs
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import (
    AXIS_PAIR_OVERLAP_EXTENSION, RELATIONAL_EXTENSION,
    build_axis_pair_overlap_space, build_relational_0923_space,
)
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.modules.readout.choice import resolve_choice_readout_config, resolve_output_noise_config
from src.Bayesian_state.model.modules.readout.oral import OralReadout, OralReadoutParameters
from src.Bayesian_state.model.modules.readout.report_space import CatalogueReportSpace, ReportKernelParameters, report_code
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report

ROOT = Path(__file__).resolve().parents[2]
COSTS = dict(feature=1., relation=1.5, reference=1., branch=2.)


def _members(h, x):
    return np.column_stack([np.logical_or.reduce([np.all(x @ p.A.T <= p.b, axis=1)
                                                for p in c.components]) for c in h.categories])


def _signature(labels):
    # Canonicalize the names only to audit geometric duplicates, never to fit labels.
    names = {}
    return bytes(names.setdefault(int(y), len(names)) for y in labels)


def test_128_rule_prefix_is_preserved_and_all_24_additions_are_distinct_nonempty_partitions():
    old, new = build_axis_pair_overlap_space(), build_relational_0923_space()
    assert len(new) == 152
    for before, after in zip(old, new):
        assert before.index == after.index and before.family == after.family
        assert before.hyperplanes == after.hyperplanes and before.parameters == after.parameters
        assert before.feedback_neighbors == after.feedback_neighbors
        assert before.label_permutation == after.label_permutation
        for first, second in zip(before.categories, after.categories):
            for p, q in zip(first.components, second.components):
                np.testing.assert_array_equal(p.A, q.A)
                np.testing.assert_array_equal(p.b, q.b)
    points = np.random.default_rng(20261011).random((8192, 4))
    signatures = set()
    for h in new:
        membership = _members(h, points)
        assert np.all(membership.sum(1) == 1)
        signature = _signature(membership.argmax(1))
        assert signature not in signatures
        signatures.add(signature)
        if h.index < 128:
            continue
        for c in h.categories:
            for p in c.components:
                A = np.vstack((p.A, np.eye(4), -np.eye(4)))
                b = np.r_[p.b, np.ones(4), np.zeros(4)]
                solution = linprog([0, 0, 0, 0, -1], A_ub=np.c_[A, np.linalg.norm(A, axis=1)],
                                   b_ub=b, bounds=[(0, 1)] * 5, method='highs')
                assert solution.success and solution.x[-1] > 1e-8
    assert {h.family for h in new.hypotheses[128:140]} == {'shared_pair_comparisons'}
    assert {h.family for h in new.hypotheses[140:]} == {'similarity_axis_disjoint'}


@pytest.mark.parametrize('tolerance', [.05, .1, .2])
def test_four_class_band_is_binary_band_crossed_with_unused_axis(tolerance):
    binary = build_continuous_hypothesis_space(4, 2, pairwise_similarity_tolerance=tolerance)
    new = build_relational_0923_space(pairwise_similarity_tolerance=tolerance)
    points = np.random.default_rng(51).random((257, 4))
    assert new.parameters['pairwise_similarity_tolerance'] == tolerance
    for h in new.hypotheses[140:]:
        pair, axis = h.parameters['related_dimensions'], h.parameters['axis_dimension']
        assert axis not in pair
        parent = next(b for b in binary if b.family == 'pairwise_similarity_band' and b.parameters['dimensions'] == pair)
        band = _members(parent, points)
        low = points[:, axis] <= .5
        expected = np.c_[low & band[:, 0], low & band[:, 1], ~low & band[:, 0], ~low & band[:, 1]]
        np.testing.assert_array_equal(_members(h, points), expected)
        assert [len(c.components) for c in h.categories] == [1, 2, 1, 2]


@pytest.mark.parametrize('tolerance', [0., -0.1, 1., np.nan])
def test_invalid_band_width_rejected_like_binary(tolerance):
    with pytest.raises(ValueError, match='pairwise_similarity_tolerance'):
        build_relational_0923_space(pairwise_similarity_tolerance=tolerance)


def test_programs_use_two_judgments_keep_old_costs_and_connect_new_rules():
    space = build_relational_0923_space()
    programs, costs = build_rule_edit_costs(space, COSTS)
    old_programs, old_costs = build_rule_edit_costs(build_axis_pair_overlap_space(), COSTS)
    assert programs[:128] == old_programs
    np.testing.assert_array_equal(costs[:128, :128], old_costs)
    points = np.r_[np.random.default_rng(92).random((257, 4)), [[0., .1, .5, .5], [.5] * 4]]
    for h, program in zip(space, programs):
        np.testing.assert_array_equal(program.evaluate(points), _members(h, points).argmax(1))
    assert np.all((np.isfinite(costs).sum(1) - 1)[128:] > 0)
    for h in space.hypotheses[140:]:
        program = programs[h.index]
        assert program.topology == 'cross' and len(program.judgments) == 2
        assert [j.kind for j in program.judgments] == ['axis', 'difference_band']
        old = next(r for r in space if r.family == '3d_axis_equality'
                   and r.parameters['axis_dimension'] == h.parameters['axis_dimension']
                   and r.parameters['related_dimensions'] == h.parameters['related_dimensions'])
        assert costs[old.index, h.index] == COSTS['relation']


def _config(condition, variant, tmp_path):
    config = yaml.safe_load((ROOT / f'configs/exp123/model_struct/model_0923_cond{condition}_{variant}.yaml').read_text())
    config['partition']['kwargs'].update(similarity_n_samples=64, similarity_cache_dir=str(tmp_path))
    config['modules']['perception_mod']['kwargs'] = {'features': 4, 'mean': [0.] * 4, 'std': [0.] * 4}
    if variant == 'full':
        config['modules']['hypo_transitions_mod']['kwargs']['relation_samples'] = 64
        config['provenance']['hypothesis_similarity']['sample_count'] = 64
    return config


@pytest.mark.parametrize('condition', [2, 3])
@pytest.mark.parametrize('variant,old_version', [('B0', '0923-B0-v2'), ('full', '0923-full-v1')])
def test_version_guards_and_saved_128_configs_remain_executable(condition, variant, old_version, tmp_path):
    config = _config(condition, variant, tmp_path)
    assert ModelConfig.from_mapping(config)
    mismatched = deepcopy(config)
    mismatched['partition']['kwargs']['structural_extension'] = AXIS_PAIR_OVERLAP_EXTENSION
    with pytest.raises(ValueError, match='structural_extension'):
        ModelConfig.from_mapping(mismatched)
    mismatched['provenance']['specification_version'] = old_version
    legacy = StateModel(mismatched, context=ModelContext(condition=condition))
    assert legacy.engine.set_size == 128
    with pytest.raises(ValueError, match='adopted catalogue'):
        StateModel(config, context=ModelContext(condition=condition), partition=legacy.partition_model)


def test_tolerance_and_catalogue_have_distinct_caches_and_injected_width_is_checked(tmp_path):
    old = ContinuousPartition(4, 4, structural_extension=AXIS_PAIR_OVERLAP_EXTENSION, similarity_n_samples=61, similarity_cache_dir=tmp_path)
    new = ContinuousPartition(4, 4, structural_extension=RELATIONAL_EXTENSION, similarity_n_samples=61, similarity_cache_dir=tmp_path)
    wider = ContinuousPartition(4, 4, structural_extension=RELATIONAL_EXTENSION, pairwise_similarity_tolerance=.2, similarity_n_samples=61, similarity_cache_dir=tmp_path)
    args = dict(distance_mode='boundary', n_samples=61, random_state=0, sample_distribution='uniform')
    keys = [p.similarity._cache_key(**args) for p in (old, new, wider)]
    assert len(set(keys)) == 3
    assert new.similarity._compatible_resource_path('boundary', 100000, 0).name == '__no_compatible_similarity_resource__'
    matrix = new.get_similarity_matrix(distance_mode='boundary')
    points = np.random.default_rng(0).random((61, 4))
    labels = np.array([_members(h, points).argmax(1) for h in new.hypothesis_space])
    np.testing.assert_array_equal(matrix, np.mean(labels[:, None] == labels[None, :], axis=-1))
    with pytest.raises(ValueError, match='similarity tolerance'):
        StateModel(_config(2, 'full', tmp_path), context=ModelContext(condition=2), partition=wider)


@pytest.mark.parametrize('condition', [2, 3])
@pytest.mark.parametrize('hypothesis,seed', [(128, 412), (140, 104)])
def test_each_new_family_is_executed_reported_and_updated_by_feedback(condition, hypothesis, seed, tmp_path):
    config = _config(condition, 'full', tmp_path)
    config['modules']['hypo_transitions_mod']['kwargs']['module_seed'] = seed
    model = StateModel(config, context=ModelContext(condition=condition))
    transition = model.engine.modules['hypo_transitions_mod']
    np.testing.assert_allclose(transition.base_prior, np.full(152, 1/152))
    assert transition.executed_hypothesis == hypothesis
    model.begin_trial([.45, .5, .2, .8])
    decision = model.predict_choice(choices=[], feedback=[],
        choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
        output_noise_config=resolve_output_noise_config(None, model.engine_config),
        rng=np.random.default_rng(0), sticky_state={}, post_error_lapse_state=0.)
    assert np.flatnonzero(decision.rule_snapshot.rule_weights).tolist() == [hypothesis]
    oral = model.predict_oral(1)
    assert oral.rule_weights_given_choice[hypothesis] == 1
    np.testing.assert_allclose(oral.probabilities.sum(), 1.)
    before = model.engine.beta[hypothesis]
    posterior, _, _ = model.complete_trial(1, .5 if condition == 3 else 1.)
    assert np.isfinite(posterior).all() and np.isfinite(model.engine.beta).all()
    assert model.engine.beta[hypothesis] != before
    np.testing.assert_allclose(posterior.sum(), 1.)
    if condition == 3:
        memory = model.engine.modules['memory_mod']
        assert memory.joint.shape == (152, 3)
        assert not np.allclose(memory.conditional_pairing()[hypothesis], 1/3)


def test_new_oral_clauses_and_unchanged_old_kernel_rows():
    old, new = build_axis_pair_overlap_space(), build_relational_0923_space()
    before, after = CatalogueReportSpace.from_catalogue(old), CatalogueReportSpace.from_catalogue(new)
    params = ReportKernelParameters(.8, .02, .5, .75, .1)
    a, b = before.matrix(params), after.matrix(params)
    np.testing.assert_array_equal(b[:128, :, [after.codes.index(c) for c in before.codes]], a)
    np.testing.assert_allclose(b.sum(-1), 1.)
    reader = OralReadout(new, OralReadoutParameters(.8, 0., 0., .75, .1, 0.))
    def code(text):
        report = encode_report(text, ('neck', 'leg', 'tail', 'head'))
        assert report.status == 'coded'
        return report_code(report.tokens)
    x = np.array([.45, .5, .2, .8])
    p, _ = reader.distribution(140, 1, x)
    assert p[reader.codes.index(code('脖子和腿差不多长，尾巴短。'))] == pytest.approx(.64)
    assert p[reader.codes.index(code('脖子和腿差不多长。'))] == pytest.approx(.16)
    p, _ = reader.distribution(128, 2, [.7, .2, .9, .4])
    assert p[reader.codes.index(code('脖子长于腿，脖子短于尾巴。'))] == pytest.approx(.64)
