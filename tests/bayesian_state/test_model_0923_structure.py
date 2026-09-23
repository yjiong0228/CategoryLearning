"""Bounded feature reuse must preserve labels, old geometry and report masses."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.hypothesis_space.geometry.boundary import BoundaryGeometry
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_axis_pair_overlap_probe_space
from src.Bayesian_state.model.oral_report_space import CatalogueReportSpace, ReportKernelParameters, report_code
from src.Bayesian_state.workflows.analysis.audit_model_0923_reports import audit_reports
from src.Bayesian_state.workflows.analysis.prepare_model_0923 import _write_csv
from src.Bayesian_state.workflows.analysis.probe_model_0923_structure import probe_structure

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ('neck', 'leg', 'tail', 'head')


def test_only_twelve_rules_added_and_old_specifications_preserved():
    base = build_continuous_hypothesis_space(4, 4)
    candidate = build_axis_pair_overlap_probe_space()
    assert len(base) == 116 and len(candidate) == 128
    assert candidate.signature != base.signature
    assert candidate.parameters['label_permutation_policy'] == 'identity_only'
    for original, retained in zip(base, candidate):
        assert original.index == retained.index
        assert original.family == retained.family
        assert original.hyperplanes == retained.hyperplanes
        assert dict(original.parameters) == dict(retained.parameters)
        assert original.feedback_neighbors == retained.feedback_neighbors
        for before, after in zip(original.categories, retained.categories):
            for a, b in zip(before.components, after.components):
                np.testing.assert_array_equal(a.A, b.A)
                np.testing.assert_array_equal(a.b, b.b)
    assert len(build_continuous_hypothesis_space(4, 2)) == 29


def test_new_rules_follow_feature_order_not_label_permutations():
    candidate = build_axis_pair_overlap_probe_space()
    seen = set()
    for h in candidate.hypotheses[116:]:
        axis = h.parameters['axis_dimension']
        pair = tuple(h.parameters['related_dimensions'])
        assert axis in pair and pair[0] < pair[1]
        assert h.label_permutation == (0, 1, 2, 3)
        assert not h.is_label_permuted
        assert h.base_hypothesis_index == h.index
        expected = np.eye(4)[pair[0]] - np.eye(4)[pair[1]]
        np.testing.assert_array_equal(h.hyperplanes[1][0], expected)
        assert h.hyperplanes[0][1] == 0.5
        seen.add((axis, pair))
    assert len(seen) == 12


@pytest.mark.parametrize('offset', range(12))
def test_four_nonempty_regions_cover_space_without_overlapping_interiors(offset):
    h = build_axis_pair_overlap_probe_space()[116 + offset]
    points = np.random.default_rng(924).uniform(size=(10000, 4))
    membership = np.column_stack([np.all(points @ c.components[0].A.T <= c.components[0].b, axis=1)
                                  for c in h.categories])
    np.testing.assert_array_equal(membership.sum(1), np.ones(len(points), dtype=int))
    # The threshold bisects a square and the diagonal splits each half into
    # triangles/trapezoids with exact areas 1/8 and 3/8, independent of axis.
    np.testing.assert_allclose(np.sort(membership.mean(0)), [.125, .125, .375, .375], atol=.02, rtol=0)


def test_boundary_geometry_agrees_with_independent_distance_calculation():
    space = build_axis_pair_overlap_probe_space()
    geometry = BoundaryGeometry(space, method=BoundaryGeometry.METHOD_KKT_ACTIVE_SET)
    # H116: threshold x0 <= .5 and comparison x0 <= x1.
    point = np.array([[.8, .2, .5, .5]])
    distance = geometry.category_distances(116, point)
    np.testing.assert_allclose(distance, [[np.sqrt(.18), .3, np.sqrt(.18), 0]], atol=1e-9)
    p = geometry.category_probabilities(116, point, 5).reshape(-1)
    expected = np.exp(-5 * np.array([np.sqrt(.18), .3, np.sqrt(.18), 0]))
    expected /= expected.sum()
    np.testing.assert_allclose(p, expected, atol=1e-12)


def test_real_report_can_gain_structure_without_gaining_its_observed_label():
    base = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, 4))
    candidate = CatalogueReportSpace.from_catalogue(build_axis_pair_overlap_probe_space())
    parsed = encode_report('躯干短于脖子，且脖子长于尾巴。', FEATURES)
    code = report_code(parsed.tokens)
    old_index, new_index = base.source_index(), candidate.source_index()
    assert all((code, y) not in old_index for y in range(4))
    assert (code, 3) in new_index  # Category 4, not S314's observed category 3.
    assert (code, 2) not in new_index
    assert code in candidate.codes


def test_appending_rules_preserves_each_old_report_probability_in_common_vocabulary():
    base = CatalogueReportSpace.from_catalogue(build_continuous_hypothesis_space(4, 4))
    candidate = CatalogueReportSpace.from_catalogue(build_axis_pair_overlap_probe_space())
    params = ReportKernelParameters(.8, .02, .5, .75, .1)
    old, new = base.matrix(params), candidate.matrix(params)
    assert len(base.codes) == 410 and len(candidate.codes) == 506
    embedded = np.zeros_like(new[:116])
    for index, code in enumerate(base.codes):
        embedded[:, :, candidate.codes.index(code)] = old[:, :, index]
    np.testing.assert_array_equal(new[:116], embedded)
    np.testing.assert_allclose(new.sum(-1), 1, rtol=0, atol=1e-12)


def _inputs(tmp_path):
    inventory = tmp_path / 'inventory'
    inventory.mkdir()
    raw_path = tmp_path / 'raw.csv'
    rows = []
    for sid, condition, texts in [(314, 3, [(3, '躯干短于脖子，且脖子长于尾巴。'),
                                           (4, '躯干短于脖子，且脖子长于尾巴。')]),
                                   (104, 1, [(1, '腿长。')])]:
        for trial, (choice, text) in enumerate(texts, 1):
            rows.append({'iSub': sid, 'condition': condition, 'iSession': 1, 'iBlock': 1, 'iTrial': trial,
                         'trial': trial, 'choice': choice, 'text': text, 'scored': trial != 1, 'oral_valid': True,
                         **{f'feature{i}_name': name for i, name in enumerate(FEATURES, 1)},
                         **{f'feature{i}': value for i, value in enumerate([.8, .3, .2, .6], 1)}})
    _write_csv(raw_path, rows)
    _write_csv(inventory / 'subjects.csv', [{'subject': 314, 'condition': 3, 'n_trials': 2},
                                           {'subject': 104, 'condition': 1, 'n_trials': 1}])
    _write_csv(inventory / 'report_review.csv', [{'status': 'unreviewed'}])
    (inventory / 'manifest.json').write_text(json.dumps({'n_trials': 3, 'sources': [
        {'path': str(raw_path), 'sha256': sha256(raw_path.read_bytes()).hexdigest()}]}))
    r1 = tmp_path / 'r1'
    audit_reports(inventory_dir=inventory, trials_path=raw_path,
                  config_path=ROOT / 'configs/exp123/oral_report_0923.yaml', output_dir=r1)
    return dict(report_dir=r1, trials_path=raw_path,
                config_path=ROOT / 'configs/exp123/model_0923_structure_probe.yaml', output_dir=tmp_path / 'r2')


def test_probe_keeps_labels_masks_and_original_files(tmp_path):
    kwargs = _inputs(tmp_path)
    before = {p: p.read_bytes() for p in kwargs['report_dir'].iterdir()}
    raw_before = kwargs['trials_path'].read_bytes()
    result = probe_structure(**kwargs)
    assert result['added_rules'] == 12
    assert result['counts']['structural_gain'] == 2
    assert result['counts']['observed_label_gain'] == 1
    assert result['counts']['structural_gain_but_label_unresolved'] == 1
    assert result['geometry_check']['valid_categories'] == 48
    assert result['interpretation']['fit_performed'] is False
    trials = pd.read_csv(kwargs['output_dir'] / 'trials.csv')
    assert trials.choice.tolist() == [3, 4, 1]
    assert trials.original_scored.tolist() == [False, True, False]
    assert kwargs['trials_path'].read_bytes() == raw_before
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError):
        probe_structure(**kwargs)


def test_probe_rejects_changed_report_identity_before_creating_output(tmp_path):
    kwargs = _inputs(tmp_path)
    path = kwargs['report_dir'] / 'trials.csv'
    records = pd.read_csv(path)
    records.loc[0, 'choice'] = 4
    records.to_csv(path, index=False)
    with pytest.raises(ValueError, match='identity/order'):
        probe_structure(**kwargs)
    assert not kwargs['output_dir'].exists()
