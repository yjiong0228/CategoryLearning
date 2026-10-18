"""Only declared coordinates change; saved parameter and trial identity persist."""
import json

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_recalled_capacity_memory_probe import (
    CAPACITY, MEMORY, descriptive_windows, points, reference_jobs,
)
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate


@pytest.mark.parametrize('memory,capacity', [(0., 8), (.3, 4)])
def test_crossed_intervention_includes_anchor_and_keeps_other_parameters(memory, capacity):
    anchor = {MEMORY: memory, CAPACITY: capacity, 'precision': 15., 'inertia': 1.}
    grid = points(anchor, [memory, .8], [2, 3, capacity])
    assert len(grid) == len({row['id'] for row in grid}) == 6
    assert sum(row['hyperparams'] == anchor for row in grid) == 1
    assert {(row['hyperparams'][MEMORY],row['hyperparams'][CAPACITY]) for row in grid} == {
        (g,c) for g in [memory,.8] for c in [2,3,capacity]}
    assert all(row['hyperparams']['precision'] == 15. and row['hyperparams']['inertia'] == 1.
               for row in grid)
    assert anchor[MEMORY] == memory and anchor[CAPACITY] == capacity


@pytest.mark.parametrize('memory,capacity', [([0.,0.],[2,8]), ([0.,float('nan')],[2,8]),
    ([0.,1.1],[2,8]), ([0.,.8],[2,2,8]), ([0.,.8],[2,8.]), ([.3,.8],[2,8]),
    ([0.,.8],[2,3]), ([0.,True],[2,8]), ([0.,.8],[1,8])])
def test_invalid_or_unanchored_grids_fail(memory,capacity):
    with pytest.raises(ValueError,match='original coordinates'):
        points({MEMORY:0.,CAPACITY:8},memory,capacity)


def save_job(source, name, seeds, value=0.):
    directory = source / 'jobs'
    directory.mkdir(exist_ok=True)
    folder = source / 'prepared'
    folder.mkdir(exist_ok=True)
    (folder / 'prepared.json').write_text(json.dumps(dict(subject=208)))
    file = directory / name
    file.write_text(json.dumps(dict(phase='autonomous', model='SEARCH', seeds=seeds, smoke=False,
                                   folder=str(folder),
                                   candidate=candidate({MEMORY:value}, 'fixture'))))
    return file


@pytest.mark.parametrize('prefix', ['S208_SEARCH_', 'AUTO_evaluate_S208_SEARCH_'])
def test_one_saved_search_point_accepts_both_source_layouts(tmp_path, prefix):
    files = [save_job(tmp_path, prefix+str(i)+'.json', [10+i]) for i in range(2)]
    assert reference_jobs(tmp_path, 208) == files
    assert reference_jobs(tmp_path, 207) == []


def test_mixed_source_layouts_fail(tmp_path):
    save_job(tmp_path, 'S208_SEARCH_000.json', [10])
    save_job(tmp_path, 'AUTO_evaluate_S208_SEARCH_000.json', [11])
    with pytest.raises(ValueError, match='Ambiguous'):
        reference_jobs(tmp_path, 208)


@pytest.mark.parametrize('seeds,value,match', [([11],.8,'parameter points'), ([10],0.,'seeds'),
                                            ([11,11],0.,'seeds')])
def test_inconsistent_reference_banks_fail(tmp_path, seeds, value, match):
    save_job(tmp_path, 'AUTO_evaluate_S208_SEARCH_000.json', [10])
    save_job(tmp_path, 'AUTO_evaluate_S208_SEARCH_001.json', seeds, value)
    with pytest.raises(ValueError, match=match):
        reference_jobs(tmp_path, 208)


def test_end_aligned_windows_preserve_every_trial_and_the_last128():
    windows = descriptive_windows(576,128,'end')
    assert windows == [(0,64),(64,192),(192,320),(320,448),(448,576)]
    assert [trial for start,stop in windows for trial in range(start,stop)] == list(range(576))
    assert descriptive_windows(16,128,'end') == [(0,16)]
    assert descriptive_windows(896,128,'end') == descriptive_windows(896,128)
    assert descriptive_windows(576,128)[-1] == (512,576)


def test_unknown_window_alignment_fails():
    with pytest.raises(ValueError, match='alignment'):
        descriptive_windows(576,128,'middle')


@pytest.mark.parametrize('smoke', [True, None])
def test_reference_must_explicitly_contain_scientific_paths(tmp_path, smoke):
    file = save_job(tmp_path, 'AUTO_evaluate_S208_SEARCH_000.json', [10])
    job = json.loads(file.read_text())
    job['smoke'] = smoke
    file.write_text(json.dumps(job))
    with pytest.raises(ValueError, match='Invalid'):
        reference_jobs(tmp_path, 208)


def test_reference_subject_cannot_be_inferred_from_a_filename_alone(tmp_path):
    save_job(tmp_path, 'AUTO_evaluate_S208_SEARCH_000.json', [10])
    (tmp_path / 'prepared/prepared.json').write_text(json.dumps(dict(subject=207)))
    with pytest.raises(ValueError, match='subject differs'):
        reference_jobs(tmp_path, 208)
