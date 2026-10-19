"""Isolated loader instrumentation preserves the frozen replay contract."""
from copy import deepcopy
import importlib.util
import json

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.particle_basis import with_particle_basis
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.workflows.analysis import model_0923_particle_basis_replay as replay
from src.Bayesian_state.workflows.analysis.model_0923_oral_reason_method_audit import contribution_example


def test_only_loaded_collector_is_extended_and_loader_restored():
    before = importlib.util.spec_from_file_location
    with replay.particle_collector_loading(with_particle_basis) as installed:
        module = replay.load_module('src.Bayesian_state.evaluation.oral._particle_collector_test', replay.ROOT / replay.COLLECTOR)
        other = replay.load_module('_particle_extension_test', replay.ROOT / replay.PARTICLE)
        assert len(installed) == 1 and other.VERSION == 1
        _, snapshot, _ = contribution_example('simultaneous_mixture')
        collector = module.DecisionBasisCollector(build_continuous_hypothesis_space(4, 2), np.array([1]))
        collector(0, 'predictive', snapshot)
        np.testing.assert_array_equal(collector.arrays()['particle_basis_choice_weights'], np.full((1, 2, 2), .5))
    assert importlib.util.spec_from_file_location is before


def test_loader_restored_when_observation_fails():
    original = importlib.util.spec_from_file_location
    with pytest.raises(RuntimeError):
        with replay.particle_collector_loading(with_particle_basis):
            raise RuntimeError('Synthetic worker failure')
    assert importlib.util.spec_from_file_location is original


@pytest.mark.parametrize('changed', ['particles', 'seeds', 'trial_limit', 'candidate', 'evaluation_use'])
def test_capture_cannot_change_the_scientific_job(changed, tmp_path):
    original = dict(target=str(tmp_path / 'old'), replay_source_sha256={}, particles=4, seeds=[12], trial_limit=8,
                    candidate={'id': 'original'}, evaluation_use='development_cohort_belief_refit')
    source = tmp_path / 'source.json'; source.write_text(json.dumps(original))
    modified = dict(deepcopy(original), target=str(tmp_path / 'new'), particle_basis_source_job=str(source),
                    particle_basis_source_job_sha256=replay.digest(source), particle_basis_inputs_sha256={})
    modified[changed] = {'particles': 8, 'seeds': [13], 'trial_limit': 4,
                         'candidate': {'id': 'other'}, 'evaluation_use': 'other_scope'}[changed]
    job = tmp_path / 'job.json'; job.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match='changed the original state experiment'):
        replay.worker(job)
    assert not (tmp_path / 'new.npz').exists()


def test_existing_output_and_unknown_scope_are_rejected_before_execution(tmp_path):
    output = tmp_path / 'output'; output.mkdir()
    with pytest.raises(FileExistsError):
        replay.run(tmp_path / 'absent.json', output)
    source = tmp_path / 'job.json'; source.write_text(json.dumps({'evaluation_use': 'unknown'}))
    with pytest.raises(ValueError, match='Unsupported'):
        replay.run(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()
