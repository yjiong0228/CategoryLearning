"""Numerical uncertainty and read-only diagnostic contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.Bayesian_state.evaluation.model_0923_parameter_precision import paired_difference
from src.Bayesian_state.workflows.analysis.model_0923_parameter_precision import (
    capture_public_arrays, check_settings, filter_seeds, prepare_jobs,
)


def test_paired_uncertainty_retains_correlated_simulation_error():
    right=np.array([[.2,.4],[.3,.5],[.4,.6],[.5,.7]])
    left=right*.8
    delta,se=paired_difference(left,right)
    assert delta==pytest.approx(-np.log(.8))
    assert se<1e-14


def test_paired_uncertainty_rejects_unaligned_or_too_few_replicates():
    with pytest.raises(ValueError):
        paired_difference(np.ones((4,2)),np.ones((4,3)))
    with pytest.raises(ValueError):
        paired_difference(np.ones((3,2)),np.ones((3,2)))


def test_capture_copies_public_arrays_and_preserves_undefined_states():
    result=SimpleNamespace(observation_probabilities={'prior_t':np.array([[.4,.6]])},
        state_probabilities={'choice_weights':np.array([[.2,.8]])},
        latent_summaries={'undefined':np.array([np.nan]),'absent':None},
        diagnostics={'resampled':np.array([False])})
    captured=capture_public_arrays(result)
    assert np.isnan(captured['latent__undefined'][0])
    assert 'latent__absent' not in captured
    captured['state__choice_weights'][0,0]=1.
    assert result.state_probabilities['choice_weights'][0,0]==.2


def test_frozen_seed_design_is_paired_and_distinct_from_other_subjects():
    settings=yaml.safe_load(Path('configs/shared/model0923_parameter_precision.yaml').read_text())
    check_settings(settings)
    first=filter_seeds(settings,207)
    assert first==filter_seeds(settings,207) and len(first)==len(set(first))==8
    assert not set(first)&set(filter_seeds(settings,305))
    settings['particle_counts']=[1024,256]
    with pytest.raises(ValueError):
        check_settings(settings)


def test_real_job_plan_uses_all_finalists_and_all_replay_paths(tmp_path):
    fit=Path('results/model_0923/h_objective_fit_20261015_v1')
    if not fit.exists():
        pytest.skip('Frozen research candidate bank is not installed')
    settings=yaml.safe_load(Path('configs/shared/model0923_parameter_precision.yaml').read_text())
    jobs,protocol=prepare_jobs(fit,tmp_path/'unused',settings,'both',False)
    filters=[j for j in jobs if j['kind']=='filter'];replays=[j for j in jobs if j['kind']=='replay']
    assert len(protocol['candidates'])==15 and len(filters)==240
    assert all(j['phase']=='training' and len(j['seeds'])==1 for j in filters)
    assert sum(len(j['seeds']) for j in replays)==2048
    assert all(j['subject']==207 for j in replays)
    assert not (tmp_path/'unused').exists()


def test_factorial_effects_use_joint_paired_simulation_errors():
    from src.Bayesian_state.evaluation.model_0923_207_search_retention_probe import factorial_effects
    values={'baseline':10.,'search':7.,'retention':9.,'both':5.}
    common=np.array([-.2,.1,.3,-.1])
    deleted={key:value+common for key,value in values.items()}
    result={row['effect']:row for row in factorial_effects(values,deleted)}
    assert result['search_average']['value']==pytest.approx(-3.5)
    assert result['retention_average']['value']==pytest.approx(-1.5)
    assert result['interaction']['value']==pytest.approx(-1.)
    assert all(row['paired_mc_se']<1e-14 for row in result.values())
