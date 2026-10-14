"""Chronological boundaries and the training search's predictive objective."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_validation import pooled_nll, training_boundary


@pytest.mark.parametrize('trials,expected', [(128,64),(192,128),(704,480),(1408,960)])
def test_chronological_boundary_retains_both_sections(trials,expected):
    assert training_boundary(trials)==expected
    assert 64<=expected<=trials-64


def test_too_short_is_excluded_by_length_only():
    with pytest.raises(ValueError):training_boundary(127)


def test_seeds_pool_probabilities_before_log_scoring():
    p=np.array([[.1,.9],[.9,.1]])
    assert pooled_nll(p)==pytest.approx(-np.log(.5))
    assert pooled_nll(p)!=pytest.approx(-np.log(p).mean())


@pytest.mark.parametrize('probability', [[[np.nan]], [[-1.]], [[1.1]], [.5,.5]])
def test_invalid_predicted_probabilities_rejected(probability):
    with pytest.raises(ValueError):pooled_nll(probability)


def test_training_scoring_does_not_open_sealed_test_arrays(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest,read,score,write
    folder=tmp_path/'subject';folder.mkdir()
    np.savez(folder/'training.npz',stimulus=np.zeros((3,4)),choices=[1,2,1],categories=[2,2,1],feedback=[0,1,1])
    # Deliberately unreadable as an NPZ. Training must never attempt to load it.
    (folder/'sealed_test.npz').write_bytes(b'sealed test sentinel')
    write(folder/'engine.json',dict(modules=dict(hypo_transitions_mod=dict(kwargs={}))))
    write(folder/'prepared.json',dict(subject=101,condition=1,training_trials=3,test_trials=2,
        engine_sha256=digest(folder/'engine.json'),
        array_sha256={name:digest(folder/name) for name in ['training.npz','sealed_test.npz']},
        paths=dict(processed_dir=str(tmp_path))))
    calls=[]
    def filter_stub(**kwargs):
        calls.append(len(kwargs['choices']))
        return SimpleNamespace(observation_probabilities={'prior_t':np.full((3,2),.5)})
    monkeypatch.setattr('src.Bayesian_state.inference.backends.particle_filter.run_state_model_particle_filter',filter_stub)
    job=dict(folder=str(folder),source_sha256={},mechanism={'kind':'B'},candidate={'hyperparams':{}},
        phase='training',smoke=False,seeds=[1,2],particles=2,target=str(tmp_path/'score'),model='B')
    score(job,None)
    assert calls==[3,3]
    assert read(tmp_path/'score.json')['mean_nll']==pytest.approx(np.log(2))


def test_heldout_scoring_runs_one_causal_sequence_and_scores_suffix(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest,read,score,write
    folder=tmp_path/'subject';folder.mkdir()
    for name,choices in [('training',[1,2,1]),('sealed_test',[2,1])]:
        np.savez(folder/(name+'.npz'),stimulus=np.zeros((len(choices),4)),choices=choices,
                 categories=np.ones(len(choices),int),feedback=np.ones(len(choices)))
    write(folder/'engine.json',dict(modules=dict(hypo_transitions_mod=dict(kwargs={}))))
    write(folder/'prepared.json',dict(subject=101,condition=1,training_trials=3,test_trials=2,
        engine_sha256=digest(folder/'engine.json'),
        array_sha256={name:digest(folder/name) for name in ['training.npz','sealed_test.npz']},
        paths=dict(processed_dir=str(tmp_path))))
    def filter_stub(**kwargs):
        assert kwargs['choices'].tolist()==[1,2,1,2,1]
        return SimpleNamespace(observation_probabilities={'prior_t':np.array([[.5,.5]]*3+[[.2,.8],[.7,.3]])})
    monkeypatch.setattr('src.Bayesian_state.inference.backends.particle_filter.run_state_model_particle_filter',filter_stub)
    job=dict(folder=str(folder),source_sha256={},mechanism={'kind':'B'},candidate={'hyperparams':{}},
        phase='evaluate',smoke=False,seeds=[1,2],particles=2,target=str(tmp_path/'score'),model='B')
    score(job,None)
    row=read(tmp_path/'score.json')
    assert row['training_nll']==pytest.approx(np.log(2))
    assert row['test_nll']==pytest.approx(-np.log([.8,.7]).mean())


def test_autonomous_generation_receives_stimuli_and_categories_only(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import autonomous,digest,write
    folder=tmp_path/'subject';folder.mkdir()
    # These files deliberately contain no observed choices or feedback.
    for name,length in [('training',3),('sealed_test',2)]:
        np.savez(folder/(name+'.npz'),stimulus=np.zeros((length,4)),categories=np.ones(length,int))
    write(folder/'engine.json',dict(modules=dict(hypo_transitions_mod=dict(kwargs={}))))
    write(folder/'prepared.json',dict(subject=101,condition=1,training_trials=3,test_trials=2,
        engine_sha256=digest(folder/'engine.json'),
        array_sha256={name:digest(folder/name) for name in ['training.npz','sealed_test.npz']},
        paths=dict(processed_dir=str(tmp_path))))
    def generator_stub(**kwargs):
        assert len(kwargs['stimulus'])==5
        assert 'choices' not in kwargs and 'feedback' not in kwargs
        return SimpleNamespace(trajectory=SimpleNamespace(observed_probabilities=np.full((5,2),.5),
            choices=np.ones(5,int),feedback=np.ones(5)))
    monkeypatch.setattr('src.Bayesian_state.simulation.autonomous.run_autonomous_category_learning',generator_stub)
    job=dict(folder=str(folder),source_sha256={},mechanism={'kind':'B'},candidate={'hyperparams':{}},
        seeds=[1,2],target=str(tmp_path/'generated'),model='B')
    autonomous(job,None)
    with np.load(tmp_path/'generated.npz') as z:assert z['correct_probability'].shape==(2,5)


@pytest.mark.parametrize('generate',[False,True])
def test_parallel_seed_chunks_combine_without_changing_scoring(tmp_path,generate):
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import combine_evaluation_chunks,digest,write
    chunks=[]
    for index,p in enumerate([.1,.9]):
        target=tmp_path/f'chunk_{index}'
        arrays=dict(seeds=np.array([index]),categories=np.ones(5,int),correct_probability=np.full((1,5),p))
        if generate:arrays.update(choices=np.ones((1,5),int),feedback=np.ones((1,5)))
        else:arrays.update(observed_probability=np.full((1,5),p),choices=np.ones(5,int),
                           probabilities=np.tile([p,1-p],(1,5,1)))
        np.savez(target.with_suffix('.npz'),**arrays)
        write(target.with_suffix('.json'),dict(seconds=1.,cpu_seconds=.8,training_trials=3,
            predictions_sha256=digest(target.with_suffix('.npz'))))
        chunks.append(dict(target=str(target)))
    row=combine_evaluation_chunks(dict(target=str(tmp_path/'pooled'),seeds=[0,1]),chunks,generate)
    with np.load(tmp_path/'pooled.npz') as z:
        np.testing.assert_array_equal(z['seeds'],[0,1])
        assert z['correct_probability'].shape==(2,5)
    if generate:assert row['paths']==2
    else:assert row['test_nll']==pytest.approx(np.log(2))
