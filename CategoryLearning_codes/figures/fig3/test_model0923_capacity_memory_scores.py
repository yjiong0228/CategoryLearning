"""Parameter comparison plots preserve cells, time ranges and observed data."""
import json

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_capacity_memory_scores import digest, load_tables, prepare_tables


def inputs():
    probabilities = np.array([.2, .4, .3, .6, .7, .8])
    original = -np.log(probabilities)
    choice, trials, autonomous, contrasts = [], [], [], []
    for index, (memory, capacity) in enumerate((g,k) for g in [0., .8, .99] for k in [2,4]):
        point = f'point{index}'
        identity = dict(subject=208, point=point, memory=memory, capacity=capacity, anchor=index==1)
        scale = 1.+.03*(index-1)
        loss = -np.log(probabilities*scale)
        for phase, start, stop in [('prefix',0,4),('later',4,6),('full',0,6)]:
            choice.append(dict(identity, model='SEARCH', particles=512, seeds=8, phase=phase,
                trials=stop-start, choice_nll=loss[start:stop].mean(), choice_nll_mc_se=.001,
                delta_choice_nll=(loss-original)[start:stop].mean(), delta_choice_nll_mc_se=.002))
        for trial in range(6):
            trials.append(dict(identity, model='SEARCH', particles=512, seeds=8, trial=trial+1,
                phase='prefix' if trial < 4 else 'later', observed_probability=probabilities[trial]*scale,
                choice_nll=loss[trial], delta_choice_nll=loss[trial]-original[trial], correct=trial%2))
        autonomous.append(dict(identity, paths=64, energy=.3+.01*(index-1), energy_mcse=.003))
        contrasts.append(dict(subject=208,point=point,reference='point1',
            delta_energy=.01*(index-1),delta_energy_mcse=.004))
    return [pd.DataFrame(rows) for rows in (choice,trials,autonomous,contrasts)]


def test_all_cells_phases_and_ordered_trials_retain_their_scores():
    frames = inputs()
    frames[1] = frames[1].iloc[::-1]
    scores, trials = prepare_tables(*frames)
    assert len(scores)==18 and len(trials)==36
    assert scores.point.nunique()==6 and set(scores.phase)=={'prefix','later','full'}
    for point, part in trials.groupby('point'):
        np.testing.assert_array_equal(part.trial,np.arange(1,7))
        full=scores[scores.point.eq(point)&scores.phase.eq('full')].iloc[0]
        assert part.cumulative_delta_choice_nll.iloc[-1] == pytest.approx(6*full.delta_choice_nll)
    assert trials[trials.anchor].cumulative_delta_choice_nll.eq(0).all()


@pytest.mark.parametrize('problem', ['grid','identity','phase','duplicate_trial','missing_trial',
    'observed','probability','loss','difference','energy_difference','reference','contrast_subject',
    'budget','nonfinite','negative_error','split'])
def test_misaligned_or_incomplete_comparisons_are_rejected(problem):
    choice,trials,auto,contrasts=inputs()
    if problem=='grid':auto=auto.iloc[:-1]
    elif problem=='identity':choice.loc[0,'memory']=.4
    elif problem=='phase':choice.loc[0,'phase']='full'
    elif problem=='duplicate_trial':trials.loc[0,'trial']=2
    elif problem=='missing_trial':trials=trials.iloc[1:]
    elif problem=='observed':trials.loc[0,'correct']=1
    elif problem=='probability':trials.loc[0,'observed_probability']=1.1
    elif problem=='loss':trials.loc[0,'choice_nll']+=.1
    elif problem=='difference':choice.loc[0,'delta_choice_nll']+=.1
    elif problem=='energy_difference':contrasts.loc[0,'delta_energy']+=.1
    elif problem=='reference':contrasts.loc[0,'reference']='point0'
    elif problem=='contrast_subject':contrasts.loc[0,'subject']=207
    elif problem=='budget':trials.loc[0,'particles']=256
    elif problem=='nonfinite':choice.loc[0,'delta_choice_nll']=np.nan
    elif problem=='negative_error':contrasts.loc[0,'delta_energy_mcse']=-.1
    elif problem=='split':trials.loc[0,'phase']='later'
    with pytest.raises((ValueError, AssertionError)):
        prepare_tables(choice,trials,auto,contrasts)


def archived_tables(tmp_path, *, wrong_observation=False, short_sequence=False):
    """Small synthetic table archive used only for integrity tests."""
    source=tmp_path/'scores';probe=tmp_path/'probe'
    source.mkdir();probe.mkdir()
    choice,trials,auto,contrasts=inputs()
    choice.to_csv(source/'scores.csv',index=False)
    trials.to_csv(source/'trial_scores.csv',index=False)
    auto.to_csv(probe/'scores.csv',index=False)
    contrasts.to_csv(probe/'contrasts.csv',index=False)
    curves=[]
    for point in auto.point:
        curves.extend([dict(point=point,trial=3,observed=1/3),
                       dict(point=point,trial=4 if short_sequence else 6,
                            observed=0 if wrong_observation else 2/3)])
    pd.DataFrame(curves).to_csv(probe/'curves.csv',index=False)
    (probe/'protocol.json').write_text('{}')
    (probe/'summary.json').write_text(json.dumps(dict(artifacts={n:digest(probe/n)
        for n in ['scores.csv','contrasts.csv','curves.csv']})))
    (probe/'completion.json').write_text(json.dumps(dict(status='complete',smoke=False,
        summary_sha256=digest(probe/'summary.json'))))
    (source/'protocol.json').write_text(json.dumps(dict(smoke=False,probe=str(probe),
        input_sha256={str(p.resolve()):digest(p) for p in probe.iterdir()})))
    (source/'completion.json').write_text(json.dumps(dict(status='complete',smoke=False,
        trials=6,training_trials=4,artifacts_sha256={n:digest(source/n) for n in ['scores.csv','trial_scores.csv']})))
    return source


def test_complete_sequence_archive_keeps_all_trial_and_score_rows(tmp_path):
    tables,context=load_tables(archived_tables(tmp_path),False)
    assert context['choice_trials']==context['autonomous_trials']==6
    assert len(tables['scores'])==18 and len(tables['trials'])==36


@pytest.mark.parametrize('problem',['wrong_observation','short_sequence'])
def test_scientific_comparison_requires_the_same_observed_sequence(tmp_path,problem):
    source=archived_tables(tmp_path,**{problem:True})
    with pytest.raises((ValueError,AssertionError)):
        load_tables(source,False)
