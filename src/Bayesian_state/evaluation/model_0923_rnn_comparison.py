"""Compare a frozen RNN pilot with completed H predictions and reason profiles.

Use --rnn RUN --oral RNN_PROFILE --h-evaluation H_RUN (repeatable) --output NEW.
Optional --h-profile READING=PROFILE (repeatable) joins exactly the same reports.
Choice scores use the probability of the observed response before its assimilation;
H filter probabilities are averaged over the original eight seeds before logging.
The two families retain their declared capacities and optimization procedures.
The report is a development comparison, with no population significance claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write


def summarize(rnn: Path, oral: Path, h_evaluations: list[Path], h_profiles: list[str], output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new comparison directory')
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return path

    def checked_artifacts(folder):
        done = read(bind(folder/'completion.json'))
        if done['status'] != 'complete':
            raise ValueError('Require completed scientific artifacts')
        verify(done.get('inputs_sha256', {}))
        inputs.update(done.get('inputs_sha256', {}))
        for name, sha in done.get('artifacts_sha256', {}).items():
            if digest(bind(folder/name)) != sha:
                raise ValueError('Completed artifact changed')
        return done

    bind(__file__)
    completed = checked_artifacts(rnn)
    checked_artifacts(oral)
    rnn_records = {row['subject']: row for row in completed['evaluations']}
    choice_rows, trial_rows = [], []
    datasets = {}
    for subject, record in rnn_records.items():
        file = bind(rnn/'evaluation'/f'evaluate_S{subject}.npz')
        if digest(file) != record['predictions_sha256']:
            raise ValueError('Frozen RNN predictions changed')
        with np.load(file) as z:
            datasets[subject] = {key: z[key] for key in z.files}
    for source in h_evaluations:
        checked_artifacts(source)
        evaluation = read(bind(source/'conditional/evaluation.json'))
        rows = [r for r in evaluation['rows'] if r['model'] == 'SEARCH']
        if len(rows) != 1:
            raise ValueError('Need one original SEARCH reference per H evaluation')
        h = rows[0]; subject = h['subject']; data = datasets[subject]
        file = bind(source/'conditional/evaluate'/f'subject_{subject}'/'SEARCH'/(h['candidate']['id']+'.npz'))
        if digest(file) != h['predictions_sha256']:
            raise ValueError('Original H probabilities changed')
        with np.load(file) as z:
            np.testing.assert_array_equal(z['choices'], data['choices'])
            np.testing.assert_array_equal(z['categories'], data['categories'])
            np.testing.assert_array_equal(z['seeds'], h['seeds'])
            hp = z['observed_probability'].astype(float).mean(axis=0)
            selected = np.take_along_axis(z['probabilities'], (data['choices']-1)[None,:,None], axis=2)[...,0]
            np.testing.assert_array_equal(selected, z['observed_probability'])
        rr = rnn_records[subject]
        if any(rr[k] != h[k] for k in ('subject','condition','training_trials','test_trials')) or len(h['seeds']) != 8:
            raise ValueError('H and RNN task or prefix boundaries differ')
        rp = data['probabilities'][np.arange(len(hp)), data['choices']-1].astype(float)
        # This pilot must reproduce both original scorers without floor differences.
        if np.any(rp < 1e-12) or np.any(hp <= 0):
            raise ValueError('Probability floors require a separate explicit comparison')
        cut = h['training_trials']
        for model, p, original in [('H_SEARCH_MIX', hp, h), ('RNN', rp, rr)]:
            for phase, mask, key in [('training',np.arange(len(hp))<cut,'training_nll'),
                                     ('later',np.arange(len(hp))>=cut,'test_nll')]:
                score = float(-np.log(p[mask]).mean())
                np.testing.assert_allclose(score, original[key], atol=1e-12, rtol=0)
                choice_rows.append(dict(subject=subject,condition=h['condition'],model=model,phase=phase,
                    trials=int(mask.sum()),nll=score))
            for t, probability in enumerate(p):
                trial_rows.append(dict(subject=subject,model=model,state_trial=t,phase='training' if t<cut else 'later',
                    choice=int(data['choices'][t]),correct=bool(data['choices'][t]==data['categories'][t]),
                    observed_probability=probability,nll=float(-np.log(probability))))
    if {r['subject'] for r in choice_rows} != set(rnn_records) or len(choice_rows) != 4*len(rnn_records):
        raise ValueError('Need exactly one H evaluation for every RNN participant')
    reports = pd.read_csv(oral/'report_trials.csv', keep_default_na=False)
    coverage = pd.read_csv(oral/'coverage.csv', keep_default_na=False)
    feature_table = pd.read_csv(oral/'reported_feature_effects.csv')
    numerical_checks = 0
    for row in reports.itertuples():
        data = datasets[row.subject]; t = row.state_trial
        q = data['reflection_probabilities'][t,:,row.choice-1].astype(float)
        reported = [i for i in range(4) if row.reported_mask & (1<<i)]
        positive, absolute, actual, negative = [], [], [], []
        for i in reported:
            contexts = [j for j in range(16) if j & ~row.reported_mask == 0 and j & (1<<i) == 0]
            delta = np.array([q[j]-q[j | (1<<i)] for j in contexts])
            positive.append(max(0., delta.max())); negative.append(max(0., -delta.min()))
            absolute.append(np.abs(delta).max()); actual.append(q[0]-q[1<<i])
        expected = [min(positive),max(0.,min(actual)),min(absolute),np.mean(positive),q[0]-q[row.reported_mask]]
        observed = [row.minimum_positive_effect_mean,row.minimum_actual_positive_effect_mean,
                    row.minimum_absolute_effect_mean,row.mean_positive_effect_mean,row.report_reflection_effect]
        np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)
        numerical_checks += len(expected)
        ft = feature_table[(feature_table.subject==row.subject)&(feature_table.reading==row.reading)&(feature_table.state_trial==t)]
        ft = ft.sort_values('feature_index')
        np.testing.assert_array_equal(ft.feature_index, reported)
        np.testing.assert_allclose(ft[['actual_effect','positive_context_effect','negative_context_effect','absolute_context_effect']],
                                   np.array([actual,positive,negative,absolute]).T,atol=1e-14,rtol=0)
        numerical_checks += 4*len(reported)
        cv = coverage[(coverage.subject==row.subject)&(coverage.reading==row.reading)&(coverage.state_trial==t)]
        if len(cv)!=1 or ft.feature.tolist()!=[json.loads(cv.feature_order.iloc[0])[i] for i in reported]:
            raise ValueError('Physical feature labels changed')
    paired = []
    identity = ['subject','state_trial','iSession','iTrial','choice','phase','text','reported_features','reported_mask']
    measures = ['minimum_positive_effect_mean','minimum_actual_positive_effect_mean','minimum_absolute_effect_mean',
                'mean_positive_effect_mean','report_reflection_effect']
    for specification in h_profiles:
        reading, path = specification.split('=', 1); source = Path(path)
        done = checked_artifacts(source)
        selected = reports[(reports.subject==done['subject'])&(reports.reading==reading)]
        h = pd.read_csv(bind(source/'report_trials.csv'), keep_default_na=False)
        for weighting in h.weighting.unique():
            part = h[h.weighting==weighting]
            joined = selected[identity+measures].merge(part[identity+measures],on=identity,how='outer',
                validate='one_to_one',indicator=True,suffixes=('_rnn','_h'))
            if len(joined)!=len(selected) or not joined['_merge'].eq('both').all():
                raise ValueError('H and RNN reason profiles must cover exactly the same reports')
            for phase, group in joined.groupby('phase'):
                paired.append(dict(subject=done['subject'],reading=reading,h_arm=done['arm'],weighting=weighting,
                    phase=phase,reports=len(group),**{k+'_'+model:float(group[k+'_'+model].mean())
                                                   for k in measures for model in ('rnn','h')}))
    output.mkdir(parents=True)
    for name, table in [('choice_scores',pd.DataFrame(choice_rows)),('choice_trials',pd.DataFrame(trial_rows)),
                        ('paired_reason_scores',pd.DataFrame(paired))]:
        table.to_csv(output/(name+'.csv'),index=False)
    verify(inputs)
    write(output/'completion.json',dict(status='complete',subjects=sorted(rnn_records),inputs_sha256=inputs,
        numerical_reason_checks=numerical_checks,original_choice_scores_reproduced=True,
        unchanged_probability_floor=True,artifacts_sha256={p.name:digest(p) for p in output.iterdir()},
        scope='Development pilot; fixed family-specific optimization, original chronological boundaries',
        interpretation='Potential feature effects under the same contrasts; no report likelihood or population claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rnn','oral','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--h-evaluation', action='append', required=True, type=Path)
    parser.add_argument('--h-profile', action='append', default=[])
    args = parser.parse_args()
    summarize(args.rnn,args.oral,args.h_evaluation,args.h_profile,args.output)
