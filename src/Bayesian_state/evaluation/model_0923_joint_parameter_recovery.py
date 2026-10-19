"""Describe joint parameter recovery and post-estimation generating-point scores.

SEARCH matches the generating mechanism; POOL is a misspecified comparator.
Keep all confirmation candidates and their exact filter seeds. NLL differences
use paired delete-one-filter Monte Carlo errors, conditional on one synthetic
sequence. These errors describe numerical integration, not population inference.
The confirmation winner was selected using the same filters; a score comparison
against truth diagnoses obvious search shortfalls and has selection uncertainty.
Use --prepared --replicate --fit --truth-score --output [--smoke].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import digest, load, nll, jackknife_scores, jackknife_se, inactive_parameter_channels
from ..workflows.analysis.model_0923_h_validation import write
from ..workflows.analysis.model_0923_joint_parameter_recovery import case_settings


def parameter_errors(truth: dict, estimate: dict, space: dict) -> list[dict]:
    """Retain both numeric-span and ordinal-grid errors, with fixed coordinates marked."""
    if set(truth) != set(estimate) or set(truth) != set(space):
        raise ValueError('Joint estimates and truth must have the same coordinates')
    rows=[]
    for key,values in space.items():
        levels=sorted(values); actual, fitted=truth[key],estimate[key]
        if actual not in levels or fitted not in levels or len(set(levels))!=len(levels):
            raise ValueError('Recovery value outside the declared support')
        span=levels[-1]-levels[0]
        rows.append(dict(parameter=key,truth=actual,estimate=fitted,variable=len(levels)>1,
            absolute_error=abs(fitted-actual),normalized_absolute_error=abs(fitted-actual)/span if span else 0.,
            grid_absolute_error=abs(levels.index(fitted)-levels.index(actual)),exact=fitted==actual,
            truth_at_boundary=len(levels)>1 and actual in (levels[0],levels[-1]),
            estimate_at_boundary=len(levels)>1 and fitted in (levels[0],levels[-1])))
    return rows


def paired_scores(estimate: dict, truth: dict) -> dict:
    """Pair complete filters, preserving trial dependence and scoring orientation."""
    np.testing.assert_array_equal(estimate['seeds'],truth['seeds'])
    a,b=estimate['observed_probability'],truth['observed_probability']
    if (a.ndim!=2 or a.shape!=b.shape or a.shape[0]<2 or not a.shape[1]
            or len(estimate['seeds'])!=a.shape[0] or len(set(estimate['seeds'].tolist()))!=a.shape[0]
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any((a<0)|(a>1)) or np.any((b<0)|(b>1))):
        raise ValueError('Require aligned finite prediction probabilities from independent filters')
    delta=nll(a)-nll(b); se=jackknife_se(jackknife_scores(a)-jackknife_scores(b))
    return dict(estimate_nll=nll(a),truth_nll=nll(b),estimate_minus_truth_nll=delta,
        difference_mc_se=se,estimate_worse_than_truth_by_two_mc_se=delta>2*se,
        trials=a.shape[1],filters=a.shape[0])


def summarize(prepared: Path, replicate: int, fit: Path, truth_score: Path, output: Path, smoke: bool=False) -> None:
    if output.exists():
        raise FileExistsError('Use a new joint-recovery summary directory')
    settings,inputs=case_settings(prepared,replicate)
    if smoke: settings['parallel_budget']=1

    def bind(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:
            raise ValueError('Recovery input changed')
        inputs[str(path)]=sha;return path

    def read(path):
        return load(bind(path))

    prep=read(prepared/'protocol.json');prep_done=read(prepared/'completion.json')
    frozen,done,protocol=[read(fit/name) for name in ('nominees.json','completion.json','protocol.json')]
    truth_done,truth_protocol=[read(truth_score/name) for name in ('completion.json','protocol.json')]
    metadata=read(fit/'joint_recovery_protocol.json')
    if (done['status']!='complete' or truth_done['status']!='complete'
            or any(record['smoke']!=smoke for record in (done,frozen,protocol,truth_done,truth_protocol))
            or frozen['settings']!=settings or protocol['settings']!=settings or truth_protocol['settings']!=settings
            or frozen['status']!='frozen_before_test' or done['test_choices_used']
            or done['nomination_sha256']!=digest(fit/'nominees.json')
            or truth_done['fit_nomination_sha256']!=digest(fit/'nominees.json')
            or truth_done['truths_sha256']!=prep_done['truths_sha256']
            or digest(prepared/'truths.json')!=prep_done['truths_sha256']
            or Path(truth_protocol['fit']).resolve()!=fit.resolve()
            or metadata['human_warm_starts_used'] or metadata['generating_parameters_used_for_initialization']):
        raise ValueError('Require frozen independent-start estimates and matching after-fit truth scoring')
    for record in (prep_done,truth_done):
        for path,sha in record['inputs_sha256'].items():
            if digest(bind(path))!=sha: raise ValueError('Preparation or truth-scoring source changed')
    for name,sha in frozen['source_sha256'].items():
        if digest(bind(name))!=sha: raise ValueError('Fitting source changed')
    space=read(fit/'parameter_spaces.json')
    if space['POOL']!=space['SEARCH'] or space['SEARCH']!=prep['parameter_space']:
        raise ValueError('Joint supports differ from the prepared protocol')
    truth_rows=[r for r in read(prepared/'truths.json')['rows'] if r['replicate']==replicate]
    if len(truth_rows)!=1: raise ValueError('Need one matching generating point')
    truth=truth_rows[0];subject=truth['subject'];point=truth['candidate']
    if sorted((r['subject'],r['model']) for r in frozen['nominees'])!=[(subject,'POOL'),(subject,'SEARCH')]:
        raise ValueError('Need both proposal arms for the same synthetic sequence')
    stages={}
    for source,label in ((fit,'estimate'),(truth_score,'truth')):
        record=read(source/'confirmation_completion.json')
        if record['status']!='complete': raise ValueError('Incomplete confirmation bank')
        stages[label]={(r['subject'],r['model']):r['rows'] for r in record['groups']}
        if set(stages[label])!={(subject,'POOL'),(subject,'SEARCH')}:
            raise ValueError('Wrong confirmation subjects or arms')

    def predictions(row,arm):
        file=Path(row['score_file']);record=read(file);array=bind(file.with_suffix('.npz'))
        if (digest(array)!=record['predictions_sha256'] or record['subject']!=subject or record['model']!=arm
                or record['candidate']['hyperparams']!=row['hyperparams']
                or record['candidate']['id']!=row['id']
                or record['particles']!=(2 if smoke else settings['budgets']['confirmation']['particles'])
                or len(record['seeds'])!=(2 if smoke else settings['budgets']['confirmation']['seeds'])):
            raise ValueError('Confirmation prediction identity or numerical budget changed')
        with np.load(array) as saved: arrays={k:saved[k] for k in saved.files}
        np.testing.assert_array_equal(arrays['seeds'],record['seeds'])
        if set(arrays['seeds'].tolist()) & set(settings['recovery_generation_seeds']):
            raise ValueError('Confirmation seeds overlap autonomous generation')
        np.testing.assert_allclose(nll(arrays['observed_probability']),row['mean_nll'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(nll(arrays['observed_probability']),record['mean_nll'],atol=1e-12,rtol=0)
        expected_trials=8 if smoke else next(c['training_trials'] for c in prep['cases'] if c['replicate']==replicate)
        if arrays['observed_probability'].shape[1]!=expected_trials:
            raise ValueError('Synthetic training sequence length changed')
        return arrays

    parameter_rows,score_rows,closure_rows=[],[],[]
    for nominee in frozen['nominees']:
        arm=nominee['model'];base=dict(subject=subject,replicate=replicate,model=arm,
            matches_generating_mechanism=arm=='SEARCH',generating_seed=truth['generation_seed'])
        reference=stages['truth'][subject,arm]
        if len(reference)!=1 or reference[0]['hyperparams']!=point['hyperparams']:
            raise ValueError('Wrong generating-point score')
        true_arrays=predictions(reference[0],arm)
        choices=stages['estimate'][subject,arm]
        if choices!=nominee['finalists'] or nominee['candidate']!=choices[0]:
            raise ValueError('Frozen winner differs from full confirmation bank')
        for row in choices:
            scores=paired_scores(predictions(row,arm),true_arrays)
            score_rows.append(dict(base,candidate_id=row['id'],selected=row['id']==nominee['candidate']['id'],**scores))
        for role,candidate in [('truth',point),('estimate',nominee['candidate'])]:
            closure=inactive_parameter_channels(dict(settings={'models':{arm:settings['mechanism']}},
                nominees=[dict(subject=subject,model=arm,candidate=candidate)]))
            closure_rows.extend(dict(base,role=role,**{k:v for k,v in r.items() if k not in ('subject','model')})
                                for r in closure.to_dict('records'))
        parameter_rows.extend(dict(base,**r) for r in parameter_errors(point['hyperparams'],nominee['candidate']['hyperparams'],space[arm]))
    parameters=pd.DataFrame(parameter_rows)
    for role in ('truth','estimate'):
        closed={(r['model'],r['parameter']) for r in closure_rows if r['role']==role}
        parameters[role+'_closed_channel']=[(r.model,r.parameter) in closed for r in parameters.itertuples()]
    output.mkdir(parents=True)
    parameters.to_csv(output/'parameter_recovery.csv',index=False)
    pd.DataFrame(score_rows).to_csv(output/'confirmation_vs_truth.csv',index=False)
    pd.DataFrame(closure_rows,columns=['subject','replicate','model','matches_generating_mechanism','generating_seed',
        'role','parameter','value','reason']).to_csv(output/'closed_channels.csv',index=False)
    for path,sha in inputs.items():
        if digest(Path(path))!=sha: raise ValueError('Joint recovery input changed during summary')
    write(output/'completion.json',dict(status='complete',subject=subject,replicate=replicate,smoke=smoke,
        inputs_sha256=inputs,source_sha256=digest(Path(__file__)),
        artifacts_sha256={p.name:digest(p) for p in sorted(output.glob('*.csv'))},
        primary_recovery_model='SEARCH',comparator='POOL changes the generating proposal mechanism',
        interpretation='Two sequences per generating environment are a pilot; errors do not establish population reliability or structural identifiability',
        score_uncertainty='Paired complete-filter Monte Carlo errors; same synthetic sequence, confirmation winner selection not accounted for',
        closure_scope='Listed exact H channels only; an unflagged coordinate can remain weakly identified'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','fit','truth-score','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--replicate',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    summarize(args.prepared,args.replicate,args.fit,args.truth_score,args.output,args.smoke)
