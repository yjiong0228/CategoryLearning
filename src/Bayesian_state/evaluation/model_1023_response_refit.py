"""Audit response refits with existing likelihood and seed-budget checks.

Use --study COMPLETE --output NEW [--allow-smoke]. Precision-grid checks mark
only the two varied coordinates as estimated. Independent audits preserve the
parameter selected before their seed bank was opened. Summaries include all
screened precision combinations and every fixed audit contender.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_1023_estimation_reliability import expected_stages,validate_stage
from .model_0923_h_validation import nll,jackknife_scores,jackknife_se
from ..workflows.analysis.model_1023_response_refit import ROOT,read,write,digest
from ..workflows.analysis.model_1021_simplification_recovery import compact_zero_effect_coordinates


def summarize(study: Path,output: Path,allow_smoke: bool=False) -> None:
    if output.exists():raise FileExistsError('Use a new response estimation audit')
    done,protocol=read(study/'completion.json'),read(study/'protocol.json')
    if done['status']!='complete' or done['smoke'] and not allow_smoke:
        raise ValueError('Require completed estimates; smoke audits need --allow-smoke')
    smoke=done['smoke'];phase=done['phase'];cases=read(study/'cases.json')['cases'];inputs={}

    def bind(file,expected=None):
        p=Path(file).resolve();sha=digest(p)
        if expected is not None and sha!=expected:raise ValueError('Estimation input changed: '+str(p))
        inputs[str(p)]=sha;return p

    for p in [study/k for k in ('completion.json','protocol.json','cases.json','selected_nominees.json','resource_status.json')]+[Path(__file__)]:bind(p)
    history,jobs,audited,parameters,grid=[],[],[],[],[]
    for case in cases:
        fit=Path(case['output']);frozen=read(bind(fit/'nominees.json'));finished=read(bind(fit/'completion.json'))
        if finished['nomination_sha256']!=digest(fit/'nominees.json') or finished['test_choices_used']:
            raise ValueError('Estimate changed or used later behavior')
        selected=frozen['nominee']['candidate']['id'];reference=frozen['nominee']['reference_id']
        if phase=='joint':budgets=expected_stages(case['settings'],smoke)
        else:
            counts=dict(screen=4 if smoke else 80,selection=2 if smoke else 8,confirmation=2 if smoke else 3)
            budgets={name:(count,2 if smoke else case['settings']['budgets'][name]['particles'],
                2 if smoke else case['settings']['budgets'][name]['seeds']) for name,count in counts.items()}
        for name,budget in budgets.items():
            rows,filters=validate_stage(fit,case,name,budget,smoke,bind,role='selection')
            history.extend(rows);jobs.extend(filters)
        if phase=='precision':
            screen=read(fit/'screen_completion.json')['groups'][0]['rows']
            coords=protocol['settings']['precision_check']['coordinates']
            anchor=case['required']['hyperparams']
            for point in screen:
                if any(point['hyperparams'][k]!=v for k,v in anchor.items() if k not in coords):
                    raise ValueError('Precision grid changed another coordinate')
                grid.append(dict(subject=case['subject'],case_id=case['case_id'],candidate=point['id'],
                    beta_init=point['hyperparams'][coords[0]],learning_rate=point['hyperparams'][coords[1]],
                    nll=point['mean_nll'],selected=point['id']==selected,reference=point['id']==reference))
        audit=study/'audit_fit'/case['case_id'];fixed=read(bind(audit/'nominees.json'))
        if not fixed['selected_before_audit'] or fixed['nominee']['candidate']['id']!=selected or fixed['selection_sha256']!=digest(fit/'nominees.json'):
            raise ValueError('Audit changed the selected parameter')
        budget=(2 if smoke else case['settings']['audit_candidates'],2 if smoke else 1024,2 if smoke else 8)
        rows,filters=validate_stage(audit,case,'confirmation',budget,smoke,bind,role='independent_audit')
        history.extend(rows);jobs.extend(filters)
        estimates={}
        for point in fixed['nominee']['finalists']:
            record=read(bind(point['score_file']))
            with np.load(bind(Path(point['score_file']).with_suffix('.npz'),record['predictions_sha256'])) as z:
                estimates[point['id']]=(nll(z['observed_probability']),jackknife_scores(z['observed_probability']))
        value0,jk0=estimates[selected]
        for point in fixed['nominee']['finalists']:
            value,jk=estimates[point['id']];se=jackknife_se(jk-jk0)
            tag=dict(case_id=case['case_id'],subject=case['subject'],condition=case['condition'],cohort=case['cohort'],
                phase=phase,response_exponent=case['response_exponent'],candidate=point['id'],
                selected=point['id']==selected,reference=point['id']==reference,nll=value,nll_mc_se=jackknife_se(jk),
                delta_to_selected=value-value0,delta_mc_se=se,no_clear_loss_at_mc_resolution=value-value0<=2*se)
            audited.append(tag);inactive=compact_zero_effect_coordinates(point['hyperparams'])
            for key,levels in case['space'].items():
                fitted=len(levels)>1 and (phase=='joint' or key in coords)
                parameters.append(dict(tag,parameter=key,value=point['hyperparams'][key],fitted=fitted,
                    known_zero_effect=key in inactive,normalized_grid_position=levels.index(point['hyperparams'][key])/max(1,len(levels)-1)))
        for folder in (fit,audit):
            for file in (folder/'jobs').glob('*.json'):
                job=read(bind(file));response=read(bind(Path(job['target']).with_suffix('.response.json')))
                if job['response_exponent']!=case['response_exponent'] or response['exponent']!=case['response_exponent']:
                    raise ValueError('Fit used a different response mechanism')
                score=read(Path(job['target']).with_suffix('.json'))
                expected=job['particles']*score['trials'] if case['response_exponent']==2 else 0
                if response['filter_reads']!=expected or response['trajectory_reads']!=0:
                    raise ValueError('Filter response transformation was not applied exactly once')
    resources=read(study/'resource_status.json')
    if resources['completed']!=len(jobs) or resources['failed'] or resources['active'] or resources['peak']>protocol['workers']:
        raise ValueError('Numerical worker accounting differs from executed filters')
    output.mkdir(parents=True)
    for name,rows in [('stage_candidates',history),('executed_filters',jobs),('audit_candidates',audited),('audit_parameters',parameters)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    if grid:pd.DataFrame(grid).to_csv(output/'precision_grid.csv',index=False)
    write(output/'completion.json',dict(status='complete',smoke=smoke,phase=phase,study=str(study.resolve()),
        cases=len(cases),checked_filters=len(jobs),peak_workers=resources['peak'],inputs_sha256=inputs,
        selected_before_independent_audit=True,recovery={},
        uncertainty='Paired filter simulation error; audited candidates are not a parameter confidence region',
        artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-smoke',action='store_true');args=parser.parse_args();summarize(args.study,args.output,args.allow_smoke)
