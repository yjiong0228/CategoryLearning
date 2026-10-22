"""Compare early and full-behavior TRANSFER estimates without relabeling a test.

Use --study COMPLETE_SCOPE_CONTROL --config ORAL_CONFIG --output NEW. Both
parameter sets are evaluated with the original trial sequence and matched
numerical seeds. Full-behavior scores are descriptive reconstructions. Original
early-fit later predictions remain identified separately. Every path, both
fixed oral readings and complete-report denominators are preserved. The common
proper scores and passive choice-basis alignment implement all statistics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import checked_arrays, block_paths, distribution_scores, jackknife_scores, jackknife_se, nll
from .model_0923_h_generation_uncertainty import path_jackknife
from .model_1022_compact_refit import training_budget
from ..workflows.analysis import model_0923_decision_basis_alignment as alignment
from ..workflows.analysis.model_1022_compact_scope_control import ROOT, read, write, digest

SCOPES=('EARLY','FULL')


def summarize(study: Path, config: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new scope-summary directory')
    complete=read(study/'completion.json');protocol=read(study/'protocol.json')
    if complete['status']!='complete' or complete['smoke'] or complete['original_later_choices_used'] is not True:
        raise ValueError('Require completed scientific full-behavior controls')
    policy=protocol['settings'];oral_policy=yaml.safe_load(config.read_text())
    early_study,early_evaluation=ROOT/policy['study'],ROOT/policy['evaluation']
    full_cases=[c for c in read(study/'cases.json')['cases'] if c['kind']=='human']
    early_cases=read(early_study/'cases.json')['cases']
    if sorted(c['subject'] for c in full_cases)!=[205,306]:raise ValueError('Require both registered participants')
    inputs={}

    def bind(path,expected=None):
        path=Path(path).resolve();value=digest(path)
        if expected is not None and value!=expected:raise ValueError('Scope-summary input changed: '+str(path))
        inputs[str(path)]=value
        return path

    for p in [study/'completion.json',study/'protocol.json',study/'cases.json',config,Path(__file__),
              Path(alignment.__file__),Path(inspect.getfile(block_paths)),Path(inspect.getfile(path_jackknife))]:bind(p)
    for name,sha in read(study/'cases.json')['inputs_sha256'].items():bind(name,sha)
    scores,curves,paths,parameters,budgets,finalists=[],[],[],[],[],[]
    banks,knives,oral_groups={}, {}, {}
    for full in full_cases:
        subject=full['subject']
        early,=[c for c in early_cases if c['kind']=='human' and c['subject']==subject and c['variant']=='TRANSFER']
        for scope,case,own in [('EARLY',early,early_evaluation/early['case_id']),('FULL',full,study/'evaluation'/full['case_id'])]:
            fit=Path(case['output']);frozen=read(bind(fit/'nominees.json'));fit_done=read(bind(fit/'completion.json'))
            eval_done=read(bind(own/'completion.json'));eval_protocol=read(bind(own/'protocol.json'))
            expected='frozen_before_test' if scope=='EARLY' else 'frozen_before_external_oral_check'
            if (frozen['status']!=expected or frozen['smoke'] or fit_done['status']!='complete'
                    or fit_done['test_choices_used'] is not (scope=='FULL')
                    or eval_done['status']!='complete' or eval_done['smoke']
                    or eval_protocol['nomination_sha256']!=digest(fit/'nominees.json')
                    or fit_done['nomination_sha256']!=digest(fit/'nominees.json')):
                raise ValueError('Estimation scope or frozen estimate differs')
            point=frozen['nominee']['candidate'];settings=frozen['settings'];arrays={};records={}
            for role in ('conditional','autonomous'):
                record,=read(bind(own/role/'evaluation.json'))['rows']
                if record['candidate']!=point or record['subject']!=subject or record['condition']!=case['condition']:
                    raise ValueError('Prediction identity differs from the fitted point')
                file=own/role/'evaluate'/f'subject_{subject}'/'COMPACT'/(point['id']+'.npz')
                bind(file,record['predictions_sha256']);arrays[role]=checked_arrays(own/role,record);records[role]=record
            c,a=arrays['conditional'],arrays['autonomous'];length=len(c['choices']);cut=records['conditional']['training_trials']
            if (c['observed_probability'].shape!=(8,length) or a['choices'].shape!=(128,length)
                    or records['conditional']['particles']!=1024):raise ValueError('Require the registered complete evaluation budget')
            np.testing.assert_array_equal(c['categories'],a['categories'])
            for r in (c,a):np.testing.assert_allclose(r['probabilities'].sum(-1),1,atol=1e-12,rtol=0)
            banks[subject,scope]=arrays
            tag=dict(subject=subject,condition=case['condition'],scope=scope,variant='TRANSFER',
                parameter_data='early_choices' if scope=='EARLY' else 'all_choices')
            actual=c['choices']==c['categories'];generated=a['choices']==a['categories'][None,:]
            for phase,first,last in [('training',0,cut),('test',cut,length),('whole',0,length)]:
                samples=block_paths(generated,first,last,settings['features']['block_width'])
                observed=block_paths(actual[None,:],first,last,settings['features']['block_width'])[0]
                probability=c['observed_probability'][:,first:last]
                jk=dict(choice_nll=jackknife_scores(probability),**path_jackknife(samples,observed));knives[subject,scope,phase]=jk
                late=max(first,last-128);late_paths=generated[:,late:last].mean(1);obs=actual[late:last].mean()
                qs=np.quantile(late_paths,[.1,.5,.9])
                scores.append(dict(tag,phase=phase,trials=last-first,original_training_trials=cut,
                    heldout_choices=scope=='EARLY' and phase=='test',choice_nll=nll(probability),
                    choice_nll_mc_se=jackknife_se(jk['choice_nll']),**distribution_scores(samples,observed),
                    energy_mc_se=jackknife_se(jk['energy']),late_accuracy=late_paths.mean(),observed_late=obs,
                    late_q10=qs[0],late_q50=qs[1],late_q90=qs[2],
                    observed_late_midrank=np.mean(late_paths<obs)+.5*np.mean(late_paths==obs),
                    target_late_present=a['present'][:,late:last].mean(),
                    target_late_contribution=a['contribution'][:,late:last].mean()))
                if phase=='whole':
                    pred=block_paths(c['correct_probability'],0,length,settings['features']['block_width']).mean(0)
                    lo,hi=np.quantile(samples,[.1,.9],axis=0)
                    for i in range(len(observed)):
                        t=min((i+1)*settings['features']['block_width'],length)
                        curves.append(dict(tag,trial=t,original_training_trials=cut,observed=observed[i],
                            autonomous=samples[:,i].mean(),lower=lo[i],upper=hi[i],conditional=pred[i]))
                        paths.extend(dict(tag,trial=t,seed=int(seed),accuracy=value) for seed,value in zip(a['seeds'],samples[:,i]))
            parameters.extend(dict(tag,parameter=k,value=v,fitted=len(case['space'][k])>1) for k,v in point['hyperparams'].items())
            budgets.extend(dict(r,scope=scope) for r in training_budget(case,bind))
            confirmed=frozen['nominee']['finalists'];seedbank=None;chosen_jk=None;confirmed_values=[]
            for candidate in confirmed:
                file=Path(candidate['score_file']);r=read(bind(file));bind(file.with_suffix('.npz'),r['predictions_sha256'])
                with np.load(file.with_suffix('.npz')) as z:
                    if seedbank is not None:np.testing.assert_array_equal(seedbank,z['seeds'])
                    seedbank=z['seeds'].copy();value=nll(z['observed_probability']);jk=jackknife_scores(z['observed_probability'])
                np.testing.assert_allclose(value,candidate['mean_nll'],atol=1e-12,rtol=0)
                if candidate['id']==point['id']:chosen_jk=jk
                confirmed_values.append((candidate,value,jk))
            if len(confirmed_values)!=4 or chosen_jk is None or set(seedbank)&set(c['seeds']):
                raise ValueError('Confirmation budget, selected candidate or independent evaluation seeds differ')
            best=min(v for _,v,_ in confirmed_values)
            np.testing.assert_allclose(point['mean_nll'],best,atol=1e-12,rtol=0)
            for candidate,value,jk in confirmed_values:
                finalists.append(dict(tag,candidate=candidate['id'],selected=candidate['id']==point['id'],
                    nll=value,delta_nll=value-best,paired_mc_se=jackknife_se(jk-chosen_jk)))
            captured=[]
            for file in sorted((own/'state_capture').glob('*.json')):
                r=read(bind(file));job=read(bind(r['original_job'],r['original_job_sha256']))
                if (r['candidate']!=point or job['compact_variant']!='TRANSFER'
                        or job['entry_control']!='PRECISION' or r['particles']!=1024):raise ValueError('Oral capture uses different parameters or mechanism')
                bind(file.with_suffix('.npz'),r['array_sha256']);alignment.validate_basis_record(file.with_suffix('.npz'),r)
                captured.append((file,dict(r,arm=scope)))
            if len(captured)!=8:raise ValueError('Incomplete original-filter oral capture')
            np.testing.assert_array_equal(sorted(r['seeds'][0] for _,r in captured),sorted(c['seeds']))
            oral_groups[subject,scope]=own,captured
    table=pd.DataFrame(scores);contrasts=[]
    for subject in (205,306):
        for role in ('conditional','autonomous'):
            for key in ('seeds','categories'):np.testing.assert_array_equal(banks[subject,'EARLY'][role][key],banks[subject,'FULL'][role][key])
        np.testing.assert_array_equal(banks[subject,'EARLY']['conditional']['choices'],banks[subject,'FULL']['conditional']['choices'])
        for phase in ('training','test','whole'):
            own=table.query('subject==@subject and phase==@phase').set_index('scope')
            contrast=dict(subject=subject,phase=phase,scope='Full minus early; full scores descriptive')
            for metric in ('choice_nll','energy','crps'):
                contrast['delta_'+metric]=own.loc['FULL',metric]-own.loc['EARLY',metric]
                contrast['delta_'+metric+'_mc_se']=jackknife_se(knives[subject,'FULL',phase][metric]-knives[subject,'EARLY',phase][metric])
            contrasts.append(contrast)
    output.mkdir(parents=True)
    for name,rows in [('scores',scores),('contrasts',contrasts),('curves',curves),('path_blocks',paths),
                      ('parameters',parameters),('compute_budgets',budgets),('finalists',finalists)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    for reading in ('midpoint','relative'):
        audit=ROOT/oral_policy['oral_audits']['development'][reading];manifest=read(bind(audit/'manifest.json'))
        if manifest['status']!='complete':raise ValueError('Oral audit incomplete')
        for name,sha in manifest['input_sha256'].items():bind(ROOT/name,sha)
        for name in ('trials.csv','component_matches.csv'):bind(audit/name,manifest['artifacts'][name])
        oral,matches=pd.read_csv(audit/'trials.csv'),pd.read_csv(audit/'component_matches.csv')
        trial_frames,summary_rows,oral_jk=[],[],{}
        for (subject,scope),(own,captured) in oral_groups.items():
            trials,rows,jk=alignment.group_scores(own,captured,oral,matches)
            trial_frames.append(trials.assign(parameter_data='early_choices' if scope=='EARLY' else 'all_choices'))
            summary_rows.extend(dict(r,parameter_data='early_choices' if scope=='EARLY' else 'all_choices') for r in rows)
            oral_jk[subject,scope]=jk
        delta=[]
        for subject in (205,306):
            for key,a in oral_jk[subject,'FULL'].items():
                b=oral_jk[subject,'EARLY'][key];np.testing.assert_array_equal(a['seeds'],b['seeds'])
                row=dict(subject=subject,window=key[0],domain=key[1],subset=key[2])
                for i,label in enumerate(('support','conflict','undetermined')):
                    row['delta_'+label]=a['point'][i]-b['point'][i]
                    row['delta_'+label+'_mc_se']=jackknife_se(a['values'][:,i]-b['values'][:,i])
                delta.append(row)
        folder=output/('oral_'+reading);folder.mkdir()
        pd.concat(trial_frames).to_csv(folder/'trials.csv',index=False)
        pd.DataFrame(summary_rows).to_csv(folder/'summary.csv',index=False)
        pd.DataFrame(delta).to_csv(folder/'contrasts.csv',index=False)
    shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    write(output/'completion.json',dict(status='complete',study=str(study.resolve()),inputs_sha256=inputs,
        scope='Two development participants; full-choice fits are descriptive, oral remains external',
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();summarize(args.study,args.config,args.output)
