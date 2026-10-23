"""Score frozen response fits with the original behavior and oral definitions.

Use --evaluation COMPLETE --config FROZEN_ORAL_CONFIG --output NEW. Include
the previously selected parameter point as an archived reference for each
participant. Joint fits also compare exponent two directly with the newly
estimated exponent-one control. Reuse score algorithms and frozen oral labels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..workflows.analysis.model_1023_response_evaluate import ROOT,read,write,digest
from ..workflows.analysis.model_1023_response_dynamics import verify
from ..workflows.analysis import model_0923_decision_basis_alignment as alignment
from .model_1023_frozen_evaluate import score_arrays
from .model_1023_state_sensitivity import read_point,compare_states
from .model_0923_h_validation import checked_arrays,jackknife_se


def summarize(evaluation: Path,config: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new external response summary')
    done,protocol=read(evaluation/'completion.json'),read(evaluation/'protocol.json')
    if done['status']!='complete' or done['smoke'] or digest(evaluation/'cases.json')!=done['cases_sha256']:
        raise ValueError('Require complete scientific frozen evaluations')
    inputs={}

    def bind(file,expected=None):
        p=Path(file).resolve();sha=digest(p)
        if expected is not None and expected!=sha:raise ValueError('Frozen evaluation input changed: '+str(p))
        inputs[str(p)]=sha;return p

    for file in (evaluation/'completion.json',evaluation/'protocol.json',evaluation/'cases.json',config,Path(__file__),
                 Path(alignment.__file__),Path(inspect.getfile(score_arrays)),Path(inspect.getfile(read_point)),
                 Path(inspect.getfile(checked_arrays)),Path(inspect.getfile(jackknife_se))):
        bind(file)
    plan=read(evaluation/'cases.json');verify(plan['inputs_sha256']);inputs.update(plan['inputs_sha256'])
    cases=deepcopy(plan['cases'])
    for c in cases:c['arm']='Q'+str(int(c['response_exponent']))
    for sid in sorted({c['subject'] for c in cases}):
        c=next(c for c in cases if c['subject']==sid)
        own=Path(c['reference_evaluation']);original=read(bind(own/'protocol.json'))
        frozen=read(bind(Path(original['fit'])/'nominees.json',original['nomination_sha256']))
        point=frozen['nominee']['candidate']
        cases.append(dict(case_id=f'S{sid}_PREVIOUS',subject=sid,condition=c['condition'],cohort=c['cohort'],
            response_exponent=1.,arm='previous',candidate=point['id'],candidate_hyperparams=point['hyperparams'],
            evaluation=str(own),reference_seeds=c['reference_seeds']))
    policy=yaml.safe_load(config.read_text())
    scores,curves,paths,states,parameters,state_trials=[],[],[],[],[],[]
    knives,oral_groups,identities,conditional_states={},{},{},{}
    for case in cases:
        own=Path(case['evaluation']);frozen_protocol=read(bind(own/'protocol.json'))
        complete=read(bind(own/'completion.json'))
        fit=Path(frozen_protocol['fit']);frozen=read(bind(fit/'nominees.json',frozen_protocol['nomination_sha256']))
        fit_done=read(bind(fit/'completion.json'))
        if complete['status']!='complete' or frozen['smoke'] or fit_done['test_choices_used']:
            raise ValueError('Evaluation requires frozen prefix-only estimates')
        point=frozen['nominee']['candidate']
        if point['hyperparams']!=case['candidate_hyperparams']:raise ValueError('Frozen parameters changed')
        banks,records={},{}
        for role in ('conditional','autonomous'):
            record,=read(bind(own/role/'evaluation.json'))['rows']
            bind(own/role/'evaluate'/f"subject_{case['subject']}"/'COMPACT'/(case['candidate']+'.npz'),record['predictions_sha256'])
            if record['candidate']!=point:raise ValueError('Numerical prediction candidate differs')
            banks[role]=checked_arrays(own/role,record);records[role]=record
            np.testing.assert_array_equal(banks[role]['seeds'],case['reference_seeds'][role])
        c,a=banks['conditional'],banks['autonomous'];cut=records['conditional']['training_trials']
        if c['probabilities'].shape[0]!=8 or len(a['choices'])!=128 or records['conditional']['particles']!=1024:
            raise ValueError('Wrong frozen evaluation budget')
        tag={k:case[k] for k in ('case_id','subject','condition','cohort','candidate','response_exponent','arm')}
        values=score_arrays(c,a,cut,32,target_rule=records['autonomous']['target_rule'])
        for dest,rows in zip((scores,curves,paths,states),values[:4]):dest.extend(dict(tag,**r) for r in rows)
        knives[case['case_id']]=values[4]
        identities[case['case_id']]={role:{k:bank[k] for k in ('seeds','categories')} for role,bank in banks.items()}
        identities[case['case_id']]['choices']=c['choices']
        parameters.extend(dict(tag,parameter=k,value=v) for k,v in point['hyperparams'].items())
        captured=[]
        for file in sorted((own/'state_capture').glob('*.json')):
            record=read(bind(file));job=read(bind(record['original_job'],record['original_job_sha256']))
            bind(file.with_suffix('.npz'),record['array_sha256'])
            if record['candidate']!=point or record['particles']!=1024:raise ValueError('Oral capture identity changed')
            if case['arm']!='previous':
                bridge=read(bind(Path(job['target']).with_suffix('.response_evaluation.json')))
                if bridge['response_exponent']!=case['response_exponent'] or bridge['additional_filters_for_capture']!=0:
                    raise ValueError('Response observation bridge differs from fitted arm')
            captured.append((file,dict(record,arm=case['case_id'])))
        if len(captured)!=8:raise ValueError('Require all eight original-filter oral captures')
        np.testing.assert_array_equal(sorted(r['seeds'][0] for _,r in captured),sorted(c['seeds']))
        oral_groups[case['case_id']]=(own,captured,tag)
        _,state,state_meta=read_point(case,bind)
        conditional_states[case['case_id']]=state
        state_trials.extend(dict(tag,trial=t+1,training_trials=cut,target_rule=state_meta['target_rule'],
            **{k:v[t] for k,v in state['trials'].items()}) for t in range(len(c['choices'])))
    pairs=[]
    for case in cases:
        if case['arm']=='previous':continue
        previous=next(c for c in cases if c['subject']==case['subject'] and c['arm']=='previous')
        pairs.append((case,previous))
        if case['arm']=='Q2':
            baseline=next(c for c in cases if c['subject']==case['subject'] and c['arm']=='Q1')
            pairs.append((case,baseline))
    table=pd.DataFrame(scores);contrasts=[];state_changes=[]
    for a,b in pairs:
        own,base=identities[a['case_id']],identities[b['case_id']]
        np.testing.assert_array_equal(own['choices'],base['choices'])
        for role in ('conditional','autonomous'):
            for k in ('seeds','categories'):np.testing.assert_array_equal(own[role][k],base[role][k])
        difference=compare_states(conditional_states[a['case_id']],conditional_states[b['case_id']])
        state_changes.extend(dict(subject=a['subject'],case_id=a['case_id'],reference=b['case_id'],
            trial=t+1,**{k:v[t] for k,v in difference.items()}) for t in range(len(own['choices'])))
        for phase in ('training','test','whole'):
            r=table[(table.case_id==a['case_id'])&(table.phase==phase)].iloc[0]
            s=table[(table.case_id==b['case_id'])&(table.phase==phase)].iloc[0]
            row=dict(subject=a['subject'],cohort=a['cohort'],case_id=a['case_id'],reference=b['case_id'],
                arm=a['arm'],reference_arm=b['arm'],phase=phase)
            for metric in ('choice_nll','energy','crps'):
                row['delta_'+metric]=r[metric]-s[metric]
                row['delta_'+metric+'_mc_se']=jackknife_se(knives[a['case_id']][phase][metric]-knives[b['case_id']][phase][metric])
            contrasts.append(row)
    output.mkdir(parents=True)
    for name,data in [('scores',scores),('curves',curves),('path_blocks',paths),('autonomous_states',states),
                      ('parameters',parameters),('contrasts',contrasts),('conditional_state_trials',state_trials),
                      ('conditional_state_changes',state_changes)]:pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    for reading in ('midpoint','relative'):
        trial_frames,summary_rows,jks=[],[],{}
        for cohort in sorted({c['cohort'] for c in cases}):
            audit=ROOT/policy['oral_audits'][cohort][reading];manifest=read(bind(audit/'manifest.json'))
            if manifest['status']!='complete':raise ValueError('Incomplete frozen oral labels')
            for name,sha in manifest['input_sha256'].items():bind(ROOT/name,sha)
            for name in ('trials.csv','component_matches.csv'):bind(audit/name,manifest['artifacts'][name])
            oral,matches=pd.read_csv(audit/'trials.csv'),pd.read_csv(audit/'component_matches.csv')
            for case in [c for c in cases if c['cohort']==cohort]:
                own,captured,tag=oral_groups[case['case_id']]
                trials,rows,jack=alignment.group_scores(own,captured,oral,matches)
                trial_frames.append(trials.assign(**tag));summary_rows.extend(dict(r,**tag) for r in rows)
                jks[case['case_id']]=jack
        differences=[]
        for a,b in pairs:
            for key,jack in jks[a['case_id']].items():
                baseline=jks[b['case_id']][key];np.testing.assert_array_equal(jack['seeds'],baseline['seeds'])
                row=dict(subject=a['subject'],case_id=a['case_id'],reference=b['case_id'],arm=a['arm'],reference_arm=b['arm'],
                    window=key[0],domain=key[1],subset=key[2])
                for i,label in enumerate(('support','conflict','undetermined')):
                    row['delta_'+label]=jack['point'][i]-baseline['point'][i]
                    row['delta_'+label+'_mc_se']=jackknife_se(jack['values'][:,i]-baseline['values'][:,i])
                differences.append(row)
        folder=output/('oral_'+reading);folder.mkdir()
        pd.concat(trial_frames).to_csv(folder/'trials.csv',index=False)
        pd.DataFrame(summary_rows).to_csv(folder/'summary.csv',index=False)
        pd.DataFrame(differences).to_csv(folder/'contrasts.csv',index=False)
    verify(inputs)
    write(output/'completion.json',dict(status='complete',evaluation=str(evaluation.resolve()),
        fitting_phase=protocol['fitting_phase'],cases=len(cases),participants=len({c['subject'] for c in cases}),
        inputs_sha256=inputs,oral_use='Frozen noisy external compatibility; explicit category-power observation bridge',
        data_scope='Development participants; original train/evaluation boundaries retained',
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('evaluation','config','output'):parser.add_argument('--'+k,type=Path,required=True)
    args=parser.parse_args();summarize(args.evaluation,args.config,args.output)
