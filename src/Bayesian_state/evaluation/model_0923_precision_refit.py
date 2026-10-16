"""Training selection, independent choice checks, and autonomous distributions.

Every configured subject/family and all saved filter seeds and paths enter the
tables. Errors delete complete seeds/paths; they describe Monte Carlo variation
at fixed fitted parameters. Scores are development diagnostics, not external
validation or population significance tests. Output directories must be new.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .model_0923_h_validation import (load, digest, checked_arrays, nll,
    jackknife_scores, jackknife_se, block_paths, distribution_scores, inactive_parameter_channels)
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_h_validation import write, REPO
from ..workflows.analysis.model_0923_precision_refit import RATE, INIT


def summarize(fit: Path, evaluation: Path, output: Path, rate_profile: Path | None = None) -> None:
    if output.exists(): raise FileExistsError('Use a new precision summary')
    frozen = load(fit/'nominees.json'); settings = frozen['settings']
    width=settings.get('features',{}).get('block_width',32)
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need frozen scientific fit')
    protocol = load(evaluation/'protocol.json')
    if protocol['nomination_sha256'] != digest(fit/'nominees.json'): raise ValueError('Evaluation used different fits')
    if load(evaluation/'completion.json')['status'] != 'complete': raise ValueError('Incomplete evaluation')
    expected = {(r['subject'], r['model']): r for r in frozen['nominees']}
    assert set(expected) == {(s,m) for s in settings['subjects'] for m in settings['arms']}
    inputs = {str(fit/'nominees.json'): digest(fit/'nominees.json'),
              str(evaluation/'protocol.json'): digest(evaluation/'protocol.json')}
    for name in ('protocol.json','parameter_spaces.json','parameter_support.yaml','screen_reuse.json'):
        if (fit/name).exists():inputs[str(fit/name)]=digest(fit/name)
    records = {}
    for role in ('conditional','autonomous'):
        file = evaluation/role/'evaluation.json'; inputs[str(file)] = digest(file)
        rows = load(file)['rows']; records[role] = {(r['subject'],r['model']):r for r in rows}
        assert len(rows)==len(expected) and set(records[role])==set(expected)
    scores, curves, paths, parameters, training, filters, flags = [], [], [], [], [], [], []
    jackknives, seed_banks, training_bank, finalist_bank = {}, {}, {}, {}
    for key, nominee in expected.items():
        s,m = key; tag = dict(subject=s,model=m)
        cr, ar = [records[role][key] for role in ('conditional','autonomous')]
        assert cr['candidate']['hyperparams']==ar['candidate']['hyperparams']==nominee['candidate']['hyperparams']
        c,a = [checked_arrays(evaluation/role, row) for role,row in [('conditional',cr),('autonomous',ar)]]
        for role,row in [('conditional',cr),('autonomous',ar)]:
            file = evaluation/role/'evaluate'/f'subject_{s}'/m/(row['candidate']['id']+'.npz')
            inputs[str(file)] = row['predictions_sha256']
        np.testing.assert_array_equal(c['categories'],a['categories'])
        n,cut = len(c['choices']),cr['training_trials']; human = c['choices']==c['categories']
        generated = a['choices']==a['categories'][None,:]
        assert generated.shape==(settings['autonomous_paths'],n)
        assert c['observed_probability'].shape==(settings['budgets']['evaluate']['seeds'],n)
        point = nominee['candidate']['hyperparams']; scale = settings['arms'][m]['rate_scale']
        parameters.extend(dict(tag,parameter=k,value=value) for k,value in point.items())
        flags.append(dict(tag,beta_init=point[INIT],configured_rate=point[RATE],effective_rate=point[RATE]*scale,
            learning_closed=point[RATE]==0.,rate_at_upper=point[RATE]==max(settings['arms'][m]['rates']),
            init_at_lower=point[INIT]==.1,init_at_upper=point[INIT]==25.))
        for phase,first,last in [('training',0,cut),('test',cut,n),('whole',0,n)]:
            p = c['observed_probability'][:,first:last]
            samples = block_paths(generated,first,last,width); observed = block_paths(human[None,:],first,last,width)[0]
            metrics = distribution_scores(samples,observed)
            jk = dict(choice_nll=jackknife_scores(p),**path_jackknife(samples,observed))
            jackknives[s,m,phase]=jk; seed_banks[s,m,phase]=(c['seeds'],a['seeds'])
            late = min(128,last-first); early=min(128,last-first)
            late_values = generated[:,last-late:last].mean(axis=1)
            row = dict(tag,phase=phase,condition=cr['condition'],trials=last-first,training_trials=cut,
                particles=cr['particles'],filter_seeds=len(c['seeds']),paths=len(a['seeds']),
                choice_nll=nll(p),choice_nll_mc_se=jackknife_se(jk['choice_nll']),
                energy_mc_se=jackknife_se(jk['energy']),crps_mc_se=jackknife_se(jk['crps']),
                late_accuracy=float(late_values.mean()),observed_late=float(human[last-late:last].mean()),
                early_accuracy=float(generated[:,first:first+early].mean()),observed_early=float(human[first:first+early].mean()),
                beta_init=point[INIT],configured_rate=point[RATE],effective_rate=point[RATE]*scale,
                beta_upper=settings['arms'][m]['upper'],**metrics)
            scores.append(row)
            filters.extend(dict(tag,phase=phase,seed=int(seed),nll=nll(p[i:i+1])) for i,seed in enumerate(c['seeds']))
            if phase=='whole':
                lower,upper=np.quantile(samples,[.1,.9],axis=0)
                conditional=block_paths(c['correct_probability'],0,n,width).mean(axis=0)
                for b in range(samples.shape[1]):
                    curves.append(dict(tag,trial=min((b+1)*width,n),training_trials=cut,observed=observed[b],
                        autonomous=float(samples[:,b].mean()),lower=lower[b],upper=upper[b],conditional=conditional[b]))
                for i,seed in enumerate(a['seeds']):
                    paths.append(dict(tag,seed=int(seed),late_accuracy=late_values[i],observed_late=row['observed_late']))
        for finalist in nominee['finalists']:
            file=Path(finalist['score_file']); record=load(file)
            if digest(file.with_suffix('.npz'))!=record['predictions_sha256']: raise ValueError('Training array changed')
            inputs[str(file.with_suffix('.npz'))]=record['predictions_sha256']
            with np.load(file.with_suffix('.npz')) as z: p=z['observed_probability']; seeds=z['seeds']
            finalist_bank[s,m,finalist['id']]=(p,seeds)
            selected=finalist['id']==nominee['candidate']['id']
            low_file=fit/'selection'/f'subject_{s}'/m/(finalist['id']+'.json')
            low=load(low_file)
            assert not set(seeds)&set(low['seeds']) and not set(seeds)&set(c['seeds'])
            with np.load(low_file.with_suffix('.npz')) as z: low_p=z['observed_probability']
            assert digest(low_file.with_suffix('.npz'))==low['predictions_sha256']
            training.append(dict(tag,candidate=finalist['id'],selected=selected,
                reference=finalist['id']==nominee['reference_id'],nll=nll(p),mc_se=jackknife_se(jackknife_scores(p)),
                selection_nll=nll(low_p),selection_mc_se=jackknife_se(jackknife_scores(low_p)),
                particles=record['particles'],seeds=len(seeds),
                beta_init=finalist['hyperparams'][INIT],effective_rate=finalist['hyperparams'][RATE]*scale))
            if selected: training_bank[key]=(p,seeds)
    frame=pd.DataFrame(scores); contrasts=[]; training_contrasts=[]; training_repeatability=[]
    indexed=frame.set_index(['subject','model','phase'])
    for row in frame.itertuples():
        key=row.subject,row.model,row.phase; base_key=row.subject,'ADD1',row.phase
        base=indexed.loc[base_key]; result=dict(subject=row.subject,model=row.model,phase=row.phase)
        for seeds,base_seeds in zip(seed_banks[key],seed_banks[base_key]): np.testing.assert_array_equal(seeds,base_seeds)
        for metric in ('choice_nll','energy','crps'):
            result['delta_'+metric]=getattr(row,metric)-base[metric]
            result['delta_'+metric+'_mc_se']=jackknife_se(jackknives[key][metric]-jackknives[base_key][metric])
        contrasts.append(result)
    for (s,m),(p,seeds) in training_bank.items():
        base,base_seeds=training_bank[s,'ADD1'];np.testing.assert_array_equal(seeds,base_seeds)
        training_contrasts.append(dict(subject=s,model=m,delta_nll=nll(p)-nll(base),
            delta_nll_mc_se=jackknife_se(jackknife_scores(p)-jackknife_scores(base))))
        repeated=indexed.loc[s,m,'training']
        confirmation_se=jackknife_se(jackknife_scores(p))
        training_repeatability.append(dict(subject=s,model=m,confirmation_nll=nll(p),
            confirmation_mc_se=confirmation_se,repeated_nll=repeated.choice_nll,
            repeated_mc_se=repeated.choice_nll_mc_se,delta_nll=repeated.choice_nll-nll(p),
            independent_mc_se=float(np.hypot(confirmation_se,repeated.choice_nll_mc_se))))
    finalist_differences=[]
    for (s,m,candidate),(p,seeds) in finalist_bank.items():
        best,best_seeds=training_bank[s,m];np.testing.assert_array_equal(seeds,best_seeds)
        finalist_differences.append(dict(subject=s,model=m,candidate=candidate,
            selected=candidate==expected[s,m]['candidate']['id'],delta_nll=nll(p)-nll(best),
            delta_nll_mc_se=jackknife_se(jackknife_scores(p)-jackknife_scores(best))))
    additive_search=[]
    for s in settings['subjects']:
        original=expected[s,'ADD1']['candidate'];expanded=expected[s,'ADD10']['candidate']
        original_p,original_seeds=training_bank[s,'ADD1'];expanded_p,expanded_seeds=training_bank[s,'ADD10']
        np.testing.assert_array_equal(original_seeds,expanded_seeds)
        mapped=original['hyperparams'][RATE]/settings['arms']['ADD10']['rate_scale']
        feasible=any(np.isclose(mapped,r,rtol=0,atol=1e-14) for r in settings['arms']['ADD10']['rates'])
        expanded_rate=expanded['hyperparams'][RATE]*settings['arms']['ADD10']['rate_scale']
        original_support=any(np.isclose(expanded_rate,r,rtol=0,atol=1e-14) for r in settings['arms']['ADD1']['rates'])
        additive_search.append(dict(subject=s,add1_candidate=original['id'],add10_candidate=expanded['id'],
            add1_rate=original['hyperparams'][RATE],add10_effective_rate=expanded_rate,
            add1_point_feasible_in_add10=feasible,add10_point_feasible_in_add1=original_support,
            delta_training_nll=nll(expanded_p)-nll(original_p),
            paired_mc_se=jackknife_se(jackknife_scores(expanded_p)-jackknife_scores(original_p))))
    baseline_budget=[]
    if 'warm_start_fit' in settings:
        old_file=REPO/settings['warm_start_fit']/'nominees.json';old=load(old_file)
        inputs[str(old_file)]=digest(old_file)
        for s in settings['subjects']:
            original=next(r for r in old['nominees'] if r['subject']==s and r['model']=='FREE')['candidate']
            current=expected[s,'ADD1'];reference=next(r for r in current['finalists'] if r['id']==current['reference_id'])
            assert original['hyperparams']==reference['hyperparams']
            file=Path(original['score_file']);record=load(file)
            if digest(file.with_suffix('.npz'))!=record['predictions_sha256']:raise ValueError('Prior reference changed')
            inputs[str(file.with_suffix('.npz'))]=record['predictions_sha256']
            with np.load(file.with_suffix('.npz')) as z:old_p=z['observed_probability'];old_seeds=z['seeds']
            new_p,new_seeds=finalist_bank[s,'ADD1',reference['id']]
            assert not set(old_seeds)&set(new_seeds)
            baseline_budget.append(dict(subject=s,old_nll=nll(old_p),new_nll=nll(new_p),
                delta_nll=nll(new_p)-nll(old_p),old_particles=record['particles'],
                new_particles=load(Path(reference['score_file']))['particles'],
                independent_mc_se=float(np.hypot(jackknife_se(jackknife_scores(old_p)),jackknife_se(jackknife_scores(new_p))))))
    candidates=[]; budgets=[]
    for stage in ('screen','refine','selection','confirmation'):
        completed=load(fit/(stage+'_completion.json')); assert completed['status']=='complete'
        inputs[str(fit/(stage+'_completion.json'))]=digest(fit/(stage+'_completion.json'))
        for group in completed['groups']:
            s,m=group['subject'],group['model']
            for row in group['rows']:
                file=Path(row['score_file']); r=load(file)
                assert digest(file.with_suffix('.npz'))==r['predictions_sha256']
                inputs[str(file.with_suffix('.npz'))]=r['predictions_sha256']
                candidates.append(dict(subject=s,model=m,stage=stage,candidate=row['id'],nll=row['mean_nll'],
                    beta_init=row['hyperparams'][INIT],effective_rate=row['hyperparams'][RATE]*settings['arms'][m]['rate_scale'],
                    source=';'.join(row['sources']),**row['hyperparams']))
        for file in (fit/stage).glob('subject_*/*/chunks/*.json'):
            if file.name.endswith(('.experiment.json','.snapshot.json')): continue
            r=load(file);assert all(pool['num_threads']==1 for pool in r['numeric_pools'])
            budgets.append(dict(subject=r['subject'],model=r['model'],stage=stage,
                state_updates=r['particles']*len(r['seeds'])*r['trials'],filter_runs=len(r['seeds']),cpu_seconds=r['cpu_seconds']))
    budget=pd.DataFrame(budgets).groupby(['subject','model','stage'],as_index=False).sum()
    assert (budget.groupby(['subject','stage']).state_updates.nunique()==1).all()
    contrast_frame=pd.DataFrame(contrasts);cohort=[]
    for (phase,model),group in frame.groupby(['phase','model']):
        group=group.sort_values('subject')
        differences=contrast_frame[(contrast_frame.phase==phase)&(contrast_frame.model==model)].sort_values('subject')
        np.testing.assert_array_equal(group.subject,differences.subject)
        weights=group.trials.to_numpy();count=len(group)
        cohort.append(dict(phase=phase,model=model,participants=count,trials=int(weights.sum()),
            participant_mean_choice_nll=float(group.choice_nll.mean()),
            trial_weighted_choice_nll=float(np.average(group.choice_nll,weights=weights)),
            participant_mean_energy=float(group.energy.mean()),
            delta_participant_choice_nll=float(differences.delta_choice_nll.mean()),
            delta_trial_weighted_choice_nll=float(np.average(differences.delta_choice_nll,weights=weights)),
            delta_participant_energy=float(differences.delta_energy.mean()),
            delta_participant_choice_nll_mc_se=float(np.linalg.norm(differences.delta_choice_nll_mc_se)/count),
            delta_trial_weighted_choice_nll_mc_se=float(np.linalg.norm(weights*differences.delta_choice_nll_mc_se)/weights.sum()),
            delta_participant_energy_mc_se=float(np.linalg.norm(differences.delta_energy_mc_se)/count)))
    profile_table=None
    if rate_profile is not None:
        record=load(rate_profile/'completion.json');profile_protocol=load(rate_profile/'protocol.json')
        if record['smoke'] or record['status']!='complete':raise ValueError('Need complete scientific rate profile')
        if record['nomination_sha256']!=digest(fit/'nominees.json'):raise ValueError('Rate profile used different fits')
        assert digest(rate_profile/'protocol.json')==record['protocol_sha256']
        assert digest(rate_profile/'rate_profile.csv')==record['summary_sha256']
        reference=expected[profile_protocol['audit']['subject'],profile_protocol['audit']['model']]['candidate']
        assert profile_protocol['reference_parameters']==reference['hyperparams']
        assert profile_protocol['reference_id']==reference['id']
        for name,sha in record['inputs_sha256'].items():
            if digest(Path(name))!=sha:raise ValueError('Rate-profile predictions changed')
            inputs[name]=sha
        for name in ('completion.json','protocol.json','rate_profile.csv'):
            inputs[str(rate_profile/name)]=digest(rate_profile/name)
        profile_table=pd.read_csv(rate_profile/'rate_profile.csv')
        assert len(profile_table)==record['points'] and profile_table.reference.sum()==1
    output.mkdir(parents=True)
    channel_input=dict(frozen,settings=dict(settings,models={m:settings['mechanism'] for m in settings['arms']}))
    tables=dict(scores=frame,contrasts=contrast_frame,cohort_scores=pd.DataFrame(cohort),curves=pd.DataFrame(curves),paths=pd.DataFrame(paths),
        parameters=pd.DataFrame(parameters),training_finalists=pd.DataFrame(training),training_contrasts=pd.DataFrame(training_contrasts),
        training_finalist_differences=pd.DataFrame(finalist_differences),
        training_repeatability=pd.DataFrame(training_repeatability),
        additive_search_comparison=pd.DataFrame(additive_search),
        candidates=pd.DataFrame(candidates),budgets=budget,filters=pd.DataFrame(filters),
        precision_flags=pd.DataFrame(flags),inactive_parameters=inactive_parameter_channels(channel_input))
    if baseline_budget:tables['baseline_budget_change']=pd.DataFrame(baseline_budget)
    if profile_table is not None:tables['rate_profile']=profile_table
    for name,table in tables.items():table.to_csv(output/(name+'.csv'),index=False)
    shutil.copy2(__file__,output/'analysis_source.py')
    write(output/'manifest.json',dict(status='complete',fit=str(fit.resolve()),evaluation=str(evaluation.resolve()),
        source_sha256=digest(Path(__file__)),inputs_sha256=inputs,
        summary_sha256={name+'.csv':digest(output/(name+'.csv')) for name in tables},
        participants=len(settings['subjects']),families=list(settings['arms']),block_width=width,
        rate_profile=str(rate_profile.resolve()) if rate_profile is not None else None,
        data_use='all participants development; training-only estimation; chronological later-trial evaluation',
        uncertainty='whole-filter or whole-path jackknife Monte Carlo standard error at fixed fitted parameters'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','evaluation','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--rate-profile',type=Path)
    args=parser.parse_args();summarize(args.fit,args.evaluation,args.output,args.rate_profile)
