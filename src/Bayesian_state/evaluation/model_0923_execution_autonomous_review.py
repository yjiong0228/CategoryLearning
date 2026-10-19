"""Review completed autonomous banks while a cohort choice evaluation continues.

Use --original MIX_EVALUATION --revised SELECT_EVALUATION --subject ID --output
NEW_DIRECTORY. Every declared autonomous chunk must be complete and hash-bound;
partial path samples are rejected. Compare confirmation likelihood and complete
autonomous distributions using the existing scientific definitions. Conditional
later-choice scores and oral alignment remain separate downstream results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_0923_h_validation as metrics
from . import model_0923_h_generation_uncertainty as uncertainty
from . import model_0923_estimation_scope as validation
from .model_0923_cohort_execution_comparison import check_settings, EXECUTION_ADDITIONS
from .model_0923_dominant_execution_refit import check_selection
from .model_0923_episodic_refit import active_precision_mean
from ..workflows.analysis.model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[3]


def combine_paths(chunks: list[dict], expected_seeds: list[int]) -> dict:
    """Require the full declared seed bank and preserve chronological arrays."""
    if not chunks or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError('Need complete unique declared trajectory seeds')
    names = set(chunks[0])
    if any(set(c) != names for c in chunks): raise ValueError('Autonomous schemas differ')
    for chunk in chunks:
        np.testing.assert_array_equal(chunk['categories'], chunks[0]['categories'])
        for name, values in chunk.items():
            if name != 'categories' and values.shape[0] != len(chunk['seeds']):
                raise ValueError('Path array does not align with trajectory seeds')
    combined = {name:(chunks[0][name] if name=='categories' else np.concatenate([c[name] for c in chunks]))
                for name in names}
    seeds = combined['seeds'].tolist()
    if len(seeds) != len(set(seeds)) or set(seeds) != set(expected_seeds):
        raise ValueError('Incomplete or repeated autonomous trajectory bank')
    order = np.argsort(combined['seeds'])
    return {name:(values if name=='categories' else values[order]) for name,values in combined.items()}


def run(original: Path, revised: Path, subject: int, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new autonomous review directory')
    inputs = {}

    def bind(path):
        path=Path(path).resolve(); sha=metrics.digest(path)
        if str(path) in inputs and inputs[str(path)] != sha: raise ValueError('Input changed')
        inputs[str(path)]=sha; return path

    def read(path): return metrics.load(bind(path))

    import inspect
    for file in (__file__,metrics.__file__,uncertainty.__file__,validation.__file__,
                 inspect.getfile(check_settings),inspect.getfile(check_selection),inspect.getfile(active_precision_mean)):
        bind(file)
    evals=[read(p/'protocol.json') for p in (original,revised)]
    fits=[Path(p['fit']) for p in evals]
    frozen=[read(p/'nominees.json') for p in fits]
    settings=[f['settings'] for f in frozen]
    check_settings(*settings,subject,fits[0])
    protocols=[read(p/'protocol.json') for p in fits]
    spaces=[read(p/'parameter_spaces.json') for p in fits]
    if spaces[0] != spaces[1]:
        raise ValueError('Parameter supports differ')
    for protocol,fr in zip(protocols,frozen):
        if (protocol['smoke'] or fr['smoke'] or protocol['settings'] != fr['settings']
                or protocol['source_sha256'] != fr['source_sha256']):
            raise ValueError('Fitting protocol and frozen nominees differ')
    if protocols[0]['initial_banks'] != protocols[1]['initial_banks']:
        raise ValueError('Actual initial parameter banks differ')
    old,new=[f['source_sha256'] for f in frozen]
    if any(new.get(k)!=v for k,v in old.items()) or set(new)-set(old)!=EXECUTION_ADDITIONS:
        raise ValueError('Undeclared changes between fitting implementations')
    banks,confirmed,parameters,closed,chunk_rows={}, {}, [], [], []
    for policy,source,fit,ev,fr,st in zip(('MIX','SELECT'),(original,revised),fits,evals,frozen,settings):
        fit_done=read(fit/'completion.json')
        if (fit_done['status']!='complete' or fit_done['smoke'] or fit_done['test_choices_used']
                or fr['status']!='frozen_before_test' or ev['nomination_sha256']!=metrics.digest(fit/'nominees.json')):
            raise ValueError('Need completed training and matching frozen evaluation nominees')
        for file,sha in fr['source_sha256'].items():
            if metrics.digest(bind(ROOT/file))!=sha: raise ValueError('Executed source changed')
        nominees={r['model']:r['candidate'] for r in fr['nominees'] if r['subject']==subject}
        if set(nominees)!=set(st['arms']): raise ValueError('Nominee proposal coverage differs')
        adapted=dict(fr,settings=dict(st,models={arm:st['mechanism'] for arm in st['arms']}))
        closed.extend(dict(row,policy=policy) for row in metrics.inactive_parameter_channels(adapted).to_dict('records'))
        confirmation_budget=st['budgets']['confirmation']
        particles={int(k):v for k,v in st['particle_overrides'].items()}.get(subject,confirmation_budget['particles'])
        for arm,point in nominees.items():
            confirmation=Path(point['score_file']); record=read(confirmation)
            file=bind(confirmation.with_suffix('.npz'))
            if metrics.digest(file)!=record['predictions_sha256']: raise ValueError('Confirmation arrays changed')
            with np.load(file) as z: c={k:z[k] for k in z.files}
            if (record['subject']!=subject or record['model']!=arm or record['candidate']['id']!=point['id']
                    or record['candidate']['hyperparams']!=point['hyperparams']
                    or record['particles']!=particles or len(record['seeds'])!=confirmation_budget['seeds']
                    or record['role']!='training_only_high_particle_audit'):
                raise ValueError('Training confirmation identity differs')
            np.testing.assert_array_equal(c['seeds'],record['seeds'])
            p=c['observed_probability']
            if (p.shape!=(len(record['seeds']),record['trials']) or not np.isfinite(p).all()
                    or np.any((p<0)|(p>1))):
                raise ValueError('Invalid confirmation probabilities')
            np.testing.assert_allclose(metrics.nll(c['observed_probability']),point['mean_nll'],atol=1e-12,rtol=0)
            confirmed[policy,arm]=c
            parameters.extend(dict(policy=policy,model=arm,parameter=k,value=v,candidate=point['id'])
                              for k,v in point['hyperparams'].items())
            declared=[j for j in ev['jobs'] if j['phase']=='autonomous' and j['model']==arm]
            expected=[seed for j in declared for seed in j['seeds']]
            if len(expected)!=st['autonomous_paths']: raise ValueError('Autonomous path budget differs')
            chunks=[]
            for template in declared:
                job=read(source/'jobs'/(template['name']+'.json')); target=Path(job['target'])
                record=read(target.with_suffix('.json')); file=bind(target.with_suffix('.npz'))
                if (job['target']!=template['target'] or job['seeds']!=template['seeds'] or job['smoke']
                        or job['cohort_belief_mechanism']!='SEPARATE' or not job['retain_precision']
                        or job.get('execution_policy')!=(None if policy=='MIX' else 'SELECT')
                        or record['subject']!=subject or record['model']!=arm
                        or record['candidate']['id']!=point['id'] or job['candidate']!=template['candidate']
                        or record['candidate']['hyperparams']!=point['hyperparams']
                        or record['role']!='autonomous_from_start' or metrics.digest(file)!=record['predictions_sha256']):
                    raise ValueError('Incomplete, changed or incorrectly executed autonomous chunk')
                if any(pool['num_threads']!=1 for pool in record['numeric_pools']): raise ValueError('Nested numerical threads')
                precision=read(target.with_suffix('.precision.json')); belief=read(target.with_suffix('.belief.json'))
                copies=len(job['seeds']); trials=record['trials']
                if (not precision['retained_precision'] or precision['transitions']!=copies*trials
                        or not belief['enabled'] or belief['updates']!=copies*(trials-1)
                        or not belief['completed_feedback_timing_exact'] or not belief['belief_matches_saved_state']):
                    raise ValueError('Belief or retained precision execution differs')
                if policy=='SELECT': check_selection(job,record,read(target.with_suffix('.execution.json')))
                with np.load(file) as z: arrays={k:z[k] for k in z.files}
                np.testing.assert_array_equal(arrays['seeds'],job['seeds']);validation.validate_generation(arrays,record)
                chunks.append(arrays)
                chunk_rows.append(dict(policy=policy,model=arm,name=job['name'],paths=copies,trials=trials,
                                       target_rule=record['target_rule'],exact_feedback=True))
            banks[policy,arm]=combine_paths(chunks,expected)
    folder=Path(frozen[0]['data'])/f'subject_{subject}'; meta=read(folder/'prepared.json')
    if (Path(frozen[0]['data']).resolve()!=Path(frozen[1]['data']).resolve()
            or meta['subject']!=subject or len({row['target_rule'] for row in chunk_rows})!=1):
        raise ValueError('Data identity or target rule differs')
    observations=[]
    for name in ('training.npz','sealed_test.npz'):
        file=bind(folder/name)
        if metrics.digest(file)!=meta['array_sha256'][name]: raise ValueError('Original observations changed')
        with np.load(file) as z: observations.append({k:z[k] for k in ('choices','categories')})
    categories=np.concatenate([a['categories'] for a in observations]); choices=np.concatenate([a['choices'] for a in observations])
    observed=(choices==categories).astype(float); cut=meta['training_trials']; length=len(observed)
    windows=[('training',0,cut),('later',cut,length),('whole',0,length),('last128',length-128,length)]
    scores,path_rows,contrasts,training_rows=[],[],[],[]; estimates={}
    for (policy,arm),data in banks.items():
        np.testing.assert_array_equal(data['categories'],categories)
        correct=(data['choices']==categories[None,:]).astype(float)
        for window,first,last in windows:
            samples=metrics.block_paths(correct,first,last); target=metrics.block_paths(observed[None,:],first,last)[0]
            score=metrics.distribution_scores(samples,target); deletions=uncertainty.path_jackknife(samples,target)
            estimates[policy,arm,window]=(score,deletions)
            scores.append(dict(policy=policy,model=arm,window=window,paths=len(data['seeds']),**score,
                realized_correct=float(correct[:,first:last].mean()),human_correct=float(observed[first:last].mean()),
                target_present=float(data['present'][:,first:last].mean()),
                target_contribution=float(data['contribution'][:,first:last].mean()),
                target_beta_when_present=active_precision_mean(data['beta'][:,first:last],data['present'][:,first:last])))
        for i,seed in enumerate(data['seeds']):
            found=np.flatnonzero(data['present'][i])
            path_rows.append(dict(policy=policy,model=arm,seed=int(seed),ever_present=bool(len(found)),
                first_present_trial=int(found[0]+1) if len(found) else None,
                last128_correct=float(correct[i,-128:].mean()),last128_present=float(data['present'][i,-128:].mean()),
                last128_contribution=float(data['contribution'][i,-128:].mean()),
                autonomous_exits=int(data['target_exited'][i].sum()),capacity_drops=int(data['target_capacity_dropped'][i].sum())))
    for arm in settings[0]['arms']:
        a,b=confirmed['SELECT',arm],confirmed['MIX',arm]
        np.testing.assert_array_equal(a['seeds'],b['seeds'])
        if a['observed_probability'].shape!=b['observed_probability'].shape or a['observed_probability'].shape[1]!=cut:
            raise ValueError('Confirmation training boundaries differ')
        training_rows.append(dict(model=arm,trials=cut,filters=len(a['seeds']),mix_nll=metrics.nll(b['observed_probability']),
            select_nll=metrics.nll(a['observed_probability']),delta_nll=metrics.nll(a['observed_probability'])-metrics.nll(b['observed_probability']),
            delta_nll_mc_se=metrics.jackknife_se(metrics.jackknife_scores(a['observed_probability'])-metrics.jackknife_scores(b['observed_probability']))))
        np.testing.assert_array_equal(banks['MIX',arm]['seeds'],banks['SELECT',arm]['seeds'])
        for window,_,_ in windows:
            old,old_j=estimates['MIX',arm,window];new,new_j=estimates['SELECT',arm,window]
            row=dict(model=arm,window=window,comparison='SELECT_minus_MIX')
            for name in ('energy','crps'):
                row['delta_'+name]=new[name]-old[name]
                row['delta_'+name+'_mc_se']=metrics.jackknife_se(new_j[name]-old_j[name])
            contrasts.append(row)
    for path,sha in inputs.items():
        if metrics.digest(Path(path))!=sha: raise ValueError('Bound input changed during review')
    output.mkdir(parents=True)
    for name,rows in [('training_confirmation',training_rows),('autonomous_scores',scores),('autonomous_contrasts',contrasts),
                      ('paths',path_rows),('parameters',parameters),('closed_channels',closed),('chunks',chunk_rows)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    write(output/'completion.json',dict(status='complete',subject=subject,training_trials=cut,total_trials=length,
        paths=len(path_rows),chunks=len(chunk_rows),inputs_sha256=inputs,
        artifacts_sha256={p.name:metrics.digest(p) for p in output.glob('*.csv')},
        scope='Completed confirmation and all declared autonomous paths; conditional later-choice and oral results excluded',
        conditional_evaluation_required_separately=True,oral_used_for_fit=False,
        uncertainty='Paired whole-filter/path Monte Carlo errors at frozen selected points; not parameter/population uncertainty'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','revised','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--subject',type=int,required=True);args=parser.parse_args()
    run(args.original,args.revised,args.subject,args.output)
