"""Fixed-parameter choice controls separating rule and pairing retention.

Run --config YAML --output NEW [--smoke]. Reuse a completed BASE/ONLINE screen
at rule gamma=0; change only the declared conditional pairing rates. Four
original filter seeds and particles are preserved. Smoke replays two original
seeds for eight trials, including both unmodified reference variants. Oral
reports and reserved participants do not enter this diagnostic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
SCORE='src/Bayesian_state/workflows/analysis/model_0923_online_evidence_scores.py'
MECHANISM='src/Bayesian_state/model/modules/pairing_retention.py'


def helpers():
    spec=importlib.util.spec_from_file_location('pairing_retention_online_scores',ROOT/SCORE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    online=helpers();retention=online.helpers();recalled=retention.helpers()
    retention.helpers=lambda:recalled
    online.helpers=lambda:retention
    original_validation=recalled.base.validation

    def validation():
        v=original_validation();original_score=v.score

        def score(job,workflow):
            rate=job['pairing_retention_override']
            if job['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma']!=0:
                raise ValueError('This diagnostic fixes rule retention at zero')
            memory=sys.modules['src.Bayesian_state.model.modules.pairing_memory'].HierarchicalPairingMemoryModule
            module=recalled.base.load('src.Bayesian_state.model.modules.pairing_retention',MECHANISM)
            with module.pairing_retention(memory,rate=rate) as counts:
                original_score(job,workflow)
            target=Path(job['target']);record=v.read(target.with_suffix('.json'))
            expected=0 if rate is None else record['trials']*job['particles']*len(job['seeds'])
            if counts['updates']!=expected or counts['recombined_updates']!=expected:
                raise ValueError('Pairing feedback update count differs from declared full filter')
            v.write(target.with_suffix('.pairing.json'),dict(rule_retention=0.,pairing_retention_override=rate,
                **counts,unit='computational feedback updates',same_pairing_cache_for_memory_and_precision=True))

        v.score=score
        return v

    recalled.base.validation=validation
    online.worker(file)


def checked_reference(protocol: dict, done: dict, subject: int, particles: int, seeds: int) -> list[dict]:
    if done['status']!='complete' or done['smoke'] or protocol['smoke']:
        raise ValueError('Require a completed full reference screen')
    plans=protocol['plans']
    if len(plans)!=2 or {p['variant'] for p in plans}!={'baseline','online'}:
        raise ValueError('Require both reference belief variants')
    for plan in plans:
        j=plan['job']
        if (j['model']!='SEARCH' or j['phase']!='evaluate' or j['particles']!=particles
                or len(j['seeds'])!=seeds or len(set(j['seeds']))!=seeds
                or j['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma']!=0
                or j['online_belief']!=(plan['variant']=='online') or not j['retain_precision']):
            raise ValueError('Reference parameters, numerical budget or mechanisms differ')
    for key in ('candidate','seeds','particles','folder','runtime','mechanism'):
        if plans[0]['job'][key]!=plans[1]['job'][key]:
            raise ValueError('Reference variants differ in frozen settings')
    if protocol['score_settings']['subject']!=subject:
        raise ValueError('Reference participant differs')
    return plans


def run(config: Path, output: Path, smoke: bool=False) -> None:
    from src.Bayesian_state.workflows.analysis.model_0923_capacity_memory_scores import choice_contrasts
    from src.Bayesian_state.workflows.analysis.model_0923_estimation_scope import smoke_data
    if output.exists():raise FileExistsError('Use a new pairing diagnostic directory')
    settings=yaml.safe_load(config.read_text())
    rates=settings['pairing_rates']
    if (not rates or len(set(rates))!=len(rates)
            or any(isinstance(x,(bool,str)) or not np.isfinite(x) or not 0<x<=1 for x in rates)
            or not 1<=settings['parallel_budget']<=128):
        raise ValueError('Require distinct positive pairing rates and a bounded worker budget')
    online=helpers();retention=online.helpers();recalled=retention.helpers();v=recalled.base.validation()
    source=ROOT/settings['reference'];inputs={}

    def bind(path):
        path=Path(path).resolve();inputs[str(path)]=v.digest(path);return path

    bind(config)
    original=v.read(bind(source/'protocol.json'));done=v.read(bind(source/'completion.json'))
    if done['protocol_sha256']!=v.digest(source/'protocol.json'):
        raise ValueError('Reference protocol changed')
    references=checked_reference(original,done,settings['subject'],settings['particles'],settings['filter_seeds'])
    hashes=dict(original['source_sha256'])
    for name,sha in {**original['input_sha256'],**hashes}.items():
        if v.digest(bind(ROOT/name))!=sha:raise ValueError('Reference implementation or input changed')
    hashes.update({n:v.digest(ROOT/n) for n in (ENTRY,MECHANISM)})
    folder=Path(references[0]['job']['folder'])
    meta=v.read(bind(folder/'prepared.json'));engine=v.read(bind(folder/'engine.json'))
    if (meta['subject']!=settings['subject'] or meta['condition']!=3
            or engine['modules']['memory_mod']['kwargs']['forgetting_mode']!='factorized'):
        raise ValueError('Require the same condition-3 factorized-memory data')
    for name in ('training.npz','sealed_test.npz'):bind(folder/name)
    originals={}
    for plan in references:
        for chunk in plan['chunks']:
            if v.read(bind(source/'jobs'/(chunk['name']+'.json')))!=chunk:
                raise ValueError('Reference planned and actual jobs differ')
        target=Path(plan['job']['target']);record=v.read(bind(target.with_suffix('.json')))
        if v.digest(bind(target.with_suffix('.npz')))!=record['predictions_sha256']:
            raise ValueError('Reference prediction changed')
        with np.load(target.with_suffix('.npz')) as z:originals[plan['variant']]={k:z[k].copy() for k in z.files}
    output.mkdir(parents=True)
    recalled.base.archive(output,config,hashes)
    if smoke:folder=smoke_data(folder.parent,output,settings['subject'],v)/f'subject_{settings["subject"]}'
    cut=v.read(folder/'prepared.json')['training_trials']
    jobs=[];plans=[]
    for reference in references:
        for rate in ([None]+rates if smoke else rates):
            label='shared' if rate is None else str(rate).replace('.','p')
            name=f"{reference['variant']}_pair_{label}"
            job=dict(deepcopy(reference['job']),name=name,folder=str(folder.resolve()),smoke=smoke,
                seeds=reference['job']['seeds'][:2] if smoke else reference['job']['seeds'],
                pairing_retention_override=rate,source_sha256=hashes,target=str((output/'conditional'/name).resolve()))
            chunks=[dict(deepcopy(job),name=name+f'_C{i:03d}',seeds=[seed],
                target=str((output/'conditional/chunks'/(name+f'_C{i:03d}')).resolve())) for i,seed in enumerate(job['seeds'])]
            plans.append(dict(variant=reference['variant'],pairing_rate=rate,job=job,chunks=chunks));jobs.extend(chunks)
    protocol=dict(settings=original['settings'],diagnostic_settings=settings,reference=str(source.resolve()),
        smoke=smoke,input_sha256=inputs,source_sha256=hashes,plans=plans,
        data_use='Fixed existing behavioral parameters; no parameter selection, oral data or reserved participants',
        intervention='Rule gamma remains zero; only conditional pairing retention changes')
    v.write(output/'protocol.json',protocol)
    recalled.ENTRY=ENTRY;recalled.execute(jobs,output,1 if smoke else settings['parallel_budget'])
    arrays={};records=[];exact_controls=0
    for plan in plans:
        record=v.combine_evaluation_chunks(plan['job'],plan['chunks'],False)
        with np.load(Path(plan['job']['target']).with_suffix('.npz')) as z:saved={k:z[k].copy() for k in z.files}
        p=saved['probabilities']
        if not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid choice probabilities')
        np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
        np.testing.assert_array_equal(saved['observed_probability'],p[:,np.arange(p.shape[1]),saved['choices']-1])
        np.testing.assert_array_equal(saved['seeds'],plan['job']['seeds'])
        arrays[(plan['variant'],plan['pairing_rate'])]=saved
        records.append(dict(variant=plan['variant'],pairing_rate=plan['pairing_rate'],record=record))
        if plan['pairing_rate'] is None:
            expected=originals[plan['variant']]
            for key in ('choices','categories'):np.testing.assert_array_equal(saved[key],expected[key][:p.shape[1]])
            for key in ('probabilities','observed_probability'):np.testing.assert_array_equal(saved[key],expected[key][:2,:p.shape[1]])
            exact_controls+=1
    feedback=[]
    for name in ('training.npz','sealed_test.npz'):
        with np.load(folder/name) as z:feedback.extend(z['feedback'].tolist())
    scores=[];trials=[]
    for reference in references:
        variant=reference['variant'];base=arrays[(variant,None)] if smoke else originals[variant]
        for rate in rates:
            saved=arrays[(variant,rate)]
            for key in ('choices','categories','seeds'):np.testing.assert_array_equal(saved[key],base[key])
            p,q=saved['observed_probability'],base['observed_probability']
            if len(feedback)!=p.shape[1]:raise ValueError('Feedback and trial identities differ')
            scores.extend(dict(variant=variant,pairing_rate=rate,comparison='same_belief_original_pairing',**r)
                          for r in choice_contrasts(p,q,cut))
            a,b=p.mean(0),q.mean(0);delta=np.log(np.maximum(b,np.finfo(float).tiny))-np.log(np.maximum(a,np.finfo(float).tiny))
            trials.extend(dict(variant=variant,pairing_rate=rate,trial=i+1,phase='prefix' if i<cut else 'later',
                feedback=feedback[i],reference_probability=b[i],probability=a[i],delta_nll=delta[i]) for i in range(len(a)))
    for rate in rates:
        p=arrays[('online',rate)]['observed_probability'];q=arrays[('baseline',rate)]['observed_probability']
        scores.extend(dict(variant='online',pairing_rate=rate,comparison='online_minus_baseline_same_pairing',**r)
                      for r in choice_contrasts(p,q,cut))
    pd.DataFrame(scores).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(trials).to_csv(output/'trial_scores.csv',index=False)
    for name,sha in inputs.items():
        if v.digest(Path(name))!=sha:raise ValueError('Input changed during execution')
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,jobs=len(jobs),records=records,
        exact_reference_controls=exact_controls,nominees_changed=False,protocol_sha256=v.digest(output/'protocol.json'),
        artifacts={p.name:v.digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.smoke)
    else:parser.error('Use --config and --output, or internal --worker')
