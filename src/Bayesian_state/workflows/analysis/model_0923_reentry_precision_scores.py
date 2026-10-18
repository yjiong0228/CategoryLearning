"""Compare original and retained precision on complete recorded choices.

Run --config CONFIG --output NEW_DIRECTORY. Each declared parameter point uses
the same eight seeds and 512 particles as its saved RESET reference. Only the
RETAIN filters are new. --smoke recomputes both mechanisms with two original
seeds, the same particle count and eight trials; RESET must match every saved
prediction exactly. Prefix/later/full comparisons are retrospective development
diagnostics. No oral data or parameter optimization enters this workflow.
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

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/reentry_precision.py'


def helpers():
    spec=importlib.util.spec_from_file_location('reentry_score_refit',ROOT/
        'src/Bayesian_state/workflows/analysis/model_0923_recalled_search_refit.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    recalled=helpers();original_validation=recalled.base.validation
    def validation():
        v=original_validation();original_score=v.score
        def score(job,workflow):
            if job['phase']!='evaluate' or job['model']!='SEARCH':
                raise ValueError('Precision choice comparison requires SEARCH evaluation')
            module=recalled.base.load('src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision',MECHANISM)
            cls=sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
            with module.reentry_precision(cls,enabled=job['retain_precision']) as events:
                original_score(job,workflow)
            v.write(Path(job['target']).with_suffix('.precision.json'),dict(
                retain_precision=job['retain_precision'],particle_transitions=len(events),
                particle_restores=sum(len(e['precision_restored_rules']) for e in events),
                role='computational particle events; not autonomous cognitive-event counts'))
        v.score=score;return v
    recalled.base.validation=validation
    recalled.worker(file)


def run(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.workflows.analysis.model_0923_estimation_scope import smoke_data
    from src.Bayesian_state.workflows.analysis.model_0923_capacity_memory_scores import choice_contrasts
    recalled=helpers();v=recalled.base.validation();settings=yaml.safe_load(config.read_text())
    if output.exists():raise FileExistsError('Use a new precision choice-score directory')
    source=ROOT/settings['reference'];automatic=ROOT/settings['autonomous']
    done=v.read(source/'completion.json');protocol=v.read(source/'protocol.json')
    auto_done=v.read(automatic/'completion.json');auto_protocol=v.read(automatic/'protocol.json')
    if any(r['status']!='complete' or r.get('smoke') is not False for r in (done,auto_done)):
        raise ValueError('Require completed scientific behavior and autonomous references')
    if settings['points']!=auto_protocol['settings']['points'] or not len(set(settings['points']))==len(settings['points']):
        raise ValueError('Score exactly the declared autonomous parameter points')
    sources=dict(protocol['source_sha256'])
    for name,expected in sources.items():
        if v.digest(ROOT/name)!=expected:raise ValueError(f'Executing dependency changed: {name}')
    sources.update({name:v.digest(ROOT/name) for name in (ENTRY,MECHANISM)})
    if auto_protocol['source_sha256'][MECHANISM]!=sources[MECHANISM]:
        raise ValueError('Behavior and autonomous precision mechanisms differ')
    inputs={str(p.resolve()):v.digest(p) for p in (config,source/'completion.json',source/'protocol.json',
        automatic/'completion.json',automatic/'protocol.json')}
    bank={r['candidate']['id']:r for r in done['records']};templates={r['candidate']['id']:r['job'] for r in protocol['plans']}
    seeds=protocol['seeds'][:2] if smoke else protocol['seeds']
    if len(protocol['seeds'])!=settings['filter_seeds'] or protocol['particles']!=settings['particles']:
        raise ValueError('Declared and original numerical budgets differ')
    output.mkdir(parents=True);recalled.base.archive(output,config,sources)
    folder=Path(templates[settings['points'][0]]['folder'])
    if smoke:folder=smoke_data(folder.parent,output,settings['subject'],v)/f'subject_{settings["subject"]}'
    cut=v.read(folder/'prepared.json')['training_trials'];jobs=[];plans=[]
    for point in settings['points']:
        original=bank[point];file=Path(original['target']);record=original['record'];template=templates[point]
        if v.digest(file)!=record['predictions_sha256'] or record['candidate']['hyperparams']!=template['candidate']['hyperparams']:
            raise ValueError('Original conditional reference changed')
        if record['subject']!=settings['subject'] or record['model']!='SEARCH' or record['seeds']!=protocol['seeds']:
            raise ValueError('Original subject, mechanism or seed bank differs')
        inputs[str(file.resolve())]=v.digest(file)
        for variant in ('RESET','RETAIN') if smoke else ('RETAIN',):
            name=f'{variant}_S{settings["subject"]}_{point}'
            job=dict(deepcopy(template),name=name,seeds=seeds,folder=str(folder.resolve()),smoke=smoke,
                retain_precision=variant=='RETAIN',source_sha256=sources,
                target=str((output/'conditional'/name).resolve()))
            chunks=[]
            for i,seed in enumerate(seeds):
                chunk=dict(job,seeds=[seed],name=f'{name}_C{i:03d}',
                    target=str((output/'conditional'/'chunks'/f'{name}_C{i:03d}').resolve()))
                chunks.append(chunk);jobs.append(chunk)
            plans.append(dict(point=point,variant=variant,job=job,chunks=chunks,reference=str(file.resolve())))
    v.write(output/'protocol.json',dict(settings=settings,smoke=smoke,source_sha256=sources,input_sha256=inputs,
        plans=plans,seeds=seeds,particles=settings['particles'],python=sys.version,numpy=np.__version__,pandas=pd.__version__,
        scope='Retrospective fixed-parameter sequence likelihood; same points, tasks, seed bank and particles',
        data_use='Complete choices; no oral data; no parameter selection or fitting'))
    recalled.ENTRY=ENTRY;recalled.execute(jobs,output,1 if smoke else settings['parallel_budget'])
    arrays={};records=[]
    for plan in plans:
        record=v.combine_evaluation_chunks(plan['job'],plan['chunks'],False)
        target=Path(plan['job']['target']).with_suffix('.npz')
        with np.load(target) as z:data={key:z[key] for key in z.files}
        np.testing.assert_array_equal(data['seeds'],seeds)
        if not np.isfinite(data['probabilities']).all() or np.any(data['probabilities']<0):raise ValueError('Invalid probabilities')
        np.testing.assert_allclose(data['probabilities'].sum(axis=-1),1,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(data['observed_probability'],data['probabilities'][:,np.arange(len(data['choices'])),data['choices']-1])
        arrays[plan['point'],plan['variant']]=data
        records.append(dict(point=plan['point'],variant=plan['variant'],record=record,target=str(target)))
    rows=[];trials=[]
    for point in settings['points']:
        with np.load(bank[point]['target']) as z:
            n=8 if smoke else len(z['choices'])
            baseline={key:z[key][:len(seeds)] if key=='seeds' else z[key][:n] if key in ('choices','categories')
                      else z[key][:len(seeds),:n] for key in z.files}
        if smoke:
            for key,value in arrays[point,'RESET'].items():np.testing.assert_array_equal(value,baseline[key])
        revised=arrays[point,'RETAIN']
        for key in ('choices','categories','seeds'):np.testing.assert_array_equal(revised[key],baseline[key])
        for score in choice_contrasts(revised['observed_probability'],baseline['observed_probability'],cut):
            rows.append(dict(subject=settings['subject'],point=point,variant='RETAIN',reference_variant='RESET',**score))
        rp,op=[x['observed_probability'].mean(axis=0) for x in (revised,baseline)]
        rloss,oloss=[-np.log(np.maximum(x,np.finfo(float).tiny)) for x in (rp,op)]
        for i in range(n):trials.append(dict(point=point,trial=i+1,phase='prefix' if i<cut else 'later',
            original_probability=op[i],revised_probability=rp[i],delta_choice_nll=rloss[i]-oloss[i]))
    pd.DataFrame(rows).to_csv(output/'scores.csv',index=False);pd.DataFrame(trials).to_csv(output/'trial_scores.csv',index=False)
    for name,expected in dict(inputs,**sources).items():
        if v.digest(Path(name))!=expected:raise ValueError('Source or input changed during scoring')
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,records=records,new_jobs=len(jobs),
        exact_reset_prefix=smoke,particles=settings['particles'],filter_seeds=len(seeds),training_trials=cut,trials=n,
        artifacts={p.name:v.digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.smoke)
    else:parser.error('Supply --config and --output, or --worker')
