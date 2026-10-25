"""Paired oracle diagnosis from frozen Model 1024 prefix particle clouds.

Run --config YAML --output NEW [--smoke]. Every arm samples identical ancestors
and future seeds within a parameter point. Fixed parameters, shared filter and
continuation lifecycle are preserved. The two prefix cutoffs are development
state diagnostics; their fitted parameters may use later training observations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import importlib.metadata
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

if __package__ in (None, ''):
    import src.Bayesian_state.workflows.analysis as analysis_package
    repository = Path(__file__).resolve().parents[4]
    analysis_package.__path__ = [str(repository/'src/Bayesian_state/workflows/analysis'), *analysis_package.__path__]
    __package__ = 'src.Bayesian_state.workflows.analysis'

import numpy as np
import yaml
from . import model_1023_execution_study as execution
from .model_1023_prefix_forecast import verify
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
HELPER = 'src/Bayesian_state/simulation/filtered_continuation.py'
OBSERVER = 'src/Bayesian_state/simulation/continuation_diagnostics.py'
PROBES = 'src/Bayesian_state/simulation/continuation_probes.py'
read, write, digest = execution.read, execution.write, execution.digest


def sources() -> dict:
    return {p: digest(ROOT/p) for p in (ENTRY, HELPER, OBSERVER, PROBES)}


def cutoffs(training_trials: int, block: int) -> dict[str, int]:
    middle = training_trials // (2*block) * block
    if not 0 < middle < training_trials:
        raise ValueError('Need two distinct valid prefix cutoffs')
    return dict(half_training_block=middle, training_end=training_trials)


def restore_selected_cloud(job: dict, kwargs: dict):
    """Restore every drawn particle using shared model construction and state IO.

    Undrawn positions share a placeholder which is never simulated. The full
    original weight vector and exact seed bank must select only saved states.
    Endpoint summaries of the whole cloud remain the original saved arrays.
    """
    from src.Bayesian_state.model import StateModel, ModelContext
    from src.Bayesian_state.inference.backends.particle_filter import _particle_config
    stem=Path(job['replay_stem'])
    with gzip.open(stem.with_suffix('.pkl.gz'),'rb') as stream:
        saved=pickle.load(stream)
    with np.load(stem.with_suffix('.npz')) as z:
        bank={k:z[k] for k in z.files}
    if saved['engine_config']!=kwargs['engine_config'] or saved['cut']!=job['cut']:
        raise ValueError('Replay changed model parameters or prefix timing')
    for key in ('stimulus','choices','feedback'):
        np.testing.assert_array_equal(saved['prefix'][key],kwargs[key])
    np.testing.assert_array_equal(saved['branch_seeds'],job['branch_seeds'])
    ancestors=np.asarray([np.random.default_rng(stable_seed(dict(role='prefix_terminal_particle',
        branch_seed=int(seed)))).choice(len(saved['weights']),p=saved['weights']) for seed in job['branch_seeds']])
    np.testing.assert_array_equal(ancestors,saved['ancestors'])
    if set(ancestors)!=set(saved['snapshots']):
        raise ValueError('Replay requires the exact saved ancestor/seed bank')
    restored={}; partition=space=None
    for index,state in saved['snapshots'].items():
        model=StateModel(_particle_config(kwargs['engine_config'],job['seeds'][0],index),
            context=ModelContext(condition=kwargs['condition'],subject_id=kwargs['subject_id'],
                processed_data_dir=kwargs['processed_data_dir'],dataset_paths=kwargs['dataset_paths']),
            partition=partition,hypotheses_set=space)
        partition,space=model.partition_model,model.hypotheses_set
        model.engine.load_state_dict(state)
        model._completed_trial_count=job['cut']; model._pending_trial=None
        restored[index]=model
    first=next(iter(restored.values()))
    models=[restored.get(i,first) for i in range(len(saved['weights']))]
    result=SimpleNamespace(final_weights=saved['weights'],
        observation_probabilities={'prior_t':bank['prefix_probabilities']})
    endpoints=[{k.removeprefix('endpoint_'):bank[k][i] for k in bank if k.startswith('endpoint_')}
               for i in range(len(models))]
    return models,result,endpoints,bank


def worker(file: Path) -> None:
    from threadpoolctl import threadpool_info
    job = read(file)
    if job['continuation_sources'] != sources():
        raise ValueError('Continuation source changed')
    verify(job['bindings'])
    helper = execution.base.load('src.Bayesian_state.simulation.filtered_continuation', HELPER)
    observer = execution.base.load('src.Bayesian_state.simulation.continuation_diagnostics', OBSERVER)
    probes = execution.base.load('src.Bayesian_state.simulation.continuation_probes', PROBES)
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    folder = Path(job['folder']); meta = read(folder/'prepared.json')
    verify({str(Path(job['runtime'])/p): h for p,h in meta['frozen_cognitive_source_sha256'].items()})
    engine = read(folder/'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    cut = job['cut']
    with np.load(folder/'training.npz') as train:
        prefix = {k: train[k][:cut].copy() for k in ('stimulus','choices','feedback')}
        if job['smoke']:
            future = {k: train[k][cut:cut+8].copy() for k in ('stimulus','categories')}
        else:
            with np.load(folder/'sealed_test.npz') as sealed:
                future = {k: np.concatenate([train[k][cut:],sealed[k]]) for k in ('stimulus','categories')}
    kwargs = dict(engine_config=engine, subject_id=meta['subject'], condition=meta['condition'], **prefix,
        particle_count=job['particles'], particle_blocks=1, filter_seed=job['seeds'][0], choice_readout_power=1.,
        processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    started, cpu = time.monotonic(), time.process_time()
    called = []

    def diagnose(installed, workflow):
        called.append(True)
        replay_bank=None
        if job.get('replay_stem'):
            models,result,endpoints,replay_bank=restore_selected_cloud(job,kwargs)
        else:
            models,result=helper.capture_prefix(**kwargs)
            endpoints=[observer.endpoint_state(m,job['target_rule']) for m in models]
        predicted = result.observation_probabilities['prior_t']
        if job['smoke']:
            independent = run_state_model_particle_filter(**kwargs)
            expected = independent.observation_probabilities['prior_t']
            np.testing.assert_array_equal(independent.final_weights, result.final_weights)
        else:
            with np.load(job['reference']) as reference:
                np.testing.assert_array_equal(reference['seeds'], job['seeds'])
                np.testing.assert_array_equal(reference['choices'][:cut],prefix['choices'])
                expected = reference['probabilities'][0,:cut]
        np.testing.assert_array_equal(predicted, expected)
        h = models[0].engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        branch_seeds = job['branch_seeds']
        ancestors = np.asarray([np.random.default_rng(stable_seed(dict(role='prefix_terminal_particle',
            branch_seed=int(seed)))).choice(len(models),p=result.final_weights) for seed in branch_seeds])
        target = Path(job['target']); target.parent.mkdir(parents=True,exist_ok=True)
        cloud = {int(i):models[int(i)].engine.state_dict() for i in set(ancestors)}
        snapshot = target.with_suffix('.pkl.gz')
        with snapshot.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as compressed:
                pickle.dump(dict(engine_config=engine, snapshots=cloud, cut=cut, weights=result.final_weights,
                    ancestors=ancestors, prefix=prefix, branch_seeds=branch_seeds),compressed,protocol=5)
        generate = dict(models=models, weights=result.final_weights, observed_choices=prefix['choices'],
            observed_feedback=prefix['feedback'], **future, condition=meta['condition'], branch_seeds=branch_seeds)
        base_capture = observer.ContinuationObserver(job['target_rule'])

        def capture(model, prediction):
            row = base_capture(model,prediction)
            transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            event = transition.transition_log[-1]
            row.update(executed=int(transition.executed_hypothesis) if job['execution_mode']!='MIX'
                or event.get('probe_arm','').startswith('AVAILABLE_EXECUTE') else -1,
                inserted=bool(event.get('probe_inserted',False)), protected=bool(event.get('probe_protected',False)))
            return row

        banks = []
        for arm in job['arms']:
            with probes.continuation_probe(type(h),arm=arm,target=job['target_rule'],cut=cut):
                arrays = helper.generate_continuations(**generate,observer=capture)
                if job['smoke']:
                    reverse = helper.generate_continuations(**dict(generate,branch_seeds=branch_seeds[::-1]),observer=capture)
                    for k in arrays:
                        other = reverse[k] if k in ('terminal_weights','categories') else reverse[k][::-1]
                        np.testing.assert_array_equal(arrays[k],other,err_msg=arm+': order changed '+k)
            np.testing.assert_array_equal(arrays['ancestors'],ancestors)
            if arm == 'BASE' and job['smoke']:
                plain = helper.generate_continuations(**generate)
                for k in plain:
                    np.testing.assert_array_equal(plain[k],arrays[k],err_msg='Capture changed '+k)
            if arm in ('AVAILABLE','AVAILABLE_PRECISION','AVAILABLE_EXECUTE','AVAILABLE_EXECUTE_PRECISION'):
                if not arrays['diagnostic_present'].all():
                    raise ValueError('Availability probe failed')
            if arm in ('PRECISION','AVAILABLE_PRECISION','AVAILABLE_EXECUTE_PRECISION'):
                np.testing.assert_array_equal(arrays['diagnostic_beta'][arrays['diagnostic_present'].astype(bool)],25.)
            banks.append(arrays)
        if job['smoke']:
            restored = helper.generate_continuations(**generate,observer=capture)
            for k in restored:
                np.testing.assert_array_equal(restored[k],banks[0][k],err_msg='Arm leaked into BASE: '+k)
            with gzip.open(snapshot,'rb') as stream:
                saved_cloud = pickle.load(stream)
            for ancestor in set(ancestors):
                models[int(ancestor)].engine.load_state_dict(saved_cloud['snapshots'][int(ancestor)])
        constant = ('terminal_weights','categories','ancestors','branch_seeds')
        output = {k: banks[0][k] if k in constant else np.stack([b[k] for b in banks]) for k in banks[0]}
        output.update({'endpoint_'+k:np.asarray([e[k] for e in endpoints]) for k in endpoints[0]})
        output.update(prefix_probabilities=predicted, filter_seed=np.asarray(job['seeds'][0]), arms=np.asarray(job['arms']))
        replay_differences=[]
        if replay_bank is not None:
            for key in output:
                if not np.array_equal(output[key],replay_bank[key],equal_nan=output[key].dtype.kind in 'fc'):
                    replay_differences.append(key)
            # BASE, protection and precision have unchanged definitions.
            for arm in ('BASE','RETAIN','PRECISION'):
                index=job['arms'].index(arm)
                for key in ('choices','feedback','probabilities'):
                    np.testing.assert_array_equal(output[key][index],replay_bank[key][index])
        with target.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream,**output)
        write(target.with_suffix('.json'),dict(status='complete',subject=job['subject'],mode=job['execution_mode'],
            candidate=job['candidate'],case_id=job['case_id'],cut=cut,filter_seed=job['seeds'][0],
            particles=job['particles'],branches=len(branch_seeds),arms=job['arms'],trials=len(future['categories']),
            prefix_exact=True,smoke=job['smoke'],branch_order_check=job['smoke'],observer_passive=job['smoke'],
            replay=bool(job.get('replay_stem')),replay_differences=replay_differences,
            snapshot_sha256=digest(snapshot),predictions_sha256=digest(target.with_suffix('.npz')),
            seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,numeric_pools=threadpool_info()))

    original = execution.driver.io.score
    execution.driver.io.score = diagnose
    try:
        execution.worker(file)
    finally:
        execution.driver.io.score = original
    if len(called)!=1:
        raise ValueError('Expected one captured prefix cloud')
    verify(job['bindings'])


def run(config: Path, output: Path, smoke: bool=False, replay: Path | None=None) -> None:
    if output.exists():
        raise FileExistsError('Use a new continuation directory')
    policy = yaml.safe_load(config.read_text())
    from ...simulation.continuation_probes import ARMS
    if tuple(policy['arms']) != ARMS or policy['filters']!=6 or policy['particles']!=512:
        raise ValueError('Require the registered probe and filter budget')
    if replay is not None:
        replay_protocol=read(replay/'protocol.json'); replay_done=read(replay/'completion.json')
        if (replay_done['status']!='complete' or replay_done['smoke']!=smoke
                or replay_protocol['policy']!=policy or replay_done['protocol_sha256']!=digest(replay/'protocol.json')):
            raise ValueError('Replay requires the same complete protocol and smoke status')
        replay_cases={c['case_id']:c for c in replay_protocol['cases']}
    evaluation = ROOT/policy['source_evaluation']; previous = read(evaluation/'protocol.json')
    if read(evaluation/'completion.json')['status']!='complete':
        raise ValueError('Complete frozen-candidate evaluation first')
    chosen = [c for c in previous['cases'] if c['subject'] in policy['subjects'] and c['mode'] in policy['modes']]
    if len(chosen)!=8:
        raise ValueError('Expected eight fixed parameter/model points')
    if smoke:
        chosen = [next(c for c in chosen if c['subject']==s and c['mode']=='STICKY') for s in policy['subjects']]
    output.mkdir(parents=True)
    bindings = {str(config.resolve()):digest(config), str((evaluation/'protocol.json').resolve()):digest(evaluation/'protocol.json')}
    jobs, cases = [], []
    seed_owners = {}
    forbidden = {s for c in previous['cases'] for role in ('filter_jobs','autonomous_jobs')
                 for j in c[role] for s in j['seeds']}
    for c in chosen:
        folder = Path(c['folder']); meta = read(folder/'prepared.json')
        geometry = ROOT/policy['geometry']/f"subject_{c['subject']}"/'audit.json'
        targets = read(geometry)['training_perfect_rules']
        if len(targets)!=1:
            raise ValueError('This protocol requires one training-defined perfect rule')
        own_bindings = {str(geometry.resolve()):digest(geometry)}
        for name in ('prepared.json','engine.json','training.npz','sealed_test.npz'):
            p=folder/name; own_bindings[str(p.resolve())]=digest(p)
        for label,cut in (dict(smoke=8) if smoke else cutoffs(meta['training_trials'],policy['block_width'])).items():
            case_id=f"S{c['subject']}_{c['mode']}_{c['candidate']['id']}_{label}"
            own=[]
            for i,template in enumerate(c['filter_jobs'][:1 if smoke else policy['filters']]):
                job=deepcopy(template); reference=Path(template['target']).with_suffix('.npz')
                if digest(reference)!=read(reference.with_suffix('.json'))['predictions_sha256']:
                    raise ValueError('Saved prefix reference changed')
                branches=[stable_seed(dict(protocol=policy['protocol'],base=policy['seed'],subject=c['subject'],
                    mode=c['mode'],cut=label,filter=i,branch=b)) for b in range(2 if smoke else policy['branches_per_filter'])]
                # The same mode/cut/filter bank is deliberately shared across
                # competing parameters, and all arms share sampled ancestors.
                for seed in branches:
                    owner=(c['subject'],c['mode'],label,i)
                    if seed in forbidden or seed in seed_owners and seed_owners[seed]!=owner:
                        raise ValueError('Future seed collides with another simulation role')
                    seed_owners[seed]=owner
                job.update(phase='training',name=case_id+f'_F{i:02d}',case_id=case_id,smoke=smoke,
                    particles=4 if smoke else policy['particles'],cut=cut,cut_label=label,
                    target_rule=targets[0],arms=policy['arms'],branch_seeds=branches,
                    reference=str(reference),continuation_sources=sources(),
                    bindings=dict(own_bindings,**{str(reference):digest(reference)}),
                    target=str((output/'chunks'/case_id/f'F{i:02d}').resolve()))
                job['source_sha256']=execution.sources(execution.driver.io)
                if replay is not None:
                    original=replay_cases[case_id]['jobs'][i]
                    stem=Path(original['target']); record=read(stem.with_suffix('.json'))
                    for suffix,sha in (('.npz',record['predictions_sha256']),('.pkl.gz',record['snapshot_sha256']),
                                       ('.json',digest(stem.with_suffix('.json')))):
                        path=stem.with_suffix(suffix)
                        if digest(path)!=sha: raise ValueError('Replay source changed')
                        job['bindings'][str(path)]=sha
                    job['replay_stem']=str(stem)
                own.append(job); jobs.append(job)
            cases.append(dict(subject=c['subject'],mode=c['mode'],candidate=c['candidate'],cut=cut,cut_label=label,
                case_id=case_id,is_original=c['is_original'],is_selected=c['is_selected'],target_rule=targets[0],jobs=own,
                autonomous_jobs=c['autonomous_jobs'],folder=str(folder)))
        bindings.update(own_bindings)
    write(output/'protocol.json',dict(status='frozen_before_simulation',policy=policy,smoke=smoke,cases=cases,
        inputs_sha256=bindings,source_sha256=sources(),versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')},
        fixed_parameters=True,scope='Development oracle diagnosis; early prefix states use retrospectively fitted parameters'))
    previous_entry=execution.ENTRY; execution.ENTRY=ENTRY
    started=time.monotonic()
    try:
        execution.execute(jobs,output,1 if smoke else policy['parallel_budget'])
    finally:
        execution.ENTRY=previous_entry
    records=[read(Path(j['target']).with_suffix('.json')) for j in jobs]
    if not all(r['prefix_exact'] and r['status']=='complete' for r in records):
        raise ValueError('Incomplete or inexact prefix replay')
    verify(bindings)
    write(output/'completion.json',dict(status='complete',smoke=smoke,seconds=time.monotonic()-started,
        jobs=len(jobs),cases=len(cases),paths=sum(r['branches']*len(r['arms']) for r in records),
        generated_trials=sum(r['branches']*len(r['arms'])*r['trials'] for r in records),
        cpu_seconds=sum(r['cpu_seconds'] for r in records),prefix_exact=True,
        replay_source=str(replay.resolve()) if replay is not None else None,
        replay_exact=all(not r['replay_differences'] for r in records) if replay is not None else None,
        protocol_sha256=digest(output/'protocol.json'),source_sha256=sources()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--replay',type=Path,help='Restore the exact saved ancestors; preserve the original seed bank')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config,args.output,args.smoke,args.replay)
    else: parser.error('Provide config/output or worker')
