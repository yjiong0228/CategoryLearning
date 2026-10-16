"""Summarize independent particle-budget checks without refitting parameters.

--runs accepts completed precision runs with disjoint subject/candidate/budget
groups. All references are the original frozen candidates. New minima describe
the audit batch and are not independent validation of newly selected parameters.
--output must be new. --catalog-worker is an internal archived-runtime helper.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('precision_metrics',
    REPO/'src/Bayesian_state/evaluation/model_0923_h_validation.py')
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
digest, load, nll = _metrics.digest, _metrics.load, _metrics.nll
jackknife_scores, jackknife_se = _metrics.jackknife_scores, _metrics.jackknife_se
inactive_parameter_channels = _metrics.inactive_parameter_channels


def paired_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if left.shape != right.shape or left.ndim != 2 or len(left) < 4:
        raise ValueError('Need at least four matched full-filter repetitions')
    return nll(left)-nll(right), jackknife_se(jackknife_scores(left)-jackknife_scores(right))


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def read_filter_groups(runs: list[Path]) -> tuple[dict, dict, dict]:
    groups, inputs, protocols = {}, {}, {}
    for run in runs:
        protocol, done = load(run/'protocol.json'), load(run/'completion.json')
        if protocol['smoke'] or done['status'] != 'complete':
            raise ValueError('Require completed scientific precision runs')
        assert done['protocol_sha256'] == digest(run/'protocol.json')
        inputs[str(run/'protocol.json')] = digest(run/'protocol.json')
        protocols[str(run)] = protocol
        batches = {}
        for job in protocol['jobs']:
            if job['kind'] == 'filter':
                key = job['subject'], job['candidate']['id'], job['particles']
                batches.setdefault(key, []).append(job)
        for key, jobs in batches.items():
            if key in groups:
                raise ValueError('Duplicate candidate/budget group across runs')
            jobs.sort(key=lambda job: job['repeat'])
            assert len(jobs) == protocol['settings']['filter_repeats']
            arrays, diagnostics, seeds = [], [], []
            for job in jobs:
                target = Path(job['target']); row = load(target.with_suffix('.json'))
                trace = Path(job['diagnostic_target']); trace_row = load(trace.with_suffix('.json'))
                for file, expected in [(target.with_suffix('.npz'), row['predictions_sha256']),
                                       (trace.with_suffix('.npz'), trace_row['predictions_sha256'])]:
                    assert digest(file) == expected
                    inputs[str(file)] = expected
                assert row['role'] == 'training' and row['candidate'] == job['candidate']
                assert row['seeds'] == job['seeds'] and len(row['seeds']) == 1
                with np.load(target.with_suffix('.npz')) as saved:
                    arrays.append({name: saved[name] for name in saved.files})
                with np.load(trace.with_suffix('.npz')) as saved:
                    diagnostics.append({name: saved[name] for name in saved.files})
                seeds.extend(row['seeds'])
            assert len(set(seeds)) == len(seeds)
            p = np.concatenate([a['observed_probability'] for a in arrays])
            assert p.shape == (len(jobs), jobs[0]['training_trials'])
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            groups[key] = dict(probability=p, seeds=seeds, jobs=jobs,
                correct=np.concatenate([a['correct_probability'] for a in arrays]), diagnostics=diagnostics)
    return groups, inputs, protocols


def summarize_filters(groups: dict, protocols: dict, output: Path) -> dict:
    fit = Path(next(iter(protocols.values()))['fit']); frozen = load(fit/'nominees.json')
    selected = {(r['subject'], r['model']): r['candidate']['id'] for r in frozen['nominees']}
    letters = {(r['subject'], r['candidate']['id']): r['letter']
               for protocol in protocols.values() for r in protocol['candidates']}
    rows, seed_rows, contrasts, trial_rows, drift_rows, state_rows = [], [], [], [], [], []
    for (subject, candidate, particles), group in groups.items():
        p = group['probability']; deleted = jackknife_scores(p)
        old_file = fit/'confirmation'/f'subject_{subject}'/'NLL'/(candidate+'.npz')
        old_row = load(old_file.with_suffix('.json'))
        assert digest(old_file) == old_row['predictions_sha256']
        with np.load(old_file) as saved:
            old = saved['observed_probability']
        rows.append(dict(subject=subject, candidate=candidate, letter=letters[subject,candidate], particles=particles,
            repetitions=len(p), trials=p.shape[1], mean_nll=nll(p), mc_se=jackknife_se(deleted),
            old_nll=nll(old), old_mc_se=jackknife_se(jackknife_scores(old)), delta_old=nll(p)-nll(old),
            independent_old_delta_mc_se=np.hypot(jackknife_se(deleted), jackknife_se(jackknife_scores(old))),
            mean_single_filter_nll=float(np.mean([nll(x[None,:]) for x in p])),
            original_nll_selected=candidate == selected[subject,'NLL'], original_gen_selected=candidate == selected[subject,'GEN']))
        for repeat, values in enumerate(p):
            seed_rows.append(dict(subject=subject, candidate=candidate, letter=letters[subject,candidate],
                particles=particles, repeat=repeat, seed=group['seeds'][repeat], nll=nll(values[None,:])))
        reference = groups[subject, selected[subject,'NLL'], particles]
        assert group['seeds'] == reference['seeds']
        delta, se = paired_difference(p, reference['probability'])
        contrasts.append(dict(subject=subject, candidate=candidate, letter=letters[subject,candidate], particles=particles,
            reference=selected[subject,'NLL'], delta_nll=delta, paired_mc_se=se))
        folder = Path(group['jobs'][0]['folder']); meta = load(folder/'prepared.json')
        assert digest(folder/'training.npz') == meta['array_sha256']['training.npz']
        with np.load(folder/'training.npz') as saved:
            choices, categories = saved['choices'], saved['categories']
        tiny = np.finfo(float).tiny
        contribution = np.log(np.maximum(reference['probability'].mean(axis=0), tiny))-np.log(np.maximum(p.mean(axis=0), tiny))
        np.testing.assert_allclose(contribution.sum(), len(choices)*delta, atol=1e-10, rtol=0)
        for t, value in enumerate(contribution):
            trial_rows.append(dict(subject=subject, candidate=candidate, letter=letters[subject,candidate], particles=particles,
                trial=t+1, block=t//32+1, choice=int(choices[t]), category=int(categories[t]),
                correct=bool(choices[t] == categories[t]), log_advantage_original_nll=value,
                original_nll_probability=reference['probability'][:,t].mean(), alternative_probability=p[:,t].mean()))
        for repeat, diag in enumerate(group['diagnostics']):
            for t in range(len(choices)):
                state_rows.append(dict(subject=subject, candidate=candidate, letter=letters[subject,candidate], particles=particles,
                    repeat=repeat, trial=t+1, pre_ess_fraction=diag['diagnostic__pre_choice_ess'][t]/particles,
                    post_ess_fraction=diag['diagnostic__post_choice_ess'][t]/particles,
                    resampled=bool(diag['diagnostic__resampled'][t]),
                    active_count=diag['state__active_probability'][t].sum(),
                    participation=diag['latent__effective_participation'][t],
                    search_probability=diag['latent__predictive_swap_probability'][t]))
    frame = pd.DataFrame(rows)
    rankings = []
    for (subject, particles), data in frame.groupby(['subject','particles']):
        best = data.sort_values(['mean_nll','candidate']).iloc[0]
        reference = groups[subject,best.candidate,particles]
        for row in data.itertuples():
            delta, se = paired_difference(groups[subject,row.candidate,particles]['probability'], reference['probability'])
            rankings.append(dict(subject=subject, particles=particles, candidate=row.candidate, letter=row.letter,
                best_candidate=best.candidate, best_letter=best.letter, delta_nll=delta, paired_mc_se=se,
                within_2_mc_se=delta <= 2*se, selected=row.candidate == best.candidate))
    for (subject,candidate), data in frame.groupby(['subject','candidate']):
        budgets = sorted(data.particles)
        for lower, upper in zip(budgets[:-1],budgets[1:]):
            a, b = groups[subject,candidate,upper], groups[subject,candidate,lower]
            assert a['seeds'] == b['seeds']
            delta, se = paired_difference(a['probability'],b['probability'])
            drift_rows.append(dict(subject=subject,candidate=candidate,letter=letters[subject,candidate],
                lower_particles=lower,upper_particles=upper,delta_nll=delta,paired_mc_se=se))
    trial = pd.DataFrame(trial_rows)
    for name, data in [('scores',frame),('seed_scores',pd.DataFrame(seed_rows)),('contrasts',pd.DataFrame(contrasts)),
        ('rankings',pd.DataFrame(rankings)),('particle_drift',pd.DataFrame(drift_rows)),('trial_contributions',trial),
        ('filter_diagnostics',pd.DataFrame(state_rows)),('inactive_parameters',inactive_parameter_channels(frozen)),
        ('block_contributions',trial.groupby(['subject','candidate','letter','particles','block'],as_index=False).agg(
            log_advantage=('log_advantage_original_nll','sum'),trials=('trial','size'))),
        ('correct_error_contributions',trial.groupby(['subject','candidate','letter','particles','correct'],as_index=False).agg(
            log_advantage=('log_advantage_original_nll','sum'),trials=('trial','size')))]:
        data.to_csv(output/(name+'.csv'),index=False)
    return dict(subjects=sorted(map(int,frame.subject.unique())),groups=len(groups),filter_runs=sum(len(g['seeds']) for g in groups.values()))


def catalog_worker(protocol_file: Path, output: Path) -> None:
    """Runs inside archived imports; current geometry helper is hash pinned."""
    import pandas as pd
    from src.Bayesian_state.model import ModelContext, StateModel
    protocol = load(protocol_file)
    for name,sha in protocol['source_sha256'].items():
        if digest(REPO/name)!=sha:
            raise ValueError('Frozen diagnostic source changed')
    path = REPO/'src/Bayesian_state/workflows/analysis/model_0923_parameter_precision.py'
    spec = importlib.util.spec_from_file_location('precision_workflow',path)
    workflow = importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
    v = workflow.validation(); interventions = v.intervention_workflow(); interventions.install_transition()
    diagnosis = interventions.cohort().load_diagnosis()
    for subject in protocol['settings']['replay_subjects']:
        folder = Path(protocol['data'])/f'subject_{subject}'
        meta = load(folder/'prepared.json');engine = load(folder/'engine.json')
        assert digest(folder/'engine.json') == meta['engine_sha256']
        assert digest(folder/'training.npz') == meta['array_sha256']['training.npz']
        with np.load(folder/'training.npz') as saved:
            arrays = SimpleNamespace(stimulus=saved['stimulus'],categories=saved['categories'])
        inputs = dict(engine=engine,condition=meta['condition'],subject=subject,paths=meta['paths'],arrays=arrays)
        model = StateModel(engine,context=ModelContext(condition=meta['condition'],subject_id=subject,
            processed_data_dir=meta['paths']['processed_dir'],dataset_paths=meta['paths']))
        catalog = pd.DataFrame([dict(rule=rule.index,family=rule.family,
            parameters=json.dumps(dict(rule.parameters),default=str,sort_keys=True)) for rule in model.partition_model.hypothesis_space])
        diagnosis.geometric_catalog(inputs,catalog,5.).to_csv(output/f'subject_{subject}_rule_catalog.csv',index=False)


def summarize_replay(groups: dict, protocol: dict, output: Path) -> dict:
    """All path/state aggregation; physical accuracy >= .9 defines useful rules."""
    jobs = [job for job in protocol['jobs'] if job['kind']=='replay']
    if not jobs:
        return {}
    summaries, path_rows, rule_rows, trajectory_rows, conditional_rows = [], [], [], [], []
    inputs = {}
    for subject, model in sorted({(j['subject'],j['model']) for j in jobs}):
        selected = [j for j in jobs if j['subject']==subject and j['model']==model]
        catalog = pd.read_csv(output/f'subject_{subject}_rule_catalog.csv')
        useful = catalog.loc[catalog.physical_argmax_accuracy >= .9,'rule'].to_numpy(int)
        geometry = catalog.set_index('rule').physical_argmax_accuracy.to_dict()
        n_rules = len(catalog);trials = selected[0]['training_trials'];candidate = selected[0]['candidate']['id']
        sums = {key:np.zeros((trials,n_rules)) for key in ('active','prior','contribution','active_beta','contribution_beta')}
        scalar = {key:[] for key in ('correct','predicted_correct','useful_count','useful_mass',
            'useful_beta_numerator','useful_correct_numerator','beta','participation','proposed')}
        folder = Path(selected[0]['folder'])
        with np.load(folder/'training.npz') as saved:
            observed = saved['choices']==saved['categories']
        last = min(128,trials);total = 0;seen = []
        for job in sorted(selected,key=lambda j:j['indices'][0]):
            target = Path(job['target']);row = load(target.with_suffix('.json'))
            assert row['exact_choice_replay'] and digest(target.with_suffix('.npz'))==row['predictions_sha256']
            inputs[str(target.with_suffix('.npz'))]=row['predictions_sha256']
            with np.load(target.with_suffix('.npz')) as saved:
                a = {key:saved[key] for key in saved.files}
            np.testing.assert_array_equal(a['seeds'],job['seeds'])
            seen.extend(job['indices']);total += len(job['seeds'])
            rules=a['rule'];active=rules>=0;time=np.broadcast_to(np.arange(trials)[:,None,None],rules.shape)
            for key in sums:
                values=(np.ones(rules.shape) if key=='active' else a['beta_pre'] if key=='active_beta'
                        else a['contribution']*a['beta_pre'] if key=='contribution_beta' else a[key])
                np.add.at(sums[key],(time[active],rules[active]),values[active])
            useful_mask=np.isin(rules,useful)&active
            useful_count=useful_mask.sum(axis=-1)
            mass=np.where(useful_mask,a['contribution'],0.).sum(axis=-1)
            useful_beta=np.where(useful_mask,a['contribution']*a['beta_pre'],0.).sum(axis=-1)
            useful_correct=np.where(useful_mask,a['contribution']*a['rule_correct'],0.).sum(axis=-1)
            correct=a['choice']==a['categories'][:,None]
            predicted=np.take_along_axis(a['prediction'],a['categories'][:,None,None]-1,axis=2)[...,0]
            beta=np.nansum(a['contribution']*a['beta_pre'],axis=-1)
            participation=1./np.nansum(a['contribution']**2,axis=-1)
            proposed=a['proposed']>=0
            for key,value in dict(correct=correct,predicted_correct=predicted,useful_count=useful_count,
                useful_mass=mass,useful_beta_numerator=useful_beta,useful_correct_numerator=useful_correct,
                beta=beta,participation=participation,proposed=proposed).items():
                scalar[key].append(value)
            for i,seed in enumerate(a['seeds']):
                present=useful_count[:,i]>0
                path_rows.append(dict(subject=subject,model=model,candidate=candidate,seed=int(seed),
                    initial_rule=int(rules[0,i,0]),initial_physical_accuracy=float(geometry[int(rules[0,i,0])]),
                    initially_useful=bool(present[0]),ever_useful=bool(present.any()),
                    first_useful_trial=int(np.flatnonzero(present)[0]+1) if present.any() else -1,
                    late_useful_presence=float(present[-last:].mean()),useful_at_end=bool(present[-1]),
                    late_accuracy=float(correct[-last:,i].mean()),observed_late_accuracy=float(observed[-last:].mean()),
                    reaches_observed_late_accuracy=bool(correct[-last:,i].mean()>=observed[-last:].mean()),
                    late_useful_mass=float(mass[-last:,i].mean()),proposals=int(proposed[:,i].sum()),
                    useful_autonomous_exits=int(np.isin(a['exits'][:,i],useful).sum()),
                    useful_capacity_drops=int(np.isin(a['capacity_dropped'][:,i],useful).sum())))
        assert seen==list(range(total)) and total==1024
        scalar={key:np.concatenate(values,axis=1) for key,values in scalar.items()}
        for t in range(trials):
            trajectory_rows.append(dict(subject=subject,model=model,trial=t+1,
                **{key:float(value[t].mean()) for key,value in scalar.items()}))
            for rule in range(n_rules):
                rule_rows.append(dict(subject=subject,model=model,trial=t+1,rule=rule,
                    **{key:float(value[t,rule]/total) for key,value in sums.items()}))
        own=pd.DataFrame([r for r in path_rows if r['subject']==subject and r['model']==model])
        summaries.append(dict(subject=subject,model=model,candidate=candidate,paths=total,useful_rules=useful.tolist(),
            initial_useful=float(own.initially_useful.mean()),ever_useful=float(own.ever_useful.mean()),
            late_accuracy=float(own.late_accuracy.mean()),observed_late_accuracy=float(observed[-last:].mean()),
            reaches_observed_late_accuracy=float(own.reaches_observed_late_accuracy.mean()),
            mean_proposals=float(own.proposals.mean()),mean_useful_exits=float(own.useful_autonomous_exits.mean()),
            mean_useful_capacity_drops=float(own.useful_capacity_drops.mean()),
            useful_at_end=float(own.useful_at_end.mean()),
            useful_weighted_beta=float(scalar['useful_beta_numerator'].sum()/scalar['useful_mass'].sum()),
            useful_weighted_correct_probability=float(scalar['useful_correct_numerator'].sum()/scalar['useful_mass'].sum())))
        for key,g in groups.items():
            if key[:2] != (subject,candidate):
                continue
            for repeat,diag in enumerate(g['diagnostics']):
                for t in range(trials):
                    conditional_rows.append(dict(subject=subject,model=model,particles=key[2],repeat=repeat,trial=t+1,
                        predicted_correct=g['correct'][repeat,t],
                        useful_count=float(diag['state__active_probability'][t,useful].sum()),
                        useful_mass=float(diag['state__choice_weights'][t,useful].sum()),
                        filtered_useful_mass=float(diag['state__filtered_choice_weights'][t,useful].sum()),
                        participation=float(diag['latent__effective_participation'][t])))
    pd.DataFrame(path_rows).to_csv(output/'autonomous_paths.csv',index=False)
    pd.DataFrame(rule_rows).to_csv(output/'autonomous_rule_states.csv',index=False)
    pd.DataFrame(trajectory_rows).to_csv(output/'autonomous_state_trajectories.csv',index=False)
    pd.DataFrame(conditional_rows).to_csv(output/'conditional_state_trajectories.csv',index=False)
    write_json(output/'autonomous_summary.json',dict(rows=summaries,inputs_sha256=inputs,
        useful_rule_definition='Physical-stimulus argmax accuracy >= 0.9 on training stimulus schedule at beta=5; analysis only.',
        late_window=128,paths_per_point=1024))
    return dict(replay_paths=len(path_rows),replay_inputs_sha256=inputs)


def summarize(runs: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new summary directory')
    groups,inputs,protocols=read_filter_groups(runs)
    output.mkdir(parents=True);shutil.copy2(__file__,output/'analysis_source.py')
    summary=summarize_filters(groups,protocols,output)
    replay_protocols=[(path,p) for path,p in protocols.items() if any(j['kind']=='replay' for j in p['jobs'])]
    if len(replay_protocols)>1:
        raise ValueError('Only one complete autonomous replay run may be summarized')
    if replay_protocols:
        path,protocol=replay_protocols[0]
        runtime=protocol['jobs'][0]['runtime']
        launcher='import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory(prefix='precision_catalog_cache_') as cache:
            env['NUMBA_CACHE_DIR']=cache
            subprocess.run([sys.executable,'-c',launcher,runtime,str(Path(__file__).resolve()),'--catalog-worker',
                            str((Path(path)/'protocol.json').resolve()),'--output',str(output.resolve())],env=env,check=True,cwd=REPO)
        summary.update(summarize_replay(groups,protocol,output))
    write_json(output/'completion.json',dict(status='complete',**summary,inputs_sha256=inputs,
        source_sha256=digest(Path(__file__)),source_runs=[str(p.resolve()) for p in runs],test_choices_used=False))
    print(json.dumps(summary,ensure_ascii=False,default=str)[:1500])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--catalog-worker',type=Path)
    args=parser.parse_args()
    if args.catalog_worker:
        catalog_worker(args.catalog_worker,args.output)
    elif args.runs:
        summarize(args.runs,args.output)
    else:
        parser.error('--runs is required')


if __name__=='__main__':
    main()
