"""Describe states and finite-candidate sensitivity without pooling parameters.

Run after the bounded controller completes. ``--allow-incomplete`` produces a
separately marked preview from existing arrays, with no completed-study claims.
All exported probabilities refer to after-choice, before-feedback states.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .model_1024_state_interpretation import MEASURES, ROOT, read, write, digest, measure_checks, verify
from .model_1024_individual_state_summary import TARGETS
from ...evaluation.independent_particle_systems import evidence_weights


def calculation_record(source: Path) -> dict | None:
    """Prefer an append-only completion record; distinguish a live repair."""
    final=source/'final_calculation_completion.json'
    if final.exists():
        return read(final)
    if (source/'completion_repair_v1'/'registration.json').exists():
        return None
    original=source/'calculation_completion.json'
    return read(original) if original.exists() else None


def conditional_predictions(probabilities: np.ndarray, choices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool pre-choice predictions using only each system's preceding evidence."""
    if probabilities.ndim!=3 or probabilities.shape[1]!=len(choices):
        raise ValueError('Unaligned independent choice predictions')
    if not np.isfinite(probabilities).all() or np.any(probabilities<0):
        raise ValueError('Invalid choice predictions')
    np.testing.assert_allclose(probabilities.sum(axis=2),1.,atol=3e-12,rtol=0)
    if np.any(choices<1) or np.any(choices>probabilities.shape[2]):
        raise ValueError('Choice outside category support')
    observed=probabilities[:,np.arange(len(choices)),choices.astype(int)-1]
    if np.any(observed<=0):
        raise ValueError('Impossible observed choice')
    logz=np.log(observed).cumsum(axis=1)
    prior=np.column_stack([np.zeros(len(probabilities)),logz[:,:-1]])
    weights=evidence_weights(prior)
    pooled=np.einsum('rt,rtc->tc',weights,probabilities)
    # The sequential mixture must reconstruct the mean sequence evidence.
    joined=np.log(pooled[np.arange(len(choices)),choices.astype(int)-1]).sum()
    expected=np.logaddexp.reduce(logz[:,-1])-np.log(len(logz))
    np.testing.assert_allclose(joined,expected,atol=2e-9,rtol=0)
    return pooled,logz


def load_arrays(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k.startswith(('filtered_', 'full_'))
                and ('ancest' not in k and 'indices' not in k and 'weights' not in k)}


def behavior(case: dict) -> dict:
    pieces = []
    for name in ('training.npz', 'sealed_test.npz'):
        with np.load(Path(case['template']['folder'])/name) as z:
            pieces.append({k: z[k] for k in ('choices', 'categories', 'feedback')})
    result = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    if len(result['choices']) != case['trials']:
        raise ValueError('Behavior/state trial count differs')
    result['correct'] = (result['choices']==result['categories']).astype(float)
    return result


def block_slices(trials: int, width: int) -> list[tuple[int, int]]:
    if trials < 1 or width < 1:
        raise ValueError('Positive record and block lengths required')
    return [(start, min(start+width, trials)) for start in range(0, trials, width)]


def probability_invariants(a: dict, case: dict) -> dict:
    n = case['trials']; targets = case['target_rules']
    for mode in ('filtered', 'full'):
        for field in MEASURES:
            value = a[mode+'_'+field]
            if len(value)!=n or not np.isfinite(value).all() or value.min() < -1e-10 or value.max() > 1+1e-10:
                raise ValueError('Invalid probability/mean state')
        present = a[mode+'_target_present']
        for field in ('target_belief', 'target_high_belief', 'target_execution'):
            if np.any(a[mode+'_'+field] > present+1e-10):
                raise ValueError('Target subset probability exceeds presence')
        for field in ('belief', 'execution', 'source'):
            np.testing.assert_allclose(a[mode+'_'+field].sum(axis=1), 1., atol=2e-10, rtol=0)
        for field in ('belief', 'execution'):
            np.testing.assert_allclose(a[mode+'_'+field][:, targets].sum(axis=1),
                                       a[mode+'_target_'+field], atol=2e-10, rtol=0)
        np.testing.assert_allclose(a[mode+'_source'][:, 0], 1-present, atol=2e-10, rtol=0)
        np.testing.assert_allclose(a[mode+'_source'][:, 2], a[mode+'_target_execution'], atol=2e-10, rtol=0)
    return dict(case=case['id'], trials=n, status='passed')


def candidate_difference(values: np.ndarray, distribution: bool) -> np.ndarray | None:
    """Return the largest pairwise difference at each trial; singleton unknown."""
    if len(values) < 2:
        return None
    if distribution:
        return np.max(np.stack([.5*np.abs(values[i]-values[j]).sum(axis=1)
                                for i, j in combinations(range(len(values)), 2)]), axis=0)
    return np.ptp(values, axis=0)


def sensitivity_status(candidates: int, all_numerically_usable: bool, mean: float,
                       q95: float, policy: dict, retained: bool = True) -> str:
    if not retained:
        return 'diagnostic_parameters_only'
    if candidates < 2:
        return 'single_candidate'
    if not all_numerically_usable:
        return 'numerically_limited'
    limits = policy['interpretation']
    return ('close_within_checked_set' if mean <= limits['candidate_mean_difference']
            and q95 <= limits['candidate_q95_difference'] else 'parameter_sensitive')


def run(source: Path, output: Path, allow_incomplete: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    verify(source)
    completion = calculation_record(source)
    complete = completion is not None
    if not complete and not allow_incomplete:
        raise ValueError('Calculation not complete; preview needs --allow-incomplete')
    output.mkdir(parents=True)
    started = time.time(); cpu = time.process_time()
    protocol = read(source/'protocol.json'); p = protocol['policy']
    registration = read(source/'interpretation_registration.json')
    cases = protocol['cases']; old = ROOT/p['source_summary']
    old_selected = {r['case']: r for r in read(old/'selected.json') if 'file' in r}
    old_rows = pd.read_csv(old/'numerical_checks.csv')
    candidates = dict(registration['anchors'])
    latest = None
    if complete:
        latest = Path(completion['latest_summary'])
    else:
        for label in ('focus_followup', 'initial'):
            if (source/'summaries'/label/'diagnostics.json').exists():
                latest = source/'summaries'/label
                break
    inputs = {str(old/'numerical_checks.csv'): digest(old/'numerical_checks.csv')}
    if latest:
        inputs[str(latest/'diagnostics.json')] = digest(latest/'diagnostics.json')
        for r in read(latest/'diagnostics.json'):
            cid = r['case']
            if r['confirmation']:
                continue
            if cid not in candidates or r['particles'] > candidates[cid]['particles']:
                candidates[cid] = r
    selected, numerical, whole, blocks, invariants, behavior_rows, rule_blocks = [], [], [], [], [], [], []
    prediction_rows=[]
    arrays = {}; checks = {}
    for c in cases:
        cid = c['id']; number = 1+sum(x['subject']==c['subject'] for x in selected)
        inherited = cid not in p['pilot_cases']
        if cid in candidates:
            r = candidates[cid]
            file = Path(r['pooled_file'])
            if digest(file)!=r['pooled_sha256']:
                raise ValueError('Selected pooled states changed')
        else:
            r = old_rows.loc[old_rows.case.eq(cid) & old_rows.bank.eq('resolution')].iloc[0].to_dict()
            for key, value in list(r.items()):
                if '_resolution_' in key:
                    r[key.replace('_resolution_', '_comparison_')] = value
            r['comparison_particles'] = 512
            file = Path(old_selected[cid]['file']); inherited = True
        inputs[str(file)] = digest(file)
        a = load_arrays(file); arrays[cid] = a
        invariants.append(probability_invariants(a, c))
        info = dict(case=cid, subject=c['subject'], candidate=c['candidate']['id'],
            number=number, parameter_status=c['parameter_status'],
            particles=int(r['particles']), systems=int(r['systems']), inherited=inherited,
            file=str(file), sha256=inputs[str(file)], comparison_particles=int(r['comparison_particles']),
            job_names=r.get('job_names',old_selected[cid]['jobs']))
        selected.append(info)
        obs = behavior(c); n = c['trials']; slices = block_slices(n, p['interpretation']['block_trials'])
        predictions=[]
        for name in info['job_names']:
            matches=[folder/'jobs'/(name+'.json') for folder in (source,ROOT/p['precision_source'],ROOT/p['source'])
                     if (folder/'jobs'/(name+'.json')).exists()]
            if len(matches)!=1:
                raise ValueError('Ambiguous or missing selected particle-system job')
            job=read(matches[0]);score=Path(job['target']).with_suffix('.npz')
            inputs[str(score)]=digest(score)
            with np.load(score) as z:
                np.testing.assert_array_equal(z['choices'],obs['choices'])
                np.testing.assert_array_equal(z['categories'],obs['categories'])
                probability=z['probabilities'][0]
                np.testing.assert_allclose(probability[np.arange(n),obs['choices'].astype(int)-1],z['observed_probability'][0],atol=1e-12,rtol=0)
                np.testing.assert_allclose(probability[np.arange(n),obs['categories'].astype(int)-1],z['correct_probability'][0],atol=1e-12,rtol=0)
                predictions.append(probability)
        predicted,logz=conditional_predictions(np.stack(predictions),obs['choices'])
        with np.load(file) as saved:
            # Original diagnostic banks can store an equivalent seed ordering;
            # compare sorted system evidence trajectories before aggregation.
            old_logz=saved['system_log_evidence']
            np.testing.assert_allclose(logz[np.argsort(logz[:,-1])],old_logz[np.argsort(old_logz[:,-1])],atol=2e-9,rtol=0)
        prediction_rows.extend(dict(case=cid,subject=c['subject'],number=number,trial=t+1,
            probability_correct=float(predicted[t,int(obs['categories'][t])-1]),
            probability_observed=float(predicted[t,int(obs['choices'][t])-1]),
            observed_correct=float(obs['correct'][t])) for t in range(n))
        if number==1:
            behavior_rows.extend(dict(subject=c['subject'], trial=t+1,
                choice=int(obs['choices'][t]), category=int(obs['categories'][t]),
                correct=float(obs['correct'][t]), feedback=float(obs['feedback'][t])) for t in range(n))
        for mode in ('filtered', 'full'):
            for field in MEASURES:
                chk = measure_checks(r, p, mode, field)
                checks[cid, mode, field] = chk['usable']
                numerical.append(dict(**{k:v for k,v in info.items() if k not in ('file','sha256')},
                    mode=mode, measure=field, **chk,
                    repeat_mean=float(r[mode+'_half_'+field+'_mean']),
                    repeat_q95=float(r[mode+'_half_'+field+'_q95']),
                    resolution_mean=float(r[mode+'_comparison_'+field+'_mean']),
                    resolution_q95=float(r[mode+'_comparison_'+field+'_q95']),
                    system_ess_min=float(r[mode+'_ess_min']),
                    maximum_system_weight=float(r[mode+'_maximum_weight'])))
            for field in TARGETS:
                value = a[mode+'_'+field]
                whole.append(dict(case=cid, subject=c['subject'], number=number, mode=mode, measure=field,
                    mean=float(value.mean()), first_block=float(value[slice(*slices[0])].mean()),
                    last_block=float(value[slice(*slices[-1])].mean()),
                    change_last_minus_first=float(value[slice(*slices[-1])].mean()-value[slice(*slices[0])].mean()),
                    numerically_usable=checks[cid,mode,field]))
                for block, (lo, hi) in enumerate(slices, 1):
                    # The pointwise half-bank distance averaged over a block
                    # bounds the distance between its two block means.
                    half = a[mode+'_half_max_'+field][lo:hi]-a[mode+'_half_min_'+field][lo:hi]
                    blocks.append(dict(case=cid, subject=c['subject'], number=number, mode=mode, measure=field,
                        block=block, first_trial=lo+1, last_trial=hi, trials=hi-lo,
                        state_mean=float(value[lo:hi].mean()), half_block_difference_bound=float(half.mean()),
                        correct_fraction=float(obs['correct'][lo:hi].mean()),
                        feedback_mean=float(obs['feedback'][lo:hi].mean()),
                        numerically_usable=checks[cid,mode,field]))
            if c['subject']==p['interpretation']['focus_subject']:
                for field in ('belief', 'execution'):
                    for block, (lo, hi) in enumerate(slices, 1):
                        mean = a[mode+'_'+field][lo:hi].mean(axis=0)
                        rule_blocks.extend(dict(case=cid, number=number, mode=mode, measure=field,
                            block=block, first_trial=lo+1, last_trial=hi, rule=i, probability=float(v),
                            target=i in c['target_rules']) for i,v in enumerate(mean))
    sensitivity, pairs, range_blocks = [], [], []
    for sid in sorted({c['subject'] for c in cases}):
        own = [c for c in cases if c['subject']==sid]; n = own[0]['trials']
        retained = all(c['parameter_status']=='retained' for c in own)
        for mode in ('filtered', 'full'):
            for field in MEASURES:
                values = np.stack([arrays[c['id']][mode+'_'+field] for c in own])
                difference = candidate_difference(values, field not in TARGETS)
                mean = float(difference.mean()) if difference is not None else np.nan
                q95 = float(np.quantile(difference,.95)) if difference is not None else np.nan
                usable = all(checks[c['id'],mode,field] for c in own)
                sensitivity.append(dict(subject=sid, mode=mode, measure=field, candidates=len(own),
                    numerically_usable_candidates=sum(checks[c['id'],mode,field] for c in own),
                    mean_difference=mean, q95_difference=q95,
                    status=sensitivity_status(len(own),usable,mean,q95,p,retained)))
                for i,j in combinations(range(len(own)), 2):
                    delta = candidate_difference(values[[i,j]],field not in TARGETS)
                    pairs.append(dict(subject=sid, mode=mode, measure=field, case_a=own[i]['id'], case_b=own[j]['id'],
                        mean_difference=float(delta.mean()),q95_difference=float(np.quantile(delta,.95)),
                        both_numerically_usable=checks[own[i]['id'],mode,field] and checks[own[j]['id'],mode,field]))
                if field in TARGETS:
                    for block,(lo,hi) in enumerate(block_slices(n,p['interpretation']['block_trials']),1):
                        means = values[:,lo:hi].mean(axis=1)
                        range_blocks.append(dict(subject=sid,mode=mode,measure=field,block=block,
                            first_trial=lo+1,last_trial=hi,minimum=float(means.min()),maximum=float(means.max()),
                            candidates=len(own),numerically_usable=usable,parameter_status='retained' if retained else 'diagnostic'))
    for name,rows in [('selected',selected),('invariants',invariants)]:
        write(output/(name+'.json'),rows)
    for name,rows in [('numerical_checks',numerical),('whole_record',whole),('blocks',blocks),
                      ('behavior',behavior_rows),('parameter_sensitivity',sensitivity),('parameter_pairs',pairs),
                      ('parameter_block_ranges',range_blocks),('S205_rule_blocks',rule_blocks)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    pd.DataFrame(prediction_rows).to_csv(output/'conditional_choice_predictions.csv',index=False)
    verify(source)
    if any(digest(Path(k))!=v for k,v in inputs.items()):
        raise ValueError('Analysis input changed during execution')
    write(output/'manifest.json',dict(source=str(source),preview=not complete,inputs=inputs,
        source_sha256=digest(Path(__file__)),protocol_sha256=digest(source/'protocol.json'),
        selection='highest available non-confirmation particle count; retain all candidates regardless of numerical status; historical diagnostic fallback',
        cases=len(cases),retained=sum(c['parameter_status']=='retained' for c in cases),
        parameter_pooling='none',oral_used=False,created_unix=time.time(),
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.time()-started,
        artifacts={f.name:digest(f) for f in output.iterdir() if f.is_file()}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();run(args.source,args.output,args.allow_incomplete)


if __name__=='__main__':
    main()
