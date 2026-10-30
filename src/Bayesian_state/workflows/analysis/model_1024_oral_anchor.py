"""Reweight archived coherent paths with one oral report per 64-trial block.

Use --phase prepare --config YAML --output NEW, then --phase run --output DIR,
then --phase summarize --output DIR. The optional --worker JOB reads one
archived independent filter. No new cognitive simulation is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_oral_development_mpl')
import numpy as np
import pandas as pd
import yaml

from ...evaluation.oral.development import (
    subset_log_probability, token_truth, anchored_weights, condition_component_distribution)
from ...evaluation.oral.partial_reason_alignment import component_geometry, predicate_coverage
from ...evaluation.particle_smoothing import ancestral_weights
from ...inference.backends.particle_filter import _trace_ancestral_indices
from .model_0923_oral_constraint_audit import catalogue
from .model_1024_oral_readout import atom_vocabulary


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def prepare(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new anchor study output')
    start = time.time()
    p = yaml.safe_load(config.read_text())
    inputs = {str(config): digest(config)}

    def bind(path: Path) -> Path:
        inputs[str(path)] = digest(path)
        return path

    folder = Path(p['diagnosis'])
    trials = pd.read_csv(bind(folder/'trials.csv'))
    selected = json.loads(bind(Path(p['state_analysis'])/'selected.json').read_text())
    readout_selected = pd.read_csv(bind(Path(p['readout'])/'selected.csv'))
    vocabulary = atom_vocabulary()
    lookup = {a:i for i,a in enumerate(vocabulary)}
    output.mkdir(parents=True)
    (output/'jobs').mkdir()
    (output/'systems').mkdir()
    all_anchors, jobs = [], []
    for sid in p['subjects']:
        frame = trials[trials.subject.eq(sid)].reset_index(drop=True)
        chosen = next(s for s in selected if s['subject'] == sid and s['number'] == p['candidate_number'])
        with np.load(bind(folder/f'S{sid}_diagnostic_arrays.npz')) as z:
            x, choices = z['stimulus'], z['choice']
            support, reference = z['values'][..., 0], z['stimulus_reference_mass']
        n, h, m = support.shape
        tokens = [tuple(json.loads(a)) for a in frame.tokens]
        eligible = frame.eligible.to_numpy() & np.array([all(a in lookup for a in ts) for ts in tokens])
        anchors = []
        for block, start_trial in enumerate(range(0, n, p['block_trials'])):
            end_trial = min(n, start_trial+p['block_trials'])
            pool = np.flatnonzero(eligible & frame.split.eq('calibration').to_numpy() &
                (np.arange(n) >= start_trial) & (np.arange(n) < end_trial))
            target = start_trial + p['anchor_target_position'] - 1
            anchor = int(pool[np.argmin(np.abs(pool-target))]) if len(pool) else -1
            anchors.append(anchor)
            all_anchors.append(dict(subject=sid, block=block+1, start=start_trial+1, end=end_trial,
                anchor=anchor+1 if anchor >= 0 else None, text=frame.text.iloc[anchor] if anchor >= 0 else '',
                selection='nearest_to_position_16_among_calibration_complete_numeric',
                assessment_reports=int((eligible[start_trial:end_trial] &
                    frame.split.iloc[start_trial:end_trial].eq('assessment').to_numpy()).sum())))
        frequency_file = bind(Path(p['readout'])/f'S{sid}_early_middle_frequency.json')
        frequency = np.asarray(json.loads(frequency_file.read_text())['frequency'])
        chosen_bg = readout_selected[readout_selected.subject.eq(sid) &
            readout_selected.selection_window.eq('early_middle') & readout_selected.branch.eq('expression_background')].iloc[0]
        stimulus_temperature = float(chosen_bg.stimulus_temperature)
        space = catalogue(2 if h == 29 else 4)
        semantics = {}
        for choice in np.unique(choices):
            entries = component_geometry(space, int(choice))[0]
            score = np.zeros((len(entries), len(vocabulary)))
            for j, (_, _, A, b) in enumerate(entries):
                for k, atom in enumerate(vocabulary[:-1]):
                    verdict = predicate_coverage((atom,), A, b)
                    score[j, k] = verdict[0]-verdict[1]
            semantics[choice] = (entries, score)
        temperatures = p['emission']['component_temperatures']
        emission = np.ones((len(temperatures), n, h, m))
        background = np.full(n, np.nan)
        for t in np.flatnonzero(eligible):
            ids = [lookup[a] for a in tokens[t]]
            truth = np.r_[token_truth(vocabulary[:-1], x[t]), 0.]
            weights = frequency * np.exp(stimulus_temperature*truth)
            background[t] = np.exp(subset_log_probability(weights[None, :], ids))[0]
            entries, semantic = semantics[choices[t]]
            for k, temperature in enumerate(temperatures):
                q = np.exp(subset_log_probability(weights[None, :]*np.exp(temperature*semantic), ids))
                q = (1-p['emission']['background_fraction'])*q + p['emission']['background_fraction']*background[t]
                for j, (hh, mm, *_geometry) in enumerate(entries):
                    emission[k, t, hh, mm] = q[j]
        fixture = output/f'S{sid}_fixture.npz'
        np.savez_compressed(fixture, emission=emission, background=background, eligible=eligible, choices=choices,
            support=support, reference=reference, anchors=anchors, temperatures=temperatures)
        # Resolve only the explicitly selected independent systems.
        for index, name in enumerate(chosen['job_names']):
            paths = [Path(p['state_source'])/'jobs'/f'{name}.json', Path(p['alternate_state_source'])/'jobs'/f'{name}.json']
            found = [path for path in paths if path.is_file()]
            if len(found) != 1:
                raise ValueError('Ambiguous or missing archived system job: '+name)
            original = json.loads(bind(found[0]).read_text())
            state = Path(original['individual_target']).with_suffix('.npz')
            basic = Path(original['state_capture_target']).with_suffix('.npz')
            meta = json.loads(bind(state.with_suffix('.json')).read_text())
            if digest(state) != meta['array_sha256']:
                raise ValueError('Archived state hash changed')
            bind(state); bind(basic)
            job = dict(subject=sid, system=index, name=name, particles=chosen['particles'],
                execution_mode=original['execution_mode'],
                execution_inertia=original['candidate']['hyperparams'][
                    'engine.modules.hypo_transitions_mod.kwargs.execution_inertia'],
                fixture=str(fixture.resolve()), fixture_sha256=digest(fixture),
                state=str(state), state_sha256=inputs[str(state)], basic=str(basic), basic_sha256=inputs[str(basic)],
                target=str((output/'systems'/name).resolve()), block_trials=p['block_trials'],
                deadline_unix=p['budget']['deadline_unix'], source_sha256=digest(Path(__file__)))
            write(output/'jobs'/f'{name}.json', job)
            jobs.append(job)
        print(f'S{sid}: {len(anchors)} oral anchors fixed; {len(chosen["job_names"])} archived systems', flush=True)
    pd.DataFrame(all_anchors).to_csv(output/'anchors.csv', index=False)
    write(output/'registration.json', dict(policy=p, started_unix=start, registered_unix=time.time(), inputs_sha256=inputs,
        jobs=jobs, vocabulary=vocabulary, source_sha256=digest(Path(__file__)),
        state_conditioning='behavior through local block end; optional one oral anchor in that block'))


def worker(file: str | Path) -> dict:
    start, cpu = time.time(), time.process_time()
    job = json.loads(Path(file).read_text())
    if digest(Path(__file__)) != job['source_sha256']:
        raise ValueError('Worker source changed')
    for key in ('fixture', 'state', 'basic'):
        if digest(Path(job[key])) != job[key+'_sha256']:
            raise ValueError('Worker input changed: '+key)
    with np.load(job['fixture']) as z:
        emission, anchors = z['emission'], z['anchors']
        fixture_choices = z['choices']
        n, h, m = z['support'].shape
    with np.load(job['basic']) as z:
        prediction, choices = z['prediction'], z['choices'].astype(int)
    np.testing.assert_array_equal(choices, fixture_choices)
    log_evidence = np.cumsum(np.log(prediction[np.arange(n), choices-1]))
    with np.load(job['state']) as z:
        ids = z['particle_rule']
        component = z['particle_component']
        filtered_weights, parents = z['filtered_weights'], z['parents']
    q = component.sum(-1)
    np.testing.assert_allclose(q.sum(-1), 1., atol=2e-12, rtol=0)
    one_hot = bool(np.all(np.sum(q > 1e-10, axis=-1) == 1))
    if job['execution_mode'] == 'STICKY':
        if not one_hot:
            raise ValueError('Persistent execution must retain sampled identities')
    elif job['execution_mode'] != 'MIX' or job['execution_inertia'] != 0:
        raise ValueError('Only zero-inertia execution may be marginalized')
    # At zero inertia, execution does not affect later workspace or learning.
    # Integrate its local report likelihood inside each particle. At the oral
    # anchor itself, also condition that particle's execution/component mass;
    # weighting particles alone would miss within-particle information.
    del q
    k = len(emission)
    means = np.zeros((k+1, n, h, m))
    block_evidence = np.zeros((k+1, len(anchors)))
    terminal_ess, anchor_ess = np.zeros_like(block_evidence), np.zeros_like(block_evidence)
    for block, start_trial in enumerate(range(0, n, job['block_trials'])):
        if time.time() > job['deadline_unix']:
            raise TimeoutError('Reached the registered reporting reserve')
        end = min(n, start_trial+job['block_trials'])
        indices = _trace_ancestral_indices(parents[start_trial:end])
        w = filtered_weights[end-1]
        weights = [w]
        anchor_local = []
        anchor = int(anchors[block])
        block_evidence[0, block] = log_evidence[end-1]
        for variant in range(k):
            likelihood = np.ones(len(w))
            if anchor >= 0:
                ai = indices[:, anchor-start_trial]
                local, particle_likelihood = condition_component_distribution(component[anchor],
                    emission[variant, anchor, np.maximum(ids[anchor], 0)])
                likelihood = particle_likelihood[ai]
                anchor_local.append(local)
            else:
                anchor_local.append(None)
            updated, scale = anchored_weights(w, likelihood)
            weights.append(updated)
            block_evidence[variant+1, block] = log_evidence[end-1] + np.log(scale)
        for variant, weight in enumerate(weights):
            ancestry = ancestral_weights(indices, weight)
            terminal_ess[variant, block] = 1/np.square(weight).sum()
            anchor_ess[variant, block] = ancestry['ancestral_ess'][max(anchor-start_trial, 0)]
            for offset, t in enumerate(range(start_trial, end)):
                aw = ancestry['weights'][offset]
                current = anchor_local[variant-1] if variant > 0 and t == anchor else component[t]
                valid = ids[t] >= 0
                for mm in range(m):
                    means[variant, t, :, mm] = np.bincount(ids[t][valid],
                        weights=(aw[:, None]*current[..., mm])[valid], minlength=h)
        np.testing.assert_allclose(means[:, start_trial:end].sum((-1, -2)), 1., atol=3e-12, rtol=0)
    target = Path(job['target'])
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, component=means, log_evidence=block_evidence,
            terminal_ess=terminal_ess, anchor_ancestral_ess=anchor_ess)
    result = dict(status='complete', subject=job['subject'], system=job['system'],
        cpu_seconds=time.process_time()-cpu, wall_seconds=time.time()-start,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        one_hot_execution=one_hot, execution_mode=job['execution_mode'],
        execution_inertia=job['execution_inertia'], local_anchor_execution_conditioned=True,
        archive_sha256=job['state_sha256'],
        output_sha256=digest(target.with_suffix('.npz')))
    write(target.with_suffix('.json'), result)
    return result


def run(output: Path) -> None:
    reg = json.loads((output/'registration.json').read_text())
    jobs = [output/'jobs'/f'{j["name"]}.json' for j in reg['jobs']]
    workers = min(reg['policy']['budget']['max_workers'], len(os.sched_getaffinity(0)), len(jobs))
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, f):f for f in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f'{len(results)}/{len(jobs)} systems complete; S{result["subject"]} system {result["system"]}', flush=True)
    cpu_seconds = sum(r['cpu_seconds'] for r in results)
    if cpu_seconds > 3600*reg['policy']['budget']['cpu_hours']:
        raise RuntimeError('Archive-analysis CPU budget was exceeded')
    write(output/'completion.json', dict(status='complete', systems=len(results), workers=workers,
        wall_seconds=time.time()-started, worker_cpu_seconds=cpu_seconds))


def summarize(output: Path) -> None:
    reg = json.loads((output/'registration.json').read_text())
    p = reg['policy']
    products = [output/name for name in ('trial_scores.csv.gz', 'numerical_checks.csv',
        'summary.csv', 'summary_validation.json')]
    for sid in p['subjects']:
        labels = ['behavior'] + [f'oral_temperature_{t:g}' for t in p['emission']['component_temperatures']]
        products.extend(output/f'S{sid}_{label}_pooled.npz' for label in labels)
    if any(path.exists() for path in products):
        raise FileExistsError('Keep existing summaries; use a new archive-study directory')
    trials = pd.read_csv(Path(p['diagnosis'])/'trials.csv')
    records, checks, summaries = [], [], []
    for sid in p['subjects']:
        own = sorted([j for j in reg['jobs'] if j['subject'] == sid], key=lambda j:j['system'])
        frame = trials[trials.subject.eq(sid)].reset_index(drop=True)
        with np.load(output/f'S{sid}_fixture.npz') as z:
            fixture = {key:z[key] for key in z.files}
        n, h, m = fixture['support'].shape
        states, evidence, terminal, ancestral = [], [], [], []
        for j in own:
            target = Path(j['target'])
            meta = json.loads(target.with_suffix('.json').read_text())
            if digest(target.with_suffix('.npz')) != meta['output_sha256']:
                raise ValueError('Completed system summary changed')
            with np.load(target.with_suffix('.npz')) as z:
                states.append(z['component']); evidence.append(z['log_evidence'])
                terminal.append(z['terminal_ess']); ancestral.append(z['anchor_ancestral_ess'])
        states, evidence, terminal, ancestral = map(np.stack, (states, evidence, terminal, ancestral))
        banks = dict(all=np.arange(len(own)), first_half=np.arange(len(own)//2), second_half=np.arange(len(own)//2, len(own)))
        for bank, indices in banks.items():
            for variant in range(states.shape[1]):
                label = 'behavior' if variant == 0 else f'oral_temperature_{fixture["temperatures"][variant-1]:g}'
                logw = evidence[indices, variant]
                w = np.exp(logw-logw.max(0, keepdims=True)); w /= w.sum(0, keepdims=True)
                mass = np.zeros((n, h, m))
                for block, start_trial in enumerate(range(0, n, p['block_trials'])):
                    end = min(n, start_trial+p['block_trials'])
                    mass[start_trial:end] = np.einsum('r,rthm->thm', w[:, block], states[indices, variant, start_trial:end])
                    effective = 1/np.square(w[:, block]).sum()
                    anc = 1/np.sum(w[:, block]**2/ancestral[indices, variant, block])
                    ess = 1/np.sum(w[:, block]**2/terminal[indices, variant, block])
                    checks.append(dict(subject=sid, bank=bank, variant=label, block64=block+1,
                        system_ess=effective, maximum_system_weight=w[:, block].max(), anchor_ancestral_ess=anc,
                        terminal_ess=ess))
                np.testing.assert_allclose(mass.sum((1, 2)), 1., atol=5e-12, rtol=0)
                support = np.einsum('thm,thm->t', mass, fixture['support'])
                log_scores = np.log(np.einsum('thm,athm->at', mass, fixture['emission']))
                primary_index = list(fixture['temperatures']).index(p['emission']['primary_temperature'])
                # Compare every state variant under the same primary observation
                # model, and retain each matched sensitivity score separately.
                score = log_scores[primary_index]
                score[~fixture['eligible']] = np.nan
                if bank == 'all':
                    np.savez_compressed(output/f'S{sid}_{label}_pooled.npz', component=mass,
                        support=support, subset_log_probability=log_scores, system_weights=w)
                for t in range(n):
                    records.append(dict(subject=sid, trial=t+1, bank=bank, variant=label,
                        split=frame.split.iloc[t], early_middle=bool(frame.early_middle.iloc[t]),
                        block64=int(frame.block64.iloc[t]), eligible=bool(fixture['eligible'][t]),
                        support=support[t] if fixture['eligible'][t] else np.nan,
                        subset_log_probability=score[t],
                        background_log_probability=np.log(fixture['background'][t]),
                        correct_rule_mass=mass[t, 42].sum()))
                windows = [('all', np.ones(n, bool)), ('early_middle', frame.early_middle.to_numpy())]
                windows += [(f'block64_{i}', frame.block64.eq(i).to_numpy()) for i in frame.block64.unique()]
                for window, mask in windows:
                    for split in ('calibration', 'assessment'):
                        use = mask & fixture['eligible'] & frame.split.eq(split).to_numpy()
                        summaries.append(dict(subject=sid, bank=bank, variant=label, window=window, split=split,
                            reports=int(use.sum()), support=float(support[use].mean()) if use.any() else np.nan,
                            subset_log_probability=float(score[use].mean()) if use.any() else np.nan,
                            background_log_probability=float(np.log(fixture['background'][use]).mean()) if use.any() else np.nan))
    pd.DataFrame(records).to_csv(output/'trial_scores.csv.gz', index=False)
    pd.DataFrame(checks).to_csv(output/'numerical_checks.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/'summary.csv', index=False)
    write(output/'summary_validation.json', dict(status='complete', trial_rows=len(records), numerical_rows=len(checks),
        summary_rows=len(summaries), state_timing='after choice before feedback; conditioned on behavior through block end',
        oral_conditioning='one complete coded report in the current block',
        limitation='Finite archived path support; separate blockwise reconstructions; development readout and unreviewed transcripts'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'prepare':
        prepare(args.config, args.output)
    elif args.phase == 'run':
        run(args.output)
    elif args.phase == 'summarize':
        summarize(args.output)
    else:
        parser.error('Require --phase or --worker')
