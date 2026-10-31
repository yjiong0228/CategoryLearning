"""Bounded single-subject comparison with behavior-only selection and fixed budgets.

Prepare first, then run. Each completed job is immutable and can be reused on
resume after its input/source manifest is checked. Models use the shared core.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc
import scipy
import yaml

from ...model import ModelContext, StateModel
from ...model.model_1024 import model_1024_runtime
from ...model.modules.base_module import ModuleRole
from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...simulation.data import SubjectTrialDataLoader
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = dict(similarity_width=.25, capacity=4, baseline_probability=.15, error_gain=1.,
               failure_decay=.8, global_search=.2, retention=.8, execution_inertia=1.,
               beta_init=5., learning_rate=.1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def bank(policy: dict) -> list[dict]:
    """Frozen discrete spread; no behavior or oral outcomes enter proposals."""
    values = policy['parameter_values']
    keys = list(values)
    count = policy['search']['candidates']
    draws = qmc.Sobol(len(keys), scramble=True, seed=policy['seed']).random_base2(
        int(np.ceil(np.log2(max(count, 2)))))
    candidates = [dict(id='c00', parameters=DEFAULT.copy())]
    for point in draws:
        candidate = {k: values[k][min(int(v * len(values[k])), len(values[k]) - 1)] for k, v in zip(keys, point)}
        if any(candidate == old['parameters'] for old in candidates):
            continue
        candidates.append(dict(id=f'c{len(candidates):02}', parameters=candidate))
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise ValueError('Insufficient unique prespecified parameter candidates')
    return candidates


def engine_config(model: str, parameters: dict) -> dict:
    if model == 'model_1031':
        cfg = yaml.safe_load((ROOT / 'configs/exp123/model_struct/model_1031_cond1_experience_v1.yaml').read_text())
        cfg['partition']['kwargs'].update(capacity=int(parameters['capacity']), similarity_width=parameters['similarity_width'])
        h = cfg['modules']['hypo_transitions_mod']['kwargs']
        for key in ('baseline_probability', 'error_gain', 'failure_decay', 'global_search', 'execution_inertia'):
            h[key] = parameters[key]
        cfg['modules']['memory_mod']['kwargs']['gamma'] = parameters['retention']
    elif model == 'model_1024':
        cfg = yaml.safe_load((ROOT / 'configs/exp123/model_struct/model_0923_cond1_full_v3.yaml').read_text())
        cfg.pop('oral_readout', None)
        h = cfg['modules']['hypo_transitions_mod']['kwargs']
        h.update(max_capacity=int(parameters['capacity']), exit_rate=0., local_structure_weight=1.,
                 execution_inertia=parameters['execution_inertia'],
                 intervention=dict(kind='E', decay=parameters['retention'], capacity='posterior', search='accumulate'))
        for key in ('baseline_probability', 'error_gain', 'failure_decay', 'global_search'):
            h['search_controller'][key] = parameters[key]
        h['search_controller']['global_search_failure_gain'] = 0.
        cfg['modules']['memory_mod']['kwargs']['gamma'] = 1.
    else:
        raise ValueError('Unsupported pilot model')
    cfg['modules']['beta_mod']['kwargs'].update(beta_init=parameters['beta_init'], learning_rate=parameters['learning_rate'])
    return cfg


def active_mask_distribution(contents: dict, rule_masks: np.ndarray) -> np.ndarray:
    """Marginal feature-set use, after choice and before feedback; no oral input."""
    weights = contents['weights'][:, None] * contents['choice_weights']
    if 'representation_kind' in contents:
        kind, rule = contents['representation_kind'], contents['catalogue_rule']
        masks = contents['template_mask'].copy()
        is_rule = kind == 2
        masks[is_rule] = rule_masks[rule[is_rule]]
        masks[kind <= 0] = False
        codes = masks.astype(int) @ (2 ** np.arange(4))
    else:
        codes = np.broadcast_to(rule_masks.astype(int) @ (2 ** np.arange(4)), weights.shape)
    result = np.bincount(codes.ravel(), weights=weights.ravel(), minlength=16)
    if not np.isclose(result.sum(), 1.) or np.any(result < 0):
        raise ValueError('Invalid feature-set distribution')
    return result


def prepare(config: Path, output: Path) -> None:
    policy = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError(output)
    if policy['condition'] != 1:
        raise ValueError('This first pilot uses binary condition 1')
    cfg = engine_config('model_1031', DEFAULT)
    loader = SubjectTrialDataLoader(cfg)
    data_path = ROOT / policy['data']
    loader.prepare_data(data_path)
    frame = loader._get_subject_frame(policy['subject'], 1.)
    if loader._get_condition_value(frame) != policy['condition']:
        raise ValueError('Subject/condition mismatch')
    arrays = loader._extract_arrays(frame, max_trials=None)
    if not np.array_equal(arrays.feedback, (arrays.choices == arrays.categories).astype(float)):
        raise ValueError('Category/choice/feedback coordinates disagree')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'behavior_input.npz', stimulus=arrays.stimulus,
                        choices=arrays.choices, feedback=arrays.feedback, categories=arrays.categories)
    frame.to_csv(output / 'subject_source.csv', index=False)
    # Retain the inspected population screen without turning it into a cognitive label.
    population = loader.learning_data.query('condition == 1')
    rows = []
    for sid, own in population.groupby('iSub'):
        texts = own.text.fillna('').astype(str)
        counts = sum(texts.str.contains(pattern, regex=True).astype(int) for pattern in ('头', '脖|颈', '腿|脚', '尾'))
        rows.append(dict(subject=int(sid), trials=len(own), first64_four_explicit_parts=int((counts.iloc[:64] == 4).sum()),
                         last64_four_explicit_parts=int((counts.iloc[-64:] == 4).sum()),
                         early_mean_named_parts=float(counts.iloc[:64].mean()), late_mean_named_parts=float(counts.iloc[-64:].mean())))
    pd.DataFrame(rows).to_csv(output / 'selection_screen.csv', index=False)
    source_paths = list((ROOT / 'src/Bayesian_state/model').rglob('*.py'))
    source_paths += list((ROOT / 'src/Bayesian_state/hypothesis_space').rglob('*.py'))
    source_paths += list((ROOT / 'src/Bayesian_state/inference').rglob('*.py'))
    source_paths += [Path(__file__), ROOT / 'src/Bayesian_state/simulation/data.py', config.resolve(),
                     ROOT / 'configs/exp123/model_struct/model_1031_cond1_experience_v1.yaml',
                     ROOT / 'configs/exp123/model_struct/model_0923_cond1_full_v3.yaml']
    candidates = bank(policy)
    register = dict(policy=policy, candidates=candidates, trials=len(frame),
        source_git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        input_sha256=digest(output / 'behavior_input.npz'), data_sha256=digest(data_path),
        created_time=time.time(), versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        scope='full_record_behavior_only_selection; oral_selected_development_case; provisional_original_oral',
        selected_subject_reason='Long specific descriptions followed by tail-based reasons; frozen before fitting',
        figure_contract=dict(backend='python', archetype='quantitative grid', format='PNG',
            evidence=['behavior_and_autonomous_distributions', 'representation_groups', 'feature_use_and_raw_oral', 'parameter_and_numerical_sensitivity'],
            claim='Evaluate whether this fixed candidate can reproduce behavior and the descriptive shift in S118; findings determine the conclusion'))
    write_new(output / 'registration.json', register)
    for model in policy['models']:
        for point in candidates:
            path = output / 'configs' / f"{model}_{point['id']}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(engine_config(model, point['parameters']), sort_keys=False))


def _work(job: dict) -> dict:
    output = Path(job['output'])
    path = output / 'jobs' / (job['id'] + '.json')
    if path.exists():
        result = json.loads(path.read_text())
        if result['job'] != job:
            raise ValueError('Existing job has different inputs')
        return result
    started, cpu = time.perf_counter(), time.process_time()
    cfg = yaml.safe_load((output / 'configs' / f"{job['model']}_{job['candidate']}.yaml").read_text())
    with np.load(output / 'behavior_input.npz') as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    runtime = model_1024_runtime() if job['model'] == 'model_1024' else nullcontext()
    with runtime:
        saved = {}
        if job['kind'] == 'fit':
            keep = job['stage'] == 'final'
            catalogue = ContinuousPartition(4, 2) if keep else None
            rule_masks = None if not keep else np.asarray([
                np.any(np.asarray([a for a, b in spec.hyperplanes]) != 0, axis=0)
                for spec in catalogue.hypothesis_space])
            feature_use, contents_log = [], []
            def observer(trial, phase, contents):
                if phase != 'filtered':
                    return
                feature_use.append(active_mask_distribution(contents, rule_masks))
                if job['model'] == 'model_1031':
                    keys = ('weights', 'choice_weights', 'representation_kind', 'representation_uid',
                            'representation_parent', 'catalogue_rule', 'template_mask', 'template_anchor',
                            'template_category_belief', 'beta', 'perceived_stimulus')
                    contents_log.append({key: contents[key].copy() for key in keys})
            result = run_state_model_particle_filter(engine_config=cfg, subject_id=job['subject'], condition=1,
                stimulus=arrays['stimulus'], choices=arrays['choices'], feedback=arrays['feedback'],
                particle_count=job['particles'], choice_readout_power=1., filter_seed=job['seed'],
                state_observer=observer if keep else None, observe_decision_basis=keep)
            prediction = result.marginal_probabilities
            chosen = prediction[np.arange(len(prediction)), arrays['choices'] - 1]
            if not np.isfinite(prediction).all() or not np.allclose(prediction.sum(axis=1), 1.):
                raise ValueError('Invalid choice probabilities')
            nll = float(-np.log(np.maximum(chosen, np.finfo(float).tiny)).sum())
            saved.update(choice_probabilities=prediction, ess=result.post_choice_ess,
                         resampled=result.resampled)
            if keep:
                saved['feature_set_probability'] = np.asarray(feature_use)
                for key, value in result.state_probabilities.items():
                    if 'representation' in key:
                        saved[key] = value
                if contents_log:
                    for key in contents_log[0]:
                        saved['particle_' + key] = np.stack([item[key] for item in contents_log])
            metrics = dict(nll=nll, nll_per_trial=nll / len(prediction),
                           mean_ess=float(np.mean(result.post_choice_ess)))
        else:
            paths, groups, feature_masks, details = [], [], [], []
            for seed in job['seeds']:
                local = deepcopy(cfg)
                local['modules']['perception_mod'].setdefault('kwargs', {})['module_seed'] = seed
                local['modules']['hypo_transitions_mod']['kwargs']['module_seed'] = seed + 1
                model = StateModel(local, context=ModelContext(condition=1, subject_id=job['subject']))
                trajectory = model.generate_step_by_step(arrays['stimulus'],
                    lambda t, choice: float(choice == arrays['categories'][t]), choice_seed=seed + 2)
                paths.append(trajectory.feedback)
                if job['model'] == 'model_1031':
                    path_groups, path_masks = [], []
                    for step in trajectory.step_log:
                        slot = int(np.argmax(step['choice_weights']))
                        node = step['representations_pre'][slot]
                        if node['kind'] == 'rule':
                            path_groups.append(3)
                            spec = model.partition_model.catalogue.hypothesis_space[node['rule_id']]
                            mask = np.any(np.asarray([a for a, b in spec.hyperplanes]) != 0, axis=0)
                        elif node['kind'] == 'template':
                            mask = np.asarray(node['mask'])
                            path_groups.append(1 if sum(mask) == 4 else 2)
                        else:
                            mask = np.zeros(4, bool); path_groups.append(0)
                        path_masks.append(int(np.asarray(mask, int) @ (2 ** np.arange(4))))
                    groups.append(path_groups); feature_masks.append(path_masks)
                    details.append(dict(seed=seed, pre_choice=[s['representations_pre'] for s in trajectory.step_log]))
            saved.update(feedback=np.asarray(paths), seeds=np.asarray(job['seeds']))
            if groups:
                saved.update(representation_group=np.asarray(groups), feature_set=np.asarray(feature_masks))
                write_new(path.with_suffix('.paths.json'), details)
            metrics = dict(paths=len(paths), mean_correct=float(np.mean(paths)))
    path.parent.mkdir(parents=True, exist_ok=True)
    array_path = path.with_suffix('.npz')
    if array_path.exists():
        raise FileExistsError('Partial job arrays exist; preserve them and use a new job identity')
    np.savez_compressed(array_path, **saved)
    record = dict(job=job, **metrics, seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu,
                  array_sha256=digest(array_path))
    write_new(path, record)
    return record


def _cpu_used(pool) -> float:
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = used.ru_utime + used.ru_stime
    for process in pool._processes.values():
        try:
            fields = Path(f'/proc/{process.pid}/stat').read_text().split(') ', 1)[1].split()
            total += (float(fields[11]) + float(fields[12])) / os.sysconf('SC_CLK_TCK')
        except (OSError, IndexError):
            pass
    return total


def run_batch(jobs: list[dict], policy: dict, deadline: float, cpu_offset: float = 0.) -> list[dict]:
    workers = min(policy['budget']['workers'], len(os.sched_getaffinity(0)), len(jobs))
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))
    futures = {pool.submit(_work, job): job for job in jobs}
    results = []
    try:
        while futures:
            if time.monotonic() >= deadline or cpu_offset + _cpu_used(pool) >= policy['budget']['cpu_seconds']:
                raise TimeoutError('Registered wall or CPU budget reached')
            done, _ = wait(futures, timeout=1., return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                record = future.result()
                results.append(record)
                print(json.dumps(dict(completed=job['id'], seconds=round(record['seconds'], 2), remaining=len(futures))), flush=True)
    except BaseException:
        for process in pool._processes.values():
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results


def rank(records: list[dict]) -> list[dict]:
    rows = []
    for model in sorted({r['job']['model'] for r in records}):
        candidates = sorted({r['job']['candidate'] for r in records if r['job']['model'] == model})
        for candidate in candidates:
            values = [r['nll'] for r in records if r['job']['model'] == model and r['job']['candidate'] == candidate]
            rows.append(dict(model=model, candidate=candidate, mean_nll=float(np.mean(values)),
                             sd_nll=float(np.std(values, ddof=1)) if len(values) > 1 else 0., repeats=len(values)))
    return sorted(rows, key=lambda r: (r['model'], r['mean_nll']))


def run(output: Path) -> None:
    register = json.loads((output / 'registration.json').read_text())
    p, candidates = register['policy'], register['candidates']
    for name, expected in register['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    if digest(output / 'behavior_input.npz') != register['input_sha256']:
        raise ValueError('Behavior inputs changed')
    if (output / 'completion.json').exists():
        raise FileExistsError('Pilot already completed')
    started = time.monotonic()
    start_path = output / 'execution_started.json'
    if not start_path.exists():
        write_new(start_path, dict(unix_time=time.time()))
    elapsed = time.time() - json.loads(start_path.read_text())['unix_time']
    deadline = started + p['budget']['wall_seconds'] - elapsed
    cpu_offset = sum(json.loads(path.read_text())['cpu_seconds'] for path in (output / 'jobs').glob('*.json')
                     if not path.name.endswith('.paths.json'))
    active = {model: [c['id'] for c in candidates] for model in p['models']}
    records = []
    search = p['search']
    for stage in ('screen', 'confirm', 'final'):
        jobs = [dict(id=f'{stage}_{model}_{candidate}_s{seed}', kind='fit', stage=stage,
                output=str(output), model=model, candidate=candidate, subject=p['subject'],
                seed=seed, particles=search[stage + '_particles'])
                for model in p['models'] for candidate in active[model] for seed in search[stage + '_seeds']]
        stage_records = run_batch(jobs, p, deadline, cpu_offset)
        records.extend(stage_records)
        ranking = rank(stage_records)
        rank_path = output / f'{stage}_ranking.json'
        if not rank_path.exists():
            write_new(rank_path, ranking)
        elif json.loads(rank_path.read_text()) != ranking:
            raise ValueError('Saved ranking differs from frozen completed records')
        if stage != 'final':
            limit = search['confirm_top' if stage == 'screen' else 'final_top']
            active = {model: [r['candidate'] for r in ranking if r['model'] == model][:limit] for model in p['models']}
    jobs = []
    for model in p['models']:
        for candidate in active[model]:
            for start in range(0, search['generation_paths'], search['generation_chunk']):
                seeds = [search['generation_seed'] + 10 * i for i in range(start, min(start + search['generation_chunk'], search['generation_paths']))]
                jobs.append(dict(id=f'generate_{model}_{candidate}_{start:03}', kind='generate', output=str(output),
                                 model=model, candidate=candidate, subject=p['subject'], seeds=seeds))
    records.extend(run_batch(jobs, p, deadline, cpu_offset))
    write_new(output / 'completion.json', dict(status='registered_pilot_complete',
        jobs=len(records), cpu_seconds=sum(r['cpu_seconds'] for r in records),
        elapsed_seconds=time.monotonic() - started, finalists=active,
        state_scope='fixed_parameter_behavior_filtering; not_state_recovery_or_independent_oral_validation'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'run'))
    parser.add_argument('--config', type=Path, default=Path('configs/exp123/specific_models/model_1031_subject118_pilot_v1.yaml'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.config, args.output) if args.phase == 'prepare' else run(args.output)


if __name__ == '__main__':
    main()
