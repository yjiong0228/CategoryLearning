"""Replay frozen admission fits and export report-independent decision bases.

Run with --config CONFIG --output NEW_DIRECTORY. --smoke uses the first subject,
arm and saved seed, sixteen trials and one worker. Original particles and seeds
are retained to check every replayed choice probability against saved results.
Workers import the archived cognitive runtime and the existing experimental
adapters; the current shared PF adds observational capture only.
The optional source_kind=scope_confirmation selects one frozen prefix/full
nominee per arm from an estimation-scope comparison. Both scopes retain their
retrospective development label; oral reports never select the parameters.
source_kind=capacity_memory_scores selects the lowest choice NLL in a completed
capacity/retention grid. The selected point and score-table hash are recorded;
the original point can bind to its reused evaluation seeds.
source_kind=reentry_precision_scores replays one declared fixed parameter point
after its retained-precision choice evaluation completes. It preserves that
structural intervention and verifies exact behavioral predictions before oral use.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_replay.py'
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
COLLECTOR = 'src/Bayesian_state/evaluation/oral/decision_basis.py'
REFIT = 'src/Bayesian_state/workflows/analysis/model_0923_episodic_refit.py'
PROPOSAL = 'src/Bayesian_state/model/modules/hypothesis_transition/episodic_proposal.py'
REENTRY = 'src/Bayesian_state/model/modules/hypothesis_transition/reentry_precision.py'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources(job: dict) -> None:
    for name, expected in {**job['source_sha256'], **job['replay_source_sha256']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Executing source changed: {name}')
    if digest(Path(job['runtime']) / PF) != job['archived_pf_sha256']:
        raise ValueError('Archived particle filter changed')
    if digest(Path(job['reference'])) != job['reference_sha256']:
        raise ValueError('Original probability reference changed')
    if digest(Path(job['original_job'])) != job['original_job_sha256']:
        raise ValueError('Original replay job changed')


def grid_source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    """Select a completed grid point using only its recorded choice score."""
    import numpy as np
    import pandas as pd
    done, protocol = read(source/'completion.json'), read(source/'protocol.json')
    objective = settings.get('estimation_objective')
    if (objective not in ('prefix', 'full') or done['status'] != 'complete'
            or done['smoke'] != protocol['smoke'] or (done['smoke'] and not smoke)
            or settings['subjects'] != [done['subject']] or settings['arms'] != ['SEARCH']):
        raise ValueError('Grid replay requires one SEARCH subject and a prefix/full choice objective')
    score_file = source/'scores.csv'
    if digest(score_file) != done['artifacts_sha256']['scores.csv']:
        raise ValueError('Grid selection scores changed')
    scores = pd.read_csv(score_file)
    plans = {plan['candidate']['id']: plan for plan in protocol['plans']}
    selected = scores[scores.phase.eq(objective)].copy()
    if (len(plans) != len(protocol['plans']) or len(plans) != done['points']
            or selected.point.duplicated().any() or set(selected.point) != set(plans)
            or selected.anchor.sum() != 1
            or not selected.subject.eq(done['subject']).all() or not selected.model.eq('SEARCH').all()
            or not np.isfinite(selected.choice_nll.to_numpy()).all()
            or not selected.particles.eq(protocol['particles']).all()
            or not selected.seeds.eq(len(protocol['seeds'])).all()):
        raise ValueError('Incomplete grid choice scores or inconsistent numerical budget')
    selected = selected.sort_values(['choice_nll', 'point'])
    winner = selected.iloc[0]
    plan = plans[winner.point]
    expected = plan['job']
    seeds = protocol['seeds']
    if (len(set(seeds)) != len(seeds) or len(seeds) != settings['seeds_per_arm']
            or expected['seeds'] != seeds or expected['particles'] != protocol['particles']
            or any(expected['candidate'][key] != plan['candidate'][key] for key in ('id','hyperparams'))):
        raise ValueError('Grid replay must retain every independent filter seed')
    if plan['reused']:
        group = sorted((Path(protocol['reference'])/'jobs').glob(
            f'evaluate_S{done["subject"]}_SEARCH_{winner.point}_C*.json'))
    else:
        group = [source/'jobs'/(chunk['name']+'.json') for chunk in plan['chunks']]
        if len(group) != len(set(group)):
            raise ValueError('Duplicate grid job references')
        for file, chunk in zip(group, plan['chunks']):
            if read(file) != chunk:
                raise ValueError('Grid scoring job differs from its saved plan')
    if len(group) != len(seeds):
        raise ValueError('Incomplete grid filter job bank')
    actual = []
    for file in group:
        job = read(file)
        if (job['phase'] != 'evaluate' or job['model'] != 'SEARCH' or job['name'] != file.stem
                or len(job['seeds']) != 1 or job['candidate']['id'] != winner.point
                or job['candidate']['hyperparams'] != plan['candidate']['hyperparams']
                or any(job[key] != expected[key] for key in ('particles','folder','runtime','mechanism','arm_settings'))
                or read(Path(job['folder'])/'prepared.json')['subject'] != done['subject']
                or any(expected['source_sha256'].get(name) != value for name,value in job['source_sha256'].items())):
            raise ValueError('Grid selected point differs from its saved filter jobs')
        actual.extend(job['seeds'])
    if actual != seeds:
        raise ValueError('Grid filter seed identities or ordering changed')
    metadata = dict(parameter_estimation_scope=objective, evaluation_use='development_retrospective_comparison',
        parameter_candidate_policy='fixed_capacity_memory_grid', selected_parameter_id=str(winner.point),
        selection_score=float(winner.choice_nll), selection_score_sha256=digest(score_file),
        selection_source_protocol_sha256=digest(source/'protocol.json'),
        selection_rule='minimum pooled choice NLL; parameter id breaks exact ties; no oral input',
        selection_candidates=selected[['point','choice_nll','choice_nll_mc_se']].to_dict('records'),
        selected_original_point=bool(winner.anchor), reused_reference_filters=bool(plan['reused']))
    return group[:1] if smoke else group, metadata


def reentry_source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    """Bind one predeclared retained-precision point without outcome selection."""
    done, protocol = read(source/'completion.json'), read(source/'protocol.json')
    point = settings.get('parameter_point')
    if (done['status'] != 'complete' or done['smoke'] != protocol['smoke']
            or (done['smoke'] and not smoke) or settings['subjects'] != [protocol['settings']['subject']]
            or settings['arms'] != ['SEARCH'] or point not in protocol['settings']['points']):
        raise ValueError('Precision replay requires a completed declared SEARCH parameter point')
    plans = [p for p in protocol['plans'] if p['point'] == point and p['variant'] == 'RETAIN']
    records = [r for r in done['records'] if r['point'] == point and r['variant'] == 'RETAIN']
    if len(plans) != 1 or len(records) != 1:
        raise ValueError('Require one completed retained-precision point')
    plan, result = plans[0], records[0]
    expected, record = plan['job'], result['record']
    seeds = protocol['seeds']
    if (len(set(seeds)) != len(seeds) or len(seeds) != settings['seeds_per_arm']
            or done['filter_seeds'] != len(seeds) or done['particles'] != protocol['particles']
            or expected['seeds'] != seeds or expected['retain_precision'] is not True
            or expected['particles'] != protocol['particles'] or expected['model'] != 'SEARCH'
            or expected['candidate']['id'] != point or expected['source_sha256'] != protocol['source_sha256']
            or any(record[key] != expected[key] for key in ('candidate', 'seeds', 'particles', 'model'))
            or Path(result['target']) != Path(expected['target']).with_suffix('.npz')
            or digest(Path(result['target'])) != record['predictions_sha256']):
        raise ValueError('Precision point identity, completion or numerical budget changed')
    if protocol['source_sha256'].get(REENTRY) != digest(ROOT/REENTRY):
        raise ValueError('Retained-precision mechanism changed after choice scoring')
    group = [source/'jobs'/(chunk['name']+'.json') for chunk in plan['chunks']]
    if len(group) != len(seeds) or len(set(group)) != len(group):
        raise ValueError('Incomplete precision filter job bank')
    actual = []
    for file, chunk in zip(group, plan['chunks']):
        job = read(file)
        if (job != chunk or job['name'] != file.stem or job['phase'] != 'evaluate'
                or job['model'] != 'SEARCH' or job['retain_precision'] is not True
                or len(job['seeds']) != 1 or job['particles'] != protocol['particles']
                or any(job[key] != expected[key] for key in
                       ('candidate', 'folder', 'runtime', 'mechanism', 'arm_settings', 'source_sha256'))
                or read(Path(job['folder'])/'prepared.json')['subject'] != settings['subjects'][0]):
            raise ValueError('Precision replay differs from its completed choice job')
        actual.extend(job['seeds'])
    if actual != seeds:
        raise ValueError('Precision replay seed identities or ordering changed')
    metadata = dict(parameter_estimation_scope='existing_behavior_parameters',
        evaluation_use='fixed_parameter_mechanism_diagnostic',
        parameter_candidate_policy='declared_point_no_new_selection', selected_parameter_id=point,
        structural_intervention='retained_rule_precision',
        precision_source_protocol_sha256=digest(source/'protocol.json'),
        precision_source_completion_sha256=digest(source/'completion.json'))
    return group[:1] if smoke else group, metadata


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    """Bind saved seeds to the declared selection scope before replaying states."""
    done = read(source / 'completion.json')
    if done['status'] != 'complete' or (done.get('smoke', False) and not smoke):
        raise ValueError('Need a completed evaluation; smoke sources require --smoke')
    kind = settings.get('source_kind', 'evaluation')
    if kind not in ('evaluation', 'scope_confirmation', 'capacity_memory_scores', 'reentry_precision_scores'):
        raise ValueError('Unknown state-replay source_kind')
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    arms = settings['arms'][:1] if smoke else settings['arms']
    if (not subjects or not arms or len(set(settings['subjects'])) != len(settings['subjects'])
            or len(set(settings['arms'])) != len(settings['arms']) or settings['seeds_per_arm'] < 1):
        raise ValueError('Need unique subjects, arms and a positive saved-seed count')
    if kind == 'capacity_memory_scores':
        return grid_source_jobs(settings, source, smoke)
    if kind == 'reentry_precision_scores':
        return reentry_source_jobs(settings, source, smoke)
    metadata = dict(parameter_estimation_scope='original_evaluation_protocol',
                    evaluation_use='original_evaluation_protocol')
    selected = {}
    if kind == 'scope_confirmation':
        objective = settings.get('estimation_objective')
        if objective not in ('prefix', 'full'):
            raise ValueError('Scope confirmation needs estimation_objective prefix or full')
        frozen = read(source / 'nominees.json')
        original = read(source / 'protocol.json')
        if (frozen['status'] != 'frozen_for_retrospective_diagnostics'
                or frozen['smoke'] != done['smoke'] or original['smoke'] != done['smoke']
                or done['nomination_sha256'] != digest(source / 'nominees.json')
                or original['source_sha256'] != frozen['source_sha256']):
            raise ValueError('Scope nomination status, provenance or checksum changed')
        for row in frozen['nominees']:
            if row['objective'] == objective:
                key = row['subject'], row['model']
                if key in selected:
                    raise ValueError('Duplicate scope nominee')
                selected[key] = row['candidate']
        if not {(s, a) for s in subjects for a in arms} <= selected.keys():
            raise ValueError('Missing requested scope nominee')
        metadata.update(parameter_estimation_scope=objective,
                        evaluation_use='development_retrospective_comparison',
                        scope_nomination_sha256=digest(source / 'nominees.json'),
                        source_protocol_sha256=digest(source / 'protocol.json'))
    files = []
    for subject in subjects:
        for arm in arms:
            point = selected.get((subject, arm))
            pattern = (f'confirmation_S{subject}_{arm}_{point["id"]}_C*.json' if point is not None
                       else f'evaluate_S{subject}_{arm}_*_C*.json')
            group = sorted((source / 'jobs').glob(pattern))
            if len(group) != settings['seeds_per_arm']:
                raise ValueError(f'Expected all saved independent seeds for {subject} {arm}')
            seeds = []
            for file in group:
                job = read(file)
                if (job['phase'] != 'evaluate' or job['model'] != arm or job['name'] != file.stem
                        or len(job['seeds']) != 1
                        or read(Path(job['folder']) / 'prepared.json')['subject'] != subject):
                    raise ValueError('Saved job has a different identity or seed grouping')
                seeds.extend(job['seeds'])
                if point is not None:
                    if (any(job['candidate'][key] != point[key] for key in ('id', 'hyperparams'))
                            or job['particles'] != point['particles']
                            or job['source_sha256'] != frozen['source_sha256']
                            or job['arm_settings'] != original['settings']['arms'][arm]
                            or job['mechanism'] != original['settings']['mechanism']
                            or Path(job['folder']).resolve() != (Path(frozen['data']) / f'subject_{subject}').resolve()):
                        raise ValueError('Saved confirmation differs from its frozen scope nominee')
            if len(set(seeds)) != len(seeds) or (point is not None and sorted(seeds) != sorted(point['seeds'])):
                raise ValueError('Missing, repeated or changed confirmation seeds')
            files.extend(group[:1] if smoke else group)
    return files, metadata


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    job = read(file)
    verify_sources(job)
    refit = load('decision_basis_refit', REFIT)
    v = refit.base.validation()
    workflow = v.intervention_workflow()
    workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', refit.base.FEEDBACK)
    precision = refit.base.load('src.Bayesian_state.model.modules.precision_interventions', refit.base.PRECISION)
    guard = refit.base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', refit.GUARD)
    history = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', refit.HELPER)
    capture = load('src.Bayesian_state.inference.backends._decision_basis_capture', PF)
    collect = load('src.Bayesian_state.evaluation.oral._decision_basis_capture', COLLECTOR)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
    from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    folder = Path(job['folder'])
    meta = read(folder / 'prepared.json')
    if digest(folder / 'engine.json') != meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    for name, expected in meta['frozen_cognitive_source_sha256'].items():
        if digest(Path(job['runtime']) / name) != expected:
            raise ValueError(f'Archived cognitive source changed: {name}')
    engine = read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        if digest(folder / name) != meta['array_sha256'][name]:
            raise ValueError('Prepared behavior changed')
        with np.load(folder / name) as saved:
            parts.append({key: saved[key] for key in ('stimulus', 'choices', 'feedback')})
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    n = min(job['trial_limit'], len(arrays['choices'])) if job['trial_limit'] else len(arrays['choices'])
    arrays = {key: value[:n] for key, value in arrays.items()}
    space = (build_continuous_hypothesis_space(4, 2) if meta['condition'] == 1 else build_relational_0923_space())
    with np.load(job['reference']) as saved:
        expected = saved['probabilities'][0, :n].copy()
        np.testing.assert_array_equal(saved['seeds'], job['seeds'])
        np.testing.assert_array_equal(saved['choices'][:n], arrays['choices'])
    if len(job['seeds']) != 1:
        raise ValueError('One independent filter seed per replay job')
    collector = collect.DecisionBasisCollector(space, arrays['choices'])
    start, cpu = time.monotonic(), time.process_time()
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        feedback_fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshot_fixes = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['arm_settings']
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'], upper=arm['upper']))
        recall_fixes = []
        if job['model'] != 'BASE':
            mode = 'POOL' if job['model'] == 'SEARCH' else job['model']
            recall_fixes = stack.enter_context(history.episodic_admission(cls, mode=mode))
        if job['model'] == 'SEARCH':
            proposal = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal', PROPOSAL)
            stack.enter_context(proposal.episodic_proposal(cls, proposal=True))
        retained_events = []
        if job.get('retain_precision', False):
            if job['model'] != 'SEARCH' or job['retain_precision'] is not True:
                raise ValueError('Retained precision requires an explicit SEARCH intervention')
            retained = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision', REENTRY)
            retained_events = stack.enter_context(retained.reentry_precision(cls))
        result = capture.run_state_model_particle_filter(
            engine_config=engine, subject_id=meta['subject'], condition=meta['condition'],
            stimulus=arrays['stimulus'], choices=arrays['choices'], feedback=arrays['feedback'],
            particle_count=job['particles'], particle_blocks=1, filter_seed=job['seeds'][0],
            choice_readout_power=1., processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'],
            state_observer=collector, observe_decision_basis=True)
    predictions = result.observation_probabilities['prior_t']
    np.testing.assert_array_equal(predictions, expected, err_msg='Saved behavioral replay differs')
    recorded = collector.arrays()
    # The observer sums weighted particle predictions; the filter additionally
    # normalizes that vector. Their difference is floating-point roundoff only.
    aggregation_roundoff = float(np.max(np.abs(recorded['prediction'] - predictions)))
    np.testing.assert_allclose(recorded['prediction'], predictions, atol=1e-14, rtol=0)
    recorded['prediction'] = predictions
    np.testing.assert_allclose(recorded['choice_component_joint_mass'].sum(axis=(1, 2)),
                               expected[np.arange(n), arrays['choices'] - 1], atol=1e-12, rtol=0)
    verify_sources(job)
    intervention_metadata = {}
    if job.get('retain_precision', False):
        intervention_metadata = dict(structural_intervention='retained_rule_precision',
            particle_transitions=len(retained_events),
            particle_precision_restores=sum(len(e['precision_restored_rules']) for e in retained_events))
    target = Path(job['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **recorded, seeds=job['seeds'])
    write(target.with_suffix('.json'), dict(
        status='complete', subject=meta['subject'], condition=meta['condition'], arm=job['model'],
        particles=job['particles'], seeds=job['seeds'], trials=n, training_trials=meta['training_trials'],
        candidate=job['candidate'], exact_behavior_replay=True, aggregation_roundoff_max=aggregation_roundoff,
        catalogue_version=space.version,
        catalogue_signature=space.signature, component_policy='nearest_internal_stimulus_uniform_ties',
        timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input',
        parameter_estimation_scope=job.get('parameter_estimation_scope', 'original_evaluation_protocol'),
        evaluation_use=job.get('evaluation_use', 'original_evaluation_protocol'),
        seconds=time.monotonic() - start, cpu_seconds=time.process_time() - cpu,
        numeric_pools=threadpool_info(), array_sha256=digest(target.with_suffix('.npz')),
        feedback_corrections=feedback_fixes, snapshot_corrections=snapshot_fixes, recall_corrections=recall_fixes,
        **intervention_metadata))


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, list[dict]]:
    import yaml
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError('Use a new decision-basis output directory')
    if settings['sample_role'] != 'development' or settings['workers'] < 1:
        raise ValueError('Expected a development protocol with positive worker budget')
    source = ROOT / settings['source']
    files, scope_metadata = source_jobs(settings, source, smoke)
    hashes = {name: digest(ROOT / name) for name in (ENTRY, PF, COLLECTOR, REFIT)}
    if 'SEARCH' in settings['arms']:
        hashes[PROPOSAL] = digest(ROOT / PROPOSAL)
    if settings.get('source_kind') == 'reentry_precision_scores':
        hashes[REENTRY] = digest(ROOT / REENTRY)
    jobs = []
    for file in files:
        job = read(file)
        reference = Path(job['target']).with_suffix('.npz')
        record = read(reference.with_suffix('.json'))
        if (digest(reference) != record['predictions_sha256']
                or any(record[key] != job[key] for key in ('candidate', 'seeds', 'particles', 'model'))):
            raise ValueError('Original evaluation array or parameter identity changed')
        job.update(reference=str(reference), reference_sha256=digest(reference),
                   original_job=str(file.resolve()), original_job_sha256=digest(file),
                   replay_source_sha256=hashes, archived_pf_sha256=digest(Path(job['runtime']) / PF),
                   trial_limit=16 if smoke else 0, **scope_metadata,
                   target=str((output / 'states' / job['name']).resolve()))
        verify_sources(job)
        jobs.append(job)
    output.mkdir(parents=True)
    (output / 'jobs').mkdir()
    for relative in hashes:
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(config, output / 'protocol.yaml')
    workers = min(1 if smoke else settings['workers'], len(os.sched_getaffinity(0)), len(jobs))
    protocol = dict(settings=settings, smoke=smoke, workers=workers, jobs=len(jobs),
                    source_sha256=hashes, config_sha256=digest(config),
                    python=sys.version,
                    versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
                    original_completion_sha256=digest(source / 'completion.json'),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    role='fixed_parameter_replay_no_oral_input', **scope_metadata)
    write(output / 'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool) -> None:
    protocol, jobs = prepare(config, output, smoke)
    print(f"{len(jobs)} saved filter seeds, {protocol['workers']} workers, one numeric thread", flush=True)
    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json')
        write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='decision_basis_worker_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                                         '--worker', str(file.resolve())], cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Replay worker failed: {file.with_suffix('.log')}")
        record = read(Path(job['target']).with_suffix('.json'))
        if digest(Path(job['target']).with_suffix('.npz')) != record['array_sha256']:
            raise ValueError('Replay artifact changed')
        return record
    records = []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            records.append(future.result())
            print(f"{len(records)}/{len(jobs)} exact replays complete", flush=True)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(records),
          exact_behavior_replays=sum(r['exact_behavior_replay'] for r in records),
          cpu_seconds=sum(r['cpu_seconds'] for r in records),
          records=[dict(subject=r['subject'], arm=r['arm'], seed=r['seeds'][0], trials=r['trials']) for r in records]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Supply --config and --output')


if __name__ == '__main__':
    main()
