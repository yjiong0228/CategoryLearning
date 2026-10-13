"""Read-only adapters for the two saved Model 0923 fit layouts.

Process completion and scientific acceptance have separate fields. Evaluation
keeps the recorded representative point, including unresolved search results.
Scientific inputs must match their saved hashes. Changes to analysis/report
source are recorded explicitly; fit artifacts remain read-only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..optimization.adaptive_runtime import cache_path, phase_seeds, read_seed
from ..optimization.search.adaptive_proposals import point_id
from ..metrics import TrialPrediction, choice_nll
from ..simulation.config import load_yaml, resolve_engine_config
from ..simulation.data import SubjectTrialDataLoader
from ..simulation.parameters import apply_fixed_hyperparams_to_engine_config
from ..utils.datasets import resolve_dataset_paths
from ..utils.paths import ROOT_DIR
from ..utils.seeding import derive_trajectory_seed, stable_seed
from ..utils.subjects import resolve_subject_config


PARAMETER_DOMAINS = {
    'baseline_probability': (None, None), 'error_gain': (0., None),
    'failure_decay': (0., None), 'global_search': (0., 1.),
    'global_search_failure_gain': (0., 1.), 'max_capacity': (2, None),
    'exit_rate': (0., 1.), 'local_structure_weight': (0., 1.),
    'transfer_scale': (None, None), 'execution_inertia': (0., None),
    'gamma': (0., 1.), 'beta_init': (.1, 25.), 'learning_rate': (0., 1.),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_kind(fit: Path) -> str:
    """Identify a saved fit layout from its manifest fields."""
    if (fit / 'run_manifest.json').is_file():
        return 'initial_cd'
    if (fit / 'manifest.json').is_file():
        manifest = _json(fit / 'manifest.json')
        if manifest.get('config', {}).get('backend') == 'model0923_adaptive':
            return 'adaptive'
    raise ValueError(f'Unrecognized Model 0923 fit source: {fit}')


def _analysis_source(path: Path) -> bool:
    try:
        name = path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return False
    return name.startswith((
        'src/Bayesian_state/evaluation/',
        'src/Bayesian_state/workflows/analysis/',
        'src/Bayesian_state/workflows/reports/',
        'src/Bayesian_state/docs/model_architecture/figures/',
        'CategoryLearning_codes/figures/',
    ))


def _current_source_counterpart(path: Path) -> Path | None:
    """Locate executing source corresponding to a queued frozen source path."""
    if path.suffix not in ('.py', '.npy'):
        return None
    parts = path.parts
    for index in range(len(parts)-1, -1, -1):
        if parts[index] == 'src':
            return ROOT_DIR.joinpath(*parts[index:])
    return None


def _checked_hash(path: Path, expected: str, *, permit_analysis_changes: bool,
                  changes: list[dict[str, str]]) -> None:
    actual = _digest(path) if path.is_file() else None
    if actual == expected:
        return
    if permit_analysis_changes and _analysis_source(path):
        changes.append({'path': str(path.resolve()), 'saved_sha256': expected,
                        'current_sha256': actual or 'missing',
                        'reason': 'evaluation_or_report_source_changed'})
        return
    raise ValueError(f'Saved scientific input hash differs: {path}')


def _check_versions(saved: Mapping[str, str], python_version: str | None = None) -> None:
    # These packages enter the model's numerical execution and input loading.
    for name in ('numpy', 'scipy', 'numba', 'pandas', 'joblib', 'PyYAML', 'threadpoolctl'):
        if name in saved and importlib.metadata.version(name) != saved[name]:
            raise ValueError(f'Saved numerical dependency version differs: {name}')
    if python_version is not None and python_version != sys.version.split()[0]:
        raise ValueError('Saved Python version differs')


def _final_audit_inputs(fit: Path, result: Mapping[str, Any]) -> dict[str, str]:
    """Check final PF arrays against the fit's atomically published receipt."""
    subject = int(result['subject'])
    expected: dict[str, str] = {}
    inputs: dict[str, str] = {}
    for plan_path in (fit / 'batches').rglob('plan.json'):
        plan = _json(plan_path)
        if plan['family'] != result['audit_family'] or plan['budget'] != result['audit_budget'] or str(subject) not in plan['banks']:
            continue
        receipt_path = plan_path.with_name('scores.json')
        receipt = _json(receipt_path)
        expected.update(receipt['cache_sha256'])
        for path in (plan_path, receipt_path):
            inputs[str(path.resolve())] = _digest(path)
    config = _json(fit / 'manifest.json')['config']
    budget = result['audit_budget']
    seeds = phase_seeds(config['base_seed'], subject, result['audit_family'],
                        budget['filter_seed_count'], namespace=config['seed_namespace'])
    for row in result['candidate_bank']:
        for seed in seeds:
            path = cache_path(fit, subject, row['id'], budget['particle_count'], seed)
            relative = str(path.relative_to(fit))
            if relative not in expected:
                raise ValueError(f'Final audit cache lacks its saved receipt: {path}')
            _checked_hash(path, expected[relative], permit_analysis_changes=False, changes=[])
            inputs[str(path.resolve())] = expected[relative]
    return inputs


def validate_fit_source(fit: Path) -> dict[str, Any]:
    """Verify fit completion, selected records and immutable scientific inputs.

    The returned acceptance fields retain the fit's own status. A completed
    adaptive fit with status ``unresolved`` remains available for evaluation.
    """
    fit = fit.resolve()
    kind = source_kind(fit)
    changes: list[dict[str, str]] = []
    inputs: dict[str, str] = {}
    smoke_only = False
    if kind == 'initial_cd':
        manifest = _json(fit / 'run_manifest.json')
        smoke_only = bool(manifest.get('smoke_only', manifest.get('smoke', False)) or
                          load_yaml(fit / 'runtime/simulation.yaml').get('max_trials') is not None)
        if _json(fit / 'run_status.json').get('status') != 'completed':
            raise ValueError('The source fit process is not complete')
        for name, expected in manifest['snapshot_files_sha256'].items():
            _checked_hash(fit / 'source' / name, expected,
                          permit_analysis_changes=False, changes=changes)
            inputs[str((fit / 'source' / name).resolve())] = expected
            current = ROOT_DIR / name
            if name.startswith('src/') and Path(name).suffix in ('.py', '.npy'):
                _checked_hash(current, expected, permit_analysis_changes=True, changes=changes)
                if current.is_file():
                    inputs[str(current.resolve())] = _digest(current)
        for name, expected in manifest['runtime_files_sha256'].items():
            _checked_hash(fit / name, expected, permit_analysis_changes=False, changes=changes)
            inputs[str((fit / name).resolve())] = expected
        _check_versions(manifest.get('versions', {}), manifest.get('python'))
        subjects = [int(row['subject']) for row in manifest['subjects']]
        inputs[str((fit / 'run_status.json').resolve())] = _digest(fit / 'run_status.json')
        for sid in subjects:
            fit_subject_record(fit, sid)
            for name in ('best_hyperparams.json', 'restart_summary.json', 'final_rescore.jsonl'):
                artifact = fit / 'hyper_cd' / f'subject_{sid}' / name
                inputs[str(artifact.resolve())] = _digest(artifact)
        acceptance = {str(sid): 'initial_search_only' for sid in subjects}
        commit = manifest.get('git_commit')
    else:
        manifest = _json(fit / 'manifest.json')
        results_path = fit / 'fit_results.json'
        if not results_path.is_file():
            raise ValueError('The source fit process is not complete')
        results = _json(results_path)
        smoke_only = bool(results.get('smoke_only', manifest.get('smoke', False)))
        if results.get('backend') != 'model0923_adaptive':
            raise ValueError('Saved results do not describe Model 0923 adaptive fitting')
        subjects = [int(row['subject']) for row in manifest['subjects']]
        if set(results['subjects']) != {str(sid) for sid in subjects}:
            raise ValueError('Completed fit does not contain every planned subject')
        for sid in subjects:
            result_path = fit / 'subjects' / str(sid) / 'fit_result.json'
            result = _json(result_path)
            if result != results['subjects'][str(sid)]:
                raise ValueError(f'Saved subject and cohort fit records differ: S{sid}')
            if result['selected'] not in {row['id'] for row in result['candidate_bank']}:
                raise ValueError(f'Selected point is absent from the candidate bank: S{sid}')
            fit_subject_record(fit, sid)
            inputs[str(result_path.resolve())] = _digest(result_path)
            inputs.update(_final_audit_inputs(fit, result))
        inputs[str(results_path.resolve())] = _digest(results_path)
        support_path = fit / 'effective_parameter_support.json'
        inputs[str(support_path.resolve())] = _digest(support_path)
        for name, expected in manifest['input_sha256'].items():
            path = Path(name)
            current = _current_source_counterpart(path)
            frozen_source = current is not None and current.resolve() != path.resolve()
            _checked_hash(path, expected, permit_analysis_changes=not frozen_source, changes=changes)
            if path.is_file():
                inputs[str(path.resolve())] = _digest(path)
            if frozen_source:
                _checked_hash(current, expected, permit_analysis_changes=True, changes=changes)
                if current.is_file():
                    inputs[str(current.resolve())] = _digest(current)
        versions = manifest.get('versions', {})
        _check_versions(versions, versions['python'].split()[0] if versions.get('python') else None)
        acceptance = {str(sid): results['subjects'][str(sid)]['status'] for sid in subjects}
        commit = manifest.get('git_commit')
        queue_path = fit.parent / 'queue_plan.json'
        if queue_path.is_file():
            queue = _json(queue_path)
            if any(Path(job['output']).resolve() == fit for job in queue['jobs']):
                commit = queue.get('source_commit', commit)
                inputs[str(queue_path.resolve())] = _digest(queue_path)
    manifest_path = fit / ('run_manifest.json' if kind == 'initial_cd' else 'manifest.json')
    inputs[str(manifest_path.resolve())] = _digest(manifest_path)
    return dict(source_kind=kind, source_fit=str(fit), process_status='completed', smoke_only=smoke_only,
                subjects=subjects, subject_acceptance=acceptance,
                source_fit_commit=commit, git_commit=commit, evaluation_source_changes=changes,
                input_sha256=inputs,
                source_manifest_sha256=_digest(manifest_path))


def _rebase_snapshot(value: Any, old_root: Path, fit: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rebase_snapshot(item, old_root, fit) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase_snapshot(item, old_root, fit) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            return str(fit / Path(value).relative_to(old_root))
        except ValueError:
            pass
    return value


def _base_config(fit: Path, kind: str) -> tuple[Path, dict[str, Any]]:
    if kind == 'initial_cd':
        path = fit / 'runtime/simulation.yaml'
        config = load_yaml(path)
        archived_root = Path(_json(fit / 'run_manifest.json')['working_directory']).parent
        return path, _rebase_snapshot(config, archived_root, fit)
    path = Path(_json(fit / 'manifest.json')['config']['base_sim_config_path'])
    return path, load_yaml(path)


def fit_subject_record(fit: Path, subject: int) -> dict[str, Any]:
    """Return the original selection and its saved finalist bank in a common shape."""
    kind = source_kind(fit)
    if kind == 'initial_cd':
        directory = fit / 'hyper_cd' / f'subject_{subject}'
        best = _json(directory / 'best_hyperparams.json')
        bank = [json.loads(line) for line in (directory / 'final_rescore.jsonl').read_text().splitlines()]
        point = best['final_rescore_best']['hyperparams']
        selected = next((i for i, row in enumerate(bank) if row['hyperparams'] == point), None)
        if selected is None:
            raise ValueError('Saved selected point is absent from the finalist bank')
        if not np.isclose(bank[selected]['aggregated_error'], best['final_rescore_best']['aggregated_error'], rtol=0, atol=1e-12):
            raise ValueError('Saved selected point and finalist scores differ')
        return dict(source_kind=kind, best=best, point=point, bank=bank,
                    selected_index=selected, status='initial_search_only', issues=[],
                    selected_mean_nll=best['final_rescore_best']['aggregated_error'])
    result = _json(fit / 'subjects' / str(subject) / 'fit_result.json')
    bank = [{**row, 'aggregated_error': result['independent_audit']['scores'][row['id']]}
            for row in result['candidate_bank']]
    selected = next((i for i, row in enumerate(bank) if row['id'] == result['selected']), None)
    if selected is None or bank[selected]['hyperparams'] != result['hyperparams']:
        raise ValueError('Saved adaptive selection differs from its candidate bank')
    if not np.isclose(bank[selected]['aggregated_error'], result['selected_mean_nll'], rtol=0, atol=1e-12):
        raise ValueError('Saved adaptive selection and audit scores differ')
    best = {'final_rescore_best': {'hyperparams': result['hyperparams'],
                                 'aggregated_error': result['selected_mean_nll']}}
    return dict(source_kind=kind, best=best, point=result['hyperparams'], bank=bank,
                selected_index=selected, status=result['status'], issues=list(result['issues']), result=result,
                selected_mean_nll=result['selected_mean_nll'])


def resolve_fit_subject(fit: Path, subject: int, cache: Path,
                        max_trials: int | None = None) -> dict[str, Any]:
    """Resolve fixed parameters and trial identities through the shared data loader."""
    kind = source_kind(fit)
    path, base = _base_config(fit, kind)
    config = resolve_subject_config(base, subject)
    engine = resolve_engine_config(config, path.parent, subject_id=subject)
    record = fit_subject_record(fit, subject)
    engine = apply_fixed_hyperparams_to_engine_config(engine, record['point'])
    engine['oral_readout'] = None
    engine['partition']['kwargs']['similarity_cache_dir'] = str(cache.resolve())
    engine['inference']['particle_blocks'] = 1
    paths = resolve_dataset_paths(config, path.parent)
    loader = SubjectTrialDataLoader(engine, dataset_paths=paths)
    loader.prepare_data(paths['learning_data'])
    frame = loader._get_subject_frame(subject, 1.)
    arrays = loader._extract_arrays(frame, max_trials)
    if kind == 'adaptive' and max_trials is None and len(arrays.choices) != record['result']['trial_count']:
        raise ValueError('Saved fit and current evaluation trial counts differ')
    if 'miss' in frame:
        frame = frame.loc[frame['miss'] != 1]
    frame = frame.iloc[:len(arrays.choices)].copy()
    if len(frame) != len(arrays.choices) or len(arrays.choices) < 2:
        raise ValueError('Source trial identifiers and model arrays are not aligned')
    return dict(subject=subject, condition=loader._get_condition_value(frame),
                engine=engine, paths=paths, arrays=arrays, frame=frame,
                point=record['point'], best=record['best'], source_kind=kind,
                fit_status=record['status'], fit_issues=record['issues'])


def original_replay_spec(fit: Path, subject: int, repeat: int = 0) -> dict[str, Any]:
    """Describe one original PF seed; replay uses its recorded particle budget."""
    record = fit_subject_record(fit, subject)
    kind = record['source_kind']
    if kind == 'initial_cd':
        search = load_yaml(fit / 'runtime/hyper.yaml')
        budget = search['final_rescore']
        base = stable_seed({'seed_role': 'hyper_cd_stage_common_random_numbers',
                            'hyper_base_seed': search['hyper_base_seed'], 'stage': 'final_rescore',
                            'subject_id': subject, 'seed_family': budget['seed_family']})
        seed = derive_trajectory_seed(base, 'simulation', repeat)
        row = record['bank'][record['selected_index']]
        expected = row['subject_metrics'][str(subject)]['simulation']['sample_errors'][repeat]
        particles = int(budget['simulation_overrides']['engine_config']['inference']['particle_count'])
        probability_file = None
    else:
        result = record['result']
        config = _json(fit / 'manifest.json')['config']
        budget = result['audit_budget']
        particles = int(budget['particle_count'])
        seed = phase_seeds(config['base_seed'], subject, result['audit_family'],
                           budget['filter_seed_count'], namespace=config['seed_namespace'])[repeat]
        path = cache_path(fit, subject, result['selected'], particles, seed)
        data = read_seed(path, particles, seed, result['selected'])
        expected = choice_nll(TrialPrediction(data['probabilities'], data['observed'], data['mask'])).value
        probability_file = str(path.resolve())
    return dict(seed=int(seed), particles=particles, expected_mean_choice_nll=float(expected),
                source_kind=kind, source_probability_file=probability_file)


def _inactive_parameters(point: Mapping[str, Any]) -> dict[str, str]:
    names = {key.split('.')[-1]: value for key, value in point.items()}
    inactive = {}
    if names['global_search'] == 1:
        inactive.update(global_search_failure_gain='global_search=1', local_structure_weight='global_search=1')
    if names['error_gain'] == 0 and (names['global_search'] == 1 or names['global_search_failure_gain'] == 0):
        inactive['failure_decay'] = 'failure has no effect on search probability or range'
    return inactive


def audit_fit_source(fit: Path, subjects: Sequence[int], output: Path) -> pd.DataFrame:
    """Write the fixed parameter/search tables without selecting another point."""
    kind = source_kind(fit)
    if kind == 'initial_cd':
        space = load_yaml(fit / 'runtime/hyper.yaml')['hyperparam_space']
        support = {path: spec['values'] for path, spec in space.items()}
        domains = {path: PARAMETER_DOMAINS[path.split('.')[-1]] for path in support}
    else:
        specs = _json(fit / 'effective_parameter_support.json')['parameters']
        support = {spec['path']: spec['values'] for spec in specs.values()}
        domains = {spec['path']: (spec['domain']['lower'] if spec['domain']['lower_closed'] else None,
                                 spec['domain']['upper'] if spec['domain']['upper_closed'] else None)
                   for spec in specs.values()}
    parameters, restarts, finalists, summaries = [], [], [], []
    for sid in subjects:
        record = fit_subject_record(fit, sid)
        _, base = _base_config(fit, kind)
        config = resolve_subject_config(base, sid)
        path, _ = _base_config(fit, kind)
        condition = resolve_engine_config(config, path.parent, subject_id=sid)['provenance']['condition']
        point = record['point']
        inactive = _inactive_parameters(point)
        for key, value in point.items():
            name, grid = key.split('.')[-1], support[key]
            edge = 'lower' if value == min(grid) else 'upper' if value == max(grid) else 'interior'
            bound = domains[key][0 if edge == 'lower' else 1] if edge != 'interior' else None
            parameters.append(dict(subject=sid, condition=condition, parameter=name, path=key,
                                   estimate=value, grid_min=min(grid), grid_max=max(grid), edge=edge,
                                   boundary_kind=('mathematical' if bound == value else 'search_support') if edge != 'interior' else 'interior',
                                   inactive_at_selected_point=name in inactive, inactive_reason=inactive.get(name, ''),
                                   parameter_uncertainty='pending_profile_or_recovery'))
        limited = False
        if kind == 'initial_cd':
            stop = _json(fit / 'hyper_cd' / f'subject_{sid}' / 'restart_summary.json')
            for stage, rows in stop.items():
                for row in rows:
                    limited |= row['stopped_by'] == 'max_outer_iters'
                    restarts.append(dict(subject=sid, stage=stage, restart=row['restart_id'],
                                         rounds=row['outer_iters_completed'], stopped_by=row['stopped_by'],
                                         start_nll=row['initial_error'], end_nll=row['best_error'],
                                         evaluated=row['num_new_evaluations'], cache_hits=row['num_cache_hits']))
        else:
            result = record['result']
            limited = 'search_budget_without_plateau' in result['issues']
            for row in result['search_trace']:
                restarts.append(dict(subject=sid, stage=f"cycle_{row['cycle']}", restart=0,
                                     rounds=row['round']+1, stopped_by='plateau' if row['plateau'] else 'continuing',
                                     end_nll=row['best_guide_nll'], gain=row['gain']))
        for rank, row in enumerate(sorted(record['bank'], key=lambda r: r['aggregated_error'])):
            finalist = dict(subject=sid, rank=rank+1, selected=row['hyperparams'] == point,
                            candidate_id=row.get('id', point_id(row['hyperparams'])),
                            candidate_id_source='saved' if 'id' in row else 'derived_parameter_point',
                            mean_probability_nll=row['aggregated_error'],
                            gap_to_selected=row['aggregated_error']-record['selected_mean_nll'],
                            mean_paired_single_run_nll_delta=np.nan, sd_paired_single_run_nll_delta=np.nan,
                            fraction_single_seeds_better=np.nan,
                            interpretation='saved finite-PF candidate scores; parameter confidence pending')
            if kind == 'initial_cd':
                winner = record['bank'][record['selected_index']]
                errors = np.asarray(winner['subject_metrics'][str(sid)]['simulation']['sample_errors'])
                sample = np.asarray(row['subject_metrics'][str(sid)]['simulation']['sample_errors'])
                delta = sample - errors
                finalist.update(mean_paired_single_run_nll_delta=float(delta.mean()),
                                sd_paired_single_run_nll_delta=float(delta.std(ddof=1)),
                                fraction_single_seeds_better=float(np.mean(delta < 0)))
            finalists.append(finalist)
        scores = sorted(row['aggregated_error'] for row in record['bank'])
        audit_status = record['result']['independent_audit']['status'] if kind == 'adaptive' else 'not_performed'
        challenge_status = record['result'].get('challenge_diagnostic', {}).get('status', 'not_performed') if kind == 'adaptive' else 'not_performed'
        boundary_review = record['result']['boundary']['review_required'] if kind == 'adaptive' else any(
            p['boundary_kind'] == 'search_support' for p in parameters if p['subject'] == sid)
        summaries.append(dict(subject=sid, condition=condition, selected_fit_nll=record['selected_mean_nll'],
                              some_restart_at_round_cap=limited, finalist_gap=scores[1]-scores[0] if len(scores)>1 else np.nan,
                              parameters_at_edge=sum(p['edge'] != 'interior' for p in parameters if p['subject']==sid),
                              inactive_parameters=','.join(sorted(inactive)), source_kind=kind,
                              fit_process_status='completed', fit_acceptance_status=record['status'],
                              fit_issues=json.dumps(record['issues']), independent_audit_status=audit_status,
                              challenge_status=challenge_status, boundary_review_required=boundary_review))
    for name, rows in [('parameters', parameters), ('restarts', restarts), ('finalists', finalists), ('search_summary', summaries)]:
        pd.DataFrame(rows).to_csv(output / f'{name}.csv', index=False)
    return pd.DataFrame(summaries)
