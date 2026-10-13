"""Model 0923 parameter and data adapter for the shared Plus fitting protocol."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .adaptive_fit import fit_subjects, validate_fit_policy
from .adaptive_runtime import digest, freeze_json, run_manifest, verify_manifest
from .search.adaptive_proposals import candidate, initial_points, select_elites
from ..model import ModelContext, StateModel
from ..model.model_0923 import is_model_0923_revised, validate_model_0923_config
from ..simulation.config import resolve_engine_config
from ..simulation.data import SubjectTrialDataLoader
from ..utils.datasets import resolve_dataset_paths
from ..utils.paths import ROOT_DIR
from ..utils.subjects import resolve_subject_config

DEFAULT_CONFIG = ROOT_DIR / 'configs/meg/model0923_adaptive_335.yaml'
CAPACITY_PATH = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'
PARAMETER_NAMES = frozenset({'baseline_probability', 'error_gain', 'failure_decay',
    'global_search', 'global_search_failure_gain', 'max_capacity', 'exit_rate',
    'local_structure_weight', 'transfer_scale', 'execution_inertia', 'gamma',
    'beta_init', 'learning_rate'})
EXPECTED_PATHS = {
    **{name: 'engine.modules.hypo_transitions_mod.kwargs.search_controller.' + name for name in (
        'baseline_probability', 'error_gain', 'failure_decay', 'global_search', 'global_search_failure_gain')},
    **{name: 'engine.modules.hypo_transitions_mod.kwargs.' + name for name in (
        'max_capacity', 'exit_rate', 'local_structure_weight', 'transfer_scale', 'execution_inertia')},
    'gamma': 'engine.modules.memory_mod.kwargs.gamma',
    'beta_init': 'engine.modules.beta_mod.kwargs.beta_init',
    'learning_rate': 'engine.modules.beta_mod.kwargs.learning_rate',
}


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path, smoke: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    required = {'schema_version', 'backend', 'analysis_id', 'parameter_space',
                'base_sim_config_path', 'subjects', 'base_seed', 'seed_namespace',
                'parallel_budget', 'search', 'precision', 'boundary', 'warm_start'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Unknown or missing Model 0923 adaptive configuration keys')
    if type(config['schema_version']) is not int or config['schema_version'] != 2 or config['backend'] != 'model0923_adaptive':
        raise ValueError('Model 0923 uses the shared schema-2 Plus audit protocol')
    for key in ('parameter_space', 'base_sim_config_path'):
        config[key] = str(resolve(path.parent, config[key]))
    if not isinstance(config['subjects'], list) or not config['subjects'] or any(type(s) is not int for s in config['subjects']) or len(set(config['subjects'])) != len(config['subjects']):
        raise ValueError('Need distinct integer subject IDs')
    if config['seed_namespace'] != 'model0923_adaptive_full_v1':
        raise ValueError('Declare the Model 0923 independent seed namespace')
    if set(config['warm_start']) != {'fit_runs'} or not isinstance(config['warm_start']['fit_runs'], list):
        raise ValueError('warm_start.fit_runs must list read-only completed fits')
    config['warm_start']['fit_runs'] = [str(resolve(path.parent, p)) for p in config['warm_start']['fit_runs']]
    support = yaml.safe_load(Path(config['parameter_space']).read_text())
    count = len(support['parameters']['max_capacity']['values'])
    return validate_fit_policy(config, smoke, smoke_cover_count=count)


def parameter_support(config: dict[str, Any]) -> tuple[dict, dict, dict]:
    support = yaml.safe_load(Path(config['parameter_space']).read_text())
    if set(support) != {'schema_version', 'model_id', 'specification_version', 'parameters'} or support['schema_version'] != 1 or support['model_id'] != 'model_0923' or support['specification_version'] != '0923-full-v3' or set(support['parameters']) != PARAMETER_NAMES:
        raise ValueError('Need the declared thirteen-parameter full-v3 support')
    support = deepcopy(support)
    for name, values in config['boundary']['extensions'].items():
        if name not in support['parameters'] or not isinstance(values, list) or not values:
            raise ValueError('Invalid explicitly declared support extension')
        support['parameters'][name]['values'] = sorted(set(support['parameters'][name]['values'] + values))
    space, anchor = {}, {}
    for name, spec in support['parameters'].items():
        if set(spec) != {'path', 'values', 'anchor', 'domain'} or spec['path'] != EXPECTED_PATHS[name]:
            raise ValueError(f'Invalid parameter declaration: {name}')
        if not spec['path'].startswith('engine.modules.') or spec['path'] in space:
            raise ValueError('Parameter paths must be distinct shared-engine fields')
        values, domain = spec['values'], spec['domain']
        if not isinstance(values, list) or len(values) < 2 or len(set(values)) != len(values) or values != sorted(values) or spec['anchor'] not in values:
            raise ValueError(f'Invalid ordered parameter support: {name}')
        if set(domain) != {'lower', 'upper', 'lower_closed', 'upper_closed', 'integer'}:
            raise ValueError(f'Incomplete mathematical domain: {name}')
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Nonfinite/nonnumeric support: {name}')
            if domain['integer'] and type(value) is not int:
                raise ValueError('Capacity values must be integers')
            for side in ('lower', 'upper'):
                bound = domain[side]
                if bound is not None and ((value < bound if side == 'lower' else value > bound) or (value == bound and not domain[side + '_closed'])):
                    raise ValueError(f'Support violates {name} mathematical domain')
        space[spec['path']], anchor[spec['path']] = values, spec['anchor']
    if CAPACITY_PATH not in space:
        raise ValueError('Candidate capacity path differs from the scientific model')
    count = len(space[CAPACITY_PATH])
    if any(config['search'][k] < count for k in ('initial_count', 'fresh_global_count', 'challenge_global_count')):
        raise ValueError('All global batches must cover all declared capacities')
    return space, anchor, support


class Model0923Policy:
    """Retain the shared proposals; interpret the full-v3 scalar parameters."""
    def initial_points(self, space: dict, anchor: dict, count: int, seed: int) -> list[dict]:
        return initial_points(space, anchor, count, seed, cover_key=CAPACITY_PATH)

    def select_elites(self, rows: list[dict], count: int) -> list[dict]:
        return select_elites(rows, count, exclude_key=CAPACITY_PATH)

    def extract_parameters(self, point: dict) -> dict:
        return {path.split('.')[-1]: value for path, value in point.items()}

    def boundary_report(self, rows: list[dict], support: dict, tolerance: float, max_candidates: int) -> dict:
        ranked = sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
        checked = [r for r in ranked if r['mean_nll'] <= ranked[0]['mean_nll'] + tolerance][:max_candidates]
        hits = []
        for row in checked:
            for name, spec in support['parameters'].items():
                value = row['hyperparams'][spec['path']]
                for side, edge in [('lower', min(spec['values'])), ('upper', max(spec['values']))]:
                    if value != edge:
                        continue
                    domain = spec['domain']
                    natural = domain[side] is not None and domain[side + '_closed'] and value == domain[side]
                    hits.append(dict(candidate=row['id'], parameter=name, value=value, side=side,
                                     kind='mechanism_boundary' if natural else 'artificial_boundary'))
        return dict(checked_candidates=[r['id'] for r in checked], hits=hits,
                    review_required=any(h['kind'] == 'artificial_boundary' for h in hits),
                    scope='Declared finite support edges; parameter recovery and identifiability remain separate checks')


def make_contexts(config: dict, selected: list[int] | None, smoke: bool,
                  cache: Path | None = None) -> tuple[list[dict], dict[int, dict], set[Path]]:
    base_path = Path(config['base_sim_config_path'])
    base = yaml.safe_load(base_path.read_text())
    ids = config['subjects'] if selected is None else selected
    if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(config['subjects']):
        raise ValueError('Requested subjects must belong to this dataset configuration')
    if smoke and len(ids) != 1:
        raise ValueError('Smoke requires exactly one subject')
    contexts, specs, files = {}, [], {base_path}
    for sid in ids:
        simulation = resolve_subject_config(base, sid)
        if simulation.get('max_trials') is not None or simulation.get('stop_at', 1.) != 1. or simulation.get('evaluation_protocol', {'mode': 'all'}) != {'mode': 'all'}:
            raise ValueError('This protocol estimates parameters on the full valid sequence')
        engine = resolve_engine_config(simulation, base_path.parent, subject_id=sid)
        if not is_model_0923_revised(engine.get('provenance', {})):
            raise ValueError('Need the declared Model 0923 full-v3 engine')
        validate_model_0923_config(engine)
        paths = resolve_dataset_paths(simulation, base_path.parent)
        loader = SubjectTrialDataLoader(engine, dataset_paths=paths)
        loader.prepare_data(paths['learning_data'])
        frame = loader._get_subject_frame(sid, 1.)
        condition = int(loader._get_condition_value(frame))
        if condition != engine['provenance']['condition']:
            raise ValueError('Behavior condition differs from the engine')
        arrays = loader._extract_arrays(frame, 32 if smoke else None)
        if len(arrays.choices) < 2:
            raise ValueError('Need at least two valid trials')
        engine['oral_readout'] = None
        engine['inference']['particle_blocks'] = 1
        if cache is not None:
            engine['partition']['kwargs']['similarity_cache_dir'] = str(cache)
        files.update(Path(p) for p in paths.values() if Path(p).is_file())
        engine_path = resolve(base_path.parent, simulation['engine_config_path'])
        files.add(engine_path)
        specs.append(dict(subject=sid, condition=condition, engine=str(engine_path)))
        contexts[sid] = dict(subject=sid, condition=condition, engine=engine, arrays=arrays,
                             processed_dir=paths['processed_dir'], dataset_paths=paths)
    return specs, contexts, files


def warm_candidates(config: dict, contexts: dict[int, dict], space: dict) -> tuple[dict[int, list[dict]], set[Path]]:
    banks = {sid: [] for sid in contexts}
    files: set[Path] = set()
    for name in config['warm_start']['fit_runs']:
        root = Path(name)
        status = root / 'run_status.json'
        if json.loads(status.read_text())['status'] != 'completed':
            raise ValueError('Warm-start source fit must have completed')
        files.add(status)
        old_base = root / 'runtime/simulation.yaml'
        files.add(old_base)
        base = yaml.safe_load(old_base.read_text())
        for sid, context in contexts.items():
            bank_file = root / 'hyper_cd' / f'subject_{sid}' / 'final_rescore.jsonl'
            if not bank_file.exists():
                continue
            simulation = resolve_subject_config(base, sid)
            old_engine = resolve_engine_config(simulation, old_base.parent, subject_id=sid)
            if not is_model_0923_revised(old_engine.get('provenance', {})):
                raise ValueError('Historical candidate uses another scientific model')
            old_paths = resolve_dataset_paths(simulation, old_base.parent)
            for key, new_path in context['dataset_paths'].items():
                if Path(new_path).is_file() and digest(Path(new_path)) != digest(Path(old_paths[key])):
                    raise ValueError(f'Historical candidate data differ: {key}')
            files.update(Path(p) for p in old_paths.values() if Path(p).is_file())
            files.add(resolve(old_base.parent, simulation['engine_config_path']))
            def scientific_engine(engine: dict) -> dict:
                value = deepcopy(engine)
                value.pop('oral_readout', None)
                for key in ('particle_count', 'particle_blocks'):
                    value['inference'].pop(key, None)
                value['partition']['kwargs'].pop('similarity_cache_dir', None)
                return value
            if scientific_engine(old_engine) != scientific_engine(context['engine']):
                raise ValueError('Historical candidate engine differs beyond PF/runtime controls')
            files.add(bank_file)
            for record in map(json.loads, bank_file.read_text().splitlines()):
                point = record['hyperparams']
                if set(point) != set(space) or any(point[k] not in values for k, values in space.items()):
                    raise ValueError('Historical candidate is outside the declared support')
                row = candidate(point, 'historical_finalist', source_fit=str(root))
                if row['id'] not in {r['id'] for r in banks[sid]}:
                    banks[sid].append(row)
    return banks, files


def run_fit(config_path: Path, output: Path | None, subjects: list[int] | None = None, *,
            smoke: bool = False, resume: bool = False, dry_run: bool = False) -> dict:
    config_path = config_path.resolve()
    config = load_config(config_path, smoke)
    space, anchor, support = parameter_support(config)
    specs, contexts, files = make_contexts(config, subjects, smoke)
    for context in contexts.values():
        rule_count = 29 if context['condition'] == 1 else 152
        if max(space[CAPACITY_PATH]) >= rule_count:
            raise ValueError('Candidate capacity must be smaller than the declared rule catalogue')
    imported, historical_files = warm_candidates(config, contexts, space)
    config['processed_dir'] = str(next(iter(contexts.values()))['processed_dir'])
    plan = dict(config=config, subjects=[dict(s, trials=len(contexts[s['subject']]['arrays'].choices),
                                            scored_trials=len(contexts[s['subject']]['arrays'].choices)-1,
                                            imported_candidates=len(imported[s['subject']])) for s in specs],
                capacities=len(space[CAPACITY_PATH]), scope='Full-v3 Plus adaptive search and independent finite-PF audit')
    if dry_run:
        return plan
    if output is None:
        raise ValueError('Choose a new output directory or explicitly resume a frozen run')
    output = output.resolve()
    if resume:
        if not (output / 'manifest.json').is_file():
            raise ValueError('Resume requires a frozen manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs, contexts, _ = make_contexts(config, subjects, smoke, output / 'resources_cache')
        # Warm immutable catalogue resources once before process workers start.
        for context in contexts.values():
            StateModel(context['engine'], context=ModelContext(condition=context['condition'], subject_id=context['subject'],
                       processed_data_dir=context['processed_dir'], dataset_paths=context['dataset_paths']))
        manifest = run_manifest(config, config_path, specs, smoke)
        manifest['input_sha256'].update({str(p.resolve()): digest(p) for p in files | historical_files})
        manifest['scope'] = plan['scope'] + '; first valid trial initializes state; internal-state precision and recovery remain separate'
        freeze_json(output / 'manifest.json', manifest)
        freeze_json(output / 'plan.json', plan)
        freeze_json(output / 'effective_parameter_support.json', support)
        freeze_json(output / 'imported_candidates.json', {str(sid): rows for sid, rows in imported.items()})
        result = fit_subjects(config, contexts, space, anchor, support, output, smoke,
                              model_policy=Model0923Policy(), seed_candidates=imported)
        verify_manifest(manifest)
        freeze_json(output / 'fit_results.json', result)
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    result = run_fit(args.config, args.output_dir, args.subjects, smoke=args.smoke,
                     resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
