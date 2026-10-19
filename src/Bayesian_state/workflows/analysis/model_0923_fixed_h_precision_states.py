"""Replay frozen fixed-H precision evaluations using the shared oral observer.

Use --phase replay --source COMPLETE_PRECISION_STUDY --output NEW [--workers N]
[--smoke], then --phase align --source REPLAY --oral-audit AUDIT --output NEW.
All four MIX/SELECT x ADD10/LOG25 nominees retain their original evaluation
seeds, particles, task boundaries and precision updates. Smoke reads one saved
seed per cell for sixteen trials in one worker; its output is ineligible for
scientific oral alignment. Oral reports enter only after exact behavior replay.
The common whole-report scores and within-policy LOG25 minus ADD10 contrasts
retain the original report set and the two separately frozen semantic readings.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
OBSERVER = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_states.py'
CELLS = ('MIX_ADD10', 'MIX_LOG25', 'SELECT_ADD10', 'SELECT_LOG25')
USE = 'fixed_h_precision_behavior_selected_development'
# Workers place the archived cognitive package first on sys.path; these are
# configuration field names, so validation does not import a newer workflow.
INIT = 'engine.modules.beta_mod.kwargs.beta_init'
RATE = 'engine.modules.beta_mod.kwargs.learning_rate'


def helpers():
    spec = importlib.util.spec_from_file_location('precision_decision_observer', ROOT / OBSERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.USE = USE
    return module


def check_point(cell: str, job: dict, nominee: dict, protocol: dict) -> None:
    """Keep H, precision mode and evaluation identity tied to the frozen point."""
    if cell not in CELLS:
        raise ValueError('Unknown fixed-H precision cell')
    declared = protocol['cells'][cell]
    family = protocol['settings']['families'][declared['family']]
    arm = {k: v for k, v in family.items() if k != 'anchor'}
    reference = declared['reference']['hyperparams']
    actual = job['candidate']['hyperparams']
    if (cell != declared['policy'] + '_' + declared['family'] or job['phase'] != 'evaluate' or job['model'] != 'SEARCH' or job['smoke']
            or job['fixed_h_precision_cell'] != cell or job['execution_policy'] != declared['policy']
            or job['candidate'] != nominee or job['arm_settings'] != arm
            or len(job['seeds']) != 1 or job['seeds'][0] not in protocol['seeds']['evaluation']
            or job['particles'] != protocol['settings']['evaluation']['particles']
            or {k: v for k, v in actual.items() if k not in (INIT, RATE)} !=
               {k: v for k, v in reference.items() if k not in (INIT, RATE)}):
        raise ValueError('Precision replay differs from the frozen evaluation or fixed H coordinates')


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
    observer = helpers()
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        value = digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Precision replay source changed')
        inputs[str(path)] = value
        return path

    done = load(bind(source / 'completion.json'))
    protocol = load(bind(source / 'protocol.json'))
    frozen = load(bind(source / 'nominees.json'))
    if (done['status'] != 'complete' or done['smoke'] or protocol['smoke']
            or done['protocol_sha256'] != digest(source / 'protocol.json')
            or done['nomination_sha256'] != digest(source / 'nominees.json')
            or frozen['status'] != 'frozen_before_test' or frozen['test_choices_used'] is not False
            or frozen['oral_used'] is not False or done['oral_used'] is not False
            or frozen['confirmation_sha256'] != digest(bind(source / 'confirmation.json'))
            or tuple(protocol['cells']) != CELLS or tuple(frozen['nominees']) != CELLS
            or settings['subjects'] != [protocol['settings']['subject']]
            or tuple(settings['arms']) != CELLS or settings['sample_role'] != 'development'
            or settings['seeds_per_arm'] != protocol['settings']['evaluation']['repeats']):
        raise ValueError('Require all four completed training-selected precision nominees')
    for path, value in done['inputs_sha256'].items():
        if digest(bind(path)) != value:
            raise ValueError('Completed precision study input changed')
    confirmed = load(source / 'confirmation.json')['rows']
    files, selected = [], {}
    for cell in CELLS:
        point = frozen['nominees'][cell]
        ranking = sorted((r for r in confirmed if r['cell'] == cell), key=lambda r: (r['training_nll'], r['id']))
        if not ranking or ranking[0]['id'] != point['id']:
            raise ValueError('Frozen precision point differs from training confirmation')
        group = sorted((source / 'jobs').glob(f'evaluation_{cell}_*.json'))
        expected_seeds = protocol['seeds']['evaluation']
        by_seed = {}
        for file in group:
            actual = load(bind(file))
            check_point(cell, actual, point, protocol)
            seed = actual['seeds'][0]
            if seed in by_seed:
                raise ValueError('Repeated precision evaluation seed')
            by_seed[seed] = file
            target = Path(actual['target'])
            record, precision, belief, execution = [load(bind(target.with_suffix(suffix)))
                for suffix in ('.json', '.precision.json', '.belief.json', '.execution.json')]
            observer.check_identity(actual['execution_policy'] + '_SEARCH', actual, actual,
                                    record, precision, belief, execution)
            if digest(bind(target.with_suffix('.npz'))) != record['predictions_sha256']:
                raise ValueError('Precision predictions changed')
            metadata = load(bind(Path(actual['folder']) / 'prepared.json'))
            if (metadata['subject'] != settings['subjects'][0]
                    or record['trials'] != metadata['training_trials'] + metadata['test_trials']):
                raise ValueError('Precision evaluation changed the subject or complete task boundary')
        if set(by_seed) != set(expected_seeds) or len(expected_seeds) != len(by_seed):
            raise ValueError('Incomplete precision evaluation seed bank')
        files.extend(by_seed[seed] for seed in (expected_seeds[:1] if smoke else expected_seeds))
        selected[cell] = point
    modules = [observer]
    for _ in range(4):
        modules.append(modules[-1].helpers())
    for module in modules:
        bind(module.__file__)
    bind(ROOT / ENTRY)
    return files, dict(parameter_estimation_scope='prefix', evaluation_use=USE,
        parameter_candidate_policy='fixed_h_training_precision_grid', selected_parameters=selected,
        precision_study=str(source.resolve()), precision_nomination_sha256=digest(source / 'nominees.json'),
        refit_replay_inputs=inputs, scientific_scope=True, original_nominations_changed=False,
        execution_policies=['MIX', 'SELECT'], precision_families=['ADD10', 'LOG25'],
        report_policy='Common behavior-only decision basis; oral excluded from parameter selection')


def worker(file: Path) -> None:
    module = helpers()
    refit = module.helpers()
    separate = refit.helpers()
    observer = separate.helpers()
    replay = observer.helpers()
    job = replay.read(file)
    cell = job.get('fixed_h_precision_cell')
    source = Path(job['precision_study'])
    protocol = replay.read(source / 'protocol.json')
    original = replay.read(Path(job['original_job']))
    check_point(cell, original, job['selected_parameters'][cell], protocol)
    for key in ('candidate', 'seeds', 'particles', 'arm_settings', 'execution_policy', 'folder', 'runtime'):
        if job[key] != original[key]:
            raise ValueError('State worker changed its original precision evaluation')
    if job['evaluation_use'] != USE or job['execution_cell'] != job['execution_policy'] + '_SEARCH':
        raise ValueError('Precision state worker lost its scientific scope')
    original_write = replay.write

    def write(path, value):
        if path == Path(job['target']).with_suffix('.json'):
            if value['arm'] != job['execution_cell']:
                raise ValueError('Unexpected executed policy label')
            value = dict(value, arm=cell, precision_family=cell.split('_')[1],
                         precision_settings=job['arm_settings'])
        original_write(path, value)

    replay.write = write
    observer.helpers = lambda: replay
    separate.helpers = lambda: observer
    refit.helpers = lambda: separate
    module.helpers = lambda: refit
    module.worker(file)


def run(source: Path, output: Path, workers: int = 14, smoke: bool = False) -> None:
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError('Use one to128 workers, bounded by ready tasks and available CPUs')
    from src.Bayesian_state.evaluation.model_0923_h_validation import load
    protocol = load(source / 'protocol.json')
    settings = protocol['settings']
    config = dict(source=str(source.resolve()), subjects=[settings['subject']], arms=list(CELLS),
        sample_role='development', workers=workers, seeds_per_arm=settings['evaluation']['repeats'])
    module = helpers()
    module.source_jobs, module.ENTRY = source_jobs, ENTRY
    with tempfile.TemporaryDirectory(prefix='precision_states_') as temporary:
        file = Path(temporary) / 'config.yaml'
        file.write_text(yaml.safe_dump(config, sort_keys=False))
        module.run(file, output, smoke)
    shutil.copy2(ROOT / OBSERVER, output / 'source' / OBSERVER)


def paired_contrasts(estimates: dict) -> pd.DataFrame:
    """Compare precision updates within each execution policy using whole filters."""
    rows = []
    for subject in sorted({key[0] for key in estimates}):
        for policy in ('MIX', 'SELECT'):
            old, new = [estimates[subject, policy + '_' + family] for family in ('ADD10', 'LOG25')]
            if set(old) != set(new):
                raise ValueError('Precision oral windows or report sets differ')
            for key in old:
                a, b = old[key], new[key]
                np.testing.assert_array_equal(a['seeds'], b['seeds'])
                delta = b['values'] - a['values']
                n = len(delta)
                if n < 2:
                    raise ValueError('Need multiple complete filters for oral uncertainty')
                uncertainty = np.sqrt((n-1) * np.var(delta, axis=0))
                row = dict(subject=subject, policy=policy, comparison='LOG25_minus_ADD10',
                           window=key[0], domain=key[1], subset=key[2])
                for i, label in enumerate(('support', 'conflict', 'undetermined')):
                    row[label+'_difference'] = b['point'][i] - a['point'][i]
                    row[label+'_difference_mcse'] = uncertainty[i]
                rows.append(row)
    return pd.DataFrame(rows)


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from . import model_0923_decision_basis_alignment as alignment
    protocol = alignment.read(source / 'protocol.json')
    done = alignment.read(source / 'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or protocol['smoke']
            or done['jobs'] != done['exact_behavior_replays'] or done['jobs'] != protocol['jobs']
            or done['jobs'] != 4 * protocol['settings']['seeds_per_arm']
            or protocol.get('scientific_scope') is not True or protocol['evaluation_use'] != USE
            or tuple(protocol['selected_parameters']) != CELLS
            or protocol['source_sha256'].get(ENTRY) != alignment.digest(ROOT / ENTRY)):
        raise ValueError('Oral comparison requires the complete exact precision state bank')
    for path, value in protocol['refit_replay_inputs'].items():
        if alignment.digest(Path(path)) != value:
            raise ValueError('Precision state-selection provenance changed')
    records = list((source / 'states').glob('*.json'))
    if len(records) != done['jobs']:
        raise ValueError('Precision state records differ from the completed bank')
    for file in records:
        record = alignment.read(file)
        job = alignment.read(source / 'jobs' / file.name)
        cell = job['fixed_h_precision_cell']
        if (cell not in CELLS or record['arm'] != cell
                or record['candidate'] != protocol['selected_parameters'][cell]
                or record['candidate'] != job['candidate'] or record['seeds'] != job['seeds']
                or record['particles'] != job['particles']
                or record['precision_settings'] != job['arm_settings']
                or record['execution_policy'] != job['execution_policy']
                or record['timing'] != 'after_choice_before_current_feedback'
                or record['data_use'] != 'behavior_only; no_oral_input'):
            raise ValueError('Precision oral state identity or timing changed')
    estimates = {}
    original_context, original_group = alignment.estimation_context, alignment.group_scores

    def context(own):
        fields = ('parameter_estimation_scope', 'evaluation_use', 'parameter_candidate_policy',
                  'selected_parameters', 'precision_study', 'precision_nomination_sha256')
        return {k: own[k] for k in fields}, (
            '<p>四格参数只用原训练选择确定，H与记忆坐标在各执行方式内固定。'
            '状态逐值复现独立行为评价，口述检验本题选择后、反馈前的局部依据。'
            '精度更新比较在MIX和SELECT内分别进行，保留共同报告、滤波种子和原试次边界。'
            '后段资料用于开发检验。</p>')

    def group(*args, **kwargs):
        table, rows, values = original_group(*args, **kwargs)
        estimates[rows[0]['subject'], rows[0]['arm']] = values
        return table, rows, values

    alignment.estimation_context, alignment.group_scores = context, group
    try:
        alignment.run(source, oral_audit, output, reference_model='MIX_ADD10')
    finally:
        alignment.estimation_context, alignment.group_scores = original_context, original_group
    target = output / 'paired_precision_contrasts.csv'
    paired_contrasts(estimates).to_csv(target, index=False)
    (output / 'precision_comparison_manifest.json').write_text(json.dumps(dict(
        status='complete', alignment_manifest_sha256=alignment.digest(output/'manifest.json'),
        paired_contrasts_sha256=alignment.digest(target), pairs=2,
        oral_used_for_parameter_selection=False), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('replay', 'align'), default='replay')
    for name in ('source', 'oral-audit', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'replay' and args.source and args.output:
        run(args.source, args.output, args.workers, args.smoke)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source, args.oral_audit, args.output)
    else:
        parser.error('Replay needs --source; align adds --oral-audit; require a new --output')
