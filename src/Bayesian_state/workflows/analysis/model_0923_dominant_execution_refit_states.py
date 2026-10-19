"""Read behavior-refitted SELECT states and apply the common oral evaluation.

Use --phase replay --config YAML --output NEW [--smoke], then --phase align
--source REPLAY --oral-audit AUDIT --output NEW. The completed SELECT fit and
its evaluation must retain the original search policy, nominees, seeds and
particle counts. The existing SEPARATE refit validator checks these boundaries;
this adapter additionally checks SELECT execution and its reference bindings.
The actual SELECT state worker reproduces the original predictions. Explicit
eight-trial fixtures are software validation and cannot enter oral analysis.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
REPLAY = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_states.py'
REFIT = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_refit.py'
USE = 'development_dominant_execution_refit'
CELLS = ('SELECT_POOL', 'SELECT_SEARCH')
POLICY = 'existing_dominant_rule_only'


def helper(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    module = helper('dominant_refit_state_observer', REPLAY)
    module.USE, module.CELLS = USE, CELLS
    return module


def reference_bindings(fit: Path, frozen: dict) -> tuple[dict, dict]:
    """Bind the search's source YAML and evaluation's reconstructed settings."""
    import yaml
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
    refit = helper('dominant_state_reference_fit', REFIT)
    original = frozen['settings']
    config = dict(reference_fit=original['execution_reference_fit'], execution_policy=original['execution_policy'],
                  parallel_budget=original['parallel_budget'], scope=original['scope'])
    settings, evaluation_inputs = refit.prepare_settings(config, refit.helpers())
    if settings != original or settings['execution_policy'] != 'SELECT':
        raise ValueError('SELECT evaluation differs from its frozen search policy')
    declaration = load(fit / 'execution_refit_protocol.json')
    search_inputs = declaration['inputs_sha256']
    if (declaration.get('execution_policy') != 'SELECT' or declaration.get('rule_execution_policy') != POLICY
            or declaration.get('execution_reference_fit') != settings['execution_reference_fit']
            or any(search_inputs.get(path) != sha for path, sha in evaluation_inputs.items())):
        raise ValueError('SELECT search and evaluation reference inputs differ')
    additional = set(search_inputs) - set(evaluation_inputs)
    if len(additional) != 1 or yaml.safe_load(Path(next(iter(additional))).read_text()) != config:
        raise ValueError('SELECT search must additionally bind its original configuration')
    for path, value in search_inputs.items():
        if digest(Path(path)) != value:
            raise ValueError('SELECT search reference changed')
    for name in (REFIT, refit.PROBE, refit.EXECUTION):
        if frozen['source_sha256'].get(name) != digest(ROOT / name):
            raise ValueError('Missing or changed SELECT implementation')
    return search_inputs, evaluation_inputs


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
    from src.Bayesian_state.evaluation.model_0923_dominant_execution_refit import check_selection
    if tuple(settings['arms']) != CELLS:
        raise ValueError('Declare both SELECT proposal arms')
    fixed = helpers()
    separate = fixed.helpers()
    fit = Path(load(source / 'protocol.json')['fit'])
    frozen = load(fit / 'nominees.json')
    search_inputs, evaluation_inputs = reference_bindings(fit, frozen)
    # Retain the existing full nominee/seed/boundary validator. The search
    # binds its input YAML; evaluation reconstructs exactly those settings.
    old_helper, old_check = separate.helper, separate.checked_execution
    fitter = old_helper('select_state_separate_fit', separate.REFIT)
    fitter.reference_inputs = lambda *args: search_inputs
    separate.helper = lambda name, relative: fitter if relative == separate.REFIT else old_helper(name, relative)
    additional = {str((fit / 'execution_refit_protocol.json').resolve()): digest(fit / 'execution_refit_protocol.json')}

    def check(planned, actual, references, coordinates, record, precision, belief):
        old_check(dict(planned, execution_policy='SELECT'), actual, evaluation_inputs,
                  coordinates, record, precision, belief)
        path = Path(actual['target']).with_suffix('.execution.json')
        check_selection(actual, record, load(path))
        additional[str(path.resolve())] = digest(path)

    separate.checked_execution = check
    original_settings = dict(deepcopy(settings), arms=['POOL', 'SEARCH'])
    files, metadata = separate.source_jobs(original_settings, source, smoke)
    for relative in (ENTRY, REPLAY, REFIT):
        additional[str((ROOT / relative).resolve())] = digest(ROOT / relative)
    metadata['refit_replay_inputs'].update(additional)
    metadata.update(evaluation_use=USE, execution_policy='SELECT', rule_execution_policy=POLICY,
                    execution_fit=str(fit.resolve()),
                    report_arm_policy='SELECT_POOL/SELECT_SEARCH retain their separate behavior-selected parameters')
    metadata['selected_parameters'] = [dict(row, arm='SELECT_' + row['arm']) for row in metadata['selected_parameters']]
    return files, metadata


def worker(file: Path) -> None:
    module = helpers()
    replay = module.helpers().helpers().helpers().helpers()
    job = replay.read(file)
    if job.get('execution_policy') != 'SELECT' or job.get('evaluation_use') != USE:
        raise ValueError('Refitted state replay must execute SELECT')
    module.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    module = helpers()
    module.source_jobs, module.ENTRY = source_jobs, ENTRY
    module.run(config, output, smoke)


def check_record(job: dict, record: dict) -> None:
    if (record.get('arm') not in CELLS or record['arm'] != 'SELECT_' + job['model']
            or record.get('execution_policy') != 'SELECT' or record.get('rule_execution_policy') != POLICY
            or record.get('execution_updates') != record['particles'] * record['trials']
            or record.get('belief_updates') != record['particles'] * (record['trials'] - 1)
            or record.get('evaluation_use') != USE or record.get('exact_behavior_replay') is not True
            or any(record[key] != job[key] for key in ('candidate', 'seeds', 'particles'))
            or record.get('timing') != 'after_choice_before_current_feedback'
            or record.get('data_use') != 'behavior_only; no_oral_input'):
        raise ValueError('Oral states differ from the completed SELECT behavior filter')


def estimation_context(protocol: dict) -> tuple[dict, str]:
    if (protocol.get('evaluation_use') != USE or protocol.get('parameter_estimation_scope') != 'prefix'
            or protocol.get('execution_policy') != 'SELECT' or protocol.get('rule_execution_policy') != POLICY
            or protocol.get('scientific_scope') is not True or not protocol.get('execution_fit')
            or not protocol.get('refit_nomination_sha256')
            or protocol.get('parameter_candidate_policy') != 'projected_common_search_policy_behavior_refit'):
        raise ValueError('Oral evaluation requires the completed scientific SELECT refit')
    keys = ('parameter_estimation_scope', 'evaluation_use', 'parameter_candidate_policy', 'execution_fit',
            'refit_nomination_sha256', 'selected_parameters', 'execution_policy', 'rule_execution_policy',
            'belief_intervention', 'structural_intervention', 'fixed_coordinates')
    return {key: protocol[key] for key in keys}, (
        '<p>参数由SELECT机制下的原训练选择似然估计。每个粒子用当前主导规则回答，候选、信念和精度继续学习。'
        'POOL与SEARCH保留原共同起点、支持范围、阶段预算和种子。全部状态先逐题复现其自身保存的行为预测，'
        '口述随后检验本题选择后、反馈前的局部分类依据。后段属于开发中的时间外推诊断；'
        '与其他重估机制的比较同时包含机制与参数的变化。</p>')


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from . import model_0923_decision_basis_alignment as alignment
    replay = helpers().helpers().helpers().helpers().helpers()
    protocol, done = (replay.read(source / name) for name in ('protocol.json', 'completion.json'))
    estimation_context(protocol)
    expected = len(protocol['settings']['subjects']) * 2 * protocol['settings']['seeds_per_arm']
    if (done['status'] != 'complete' or done['smoke'] or done['jobs'] != expected
            or done['exact_behavior_replays'] != expected or tuple(protocol['settings']['arms']) != CELLS
            or protocol['source_sha256'].get(ENTRY) != replay.digest(ROOT / ENTRY)):
        raise ValueError('Require all exact SELECT refit state replays')
    files = list((source / 'states').glob('*.json'))
    if len(files) != expected:
        raise ValueError('SELECT state coverage differs from the declared filter bank')
    for file in files:
        job = replay.read(source / 'jobs' / file.name)
        replay.verify_sources(job)
        check_record(job, replay.read(file))
    original = alignment.estimation_context
    alignment.estimation_context = estimation_context
    try:
        alignment.run(source, oral_audit, output, reference_model='SELECT_POOL')
    finally:
        alignment.estimation_context = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('replay', 'align'), default='replay')
    for name in ('config', 'source', 'oral-audit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'replay' and args.config and args.output:
        run(args.config, args.output, args.smoke)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source, args.oral_audit, args.output)
    else:
        parser.error('Use replay with --config or align with --source and --oral-audit; require a new --output')
