"""Replay MIX/SELECT filters and compare their behavior-only states with oral reports.

Use --phase replay --config YAML --output NEW [--smoke], followed by --phase
align --source REPLAY --oral-audit AUDIT --output NEW. The four fixed-parameter
cells preserve every original filter seed and particle count. SELECT executes
inside the actual filter; the existing decision-basis observer must reproduce
the original category predictions exactly. The shared oral readout evaluates
states after the observed choice and before its feedback. Paired contrasts
compare SELECT with MIX separately for POOL and SEARCH. A smoke source remains
ineligible for scientific oral alignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
REFIT_STATES = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_refit_states.py'
EXECUTION = 'src/Bayesian_state/model/modules/hypothesis_transition/dominant_rule_execution.py'
CELLS = ('MIX_POOL', 'SELECT_POOL', 'MIX_SEARCH', 'SELECT_SEARCH')
USE = 'dominant_execution_fixed_parameter_development'
POLICY = 'existing_dominant_rule_only'


def helpers():
    spec = importlib.util.spec_from_file_location('dominant_execution_refit_states', ROOT / REFIT_STATES)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_identity(cell: str, planned: dict, actual: dict, record: dict,
                   precision: dict, belief: dict, execution: dict) -> None:
    from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import check_execution
    if cell not in CELLS:
        raise ValueError('Unknown execution comparison cell')
    policy, arm = cell.split('_')
    enabled = policy == 'SELECT'
    if (actual != planned or actual.get('phase') != 'evaluate' or actual.get('model') != arm
            or actual.get('execution_policy') != policy or len(actual['seeds']) != 1
            or any(record[key] != actual[key] for key in ('candidate', 'model', 'seeds', 'particles'))
            or execution.get('enabled') is not enabled or execution.get('policy') != policy
            or execution.get('rule_execution_policy') != (POLICY if enabled else 'mixture')
            or execution.get('uses_existing_dominant') is not True
            or execution.get('task_categories_used_for_selection') is not False
            or execution.get('updates') != (actual['particles'] * record['trials'] if enabled else 0)):
        raise ValueError('Execution replay reference differs from the actual filter')
    check_execution(actual, record, belief, precision)


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    refit = helpers()
    separate = refit.helpers()
    replay = separate.helpers().helpers()
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        value = replay.digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Execution-state input changed')
        inputs[str(path)] = value
        return path

    if (settings['subjects'] != [208] or tuple(settings['arms']) != CELLS
            or settings['sample_role'] != 'development' or not 1 <= settings['workers'] <= 128):
        raise ValueError('Declare all four development execution cells')
    done = replay.read(bind(source / 'completion.json'))
    protocol = replay.read(bind(source / 'protocol.json'))
    fixture = done['smoke']
    if (done['status'] != 'complete' or done['parameters_changed'] or done['oral_used']
            or done['cells'] != 4 or done['protocol_sha256'] != replay.digest(source / 'protocol.json')
            or fixture is not protocol['smoke'] or fixture and not smoke):
        raise ValueError('Require a completed fixed-parameter screen; fixtures require smoke')
    for path, value in done['inputs_sha256'].items():
        if replay.digest(bind(path)) != value:
            raise ValueError('Completed execution-screen input changed')
    for relative in (ENTRY, REFIT_STATES, refit.STATES, separate.OBSERVER, separate.BELIEF, EXECUTION,
                     replay.PF, replay.COLLECTOR, replay.REFIT, replay.PROPOSAL, replay.REENTRY,
                     str(Path(replay.__file__).relative_to(ROOT))):
        bind(ROOT / relative)
    banks = {bank['cell']: bank for bank in protocol['banks']}
    if tuple(banks) != CELLS or len(banks) != len(protocol['banks']):
        raise ValueError('Execution cells are missing, duplicated or reordered')
    expected_seeds = 1 if fixture else protocol['settings']['filter_repeats']
    expected_particles = 2 if fixture else protocol['settings']['particles']
    if settings['seeds_per_arm'] != expected_seeds:
        raise ValueError('Preserve all declared independent filter seeds')
    files, seeds, selected = [], [], {}
    for cell, bank in banks.items():
        policy, arm = cell.split('_')
        if bank['policy'] != policy or bank['arm'] != arm:
            raise ValueError('Cell policy or proposal arm changed')
        selected[cell] = bank['candidate']
        items = bank['roles']['evaluate']
        group = [item['job']['seeds'][0] for item in items]
        if len(group) != expected_seeds or len(set(group)) != len(group):
            raise ValueError('Missing or duplicate execution-screen seeds')
        seeds.append(group)
        for index, item in enumerate(items):
            planned = item['job']
            file = bind(source / 'jobs' / (planned['name'] + '.json'))
            actual = replay.read(file)
            target = Path(actual['target'])
            record, precision, belief, execution = [replay.read(bind(target.with_suffix(suffix)))
                for suffix in ('.json', '.precision.json', '.belief.json', '.execution.json')]
            check_identity(cell, planned, actual, record, precision, belief, execution)
            if (actual['candidate'] != bank['candidate'] or actual['particles'] != expected_particles
                    or replay.read(bind(Path(actual['folder']) / 'prepared.json'))['subject'] != 208
                    or fixture and record['trials'] != 8):
                raise ValueError('Execution bank changed its parameter, participant or particle count')
            if not smoke or index == 0:
                files.append(file)
    if any(group != seeds[0] for group in seeds):
        raise ValueError('Execution policies have different filter seeds')
    for arm in ('POOL', 'SEARCH'):
        if selected['MIX_' + arm] != selected['SELECT_' + arm]:
            raise ValueError('Execution comparison changed its parameter point')
    return files, dict(parameter_estimation_scope='prefix', evaluation_use=USE,
        parameter_candidate_policy='fixed_behavior_nominees_execution_comparison',
        selected_parameters=selected, source_probe=str(source.resolve()),
        original_nominations_changed=False, scientific_scope=not fixture,
        refit_replay_inputs=inputs, execution_policies=['MIX', 'SELECT'],
        report_policy='Same decision-basis readout and oral constraints for all four cells')


def worker(file: Path) -> None:
    refit = helpers()
    separate = refit.helpers()
    observer = separate.helpers()
    replay = observer.helpers()
    job = replay.read(file)
    cell = job.get('execution_cell')
    if (cell not in CELLS or cell != job['execution_policy'] + '_' + job['model']
            or job.get('evaluation_use') != USE):
        raise ValueError('Execution-state worker lost its policy identity')
    enabled = job['execution_policy'] == 'SELECT'
    events = []
    original_load, original_write = replay.load, replay.write

    def load(name, relative):
        module = original_load(name, relative)
        if relative == replay.PF:
            capture = module.run_state_model_particle_filter
            execution = original_load(
                'src.Bayesian_state.model.modules.hypothesis_transition.dominant_rule_execution', EXECUTION)

            def run_filter(**kwargs):
                cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
                with execution.dominant_rule_execution(cls, enabled=enabled) as own:
                    result = capture(**kwargs)
                    events.extend(own)
                return result

            module.run_state_model_particle_filter = run_filter
        return module

    def write(path, value):
        if path == Path(job['target']).with_suffix('.json'):
            expected = {t: job['particles'] for t in range(value['trials'])} if enabled else {}
            if Counter(event['trial'] for event in events) != expected:
                raise ValueError('State replay did not execute the declared policy before every choice')
            value = dict(value, arm=cell, proposal_arm=job['model'],
                         execution_policy=job['execution_policy'], execution_updates=len(events),
                         rule_execution_policy=POLICY if enabled else 'mixture')
        original_write(path, value)

    replay.load, replay.write = load, write
    observer.helpers = lambda: replay
    separate.helpers = lambda: observer
    refit.helpers = lambda: separate
    refit.USE = USE
    refit.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    refit = helpers()
    separate = refit.helpers()
    replay = separate.helpers().helpers()
    original_prepare = replay.prepare
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY

    def prepare(own_config, own_output, own_smoke):
        protocol, jobs = original_prepare(own_config, own_output, own_smoke)
        for job in jobs:
            job['execution_cell'] = job['execution_policy'] + '_' + job['model']
            job['reported_arm'] = 'SEPARATE_' + job['model']
            if own_smoke:
                job['trial_limit'] = min(16, replay.read(Path(job['reference']).with_suffix('.json'))['trials'])
        for relative in (REFIT_STATES, refit.STATES, separate.OBSERVER, EXECUTION, replay.PROPOSAL, replay.REENTRY):
            target = own_output / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            replay.shutil.copy2(ROOT / relative, target)
        return protocol, jobs

    replay.prepare = prepare
    replay.run(config, output, smoke)


def paired_contrasts(estimates: dict):
    """Pair whole-filter deletions within each unchanged parameter point."""
    import numpy as np
    import pandas as pd
    rows = []
    for subject in sorted({key[0] for key in estimates}):
        for arm in ('POOL', 'SEARCH'):
            old, new = [estimates[subject, policy + '_' + arm] for policy in ('MIX', 'SELECT')]
            if set(old) != set(new):
                raise ValueError('Oral windows or report subsets differ')
            for key in old:
                a, b = old[key], new[key]
                np.testing.assert_array_equal(a['seeds'], b['seeds'])
                delta = b['values'] - a['values']
                n = len(delta)
                se = np.sqrt((n - 1) / n * ((delta - delta.mean(0)) ** 2).sum(0)) if n > 1 else np.full(3, np.nan)
                row = dict(subject=subject, proposal_arm=arm, arm='SELECT_' + arm,
                           reference='MIX_' + arm, window=key[0], domain=key[1], subset=key[2])
                for i, label in enumerate(('support', 'conflict', 'undetermined')):
                    row[label + '_difference'] = b['point'][i] - a['point'][i]
                    row[label + '_difference_mcse'] = se[i]
                rows.append(row)
    return pd.DataFrame(rows)


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from . import model_0923_decision_basis_alignment as alignment
    replay = helpers().helpers().helpers().helpers()
    protocol, done = (replay.read(source / name) for name in ('protocol.json', 'completion.json'))
    if (done['status'] != 'complete' or done['smoke'] or done['jobs'] != done['exact_behavior_replays']
            or protocol.get('evaluation_use') != USE or tuple(protocol['selected_parameters']) != CELLS
            or protocol.get('scientific_scope') is not True or protocol['original_nominations_changed'] is not False
            or protocol['source_sha256'].get(ENTRY) != replay.digest(ROOT / ENTRY)):
        raise ValueError('Oral evaluation needs all exact scientific execution-state replays')
    estimates = {}
    original_context, original_group = alignment.estimation_context, alignment.group_scores

    def context(own):
        keys = ('parameter_estimation_scope', 'evaluation_use', 'parameter_candidate_policy',
                'selected_parameters', 'source_probe', 'original_nominations_changed', 'execution_policies')
        return {key: own[key] for key in keys}, (
            '<p>MIX与SELECT在POOL、SEARCH各自相同的行为参数下比较。SELECT在每个粒子中执行其现有主导规则，'
            '所有候选继续更新信念与精度。全部状态先逐题复现原行为预测，口述随后检验本题选择后、反馈前的局部分类依据。'
            '四组保留共同报告、粒子数、滤波种子及原训练边界。同参数差值见paired_execution_contrasts.csv；'
            '本轮属于208的固定参数开发诊断。</p>')

    def group(*args, **kwargs):
        table, rows, values = original_group(*args, **kwargs)
        estimates[rows[0]['subject'], rows[0]['arm']] = values
        return table, rows, values

    alignment.estimation_context, alignment.group_scores = context, group
    try:
        alignment.run(source, oral_audit, output, reference_model='MIX_POOL')
    finally:
        alignment.estimation_context, alignment.group_scores = original_context, original_group
    file = output / 'paired_execution_contrasts.csv'
    paired_contrasts(estimates).to_csv(file, index=False)
    replay.write(output / 'execution_comparison_manifest.json', dict(status='complete',
        alignment_manifest_sha256=replay.digest(output / 'manifest.json'),
        paired_contrasts_sha256=replay.digest(file), source_sha256=replay.digest(Path(__file__)),
        pairs=2, parameters_changed=False, oral_used_for_parameter_selection=False))


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
