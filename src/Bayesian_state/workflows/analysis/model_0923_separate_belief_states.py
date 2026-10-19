"""Replay the fixed-parameter separate-belief probe and align oral constraints.

Use --phase replay --config YAML --output NEW [--smoke], then --phase align
--source REPLAY --oral-audit AUDIT --output NEW. All six parameter/mechanism
cells retain their executed seeds and particle count. The existing decision-
basis observer must exactly reproduce saved pre-choice predictions. Oral
alignment uses the existing common readout; paired_mechanism_contrasts.csv
compares SEPARATE with ONLINE at each identical parameter point.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
OBSERVER = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_states.py'
BELIEF = 'src/Bayesian_state/model/modules/hypothesis_transition/separate_belief_evidence.py'
CELLS = tuple(f'{mechanism}_{parameter}' for parameter in ('RETAIN', 'ONLINE', 'NEAR')
              for mechanism in ('ONLINE', 'SEPARATE'))
SCOPE = 'prefix_selected_parameter_cross'
USE = 'separate_belief_fixed_parameter_development'


def helpers():
    spec = importlib.util.spec_from_file_location('separate_belief_state_observer', ROOT / OBSERVER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_identity(cell: str, planned: dict, actual: dict, record: dict,
                   precision: dict, belief: dict) -> None:
    """Check executed mechanism, unchanged parameter point and feedback timing."""
    if cell not in CELLS:
        raise ValueError('Unknown separate-belief comparison cell')
    mechanism = cell.split('_')[0]
    if (actual != planned or actual.get('probe_mechanism') != mechanism
            or actual.get('model') != 'SEARCH' or actual.get('phase') != 'evaluate'
            or actual.get('retain_precision') is not True or len(actual['seeds']) != 1
            or actual.get('online_belief') is not (mechanism == 'ONLINE')
            or any(record[k] != actual[k] for k in ('candidate', 'model', 'seeds', 'particles'))
            or precision.get('retained_precision') is not True
            or precision.get('structural_intervention') != 'retained_rule_precision'
            or precision.get('transitions') != actual['particles'] * record['trials']
            or belief.get('enabled') is not True):
        raise ValueError('Separate-belief state reference differs from the actual filter')
    count = actual['particles'] * (record['trials'] - 1)
    if mechanism == 'ONLINE':
        if (actual.get('belief_intervention') != 'online_retention_evidence'
                or belief.get('belief_intervention') != actual['belief_intervention']
                or belief.get('exact_transition_timing') is not True
                or belief.get('evidence_matches_retained_state') is not True
                or belief.get('reconstructions') != count):
            raise ValueError('ONLINE reference evidence or timing differs')
    else:
        point = actual['candidate']['hyperparams']
        if (actual.get('belief_intervention') != 'separate_gamma_belief_evidence'
                or belief.get('belief_intervention') != actual['belief_intervention']
                or belief.get('completed_feedback_timing_exact') is not True
                or belief.get('belief_matches_saved_state') is not True or belief.get('updates') != count
                or belief.get('gamma') != point['engine.modules.memory_mod.kwargs.gamma']
                or belief.get('delta') != [point['engine.modules.hypo_transitions_mod.kwargs.intervention.decay']]):
            raise ValueError('SEPARATE reference memory, evidence or timing differs')


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    replay = helpers().helpers(); inputs = {}

    def bind(path):
        path = Path(path).resolve(); value = replay.digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('State replay input changed')
        inputs[str(path)] = value
        return path

    if (settings['subjects'] != [208] or settings['arms'] != list(CELLS)
            or settings['sample_role'] != 'development' or not 1 <= settings['workers'] <= 128):
        raise ValueError('Replay all six declared development cells')
    done = replay.read(bind(source / 'completion.json'))
    protocol = replay.read(bind(source / 'protocol.json'))
    if (done['status'] != 'complete' or done['parameters_changed'] or done['cells'] != 6
            or done['protocol_sha256'] != replay.digest(source / 'protocol.json')
            or done['smoke'] != protocol['smoke'] or (done['smoke'] and not smoke)):
        raise ValueError('Require a complete unchanged probe; short fixtures require --smoke')
    for name, sha in done['inputs_sha256'].items():
        if replay.digest(bind(name)) != sha: raise ValueError('Probe execution inputs changed')
    for relative in (ENTRY, OBSERVER, BELIEF, replay.PF, replay.COLLECTOR, replay.REFIT,
                     replay.PROPOSAL, replay.REENTRY, str(Path(replay.__file__).relative_to(ROOT))):
        bind(ROOT / relative)
    banks = {b['cell']: b for b in protocol['banks']}
    if tuple(banks) != CELLS: raise ValueError('Missing or reordered probe cells')
    files, seeds, selected = [], [], {}
    for cell, bank in banks.items():
        items = bank['roles']['evaluate']; selected[cell] = bank['candidate']
        actual_seeds = [item['job']['seeds'][0] for item in items]
        if len(actual_seeds) != settings['seeds_per_arm'] or len(set(actual_seeds)) != len(actual_seeds):
            raise ValueError('Preserve every independent filter seed')
        seeds.append(actual_seeds)
        for item in items[:1] if smoke else items:
            expected = item['job']; file = bind(source / 'jobs' / (expected['name'] + '.json'))
            actual = replay.read(file); target = Path(actual['target'])
            record = replay.read(bind(target.with_suffix('.json')))
            precision = replay.read(bind(target.with_suffix('.precision.json')))
            belief = replay.read(bind(target.with_suffix('.online.json' if bank['mechanism'] == 'ONLINE' else '.belief.json')))
            check_identity(cell, expected, actual, record, precision, belief)
            if actual['candidate'] != bank['candidate'] or replay.read(Path(actual['folder']) / 'prepared.json')['subject'] != 208:
                raise ValueError('State bank refers to another parameter or participant')
            files.append(file)
    if any(group != seeds[0] for group in seeds): raise ValueError('Cells have different seed banks')
    if len({replay.read(file)['particles'] for file in files}) != 1:
        raise ValueError('Cells have different particle counts')
    return files, dict(parameter_estimation_scope=SCOPE, evaluation_use=USE,
        parameter_candidate_policy='frozen_behavior_nominees_and_declared_near_tie',
        selected_parameters=selected, source_probe=str(source.resolve()), original_nominations_changed=False,
        refit_replay_inputs=inputs, report_policy='Same oral constraints and behavior-only state inference in all six cells')


def worker(file: Path) -> None:
    observer = helpers(); replay = observer.helpers(); job = replay.read(file)
    if (job.get('reported_arm') not in CELLS or job.get('parameter_estimation_scope') != SCOPE
            or job.get('evaluation_use') != USE
            or job['reported_arm'].split('_')[0] != job['probe_mechanism']):
        raise ValueError('State worker lost its comparison identity')
    if job['probe_mechanism'] == 'ONLINE':
        observer.worker(file)
        return
    if job.get('online_belief') is not False or job['belief_intervention'] != 'separate_gamma_belief_evidence':
        raise ValueError('SEPARATE replay must execute its own evidence mechanism')
    original_load, original_write = replay.load, replay.write
    events = []

    def load(name, relative):
        module = original_load(name, relative)
        if relative == replay.REFIT:
            base_load = module.base.load

            def with_belief(child_name, child_relative):
                child = base_load(child_name, child_relative)
                if child_relative == replay.REENTRY:
                    retained = child.reentry_precision
                    separate = base_load('src.Bayesian_state.model.modules.hypothesis_transition.separate_belief_evidence', BELIEF)

                    @contextmanager
                    def combined(cls, **kwargs):
                        with retained(cls, **kwargs) as precision_events:
                            with separate.separate_belief_evidence(cls, enabled=True) as own:
                                yield precision_events
                                events.extend(own)
                    child.reentry_precision = combined
                return child
            module.base.load = with_belief
        return module

    def write(path, value):
        if path == Path(job['target']).with_suffix('.json'):
            if Counter(e['trial'] for e in events) != {t: job['particles'] for t in range(1, value['trials'])}:
                raise ValueError('State replay belief timing differs from original choices')
            value = dict(value, arm=job['reported_arm'], proposal_arm=job['model'],
                online_belief=False, separate_belief=True, belief_intervention=job['belief_intervention'],
                belief_updates=len(events), belief_matches_saved_state=True,
                belief_gamma=job['candidate']['hyperparams']['engine.modules.memory_mod.kwargs.gamma'])
        original_write(path, value)

    replay.load, replay.write = load, write
    replay.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    replay = helpers().helpers(); original_prepare = replay.prepare
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY

    def prepare(own_config, own_output, own_smoke):
        protocol, jobs = original_prepare(own_config, own_output, own_smoke)
        for job in jobs:
            parameter = job['name'].split('_')[1]
            job['reported_arm'] = job['probe_mechanism'] + '_' + parameter
            if job['reported_arm'] not in CELLS: raise ValueError('Unrecognized saved cell name')
        return protocol, jobs

    replay.prepare = prepare
    replay.run(config, output, smoke)


def paired_contrasts(estimates: dict) -> pd.DataFrame:
    """Pair complete-filter deletions at the same frozen parameter point."""
    rows = []
    for subject in sorted({key[0] for key in estimates}):
        for parameter in ('RETAIN', 'ONLINE', 'NEAR'):
            old, new = [estimates[subject, f'{mechanism}_{parameter}'] for mechanism in ('ONLINE', 'SEPARATE')]
            if set(old) != set(new): raise ValueError('Oral comparison windows or report subsets differ')
            for key in old:
                a, b = old[key], new[key]
                np.testing.assert_array_equal(a['seeds'], b['seeds'])
                delta = b['values'] - a['values']; n = len(delta)
                se = np.sqrt((n - 1) / n * ((delta - delta.mean(0)) ** 2).sum(0)) if n > 1 else np.full(3, np.nan)
                row = dict(subject=subject, parameter=parameter, arm='SEPARATE_' + parameter,
                    reference='ONLINE_' + parameter, window=key[0], domain=key[1], subset=key[2])
                for i, label in enumerate(('support', 'conflict', 'undetermined')):
                    row[label + '_difference'] = b['point'][i] - a['point'][i]
                    row[label + '_difference_mcse'] = se[i]
                rows.append(row)
    return pd.DataFrame(rows)


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from . import model_0923_decision_basis_alignment as alignment
    replay = helpers().helpers(); protocol = replay.read(source / 'protocol.json'); done = replay.read(source / 'completion.json')
    if (done['status'] != 'complete' or done['smoke'] or done['jobs'] != done['exact_behavior_replays']
            or protocol['evaluation_use'] != USE or tuple(protocol['selected_parameters']) != CELLS
            or protocol['original_nominations_changed'] is not False
            or protocol['source_sha256'][ENTRY] != replay.digest(ROOT / ENTRY)):
        raise ValueError('Oral readout needs complete, unchanged exact state replays')
    estimates = {}; original_context, original_group = alignment.estimation_context, alignment.group_scores

    def context(own):
        keys = ('parameter_estimation_scope', 'evaluation_use', 'parameter_candidate_policy',
                'selected_parameters', 'source_probe', 'original_nominations_changed')
        return {k: own[k] for k in keys}, ('<p>六组均沿用三组已保存的行为参数；ONLINE与SEPARATE在每一参数点成对比较。'
            'SEPARATE采用γ保持作答信念证据，δ继续管理候选。口述用于冻结参数后的表征检验。'
            '同一参数下的机制差见paired_mechanism_contrasts.csv；contrast表另列相对首组的完整差异。'
            '全部组使用相同报告、粒子数、滤波种子及384／192题分段，属于208的开发诊断。</p>')

    def group(*args, **kwargs):
        table, rows, values = original_group(*args, **kwargs)
        estimates[rows[0]['subject'], rows[0]['arm']] = values
        return table, rows, values

    alignment.estimation_context, alignment.group_scores = context, group
    try: alignment.run(source, oral_audit, output, reference_model='ONLINE_RETAIN')
    finally: alignment.estimation_context, alignment.group_scores = original_context, original_group
    table = paired_contrasts(estimates); file = output / 'paired_mechanism_contrasts.csv'
    table.to_csv(file, index=False)
    replay.write(output / 'comparison_manifest.json', dict(status='complete',
        alignment_manifest_sha256=replay.digest(output / 'manifest.json'),
        paired_contrasts_sha256=replay.digest(file), source_sha256=replay.digest(Path(__file__)),
        pairs=3, rows=len(table), parameters_changed=False, oral_used_for_parameter_selection=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('replay', 'align'), default='replay')
    for name in ('config', 'source', 'oral-audit', 'output', 'worker'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.phase == 'replay' and args.config and args.output: run(args.config, args.output, args.smoke)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output: align(args.source, args.oral_audit, args.output)
    else: parser.error('Replay needs --config; align needs --source and --oral-audit; supply --output')
