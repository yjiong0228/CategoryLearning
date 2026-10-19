"""Assemble five exact state banks and apply the existing oral readout.

Use --phase assemble --config YAML --output NEW_DIRECTORY after the three new
state cells and both original nominee banks finish. Arrays are linked into a
new index; source results are preserved. Each linked prediction is compared to
its corresponding completed cross filter. Use --phase align --source INDEX
--oral-audit AUDIT --output NEW_DIRECTORY for each uniform oral reading.
Parameter provenance is retained per cell. No report selects a parameter.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from . import model_0923_decision_basis_alignment as alignment
from . import model_0923_online_parameter_cross_states as states
from .model_0923_decision_basis_replay import digest, read, write

ROOT = states.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
CELLS = ('RETAIN_RETAIN', 'ONLINE_RETAIN', 'RETAIN_ONLINE', 'ONLINE_ONLINE', 'ONLINE_NEAR')


def same_execution(expected: dict, actual: dict, record: dict, cell: str) -> None:
    """Match cognitive inputs even when an original bank has a different label."""
    if cell not in CELLS:
        raise ValueError('Unknown cross cell')
    keys = ('phase', 'folder', 'runtime', 'model', 'mechanism', 'seeds', 'particles', 'arm_settings',
            'retain_precision', 'structural_intervention')
    if (any(expected[k] != actual[k] for k in keys)
            or actual['candidate'] != expected['candidate'] or record['candidate'] != expected['candidate']
            or record['seeds'] != expected['seeds'] or record['particles'] != expected['particles']
            or record['exact_behavior_replay'] is not True
            or bool(actual.get('online_belief')) != cell.startswith('ONLINE_')
            or bool(record.get('online_belief')) != cell.startswith('ONLINE_')):
        raise ValueError('Reused state bank differs from its cross execution')


def assemble(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new combined state index')
    settings = yaml.safe_load(config.read_text()); inputs = {}

    def bind(path):
        path = Path(path).resolve(); sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError('Assembly input changed')
        inputs[str(path)] = sha
        return path

    def load(path):
        return read(bind(path))

    bind(config); cross = ROOT / settings['cross']
    done, protocol = load(cross / 'completion.json'), load(cross / 'protocol.json')
    if (done['status'] != 'complete' or done['smoke'] or done['nominees_changed']
            or done['protocol_sha256'] != digest(cross / 'protocol.json')):
        raise ValueError('Require the completed full cross')
    banks = {b['cell']: b for b in protocol['banks']}
    if set(banks) != set(CELLS):
        raise ValueError('Need all five declared cells')
    sources = {}; entries = []
    for cell in CELLS:
        key = 'retained_states' if cell == 'RETAIN_RETAIN' else 'online_states' if cell == 'ONLINE_ONLINE' else 'cross_states'
        source = ROOT / settings[key]
        if str(source) not in sources:
            completed, declared = load(source / 'completion.json'), load(source / 'protocol.json')
            if (completed['status'] != 'complete' or completed['smoke']
                    or completed['jobs'] != completed['exact_behavior_replays']):
                raise ValueError('Need a complete exact state bank')
            for name, sha in declared['source_sha256'].items():
                if digest(bind(ROOT / name)) != sha:
                    raise ValueError('State observer source changed')
            sources[str(source)] = declared
        arm = 'SEARCH' if key != 'cross_states' else cell
        matched = []
        for file in sorted((source / 'states').glob('*.json')):
            record = read(file)
            if record['subject'] == settings['subject'] and record['arm'] == arm:
                matched.append((file, record))
        plans = {tuple(item['job']['seeds']): item['job'] for item in banks[cell]['roles']['evaluate']}
        if len(matched) != 8 or len(plans) != 8 or {tuple(r['seeds']) for _, r in matched} != set(plans):
            raise ValueError('State index must retain every original filter seed')
        for file, record in matched:
            bind(file); job = load(source / 'jobs' / file.name); expected = plans[tuple(record['seeds'])]
            same_execution(expected, job, record, cell)
            state_array = bind(file.with_suffix('.npz'))
            choice_array = bind(Path(expected['target']).with_suffix('.npz'))
            choice_record = load(choice_array.with_suffix('.json'))
            if (digest(state_array) != record['array_sha256']
                    or digest(choice_array) != choice_record['predictions_sha256']):
                raise ValueError('Saved state or choice arrays changed')
            with np.load(state_array) as s, np.load(choice_array) as c:
                np.testing.assert_array_equal(s['seeds'], c['seeds'])
                np.testing.assert_array_equal(s['prediction'], c['probabilities'][0])
                np.testing.assert_array_equal(s['choices'], c['choices'])
                if record['trials'] != len(s['choices']):
                    raise ValueError('State trial count differs')
            entries.append((cell, file, state_array, job, record))
    output.mkdir(parents=True); (output / 'jobs').mkdir(); (output / 'states').mkdir()
    for cell, file, array, job, record in entries:
        name = cell + '_S' + str(record['seeds'][0]); target = output / 'states' / name
        target.with_suffix('.npz').symlink_to(os.path.relpath(array, target.parent.resolve()))
        write(target.with_suffix('.json'), dict(record, arm=cell, original_state_record=str(file.resolve()),
            parameter_estimation_scope=states.SCOPE, evaluation_use=states.USE))
        write(output / 'jobs' / (name + '.json'), dict(job, name=name, target=str(target.resolve()), reported_arm=cell))
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('State input changed during assembly')
    exported = dict(subjects=[settings['subject']], arms=list(CELLS), seeds_per_arm=8)
    write(output / 'protocol.json', dict(settings=exported, smoke=False,
        source_sha256={ENTRY: digest(ROOT / ENTRY)}, input_sha256=inputs,
        parameter_estimation_scope=states.SCOPE, evaluation_use=states.USE,
        parameter_candidate_policy='frozen_behavior_nominees_and_declared_near_tie',
        selected_parameters={cell: banks[cell]['candidate'] for cell in CELLS},
        original_nominations_changed=False, source_cross=str(cross.resolve())))
    write(output / 'completion.json', dict(status='complete', smoke=False, jobs=len(entries),
        exact_behavior_replays=len(entries), new_replays=24, reused_replays=16,
        protocol_sha256=digest(output / 'protocol.json')))
    print('Five cross cells assembled; 40 saved filters matched exactly', flush=True)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    if (protocol.get('parameter_estimation_scope') != states.SCOPE
            or protocol.get('evaluation_use') != states.USE
            or protocol.get('original_nominations_changed') is not False
            or set(protocol.get('selected_parameters', {})) != set(CELLS)):
        raise ValueError('Oral comparison needs explicit five-cell diagnostic provenance')
    keys = ('parameter_estimation_scope', 'evaluation_use', 'parameter_candidate_policy',
            'selected_parameters', 'original_nominations_changed', 'source_cross')
    return ({k: protocol[k] for k in keys},
            '<p>五组均采用SEARCH候选提议。组名前半部分表示RETAIN或ONLINE信念机制，后半部分表示旧参数、'
            '新参数或接近候选。三组参数来自已完成的前段行为估计与确认，原提名保持一致。'
            '口述用于比较这些固定组合的表征推断。全部组合共享原试次、前后段边界及八次滤波种子；'
            '本次属于被试208的开发诊断。</p>')


def align(source: Path, oral_audit: Path, output: Path) -> None:
    protocol, done = read(source / 'protocol.json'), read(source / 'completion.json')
    if (done['status'] != 'complete' or done['jobs'] != 40
            or done['protocol_sha256'] != digest(source / 'protocol.json')
            or protocol['source_sha256'][ENTRY] != digest(ROOT / ENTRY)):
        raise ValueError('Combined state index or readout adapter changed')
    original = alignment.estimation_context
    alignment.estimation_context = estimation_context
    try:
        alignment.run(source, oral_audit, output, reference_model='RETAIN_RETAIN')
    finally:
        alignment.estimation_context = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('assemble', 'align'), required=True)
    for name in ('config', 'source', 'oral-audit', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.phase == 'assemble' and args.config and args.output: assemble(args.config, args.output)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output: align(args.source, args.oral_audit, args.output)
    else: parser.error('Supply --config for assembly, or --source and --oral-audit for alignment, with --output')
