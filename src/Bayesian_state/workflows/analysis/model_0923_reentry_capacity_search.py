"""Cross capacity and global proposal probability within a fixed RETAIN model.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Reuse both completed
behavioral-finalist path banks; generate only the missing cells of a declared
2 x 2 parameter intervention. All other coordinates, task order and path seeds
stay fixed. The source finalists remain the behavioral nominations. New cells
have no fitted likelihood attached. The original precision-retention worker
executes generation; --smoke uses one missing cell, one seed and 16 trials.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

from . import model_0923_reentry_finalist_generation as finalists
from ...optimization.search.adaptive_proposals import candidate

ROOT = finalists.retention.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'
GLOBAL = 'engine.modules.hypo_transitions_mod.kwargs.search_controller.global_search'


def crossed_points(bank: list[dict], levels: dict) -> list[dict]:
    """Require a two-coordinate comparison with identical remaining parameters."""
    if (set(levels) != {CAPACITY, GLOBAL} or any(len(values) != 2 for values in levels.values())
            or any(len(set(values)) != 2 for values in levels.values())
            or any(type(value) is not int or value < 2 for value in levels[CAPACITY])
            or any(isinstance(value, bool) or not 0 <= value <= 1 for value in levels[GLOBAL])):
        raise ValueError('Require two valid distinct capacity and global-proposal levels')
    if len(bank) != 2 or len({row['id'] for row in bank}) != 2:
        raise ValueError('Require two distinct behavior finalists')
    anchor = bank[0]['hyperparams']
    unchanged = {key: value for key, value in anchor.items() if key not in levels}
    for row in bank:
        point = row['hyperparams']
        if ({key: value for key, value in point.items() if key not in levels} != unchanged
                or any(point[key] not in values for key, values in levels.items())
                or candidate(point, 'check')['id'] != row['id']):
            raise ValueError('Finalists differ outside the declared coordinates or grid')
    if any({row['hyperparams'][key] for row in bank} != set(values) for key, values in levels.items()):
        raise ValueError('Finalists must span both declared levels on each coordinate')
    known = {row['id']: row for row in bank}
    result = []
    for capacity, global_probability in product(levels[CAPACITY], levels[GLOBAL]):
        point = candidate(dict(anchor, **{CAPACITY: capacity, GLOBAL: global_probability}),
                          'capacity_global_proposal_intervention')
        result.append(deepcopy(known.get(point['id'], point)))
    return result


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new capacity-search intervention output')
    settings = yaml.safe_load(config.read_text())
    source = ROOT / settings['source']
    recalled = finalists.retention.helpers()
    v = recalled.base.validation()
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = v.digest(path)
        return path

    original = v.read(bind(source / 'protocol.json'))
    completion = v.read(bind(source / 'completion.json'))
    generation = v.read(bind(source / 'generation.json'))
    if (completion['status'] != 'complete' or completion['smoke']
            or completion['nominees_changed'] or generation['status'] != 'complete'
            or completion['protocol_sha256'] != v.digest(source / 'protocol.json')):
        raise ValueError('Require complete unchanged RETAIN finalist generation')
    for field in ('source_sha256', 'input_sha256'):
        for name, sha in original[field].items():
            if v.digest(bind(ROOT / name)) != sha:
                raise ValueError('Finalist implementation or input changed')
    bind(config)
    groups = {}
    for bank in original['banks']:
        for job in bank['chunks']:
            finalists.checked_chunk(job, v)
            for suffix in ('.json', '.npz', '.precision.json', '.experiment.json'):
                bind(Path(job['target']).with_suffix(suffix))
        groups.setdefault((bank['subject'], bank['model']), []).append(bank)
    hashes = dict(original['source_sha256'], **{ENTRY: v.digest(Path(__file__))})
    jobs, banks = [], []
    for (subject, arm), old in groups.items():
        grid = crossed_points([bank['candidate'] for bank in old], settings['levels'])
        known = {bank['candidate']['id']: bank for bank in old}
        templates = old[0]['chunks']
        seeds = [seed for job in templates for seed in job['seeds']]
        if len(seeds) != len(set(seeds)) or len(seeds) != original['settings']['autonomous_paths']:
            raise ValueError('Incomplete or repeated source trajectory seeds')
        if any([seed for job in bank['chunks'] for seed in job['seeds']] != seeds for bank in old):
            raise ValueError('Source finalists use different path seeds')
        for point in grid:
            if point['id'] in known:
                if not smoke:
                    banks.append(dict(deepcopy(known[point['id']]), reused=True))
                continue
            chunks = []
            for index, template in enumerate(templates[:1] if smoke else templates):
                job = deepcopy(template)
                name = f"CROSS_S{subject}_{arm}_{point['id']}_C{index:03d}"
                job.update(candidate=point, name=name, source_sha256=hashes, smoke=smoke,
                           target=str((output / 'chunks' / name).resolve()))
                if smoke:
                    job['seeds'] = job['seeds'][:1]
                chunks.append(job)
                jobs.append(job)
            banks.append(dict(subject=subject, model=arm, candidate=point,
                              frozen_nominee=False, reused=False, chunks=chunks))
            if smoke:
                break
        if smoke:
            break
    references = finalists.retention.reference_inputs(recalled, original['settings'])
    inputs.update(references)
    finalists.retention.configure(recalled, references)
    output.mkdir(parents=True)
    v.write(output / 'protocol.json', dict(settings=original['settings'], factorial_settings=settings,
        source=str(source.resolve()), smoke=smoke, input_sha256=inputs, source_sha256=hashes,
        jobs=jobs, banks=banks, role='fixed_parameter_capacity_global_proposal_intervention',
        data_use='development; autonomous own-choice feedback; no parameter fitting; oral excluded'))
    recalled.base.execute(jobs, output, 1 if smoke else original['settings']['parallel_budget'])
    rows = []
    for bank in banks:
        for job in bank['chunks']:
            finalists.checked_chunk(job, v)
        row = {key: value for key, value in bank.items() if key != 'chunks'}
        row.update(chunk_files=[job['target'] for job in bank['chunks']],
                   seeds=[seed for job in bank['chunks'] for seed in job['seeds']])
        rows.append(row)
    for name, sha in inputs.items():
        if v.digest(Path(name)) != sha:
            raise ValueError('Bound source changed during generation')
    v.write(output / 'generation.json', dict(status='complete', rows=rows))
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke,
        cells=len(banks), generated_paths=sum(len(job['seeds']) for job in jobs),
        reused_paths=sum(len(row['seeds']) for row in rows if row['reused']),
        nominees_changed=False, protocol_sha256=v.digest(output / 'protocol.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)
