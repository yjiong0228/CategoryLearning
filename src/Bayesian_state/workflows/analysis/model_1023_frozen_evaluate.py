"""Evaluate selected, reference and audited alternative Model 1022 parameters.

Use --study COMPLETED_HUMAN_REFIT --output NEW [--prepare-only | --smoke].
All parameter identities freeze before later choices and oral reports are read.
The three fixed-audit candidates per participant define the sensitivity set;
audit scores do not select a replacement estimate. Matching archived parameter
points reuse their complete evaluations. New points reuse the archived filter
and autonomous seeds for paired comparisons. The shared Model 1022 evaluator
collects oral decision bases in its original filters, with no extra simulations.

One common queue caps independent numerical processes at 128; each process has
one numeric thread. Short smoke evaluations use the existing 4+12 trial fixture
and two seeds, and never provide scientific evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import os

import numpy as np

from . import model_1022_compact_evaluate as evaluate
from . import model_1023_estimation_reliability as reliability
from ...evaluation.model_0923_h_validation import checked_arrays
from ...optimization.search.compact_joint import behavior_key

ROOT, read, write, digest = reliability.ROOT, reliability.read, reliability.write, reliability.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def paired_seed_jobs(groups: dict, reference_seeds: dict[str, list[int]]) -> None:
    """Replace seed banks jointly in aggregate jobs and every disjoint chunk."""
    for role, records in groups.items():
        seeds = list(reference_seeds[role])
        if len(seeds) != len(set(seeds)):
            raise ValueError('Archived evaluation repeats a seed')
        for job, parts in records:
            if len(job['seeds']) != len(seeds) or sum(len(p['seeds']) for p in parts) != len(seeds):
                raise ValueError('Paired evaluation has a different seed budget')
            job['seeds'] = seeds
            offset = 0
            for part in parts:
                width = len(part['seeds'])
                part['seeds'] = seeds[offset:offset + width]
                offset += width


def prepare(study: Path, output: Path, smoke: bool) -> tuple[list[dict], dict]:
    finished, protocol = read(study/'completion.json'), read(study/'protocol.json')
    if finished['status'] != 'complete' or finished['phase'] != 'human' or finished['smoke']:
        raise ValueError('Require completed scientific human prefix fits')
    inputs = {str((study/name).resolve()): digest(study/name)
              for name in ('completion.json', 'protocol.json', 'cases.json', 'selected_nominees.json')}
    cases = read(study/'cases.json')['cases']
    if sorted(c['subject'] for c in cases) != sorted(protocol['settings']['human_subjects']):
        raise ValueError('Missing registered human cases')
    archive = ROOT/protocol['settings']['evaluation']
    planned = []
    for case in cases[:1] if smoke else cases:
        fit = Path(case['output'])
        frozen = read(fit/'nominees.json')
        done = read(fit/'completion.json')
        audit_dir = study/'audit_fit'/case['case_id']
        audited = read(audit_dir/'nominees.json')
        audit_done = read(audit_dir/'completion.json')
        if (frozen['status'] != 'frozen_before_test' or frozen['smoke'] or done['test_choices_used']
                or done['nomination_sha256'] != digest(fit/'nominees.json')
                or audited['selected_before_audit'] is not True
                or audited['selection_sha256'] != digest(fit/'nominees.json')
                or audit_done['nomination_sha256'] != digest(audit_dir/'nominees.json')):
            raise ValueError('Parameter selection, independent audit or data boundary changed')
        original = archive/case['case_id']
        original_fit = Path(case['source_fit'])
        previous = read(original_fit/'nominees.json')
        original_done = read(original/'completion.json')
        original_protocol = read(original/'protocol.json')
        if (original_done['status'] != 'complete' or original_done['smoke']
                or original_protocol['nomination_sha256'] != digest(original_fit/'nominees.json')
                or original_protocol['compact_variant'] != 'TRANSFER'):
            raise ValueError('Reference evaluation does not match the archived TRANSFER fit')
        selected = frozen['nominee']['candidate']
        reference = previous['nominee']['candidate']
        points = audited['nominee']['finalists']
        confirmation = {p['id']: p for p in frozen['nominee']['finalists']}
        if len(points) != case['settings']['audit_candidates'] or any(p['id'] not in confirmation for p in points):
            raise ValueError('Sensitivity set differs from the pre-audit contenders')
        if selected['id'] not in {p['id'] for p in points} or reference['id'] not in {p['id'] for p in points}:
            raise ValueError('Selected point or original reference missing from audit')
        seeds = {}
        for role in ('conditional', 'autonomous'):
            record, = read(original/role/'evaluation.json')['rows']
            if record['candidate']['hyperparams'] != reference['hyperparams']:
                raise ValueError('Reference arrays use different parameters')
            arrays = checked_arrays(original/role, record)
            expected = 8 if role == 'conditional' else 128
            if len(arrays['seeds']) != expected:
                raise ValueError('Reference evaluation has an unexpected seed budget')
            seeds[role] = [int(s) for s in arrays['seeds'][:2 if smoke else expected]]
            file = original/role/'evaluate'/f'subject_{case["subject"]}'/'COMPACT'/(reference['id']+'.npz')
            inputs[str(file.resolve())] = digest(file)
            inputs[str((original/role/'evaluation.json').resolve())] = digest(original/role/'evaluation.json')
        fitting_seeds = {s for name, bank in case['independent_seed_banks'].items()
                         if name != 'evaluate' for s in bank}
        if any(set(bank) & fitting_seeds for bank in seeds.values()):
            raise ValueError('Paired evaluation overlaps new fitting or audit seeds')
        for file in (fit/'nominees.json', fit/'completion.json', audit_dir/'nominees.json',
                     audit_dir/'completion.json', original/'protocol.json', original/'completion.json',
                     original_fit/'nominees.json'):
            inputs[str(file.resolve())] = digest(file)
        for point in points:
            roles = []
            if point['id'] == selected['id']:
                roles.append('selected')
            if point['id'] == reference['id']:
                roles.append('reference')
            if not roles:
                roles.append('alternative')
            reuse = behavior_key(point['hyperparams']) == behavior_key(reference['hyperparams']) and not smoke
            label = case['case_id']+'_'+point['id']
            record = dict(case_id=label, original_case_id=case['case_id'], subject=case['subject'],
                cohort=case['cohort'], condition=case['condition'], candidate=point['id'], roles=roles,
                reuse_archive=reuse, reference_seeds=seeds,
                data_scope='Original prefix estimation; later behavior evaluated after freezing',
                selection_nll=confirmation[point['id']]['mean_nll'], audit_nll=point['mean_nll'])
            if reuse:
                record['evaluation'] = str(original.resolve())
                record['candidate_hyperparams'] = deepcopy(point['hyperparams'])
            else:
                proxy = output/'frozen_parameters'/label
                proxy.mkdir(parents=True)
                nominee = dict(frozen['nominee'], candidate=confirmation[point['id']])
                own = dict(frozen, smoke=smoke, nominee=nominee, nominees=[nominee],
                    sensitivity_roles=roles, selection_used_only_training=True,
                    source_nomination_sha256=digest(fit/'nominees.json'))
                own['source_sha256'] = dict(own['source_sha256'], **{ENTRY: digest(ROOT/ENTRY)})
                write(proxy/'nominees.json', own)
                write(proxy/'completion.json', dict(status='complete', smoke=smoke, test_choices_used=False,
                    nomination_sha256=digest(proxy/'nominees.json')))
                record['case'] = dict(case, output=str(proxy.resolve()), case_id=label)
                record['candidate_hyperparams'] = deepcopy(point['hyperparams'])
                record['evaluation'] = str((output/'predictions'/label).resolve())
            planned.append(record)
    write(output/'cases.json', dict(cases=planned, inputs_sha256=inputs, oral_used_for_selection=False,
        candidates_selected_before_later_behavior=True, evaluation_seeds='Reused archived banks for paired contrasts'))
    return planned, inputs


def evaluate_point(record: dict, smoke: bool, pool, gate) -> None:
    own = Path(record['evaluation'])
    own.mkdir(parents=True)
    case = record['case']
    chunks, groups = evaluate.make_jobs(case, own, smoke)
    paired_seed_jobs(groups, record['reference_seeds'])
    fit = Path(case['output'])
    write(own/'protocol.json', dict(fit=str(fit), nomination_sha256=digest(fit/'nominees.json'),
        compact_variant='TRANSFER', case_id=case['case_id'], cohort=case['cohort'], smoke=smoke,
        sensitivity_roles=record['roles'], data_use=record['data_scope'],
        route_source_sha256=evaluate.sources(), paired_route_sha256=digest(ROOT/ENTRY), jobs=chunks))
    write(own/'decision_basis_capture_protocol.json', dict(source_sha256=evaluate.capture.sources(),
        route_source_sha256=evaluate.sources(), additional_filters=0, timing='after_choice_before_current_feedback'))
    for future in as_completed([pool.submit(evaluate.launch, job, own, gate) for job in chunks]):
        future.result()
    for role, items in groups.items():
        rows = [evaluate.driver.io.combine_evaluation_chunks(job, parts, role == 'autonomous') for job, parts in items]
        write(own/role/'evaluation.json', dict(status='complete', rows=rows))
    write(own/'completion.json', dict(status='complete', smoke=smoke, jobs=len(chunks), compact_variant='TRANSFER',
        nomination_sha256=digest(fit/'nominees.json'), sensitivity_roles=record['roles'],
        evaluation_seeds='Archived conditional and autonomous seeds; independent of the new parameter fit'))
    print(case['case_id']+': paired frozen evaluation complete', flush=True)


def run(study: Path, output: Path, smoke: bool, prepare_only: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new frozen-evaluation directory')
    output.mkdir(parents=True)
    planned, inputs = prepare(study, output, smoke)
    workers = 1 if smoke else min(128, len(os.sched_getaffinity(0)))
    write(output/'protocol.json', dict(study=str(study.resolve()), smoke=smoke, maximum_workers=workers,
        inputs_sha256=inputs, source_sha256={ENTRY: digest(ROOT/ENTRY), **evaluate.sources()},
        selection='Three pre-audit contenders per subject; chosen estimate fixed before audit',
        archived_reuses=sum(r['reuse_archive'] for r in planned)))
    if prepare_only:
        write(output/'preparation_complete.json', dict(status='complete', numeric_jobs=0, candidates=len(planned)))
        return
    pending = [r for r in planned if not r['reuse_archive']]
    gate = evaluate.ResourceGate(output, workers, 0)
    with reliability.NumericQueue(workers, output) as numerical:
        with ThreadPoolExecutor(max_workers=1 if smoke else max(1, len(pending))) as controls:
            futures = [controls.submit(evaluate_point, r, smoke, numerical, gate) for r in pending]
            for future in futures:
                future.result()
    write(output/'completion.json', dict(status='complete', smoke=smoke, candidates=len(planned),
        archived_reuses=len(planned)-len(pending), new_evaluations=len(pending),
        cases_sha256=digest(output/'cases.json'), additional_filters_for_oral_capture=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.study, args.output, args.smoke, args.prepare_only)
