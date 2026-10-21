"""One bounded search extension for the four completed COMPACT recovery cases.

Use --parent FOLLOWTHROUGH --output NEW. Each case receives 64 new proposals
from eight diverse behavior-fit elites, plus reevaluation of the elites and
previous nominee. New independent seed banks score screening, selection and
confirmation. Generating parameters are opened only after all extended estimates
freeze. This retrospective optimizer diagnostic leaves original recovery intact.
--smoke checks one case using short two-particle filters and one worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import shutil

from . import model_1021_simplification_recovery as recovery
from ...optimization.search.adaptive_proposals import propose_round, select_elites
from ...utils.seeding import stable_seed

read, write, digest = recovery.read, recovery.write, recovery.digest
ENTRY = str(Path(__file__).resolve().relative_to(recovery.ROOT))


def proposal_bank(case: dict, previous: Path) -> tuple[list[dict], dict]:
    """Candidate generation reads only previous synthetic-behavior scores."""
    rows = {}
    for stage in ('screen', 'refine'):
        complete = read(previous/(stage+'_completion.json'))
        if complete['status'] != 'complete':
            raise ValueError('Original recovery search is incomplete')
        groups = complete['groups']
        if len(groups) != 1 or (groups[0]['subject'], groups[0]['model']) != (case['subject'], 'COMPACT'):
            raise ValueError('Previous scores belong to another case')
        rows.update({r['id']: r for r in groups[0]['rows']})
    required = read(previous/'nominees.json')['nominee']['candidate']
    if required['id'] not in rows:
        raise ValueError('Original nominee absent from the evaluated search bank')
    elites = select_elites(list(rows.values()), 8, exclude_key=recovery.driver.H+'max_capacity')
    proposed = propose_round(elites, case['space'], set(rows), 8,
        stable_seed(dict(base=case['settings']['seed'], role='bounded_recovery_extension_proposals')), .5, .25)
    if len(elites) != 8 or len(proposed) != 64:
        raise ValueError('The bounded extension requires eight elites and 64 distinct new proposals')
    bank = {r['id']: r for r in [required]+elites+proposed}
    if set(r['id'] for r in proposed) & set(rows):
        raise ValueError('Extension repeats an already evaluated point')
    return list(bank.values()), required


def prepare(parent: Path, output: Path) -> list[dict]:
    if output.exists():
        raise FileExistsError('Use a new extension preparation directory')
    original = parent/'recovery_prepared'
    if read(parent/'recovery_fit/completion.json')['status'] != 'complete':
        raise ValueError('Complete the original recovery first')
    cases = recovery.checked_cases(original)
    output.mkdir(parents=True)
    declared, result = [], []
    for case in cases:
        own = deepcopy(case)
        bank, required = proposal_bank(case, parent/'recovery_fit'/case['case_id'])
        own.update(starts=bank, required=required, original_unique_candidates=56, new_unique_candidates=64,
                   generating_parameters_used_for_starts=False)
        own['settings'].update(protocol='model1021_recovery_rescue_v1', initial_candidates=len(bank),
            seed=stable_seed(dict(base=case['settings']['seed'], role='bounded_recovery_extension_scoring')))
        file = output/case['case_id']/'case.json'
        file.parent.mkdir()
        write(file, own)
        declared.append(dict(case_id=case['case_id'], file=str(file.resolve()), sha256=digest(file)))
        result.append(own)
    write(output/'protocol.json', dict(cases=declared, parent=str(parent.resolve()),
        proposal_policy='Eight diverse previous behavior-fit elites; 64 new local/jump/joint proposals per case',
        truth_policy='Copied opaquely; numerical estimates freeze before truth scoring',
        scope='One retrospective bounded optimizer extension, all four original cases'))
    # The planner does not parse generating parameters or use them to propose.
    shutil.copyfile(original/'truths.json', output/'truths.json')
    write(output/'completion.json', dict(status='complete', cases=len(cases),
        protocol_sha256=digest(output/'protocol.json'), truths_sha256=digest(output/'truths.json'),
        inputs_sha256={str(p.resolve()): digest(p) for p in (original/'completion.json',
            parent/'recovery_fit/nominees.json', parent/'recovery_fit/completion.json')}))
    return result


def fit_case(case: dict, output: Path, smoke: bool) -> dict:
    output.mkdir()
    settings = case['settings']
    key = case['subject'], 'COMPACT'
    bank = case['starts'][:4] if smoke else case['starts']
    write(output/'protocol.json', dict(settings=settings, case_id=case['case_id'], smoke=smoke,
        source_sha256=recovery.recovery_hashes(recovery.driver.io), space=case['space'],
        initialization='Previous behavior-fit elites and bounded new proposals; generating point unused'))
    rows = recovery.driver.base.stage(settings, Path(settings['data']), output, {key: bank}, 'screen', smoke)
    for stage, count in [('selection', 4), ('confirmation', 2)]:
        shortlisted = recovery.driver.base.shortlist(rows[key], case['required'], 2 if smoke else count)
        rows = recovery.driver.base.stage(settings, Path(settings['data']), output, {key: shortlisted}, stage, smoke)
    nominee = dict(case_id=case['case_id'], subject=case['subject'], replicate=case['replicate'],
        model='COMPACT', candidate=rows[key][0], finalists=rows[key],
        unique_candidates=120 if not smoke else None, extension_candidates=64 if not smoke else None)
    write(output/'nominees.json', dict(status='frozen_before_truth', settings=settings, smoke=smoke,
        nominee=nominee, source_sha256=recovery.recovery_hashes(recovery.driver.io)))
    write(output/'completion.json', dict(status='complete', smoke=smoke, truths_opened=False,
        nomination_sha256=digest(output/'nominees.json')))
    print(case['case_id']+': bounded extension frozen', flush=True)
    return nominee


def run(parent: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new bounded recovery output directory')
    # The original four recovery controllers are finished. Only the eight slow
    # S205 reference filters remain in the original numerical pool at launch.
    if read(parent/'recovery_states/completion.json')['status'] != 'complete':
        raise ValueError('Finish original recovery and state summaries first')
    evaluation = parent/'evaluation'
    completed_filters = list((evaluation/'conditional/evaluate').glob('**/chunks/*.npz'))
    for subject, model in [(108, 'REFERENCE'), (108, 'COMPACT'), (205, 'COMPACT'),
                           (306, 'REFERENCE'), (306, 'COMPACT')]:
        files = list((evaluation/f'conditional/evaluate/subject_{subject}/{model}/chunks').glob('*.mechanism.json'))
        if len(files) != 8:
            raise ValueError('Five primary evaluation cells must finish before the 120-worker extension')
    output.mkdir(parents=True)
    prepared, fitted, scored = [output/s for s in ('prepared', 'fit', 'truth_scores')]
    cases = prepare(parent, prepared)
    cases = cases[:1] if smoke else cases
    workers = 1 if smoke else min(120, len(os.sched_getaffinity(0)))
    hashes = recovery.recovery_hashes

    def bound_sources(v):
        return dict(hashes(v), **{ENTRY: digest(recovery.ROOT/ENTRY)})

    recovery.recovery_hashes = bound_sources
    fitted.mkdir()
    write(output/'protocol.json', dict(parent=str(parent.resolve()), smoke=smoke,
        workers=workers, remaining_primary_filters_upper_bound=8, completed_primary_filters=len(completed_filters),
        source_sha256=bound_sources(recovery.driver.io), additional_search_rounds=1,
        oral_used=False, new_candidates_per_case=64, previous_nominee_always_confirmed=True))
    write(fitted/'protocol.json', dict(prepared=str(prepared.resolve()), smoke=smoke, workers=workers,
        source_sha256=bound_sources(recovery.driver.io), estimation='Bounded extension from previous behavior fits'))
    print(f'{len(cases)} cases; {workers} shared numeric workers; one bounded extension', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        recovery.configure_executor(pool)
        with ThreadPoolExecutor(max_workers=len(cases)) as control:
            futures = [control.submit(fit_case, case, fitted/case['case_id'], smoke) for case in cases]
            nominees = [future.result() for future in futures]
        write(fitted/'nominees.json', dict(status='frozen_before_truth', smoke=smoke, nominees=nominees))
        write(fitted/'completion.json', dict(status='complete', smoke=smoke, cases=len(cases), truths_opened=False,
            nomination_sha256=digest(fitted/'nominees.json')))
        recovery.score_truth(prepared, fitted, scored, numerical_pool=pool)
    from ...evaluation.model_1021_simplification_recovery_states import summarize
    summarize(prepared, fitted, scored, output/'states')
    write(output/'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        scope='One bounded optimizer extension; original scientific comparison and recovery retained'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.parent, args.output, args.smoke)
