"""Diagnose the two TRANSFER recovery shortfalls with a bounded joint profile.

Use --config YAML --output NEW [--prepare-only | --smoke]. At each previously
frozen S306 estimate, cross the full existing beta_init and gamma supports; all
other fitted coordinates stay fixed. The previous estimate is always retained
in selection and confirmation. New independent seed banks score each stage.

This is a retrospective numerical diagnostic designed after reviewing the
original generating-point gaps. Candidate construction uses only the previous
estimate and declared parameter support. New generating-point scores follow
freezing of both profile estimates. It is not independent recovery validation.
The shared scorer, precision-transfer adapter and state capture are reused.
One numerical queue shares the CPU allocation with the final primary filters.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import itertools
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

from . import model_1022_compact_refit as refit
from .model_1022_compact_evaluate import ResourceGate

ROOT, read, write, digest = refit.ROOT, refit.read, refit.write, refit.digest
recovery, driver = refit.recovery, refit.driver
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
EVALUATE = 'src/Bayesian_state/workflows/analysis/model_1022_compact_evaluate.py'
BETA = 'engine.modules.beta_mod.kwargs.beta_init'
GAMMA = 'engine.modules.memory_mod.kwargs.gamma'
SCOPE = ('Retrospective beta_init by gamma profile at two fixed estimates of one '
         'generating point; all other coordinates fixed; not independent joint recovery')


def profile_bank(space: dict, anchor: dict) -> tuple[list[dict], dict]:
    """Cross both complete supports around a frozen behavior-only estimate."""
    point = anchor['hyperparams']
    if set(point) != set(space) or any(point[k] not in values for k, values in space.items()):
        raise ValueError('Frozen estimate lies outside the declared support')
    if len(space[BETA]) < 2 or len(space[GAMMA]) < 2:
        raise ValueError('Both profile coordinates must vary')
    required = refit.candidate(point, 'previous_behavior_estimate')
    bank = {required['id']: required}
    for beta, gamma in itertools.product(space[BETA], space[GAMMA]):
        own = dict(point, **{BETA: beta, GAMMA: gamma})
        candidate = refit.candidate(own, 'precision_pairing_grid')
        bank.setdefault(candidate['id'], candidate)
    if len(bank) != len(space[BETA])*len(space[GAMMA]):
        raise ValueError('Profile grid contains duplicate parameter points')
    return list(bank.values()), required


def seed_bank(settings: dict, subject: int) -> set[int]:
    return {refit.stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
        stage=stage, subject=subject, repeat=i)) for stage in ('screen','selection','confirmation')
        for i in range(settings['budgets'][stage]['seeds'])}


def prepare(policy: dict, output: Path, smoke: bool) -> tuple[list[dict], Path]:
    study = ROOT/policy['study']
    if (policy['cases'] != ['S306_TRANSFER_R0','S306_TRANSFER_R1']
            or read(study/'recovery_scores/completion.json')['status'] != 'complete'):
        raise ValueError('Complete the registered original generating-point comparison first')
    inputs = {}; cases = []; declared = []
    def bind(file):
        inputs[str(file.resolve())] = digest(file)
        return file
    original = read(bind(study/'cases.json'))['cases']
    prepared = output/'prepared'; prepared.mkdir()
    for old in original:
        if old['case_id'] not in policy['cases'][:1 if smoke else 2]:
            continue
        previous = Path(old['output'])
        frozen = read(bind(previous/'nominees.json')); done = read(bind(previous/'completion.json'))
        if (old['kind'] != 'recovery' or old['variant'] != 'TRANSFER' or old['condition'] != 3
                or frozen['smoke'] or frozen['status'] != 'frozen_before_truth'
                or done['status'] != 'complete' or done['nomination_sha256'] != digest(previous/'nominees.json')):
            raise ValueError('Profile requires the original frozen scientific TRANSFER estimate')
        for name, sha in frozen['source_sha256'].items():
            if digest(ROOT/name) != sha: raise ValueError('Scientific implementation changed: '+name)
        starts, required = profile_bank(old['space'], frozen['nominee']['candidate'])
        if len(starts) != 56: raise ValueError('Require the full registered 8 by 7 support')
        settings = deepcopy(old['settings'])
        settings.update(protocol=policy['protocol'], seed=refit.stable_seed(dict(
            base=policy['seed'], case=old['case_id'], role='precision_pairing_profile')),
            initial_candidates=56, rounds=0, selection_candidates=policy['selection_candidates'],
            confirmation_candidates=policy['confirmation_candidates'], budgets=deepcopy(policy['budgets']),
            scope=SCOPE, parallel_budget=policy['parallel_budget'])
        seeds = seed_bank(settings, old['subject'])
        if len(seeds) != 14 or old['generation_seed'] in seeds:
            raise ValueError('New profile stage seeds overlap')
        previous_seeds = set()
        for file in (previous/'jobs').glob('*.json'):
            previous_seeds.update(read(bind(file))['seeds'])
        if seeds & previous_seeds: raise ValueError('New scoring reuses old fitting seeds')
        own = dict(deepcopy(old), settings=settings, starts=starts[:4] if smoke else starts,
            required=required, previous_fit=str(previous.resolve()),
            output=str((output/'fit'/old['case_id']).resolve()),
            profile_coordinates=[BETA,GAMMA], original_joint_recovery=False,
            design_informed_by_previous_recovery_failure=True, generating_parameters_used_for_starts=False)
        file = prepared/old['case_id']/'case.json'; file.parent.mkdir(); write(file, own)
        declared.append(dict(case_id=own['case_id'], file=str(file.resolve()), sha256=digest(file)))
        cases.append(own)
    if len(cases) != (1 if smoke else 2): raise ValueError('Missing registered profile cases')
    # Preserve the original truths opaquely here. Their prior inspection is
    # explicitly disclosed in the diagnostic scope; values never seed this grid.
    truth_file = bind(study/'recovery_prepared/truths.json')
    shutil.copyfile(truth_file, prepared/'truths.json')
    write(prepared/'protocol.json', dict(cases=declared, inputs_sha256=inputs, scope=SCOPE,
        candidate_policy='Full existing beta_init by gamma support around each frozen estimate',
        new_truth_scoring='After both new profile estimates freeze', oral_used=False))
    write(prepared/'completion.json', dict(status='complete', cases=len(cases),
        protocol_sha256=digest(prepared/'protocol.json'), truths_sha256=digest(prepared/'truths.json')))
    recovery.checked_cases(prepared)
    write(output/'cases.json', dict(cases=cases, inputs_sha256=inputs, scope=SCOPE))
    return cases, prepared


class GatedPool:
    """Use the common numerical executor while accounting for external filters."""
    def __init__(self, pool: ThreadPoolExecutor, gate: ResourceGate):
        self.pool, self.gate, self._max_workers = pool, gate, pool._max_workers

    def submit(self, function, *args, **kwargs):
        def run():
            self.gate.enter()
            try: return function(*args, **kwargs)
            finally: self.gate.leave()
        return self.pool.submit(run)


def fit_profile(case: dict, smoke: bool) -> dict:
    settings, output = case['settings'], Path(case['output']); output.mkdir(parents=True)
    hashes = refit.sources(driver.io)
    write(output/'protocol.json', dict(settings=settings, case_id=case['case_id'], smoke=smoke,
        source_sha256=hashes, space=case['space'], scope=SCOPE))
    key = case['subject'], 'COMPACT'
    rows = driver.base.stage(settings, Path(settings['data']), output, {key:case['starts']}, 'screen', smoke)[key]
    print(case['case_id']+': two-coordinate grid complete', flush=True)
    for stage, count in [('selection',settings['selection_candidates']),('confirmation',settings['confirmation_candidates'])]:
        bank = driver.base.shortlist(rows, case['required'], 2 if smoke else count)
        rows = driver.base.stage(settings, Path(settings['data']), output, {key:bank}, stage, smoke)[key]
    nominee = dict(case_id=case['case_id'], subject=case['subject'], replicate=case['replicate'],
        model='COMPACT', compact_variant='TRANSFER', candidate=rows[0], finalists=rows,
        unique_candidates=len(case['starts']), reference_id=case['required']['id'])
    write(output/'nominees.json', dict(status='frozen_before_truth', settings=settings, smoke=smoke,
        source_sha256=hashes, nominee=nominee, nominees=[nominee], scope=SCOPE,
        previous_generating_comparison_reviewed=True, data=settings['data']))
    write(output/'completion.json', dict(status='complete', smoke=smoke, test_choices_used=False,
        nomination_sha256=digest(output/'nominees.json'), scope=SCOPE))
    print(case['case_id']+': profile estimate frozen', flush=True)
    return nominee


@contextmanager
def truth_scope(output: Path):
    original = recovery.write
    def scoped(file, record):
        if Path(file) == output/'completion.json':
            record = dict(record, scope=SCOPE, previous_generating_comparison_reviewed=True)
        return original(file, record)
    recovery.write = scoped
    try: yield
    finally: recovery.write = original


def run(config: Path, output: Path, smoke: bool, prepare_only: bool) -> None:
    if output.exists(): raise FileExistsError('Use a new profile output directory')
    policy = yaml.safe_load(config.read_text()); study = ROOT/policy['study']
    if not 1 <= policy['parallel_budget'] <= 128: raise ValueError('Invalid process budget')
    if (policy['selection_candidates'],policy['confirmation_candidates']) != (4,3):
        raise ValueError('Require the bounded four-selection, three-confirmation design')
    output.mkdir(parents=True); shutil.copyfile(config,output/'config.yaml')
    shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    cases,prepared = prepare(policy,output,smoke)
    original_sources = refit.sources
    def bound_sources(v):
        return dict(original_sources(v), **{name:digest(ROOT/name) for name in (ENTRY,EVALUATE)})
    refit.sources = bound_sources
    driver.base.archive(output,config,bound_sources(driver.io))
    workers = 1 if smoke else min(policy['parallel_budget'],len(os.sched_getaffinity(0)))
    if read(study/'evaluation/S306_FULL_TRANSFER/completion.json')['status'] != 'complete':
        raise ValueError('Only the final S205 filters may remain in the parent numerical pool')
    remaining = 0 if smoke or (study/'completion.json').exists() else 8
    gate = ResourceGate(study,workers,remaining)
    write(output/'protocol.json',dict(settings=policy,smoke=smoke,workers=workers,
        scope=SCOPE,parent_remaining_filter_bound=remaining,source_sha256=bound_sources(driver.io),
        python=sys.version,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','PyYAML')},
        oral_used=False,additional_grid_rounds=1,new_points_per_case=55,
        previous_estimate_always_confirmed=True,extra_refinement_rounds=0))
    if prepare_only:
        write(output/'preparation_complete.json',dict(status='complete',cases=len(cases),numeric_jobs=0));return
    fit,scored=output/'fit',output/'truth_scores';fit.mkdir()
    write(fit/'protocol.json',dict(prepared=str(prepared.resolve()),smoke=smoke,workers=workers,
        source_sha256=bound_sources(driver.io),scope=SCOPE))
    print(f'{len(cases)} profile cases; up to {workers} numerical workers; {remaining} parent filters accounted for',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as actual:
        pool=GatedPool(actual,gate);refit.configure_executor(pool)
        with ThreadPoolExecutor(max_workers=len(cases)) as controllers:
            nominees=[f.result() for f in [controllers.submit(fit_profile,c,smoke) for c in cases]]
        write(fit/'nominees.json',dict(status='frozen_before_truth',smoke=smoke,nominees=nominees,scope=SCOPE))
        write(fit/'completion.json',dict(status='complete',smoke=smoke,cases=len(cases),scope=SCOPE,
            nomination_sha256=digest(fit/'nominees.json')))
        original_executor=recovery.configure_executor
        recovery.configure_executor=refit.configure_executor
        try:
            with truth_scope(scored): recovery.score_truth(prepared,fit,scored,numerical_pool=pool)
        finally: recovery.configure_executor=original_executor
    from ...evaluation.model_1021_simplification_recovery_states import summarize
    summarize(prepared,fit,scored,output/'states')
    write(output/'resources.json',dict(maximum=workers,peak=gate.peak,events=gate.events,numeric_threads=1))
    write(output/'completion.json',dict(status='complete',smoke=smoke,cases=len(cases),scope=SCOPE,
        original_joint_recovery_retained=True,additional_search_rounds=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    group=parser.add_mutually_exclusive_group();group.add_argument('--prepare-only',action='store_true');group.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.config,args.output,args.smoke,args.prepare_only)
