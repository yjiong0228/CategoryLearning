"""Bounded joint-search development, new sequence checks and human prefix fits.

Run --config YAML --phase diagnostic|validation|human --output NEW, optionally
--prepare-only or --smoke. Diagnostic cases reuse the two failed S306 sequences;
validation uses registered, previously unused recovery seeds from the saved
autonomous bank. Human fits retain the original training/test boundaries.

The COMPACT+TRANSFER equations, support and likelihood are frozen. A population
combines coordinate, pair and multi-coordinate changes with global restarts.
Leaders from all screening stages receive a common independent recheck. Fixed
post-selection audits use another seed bank and preserve the selected estimate.
Generating parameters are scored after selection and audit freeze. All numerical
jobs share one queue capped by the available allocation (at most 128 processes),
with one numeric thread per process. Existing scientific artifacts stay intact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
from threading import Event, Lock, Thread
import time

import numpy as np
import yaml

from . import model_1022_compact_refit as refit
from . import model_1022_compact_scope_control as fixtures
from ...optimization.search import compact_joint as proposals

ROOT, read, write, digest = refit.ROOT, refit.read, refit.write, refit.digest
driver, recovery = refit.driver, refit.recovery
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PROPOSALS = str(Path(proposals.__file__).resolve().relative_to(ROOT))


class NumericQueue:
    """Measure shared numerical slots, including blocked case controllers."""
    def __init__(self, workers: int, output: Path):
        self._max_workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.output, self.lock, self.stop = output, Lock(), Event()
        self.active = self.peak = self.completed = self.failed = 0
        self.snapshot_count = 0
        self.monitor = Thread(target=self.watch, daemon=True)

    def snapshot(self, final: bool = False):
        with self.lock:
            state = dict(maximum=self._max_workers, active=self.active, peak=self.peak,
                         completed=self.completed, failed=self.failed, unix_time=time.time())
            number = self.snapshot_count
            self.snapshot_count += 1
        destination = (self.output/'resource_status.json' if final else
                       self.output/'resource_snapshots'/f'{number:06d}.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        write(destination, state)

    def watch(self):
        while not self.stop.wait(30):
            self.snapshot()

    def submit(self, function, *args, **kwargs):
        def execute():
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                result = function(*args, **kwargs)
            except BaseException:
                with self.lock:
                    self.failed += 1
                raise
            else:
                with self.lock:
                    self.completed += 1
                return result
            finally:
                with self.lock:
                    self.active -= 1
        return self.pool.submit(execute)

    def __enter__(self):
        self.snapshot()
        self.monitor.start()
        return self

    def __exit__(self, *error):
        self.pool.shutdown(wait=True, cancel_futures=error[0] is not None)
        self.stop.set()
        self.monitor.join()
        self.snapshot(final=True)


def independent_seed_banks(settings: dict, subject: int, generation_seed: int | None = None) -> dict:
    banks = {}
    for stage, budget in settings['budgets'].items():
        banks[stage] = [refit.stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
            stage=stage, subject=subject, repeat=i)) for i in range(budget['seeds'])]
    audit_seed = refit.stable_seed(dict(base=settings['seed'], role='post_freeze_audit'))
    banks['post_freeze_audit'] = [refit.stable_seed(dict(protocol=settings['protocol'], base=audit_seed,
        stage='confirmation', subject=subject, repeat=i)) for i in range(settings['budgets']['confirmation']['seeds'])]
    flat = [seed for values in banks.values() for seed in values]
    if len(flat) != len(set(flat)) or generation_seed in flat:
        raise ValueError('Screening, selection, audit, evaluation or generation seeds overlap')
    return banks


def settings_for_case(old: dict, policy: dict, case_id: str) -> dict:
    settings = deepcopy(old)
    for key in ('protocol', 'parallel_budget', 'initial_candidates', 'rounds', 'proposals',
                'carry_candidates', 'carry_top', 'archive_per_stage', 'selection_candidates',
                'confirmation_candidates', 'confirmation_top', 'audit_candidates', 'budgets',
                'autonomous_paths', 'autonomous_chunk', 'scope'):
        settings[key] = deepcopy(policy[key])
    settings.update(seed=refit.stable_seed(dict(base=policy['seed'], case=case_id, role='joint_search')),
                    case_id=case_id, compact_variant='TRANSFER', particle_overrides={})
    return settings


def checked_frozen(case: dict) -> dict:
    folder = Path(case['output'])
    frozen, done = read(folder/'nominees.json'), read(folder/'completion.json')
    expected = 'frozen_before_test' if case['kind'] == 'human' else 'frozen_before_truth'
    if (frozen['status'] != expected or frozen['smoke'] or done['status'] != 'complete'
            or done['test_choices_used'] or done['nomination_sha256'] != digest(folder/'nominees.json')
            or case['variant'] != 'TRANSFER'):
        raise ValueError('Require a completed prefix-only TRANSFER estimate')
    for name, sha in frozen['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Frozen scientific source changed: ' + name)
    return frozen


def save_prepared(prepared: Path, cases: list[dict], *, truth_source: Path | None = None,
                  truths: list[dict] | None = None, inputs: dict | None = None) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    declared = []
    for case in cases:
        file = prepared/case['case_id']/'case.json'
        file.parent.mkdir(parents=True, exist_ok=True)
        write(file, case)
        declared.append(dict(case_id=case['case_id'], file=str(file.resolve()), sha256=digest(file)))
    if truth_source is not None:
        shutil.copyfile(truth_source, prepared/'truths.json')
    else:
        write(prepared/'truths.json', dict(rows=truths, use='Post-freeze scoring only'))
    write(prepared/'protocol.json', dict(cases=declared, inputs_sha256=inputs or {},
        truth_policy='Preparation stores generating parameters separately; search reads case definitions and behavior only'))
    write(prepared/'completion.json', dict(status='complete', cases=len(cases),
        protocol_sha256=digest(prepared/'protocol.json'), truths_sha256=digest(prepared/'truths.json')))
    recovery.checked_cases(prepared)


def prepare(policy: dict, phase: str, output: Path, smoke: bool) -> tuple[list[dict], Path | None]:
    cases, inputs, truths = [], {}, []
    study = ROOT/policy['study']
    original = read(study/'cases.json')['cases']
    inputs[str(study/'cases.json')] = digest(study/'cases.json')
    prepared = None if phase == 'human' else output/'recovery_prepared'
    if phase == 'diagnostic':
        source = ROOT/policy['diagnostic_study']
        old_cases = recovery.checked_cases(source/'recovery_prepared')
        selected = [c for name in policy['diagnostic_cases'] for c in old_cases if c['case_id'] == name]
        if len(selected) != len(policy['diagnostic_cases']):
            raise ValueError('Missing declared diagnostic cases')
    else:
        selected = [c for c in original if c['kind'] == 'human' and c['variant'] == 'TRANSFER'
                    and c['subject'] in (policy['human_subjects'] if phase == 'human'
                                         else [r['subject'] for r in policy['validation_sequences']])]
    if smoke:
        selected = selected[:1]
    for old in selected:
        frozen = checked_frozen(old)
        subject, case_id = old['subject'], old['case_id']
        inputs[str(Path(old['output'])/'nominees.json')] = digest(Path(old['output'])/'nominees.json')
        own = deepcopy(old)
        if phase == 'validation':
            specification, = [r for r in policy['validation_sequences'] if r['subject'] == subject]
            rank = specification['seed_rank']
            case_id = f'S{subject}_TRANSFER_V{rank}'
            evaluation = ROOT/policy['evaluation']/old['case_id']
            record, = read(evaluation/'autonomous/evaluation.json')['rows']
            if record['candidate'] != frozen['nominee']['candidate']:
                raise ValueError('Generation differs from frozen model point')
            bank = fixtures.checked_arrays(evaluation/'autonomous', record)
            if len(bank['seeds']) != 128 or len(set(bank['seeds'])) != 128:
                raise ValueError('Require the complete archived generation bank')
            index = np.argsort(bank['seeds'])[rank]
            original_data = Path(frozen['data'])/f'subject_{subject}'
            meta = read(original_data/'prepared.json')
            cut = meta['training_trials']
            if digest(original_data/'training.npz') != meta['array_sha256']['training.npz']:
                raise ValueError('Original training sequence changed')
            with np.load(original_data/'training.npz') as z:
                arrays = {k: z[k].copy() for k in ('stimulus', 'categories')}
            np.testing.assert_array_equal(arrays['categories'], bank['categories'][:cut])
            arrays.update(choices=bank['choices'][index, :cut], feedback=bank['feedback'][index, :cut])
            job = read(next((evaluation/'jobs').glob('AUTO_*.json')))
            if job['compact_variant'] != 'TRANSFER' or job['entry_control'] != 'PRECISION':
                raise ValueError('Wrong generating mechanism')
            data = prepared/case_id/'data'/f'subject_{subject}'
            fixtures.save_fixture(original_data, data, arrays, job['runtime'], 'synthetic_training_only')
            truth_file = prepared/case_id/'states_truth.npz'
            with truth_file.open('xb') as stream:
                np.savez_compressed(stream, **{k: bank[k][index, :cut] for k in ('present', 'contribution', 'beta')})
            truths.append(dict(case_id=case_id, subject=subject, replicate=rank, model='COMPACT',
                compact_variant='TRANSFER', point=frozen['nominee']['candidate']['hyperparams'],
                generation_seed=int(bank['seeds'][index]), states_truth=str(truth_file.resolve()),
                states_truth_sha256=digest(truth_file)))
            own.update(kind='recovery', replicate=rank, generation_seed=int(bank['seeds'][index]),
                trials=cut, generating_parameters_used_for_starts=False,
                arrays_sha256={n: digest(data/n) for n in ('training.npz', 'sealed_test.npz', 'engine.json', 'prepared.json')})
            own['settings'].update(data=str(data.parent.resolve()), recovery_target_rule=int(record['target_rule']))
            for file in [evaluation/'autonomous/evaluation.json', original_data/'prepared.json', original_data/'training.npz']:
                inputs[str(file.resolve())] = digest(file)
        settings = settings_for_case(own['settings'], policy, case_id)
        banks = independent_seed_banks(settings, subject, own.get('generation_seed'))
        if phase == 'validation':
            anchors = []
        else:
            anchors = [frozen['nominee']['candidate']['hyperparams']]
            anchors += [p['hyperparams'] for p in frozen['nominee']['finalists']
                        if proposals.behavior_key(p['hyperparams']) != proposals.behavior_key(anchors[0])]
        if smoke:
            anchors = anchors[:1]
        starts = proposals.initial_bank(old['space'], anchors, 4 if smoke else policy['initial_candidates'], settings['seed'])
        own.update(case_id=case_id, settings=settings, starts=starts, required=starts[0], variant='TRANSFER',
                   output=str((output/('human_fit' if phase == 'human' else 'recovery_fit')/case_id).resolve()),
                   source_fit=old['output'], estimation_phase=phase,
                   independent_seed_banks=banks,
                   optimizer_development_case=phase == 'diagnostic')
        cases.append(own)
    if prepared is not None:
        save_prepared(prepared, cases, truth_source=(source/'recovery_prepared/truths.json')
                      if phase == 'diagnostic' else None, truths=truths, inputs=inputs)
    write(output/'cases.json', dict(cases=cases, inputs_sha256=inputs, phase=phase,
        oral_used=False, human_later_choices_used=False, generating_parameters_used_for_proposals=False))
    return cases, prepared


def fit_case(case: dict, smoke: bool) -> dict:
    settings, output = case['settings'], Path(case['output'])
    output.mkdir(parents=True)
    hashes = refit.sources(driver.io)
    write(output/'protocol.json', dict(settings=settings, case_id=case['case_id'], smoke=smoke,
        source_sha256=hashes, space=case['space'], phase=case['estimation_phase']))
    key = case['subject'], 'COMPACT'
    seen = {proposals.behavior_key(r['hyperparams']) for r in case['starts']}
    stage = lambda bank, name: driver.base.stage(settings, ROOT/settings['data'], output, {key: bank}, name, smoke)[key]
    rows = stage(case['starts'], 'screen')
    history = [rows]
    print(case['case_id'] + ': initial population scored', flush=True)
    for index in range(1, (1 if smoke else settings['rounds']) + 1):
        plan = dict(elites=1, rays=1, pairs=1, multi=1, global_starts=1) if smoke else settings['proposals']
        new = proposals.population_round(rows, case['space'], seen,
            seed=refit.stable_seed(dict(base=settings['seed'], round=index, role='joint_population')), **plan)
        seen.update(proposals.behavior_key(r['hyperparams']) for r in new)
        carry = proposals.diverse_shortlist(rows, case['required'], 2 if smoke else settings['carry_candidates'],
                                             1 if smoke else settings['carry_top'])
        write(output/f'refine_{index}_proposals.json', dict(new=new, carry=carry, behavioral_candidates=len(seen)))
        rows = stage(new + carry, f'refine_{index}')
        history.append(rows)
        print(case['case_id'] + f': population round {index} scored', flush=True)
    bank = (driver.base.shortlist(rows, case['required'], 2) if smoke else
            proposals.stage_archive_shortlist(history, case['required'], settings['selection_candidates'], settings['archive_per_stage']))
    rows = stage(bank, 'selection')
    bank = proposals.diverse_shortlist(rows, case['required'], 2 if smoke else settings['confirmation_candidates'],
                                      1 if smoke else settings['confirmation_top'])
    rows = stage(bank, 'confirmation')
    nominee = dict(case_id=case['case_id'], subject=case['subject'], model='COMPACT', compact_variant='TRANSFER',
        candidate=rows[0], finalists=rows, unique_candidates=len(seen), reference_id=case['required']['id'])
    if case['kind'] == 'recovery':
        nominee['replicate'] = case['replicate']
    write(output/'nominees.json', dict(status='frozen_before_test' if case['kind'] == 'human' else 'frozen_before_truth',
        settings=settings, smoke=smoke, source_sha256=hashes, data=str((ROOT/settings['data']).resolve()),
        nominee=nominee, nominees=[nominee]))
    write(output/'completion.json', dict(status='complete', smoke=smoke, test_choices_used=False, truths_opened=False,
        nomination_sha256=digest(output/'nominees.json'), bounded_rounds=len(history)-1))
    print(case['case_id'] + ': estimate frozen', flush=True)
    return nominee


def audit_case(case: dict, output: Path, smoke: bool) -> tuple[dict, dict]:
    """Independently score fixed contenders; do not choose again on audit data."""
    frozen = read(Path(case['output'])/'nominees.json')
    chosen = frozen['nominee']['candidate']
    count = 2 if smoke else case['settings']['audit_candidates']
    reference = next(r for r in frozen['nominee']['finalists'] if r['id'] == frozen['nominee']['reference_id'])
    bank = {chosen['id']: chosen, reference['id']: reference}
    for row in frozen['nominee']['finalists']:
        if len(bank) == count:
            break
        bank.setdefault(row['id'], row)
    own = deepcopy(case)
    own['settings']['seed'] = refit.stable_seed(dict(base=case['settings']['seed'], role='post_freeze_audit'))
    own['output'] = str(output.resolve())
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(settings=own['settings'], smoke=smoke,
        source_sha256=frozen['source_sha256'], selection_sha256=digest(Path(case['output'])/'nominees.json'),
        candidate_selection='Frozen before independent audit; no audit reselection'))
    rows = driver.base.stage(own['settings'], ROOT/own['settings']['data'], output,
                            {(case['subject'], 'COMPACT'): list(bank.values())}, 'confirmation', smoke)[case['subject'], 'COMPACT']
    nominee = dict(frozen['nominee'], candidate=next(r for r in rows if r['id'] == chosen['id']), finalists=rows)
    write(output/'nominees.json', dict(status=frozen['status'], settings=own['settings'], smoke=smoke,
        source_sha256=frozen['source_sha256'], data=frozen['data'], nominee=nominee, nominees=[nominee],
        selected_before_audit=True, selection_sha256=digest(Path(case['output'])/'nominees.json')))
    write(output/'completion.json', dict(status='complete', smoke=smoke, test_choices_used=False,
        nomination_sha256=digest(output/'nominees.json'), selected_before_audit=True))
    print(case['case_id'] + ': independent fixed-candidate audit complete', flush=True)
    return own, nominee


def run(config: Path, phase: str, output: Path, smoke: bool, prepare_only: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new reliability-study directory')
    policy = yaml.safe_load(config.read_text())
    if not 1 <= policy['parallel_budget'] <= 128 or policy['rounds'] != 3:
        raise ValueError('Require the registered bounded population budget')
    output.mkdir(parents=True)
    source_function = refit.sources
    refit.sources = lambda v: dict(source_function(v), **{name: digest(ROOT/name) for name in (ENTRY, PROPOSALS)})
    driver.base.archive(output, config, refit.sources(driver.io))
    cases, prepared = prepare(policy, phase, output, smoke)
    workers = 1 if smoke else min(policy['parallel_budget'], len(os.sched_getaffinity(0)))
    write(output/'protocol.json', dict(settings=policy, phase=phase, smoke=smoke, workers=workers,
        source_sha256=refit.sources(driver.io), python=sys.version,
        versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        mechanism='COMPACT + precision transfer + mixture + ADD10',
        objective='Original training choice likelihood; oral and later behavior excluded',
        starting_time=time.time()))
    if prepare_only:
        write(output/'preparation_complete.json', dict(status='complete', cases=len(cases), numeric_jobs=0))
        return
    audit_fit = output/'audit_fit'
    audit_fit.mkdir()
    write(audit_fit/'protocol.json', dict(smoke=smoke, source_sha256=refit.sources(driver.io),
        independent_audit=True, original_fit=str(output.resolve())))
    print(f'{phase}: {len(cases)} cases, {workers} shared single-thread numeric workers', flush=True)
    with NumericQueue(workers, output) as pool:
        refit.configure_executor(pool)
        with ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controls:
            futures = [controls.submit(fit_case, c, smoke) for c in cases]
            nominees = [f.result() for f in futures]
        write(output/'selected_nominees.json', dict(status='frozen_before_external_evaluation', smoke=smoke, nominees=nominees))
        with ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controls:
            audited = [f.result() for f in [controls.submit(audit_case, c, audit_fit/c['case_id'], smoke) for c in cases]]
        audit_cases, audit_nominees = zip(*audited)
        write(audit_fit/'nominees.json', dict(status='frozen_before_truth' if prepared else 'frozen_before_test',
            smoke=smoke, nominees=list(audit_nominees)))
        write(audit_fit/'completion.json', dict(status='complete', smoke=smoke, truths_opened=False,
            nomination_sha256=digest(audit_fit/'nominees.json'), selected_before_audit=True))
        if prepared:
            audit_prepared = output/'audit_prepared'
            save_prepared(audit_prepared, list(audit_cases), truth_source=prepared/'truths.json')
            original_executor = recovery.configure_executor
            recovery.configure_executor = refit.configure_executor
            try:
                recovery.score_truth(audit_prepared, audit_fit, output/'recovery_scores', numerical_pool=pool)
            finally:
                recovery.configure_executor = original_executor
            from ...evaluation.model_1021_simplification_recovery_states import summarize
            summarize(audit_prepared, audit_fit, output/'recovery_scores', output/'recovery_states')
            write(output/'recovery_scores/scope_note.json', dict(phase=phase,
                cases=[c['case_id'] for c in cases], selected_before_independent_audit=True,
                prior_recovery_failure_reviewed=phase == 'diagnostic',
                scope='Registered finite-sequence optimizer check; see case definitions for actual generating points',
                inherited_scope_text='The shared scorer retains its historical two-point wording; this note states the current scope'))
        else:
            write(output/'human_nominees.json', dict(status='frozen_before_test', smoke=smoke, nominees=nominees))
    write(output/'completion.json', dict(status='complete', phase=phase, smoke=smoke, cases=len(cases),
        human_evaluation='pending' if phase == 'human' else 'not_applicable',
        original_results_preserved=True, oral_used_for_fitting=False, selected_before_audit=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('diagnostic', 'validation', 'human'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--smoke', action='store_true')
    group.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.config, args.phase, args.output, args.smoke, args.prepare_only)
