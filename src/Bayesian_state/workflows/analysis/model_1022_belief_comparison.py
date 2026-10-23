"""Bounded, separately fitted SUPPORT/BELIEF comparisons on exp123 prefixes.

Run --config YAML --output NEW [--smoke]. All arms share starting candidates,
parameter ranges, filter seeds and budgets. Each receives adaptive rounds;
the union of leading candidates is independently confirmed for every arm.
Parameters freeze before later choices and task-start autonomous paths open.
The existing frozen numerical runtime and all shared cognitive modules are
reused. Outputs include provenance, per-seed predictions and a summary table.
OUTCOME, ENTRY and ADAPT opt into feedback-dependent belief mechanisms.
For these arms use entry_belief: by_variant. warm_comparisons accepts several
completed training-nomination sources; warm_comparison retains its single
source meaning. confirm_training_anchors: true also confirms the original
SUPPORT training anchor when no additional warm source is configured.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ADAPTER = 'src/Bayesian_state/model/modules/hypothesis_transition/compact_belief.py'
PRECISION = 'src/Bayesian_state/model/modules/hypothesis_transition/entry_precision.py'
SUMMARY = 'src/Bayesian_state/workflows/analysis/model_1022_belief_summary.py'
SLOT = 'src/Bayesian_state/model/modules/hypothesis_transition/belief_slot_transfer.py'
OUTCOME = 'src/Bayesian_state/model/modules/hypothesis_transition/belief_outcome_retention.py'
ERROR_ENTRY = 'src/Bayesian_state/model/modules/hypothesis_transition/belief_error_entry.py'
ARMS = {'SUPPORT', 'BELIEF', 'SLOT', 'OUTCOME', 'ENTRY', 'ADAPT'}
spec = importlib.util.spec_from_file_location('belief_comparison_driver',
    ROOT/'src/Bayesian_state/workflows/analysis/model_1021_simplification.py')
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
read, write, digest = driver.io.read, driver.io.write, driver.io.digest


def sources(v) -> dict:
    return dict(driver.source_hashes(v), **{k: digest(ROOT/k)
                for k in (ENTRY, ADAPTER, PRECISION, SUMMARY, SLOT, OUTCOME, ERROR_ENTRY)})


def valid_variants(variants: list[str]) -> bool:
    """Require a named SUPPORT reference and distinct registered mechanisms."""
    return (isinstance(variants, list) and len(variants) >= 2 and variants[0] == 'SUPPORT'
            and all(isinstance(arm, str) for arm in variants)
            and len(set(variants)) == len(variants) and set(variants) <= ARMS)


def load_warm_nominees(policy: dict, data: str) -> tuple[dict | None, dict]:
    """Combine unchanged training nominations without reading evaluation scores."""
    if policy.get('warm_comparison') and policy.get('warm_comparisons'):
        raise ValueError('Choose warm_comparison or warm_comparisons')
    paths = ([policy['warm_comparison']] if policy.get('warm_comparison')
             else policy.get('warm_comparisons', []))
    if (not isinstance(paths, list) or any(not isinstance(p, str) for p in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError('Warm comparisons require distinct source paths')
    rows, inputs = [], {}
    for path in paths:
        source = ROOT / path
        warm = read(source/'nominees.json')
        if (warm['smoke'] or warm['status'] != 'frozen_before_test' or warm['later_choices_used']
                or warm['oral_used'] or read(source/'completion.json')['status'] != 'complete'
                or read(source/'protocol.json')['settings']['data'] != data
                or read(source/'summary/completion.json')['nominee_sha256'] != digest(source/'nominees.json')):
            raise ValueError('Warm starts require matching intact training-only nominations')
        rows.extend(warm['nominees'])
        for name in ('nominees.json', 'completion.json', 'protocol.json', 'summary/completion.json'):
            inputs[str(source/name)] = digest(source/name)
    return (dict(nominees=rows) if paths else None), inputs


def worker(file: Path) -> None:
    job = read(file)
    original_load = driver.base.load

    def load(name, path):
        module = original_load(name, path)
        if path == driver.COMPACT:
            if job['comparison_arm'] in ('BELIEF', 'SLOT', 'OUTCOME', 'ENTRY', 'ADAPT'):
                belief = original_load('src.Bayesian_state.model.modules.hypothesis_transition.compact_belief', ADAPTER)
                module.compact_evidence = belief.compact_belief
                if job['comparison_arm'] != 'BELIEF':
                    slot = original_load('src.Bayesian_state.model.modules.hypothesis_transition.belief_slot_transfer', SLOT)
                    outcome = (original_load('src.Bayesian_state.model.modules.hypothesis_transition.belief_outcome_retention', OUTCOME)
                               if job['comparison_arm'] in ('OUTCOME', 'ADAPT') else None)
                    admission = (original_load('src.Bayesian_state.model.modules.hypothesis_transition.belief_error_entry', ERROR_ENTRY)
                                 if job['comparison_arm'] in ('ENTRY', 'ADAPT') else None)

                    @contextmanager
                    def conserved(cls):
                        with ExitStack() as stack:
                            audit = stack.enter_context(belief.compact_belief(cls))
                            audit['slot_transfer'] = stack.enter_context(slot.belief_slot_transfer(cls))
                            if outcome is not None:
                                audit['outcome_retention'] = stack.enter_context(outcome.belief_outcome_retention(cls))
                            if admission is not None:
                                audit['error_entry'] = stack.enter_context(admission.belief_error_entry(cls))
                            yield audit

                    module.compact_evidence = conserved
            elif job['comparison_arm'] == 'SUPPORT':
                precision = original_load('src.Bayesian_state.model.modules.hypothesis_transition.entry_precision', PRECISION)
                original = module.compact_evidence

                @contextmanager
                def transfer(cls):
                    with original(cls) as audit, precision.entry_precision(cls):
                        yield audit

                module.compact_evidence = transfer
            else:
                raise ValueError('Unknown belief comparison arm')
        return module

    driver.base.load = load
    try:
        driver.worker(file)
    finally:
        driver.base.load = original_load


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    directory = output/'jobs'
    directory.mkdir(exist_ok=True)

    def launch(original):
        job = deepcopy(original)
        job.update(comparison_arm=job['model'], model='COMPACT')
        job['mechanism']['capacity'] = 'evidence'
        file = directory/(job['name']+'.json')
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='belief_comparison_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                       NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT/ENTRY),
                                '--worker', str(file.resolve())], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)

    workers = min(budget, len(os.sched_getaffinity(0)), len(jobs))
    print(f'{len(jobs)} numerical jobs; {workers} single-thread workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(launch, j) for j in jobs]), 1):
            future.result()
            if index % 16 == 0 or index == len(jobs):
                print(f'{index}/{len(jobs)} completed', flush=True)


def summarize(output: Path) -> None:
    """Audit saved identities and probabilities before writing completion."""
    from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import run as audit
    summary = output/'summary'
    audit(output, summary)
    write(output/'completion.json', dict(status='complete', smoke=False,
        cases=read(summary/'completion.json')['cases'],
        scope=read(output/'protocol.json')['policy']['scope'],
        oral_used_for_fitting=False, later_choices_used_for_fitting=False,
        numerical_protocol_sha256=digest(output/'protocol.json'),
        summary_sha256=digest(summary/'summary.csv'),
        postprocessing_source_sha256=digest(ROOT/SUMMARY)))


def run(config: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import (
        candidate, initial_points, propose_round, select_elites)
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new belief comparison output directory')
    policy = yaml.safe_load(config.read_text())
    if (not valid_variants(policy['variants'])
            or policy['entry_belief'] not in ('uniform_share_after_replacement', 'by_variant')
            or (set(policy['variants']) & {'OUTCOME', 'ENTRY', 'ADAPT'} and policy['entry_belief'] != 'by_variant')
            or not isinstance(policy.get('confirm_training_anchors', False), bool)
            or not 1 <= policy['parallel_budget'] <= 128):
        raise ValueError('Require the declared belief mechanisms and a bounded worker budget')
    settings = yaml.safe_load((ROOT/policy['base_config']).read_text())
    precision = deepcopy(settings['arms']['COMPACT'])
    settings.update({k: deepcopy(policy[k]) for k in ('protocol', 'seed', 'subjects', 'parallel_budget',
                                                     'budgets', 'autonomous_paths', 'autonomous_chunk')})
    settings['arms'] = {arm: deepcopy(precision) for arm in policy['variants']}
    if smoke:
        settings['subjects'] = settings['subjects'][:1]
        settings['parallel_budget'] = 1
    output.mkdir(parents=True)
    driver.base.source_hashes, driver.base.execute = sources, execute
    driver.base.archive(output, config, sources(driver.io))
    write(output/'protocol.json', dict(settings=settings, policy=policy, smoke=smoke,
        source_sha256=sources(driver.io), python=sys.version, started=time.time(),
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        belief_formula='normalize(prior ** rule_exponent * feedback_likelihood)',
        rule_exponent=dict(BELIEF='delta', SLOT='delta', ENTRY='delta',
                           OUTCOME='delta+(1-delta)*current_feedback',
                           ADAPT='delta+(1-delta)*current_feedback'),
        condition3='joint belief uses rule_exponent for rules and gamma for conditional pairings',
        entry_belief=dict(BELIEF='1 / active_count_after; preserve survivor odds',
                         SLOT='full workspace: preserve survivors and transfer removed mass; growth: uniform entry',
                         OUTCOME='SLOT entry; feedback-dependent rule exponent delta+(1-delta)*feedback',
                         ENTRY='full workspace: previous_feedback*removed_mass+(1-previous_feedback)/capacity',
                         ADAPT='OUTCOME rule retention and ENTRY admission; conditional pairing retention unchanged'),
        decision='separate training NLL fits; nominees frozen before later evaluation'))
    banks, spaces, required = {}, {}, {}
    warm, inputs = load_warm_nominees(policy, settings['data'])
    for subject in settings['subjects']:
        source = ROOT/policy['warm_fit']/f'S{subject}_TRANSFER'
        previous = read(source/'nominees.json')
        done = read(source/'completion.json')
        if (previous['smoke'] or done['status'] != 'complete' or done['test_choices_used'] is not False
                or done['nomination_sha256'] != digest(source/'nominees.json')):
            raise ValueError('Warm starts must be frozen training-only estimates')
        inputs[str(source/'nominees.json')] = digest(source/'nominees.json')
        meta = read(ROOT/settings['data']/f'subject_{subject}'/'prepared.json')
        space = spaces[subject] = driver.model_space(previous['settings'], 'COMPACT', meta['condition'])
        anchor = previous['nominee']['candidate']['hyperparams']
        count = 5 if smoke else policy['initial_candidates']
        points = initial_points(space, anchor, count if warm is None else max(count * 2, 5),
            stable_seed(dict(base=policy['seed'], subject=subject, stage='initial')),
            cover_key=driver.H+'max_capacity')
        anchors = ([anchor] + [r['candidate']['hyperparams'] for r in warm['nominees'] if r['subject'] == subject]
                   if warm else [anchor] if policy.get('confirm_training_anchors', False) else [])
        if any(set(p) != set(space) or any(p[k] not in values for k, values in space.items()) for p in anchors):
            raise ValueError('Additional warm start is outside the common parameter support')
        if warm:
            unique = {}
            for p in points[:5] + anchors + points[5:]:
                unique.setdefault(candidate(p, 'common_training_start')['id'], p)
            points = list(unique.values())[:count]
            if len(points) != count:
                raise ValueError('Insufficient common initial candidates')
        required[subject] = [candidate(p, 'training_anchor_confirmation') for p in anchors]
        for arm in policy['variants']:
            banks[subject, arm] = [candidate(p, 'common_training_start') for p in points]
    write(output/'starting_banks.json', dict(inputs_sha256=inputs,
        banks=[dict(subject=s, arm=a, candidates=b) for (s, a), b in banks.items()]))
    stage = lambda bank, name: driver.base.stage(settings, ROOT/settings['data'], output, bank, name, smoke)
    screen = stage(banks, 'screen')
    rounds = 1 if smoke else policy.get('refinement_rounds', 1)
    seen = {key: {r['id'] for r in rows} for key, rows in screen.items()}
    refined = screen
    for index in range(rounds):
        name = 'refine' if policy.get('refinement_rounds', 1) == 1 else f'refine_{index+1}'
        refinement = {}
        fractions = policy.get('proposal_fractions', dict(local=0., jump=0.))
        for key, rows in refined.items():
            elites = select_elites(rows, 1 if smoke else policy['elites'], exclude_key=driver.H+'max_capacity')
            proposed = propose_round(elites, spaces[key[0]], seen[key],
                2 if smoke else policy['refinement_candidates'] // len(elites),
                stable_seed(dict(base=policy['seed'], subject=key[0], stage=name)),
                fractions['local'], fractions['jump'])
            expected = 2 if smoke else policy['refinement_candidates']
            if len(proposed) != expected:
                raise ValueError('Adaptive proposal quota was not met')
            seen[key].update(r['id'] for r in proposed)
            refinement[key] = proposed + rows[:1 if smoke else policy['elites']]
        refined = stage(refinement, name)
    confirmation = {}
    for subject in settings['subjects']:
        union = {}
        for arm in policy['variants']:
            # Compare stage-specific ranks, then use independent seeds to
            # select among both arms' best candidates and the common anchor.
            for rows in (screen[subject, arm], refined[subject, arm]):
                for row in rows[:1 if smoke else policy['finalists_per_arm']]:
                    union.setdefault(row['id'], candidate(row['hyperparams'], 'cross_arm_confirmation'))
        for row in required[subject]:
            union.setdefault(row['id'], row)
        for arm in policy['variants']:
            confirmation[subject, arm] = list(union.values())
    confirmed = stage(confirmation, 'confirmation')
    nominees = [dict(subject=s, arm=a, candidate=rows[0], finalists=rows) for (s, a), rows in confirmed.items()]
    write(output/'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
                                     later_choices_used=False, oral_used=False))
    if not smoke:
        stage({(r['subject'], r['arm']): [r['candidate']] for r in nominees}, 'evaluate')
    preparation = read(ROOT/settings['data']/'preparation_manifest.json')
    runtime = str((ROOT/preparation['settings']['source_fit']/'source').resolve())
    jobs = []
    for row in nominees:
        subject, arm = row['subject'], row['arm']
        seeds = [stable_seed(dict(base=policy['seed'], subject=subject, stage='autonomous', repeat=i))
                 for i in range(2 if smoke else policy['autonomous_paths'])]
        audit = ROOT/settings['audit']/f'audit/subject_{subject}/audit.json'
        width = policy['autonomous_chunk']
        for i in range(0, len(seeds), width):
            name = f'autonomous_S{subject}_{arm}_{i:03d}'
            jobs.append(dict(phase='autonomous', name=name, subject=subject, model=arm,
                candidate=row['candidate'], mechanism=dict(kind='E', decay=.95, capacity='evidence', search='accumulate'),
                folder=str((ROOT/settings['data']/f'subject_{subject}').resolve()), runtime=runtime,
                arm_settings=precision, seeds=seeds[i:i+width], smoke=smoke, source_sha256=sources(driver.io),
                audit=str(audit), audit_sha256=digest(audit), target=str((output/'autonomous'/name).resolve())))
    write(output/'autonomous_jobs.json', dict(jobs=jobs))
    execute(jobs, output, settings['parallel_budget'])
    if smoke:
        write(output/'completion.json', dict(status='complete', smoke=True, cases=len(nominees),
              scope='eight-trial filters and sixteen-trial autonomous paths; scientific evaluation pending'))
    else:
        summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--summary-only', action='store_true', help='Audit an existing numerical study without rerunning it')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.summary_only and args.output:
        summarize(args.output)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Supply --config and --output, or --worker')
