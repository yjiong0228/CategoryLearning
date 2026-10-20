"""Bounded joint recovery for the frozen ONLINE recommendation.

Prepare two existing generation points (205 and306), selecting the two lowest
original generation seeds at each point. Fit all thirteen variable coordinates
with forty support-only starts and the original five-stage numerical policy.
Only the recommended proposal/precision arm is fitted. Generating parameters
are read for scoring after estimates freeze; oral data are unused.

Use --phase prepare --config YAML --output NEW, then search or score-truth
with --prepared PREPARED --subject ID --replicate INDEX --output NEW. Truth
scoring also needs --fit FIT. --smoke requires a software_validation preparation
and runs the existing eight-trial two-particle policy on one worker. Formal
preparation requires a recommendation based on all nine development cells.
Every phase preserves existing outputs and writes a new directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = 'src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_refit.py'
IO = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
JOINT = 'src/Bayesian_state/workflows/analysis/model_0923_joint_parameter_recovery.py'
SCOPES = {'software_validation', 'finalization_recovery'}
CANDIDATES = {'SEARCH_ADD10': ('SEARCH', 'fast', 10.),
              'SEARCH_LOG25': ('SEARCH', 'log', 1.), 'POOL_LOG25': ('POOL', 'log', 1.)}


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cohort = load_module('finalization_recovery_cohort', COHORT)
io = load_module('finalization_recovery_io', IO)
read, write, digest = io.read, io.write, io.digest


def validate_config(config: dict) -> bool:
    """Keep the scientific batch at two points and two sequences per point."""
    keys = {'scope', 'recommendation', 'software_candidate', 'evidence_config',
            'subjects', 'replicates', 'parallel_budget', 'seed'}
    if set(config) != keys or config['scope'] not in SCOPES:
        raise ValueError('Declare the recovery scope and its fixed preparation fields')
    smoke = config['scope'] == 'software_validation'
    if (type(config['seed']) is not int or type(config['parallel_budget']) is not int
            or type(config['replicates']) is not int
            or (smoke and (config['subjects'] != [205] or config['replicates'] != 1
                or config['parallel_budget'] != 1 or config['recommendation'] is not None
                or config['software_candidate'] not in CANDIDATES))
            or (not smoke and (config['subjects'] != [205, 306] or config['replicates'] != 2
                or config['parallel_budget'] != 32 or not config['recommendation']
                or config['software_candidate'] is not None))):
        raise ValueError('Use two registered points times two sequences, or the single-worker software case')
    return smoke


def prepare(config_file: Path, output: Path) -> None:
    """Reuse complete autonomous paths, keeping truth outside fitting inputs."""
    from src.Bayesian_state.evaluation.model_0923_h_validation import checked_arrays
    from src.Bayesian_state.evaluation.model_0923_estimation_scope import validate_generation
    from src.Bayesian_state.workflows.analysis.model_0923_finalization_extension_fit import validate_recommendation
    from src.Bayesian_state.workflows.analysis.model_0923_h_objective_recovery import fixture
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new recovery preparation directory')
    config = yaml.safe_load(config_file.read_text())
    smoke = validate_config(config)
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f'Recovery source changed: {path}')
        if str(path) in bindings and bindings[str(path)] != sha:
            raise ValueError('Recovery preparation changed during reading')
        bindings[str(path)] = sha
        return path

    bind(config_file); bind(__file__)
    evidence_config = yaml.safe_load(bind(ROOT / config['evidence_config']).read_text())
    if smoke:
        chosen = config['software_candidate']
    else:
        decision = read(bind(ROOT / config['recommendation']))
        evidence = read(bind(ROOT / decision['evidence_manifest'], decision['evidence_manifest_sha256']))
        chosen = validate_recommendation(decision, evidence)
        for path, sha in evidence['inputs_sha256'].items():
            bind(path, sha)
    arm, mode, scale = CANDIDATES[chosen]
    family = chosen.split('_', 1)[1]
    recovery = load_module('finalization_recovery_neutral', JOINT)
    bind(ROOT / JOINT)
    output.mkdir(parents=True)
    cases, truths, spaces = [], [], {}
    for subject in config['subjects']:
        source = ROOT / evidence_config['subjects'][subject][family]['summary']
        summary = read(bind(source / 'manifest.json'))
        fit, evaluation = Path(summary['fit']), Path(summary['evaluation'])
        frozen, done = [read(bind(fit / name)) for name in ('nominees.json', 'completion.json')]
        if (summary['status'] != 'complete' or done['status'] != 'complete' or done['smoke']
                or done['test_choices_used'] or frozen['smoke'] or frozen['status'] != 'frozen_before_test'
                or done['nomination_sha256'] != digest(fit / 'nominees.json')
                or frozen['settings']['belief_mechanism'] != 'ONLINE'
                or frozen['settings']['structural_intervention'] != 'retained_rule_precision'):
            raise ValueError('Require the complete behavior-selected ONLINE generation point')
        nominee, = [r for r in frozen['nominees'] if (r['subject'], r['model']) == (subject, arm)]
        bank_row, = [r for r in read(bind(evaluation / 'autonomous/evaluation.json'))['rows']
                    if (r['subject'], r['model']) == (subject, arm)]
        if bank_row['candidate'] != nominee['candidate']:
            raise ValueError('Autonomous generation differs from the selected point')
        generated = checked_arrays(evaluation / 'autonomous', bank_row)
        validate_generation(generated, bank_row)
        source_jobs = sorted((evaluation / 'jobs').glob(f'AUTO_evaluate_S{subject}_{arm}_*.json'))
        if len(source_jobs) != 32:
            raise ValueError('Require all128 original autonomous paths')
        jobs = [read(bind(path)) for path in source_jobs]
        for job in jobs:
            if (job['candidate'] != nominee['candidate'] or job['cohort_belief_mechanism'] != 'ONLINE'
                    or job['online_belief'] is not True or job['retain_precision'] is not True
                    or job.get('execution_policy', 'MIX') != 'MIX'
                    or (job['arm_settings']['precision'], job['arm_settings']['rate_scale'], job['arm_settings']['upper'])
                    != (mode, scale, 25.)):
                raise ValueError('Generating worker differs from the frozen candidate')
            target = Path(job['target'])
            record = read(bind(target.with_suffix('.json')))
            bind(target.with_suffix('.npz'), record['predictions_sha256'])
            for path, sha in job['source_sha256'].items():
                bind(ROOT / path, sha)
        original = Path(jobs[0]['folder'])
        metadata = read(bind(original / 'prepared.json'))
        cut = metadata['training_trials']
        if cut != {205: 480, 306: 512}[subject] or bank_row['training_trials'] != cut:
            raise ValueError('Preserve the original chronological training length')
        with np.load(bind(original / 'training.npz', metadata['array_sha256']['training.npz'])) as saved:
            physical = {k: saved[k].copy() for k in ('stimulus', 'categories')}
        np.testing.assert_array_equal(generated['categories'][:cut], physical['categories'])
        replaced = cohort.REFERENCE_KEYS | {'warm_start_fit', 'initial_ray_coordinates'}
        settings = {k: deepcopy(v) for k, v in frozen['settings'].items() if k not in replaced}
        settings.update(subjects=[subject], arms={arm: deepcopy(settings['arms'][arm])},
            parallel_budget=config['parallel_budget'], scope='synthetic_online_joint_recovery',
            recovery_scope=config['scope'], recovery_candidate=chosen,
            recovery_recommendation=config['recommendation'], recovery_generation_seeds=generated['seeds'].tolist(),
            recovery_initialization='support_midpoint_capacity_rays_independent_stratified',
            policy_reference_fit=str(fit.resolve()))
        if settings['initial_candidates'] != 40 or settings['fixed_coordinates'] != {recovery.TRANSFER: .01}:
            raise ValueError('Preserve forty starts and the single fixed transfer coordinate')
        modules = cohort.helpers()
        _, online, _, recalled = modules
        space = online.fixed_space(recalled.base.model_space(settings, arm), settings['fixed_coordinates'])
        if space != read(bind(fit / 'parameter_spaces.json'))[arm]:
            raise ValueError('Recovery parameter support differs from its fitted candidate')
        point = nominee['candidate']['hyperparams']
        if set(point) != set(space) or any(point[k] not in levels for k, levels in space.items()):
            raise ValueError('Generating point is outside the fitting support')
        spaces[str(subject)] = space
        for replicate, index in enumerate(recovery.selected_path_indices(generated['seeds'], config['replicates'])):
            case = output / f'subject_{subject}_replicate_{replicate}'
            folder = case / 'data' / f'subject_{subject}'
            arrays = dict(physical, choices=generated['choices'][index, :cut].copy(),
                          feedback=generated['feedback'][index, :cut].copy())
            fixture(original, folder, arrays)
            runtime = Path(jobs[0]['runtime'])
            write(folder.parent / 'preparation_manifest.json', dict(status='complete',
                settings={'source_fit': str(runtime.parent.resolve())}, subjects=[subject],
                data_role='synthetic_training_only', source_runtime=str(runtime.resolve())))
            own = dict(deepcopy(settings), data=str(folder.parent.resolve()),
                seed=stable_seed(dict(protocol='finalization_online_recovery_v1', base=config['seed'],
                                     subject=subject, replicate=replicate, role='search')),
                recovery_case=dict(subject=subject, replicate=replicate))
            neutral = recovery.independent_starts(space, 40, stable_seed(dict(base=own['seed'], role='neutral_starts')))
            write(case / 'initial_bank.json', dict(rows=neutral, required_reference_id=neutral[0]['id'],
                generating_parameters_used=False, human_parameters_used=False))
            config_path = case / 'fit.yaml'
            config_path.write_text(yaml.safe_dump(own, sort_keys=False))
            truth_path = case / 'states_truth.npz'
            with truth_path.open('xb') as stream:
                np.savez_compressed(stream, **{k: generated[k][index, :cut] for k in ('present', 'contribution', 'beta')})
            cases.append(dict(subject=subject, replicate=replicate, fit_config=str(config_path.resolve()),
                fit_config_sha256=digest(config_path), initial_bank=str((case / 'initial_bank.json').resolve()),
                initial_bank_sha256=digest(case / 'initial_bank.json'), training_trials=cut,
                generation_seed=int(generated['seeds'][index]), source_runtime=str(runtime.resolve()),
                target_rule=int(bank_row['target_rule']), states_truth=str(truth_path.resolve()),
                states_truth_sha256=digest(truth_path),
                prepared_sha256={n: digest(folder / n) for n in ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz')}))
            truths.append(dict(subject=subject, replicate=replicate, model=arm, candidate=nominee['candidate'],
                generation_seed=int(generated['seeds'][index])))
    write(output / 'protocol.json', dict(config=config, candidate=chosen, cases=cases, spaces=spaces,
        point_selection='205and306frozen recommended nominees; contrasting evidence and precision regimes',
        path_selection='Two lowest original generation seeds per point; one in software validation',
        scope='One bounded ONLINE joint-recovery batch; no search-budget extension', oral_used=False))
    write(output / 'truths.json', dict(rows=truths, role='After-estimation evaluation only'))
    for path, sha in bindings.items():
        if digest(Path(path)) != sha:
            raise ValueError('Recovery generation source changed during preparation')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        protocol_sha256=digest(output / 'protocol.json'), truths_sha256=digest(output / 'truths.json'),
        inputs_sha256=bindings, new_generation_runs=0))


def case_settings(prepared: Path, subject: int, replicate: int, smoke: bool) -> tuple[dict, dict, dict, dict]:
    """Read only synthetic fitting data and support-only starts; leave truth unopened."""
    done, protocol = [read(prepared / name) for name in ('completion.json', 'protocol.json')]
    if (done['status'] != 'complete' or done['smoke'] is not smoke
            or done['protocol_sha256'] != digest(prepared / 'protocol.json')
            or validate_config(protocol['config']) is not smoke):
        raise ValueError('Scientific and software recovery scopes must match')
    case, = [r for r in protocol['cases'] if (r['subject'], r['replicate']) == (subject, replicate)]
    inputs = {str((prepared / n).resolve()): digest(prepared / n) for n in ('completion.json', 'protocol.json')}
    for key in ('fit_config', 'initial_bank'):
        file = Path(case[key])
        if digest(file) != case[key + '_sha256']:
            raise ValueError('Frozen recovery fitting policy changed')
        inputs[str(file)] = digest(file)
    settings = yaml.safe_load(Path(case['fit_config']).read_text())
    folder = Path(settings['data']) / f'subject_{subject}'
    for name, sha in case['prepared_sha256'].items():
        if digest(folder / name) != sha:
            raise ValueError('Synthetic fitting data changed')
        inputs[str(folder / name)] = sha
    neutral = read(Path(case['initial_bank']))
    if neutral['generating_parameters_used'] or neutral['human_parameters_used']:
        raise ValueError('Recovery initialization must depend only on the support')
    return settings, inputs, neutral, case


def check_job(job: dict) -> None:
    chosen = job.get('recovery_candidate')
    if chosen not in CANDIDATES or job.get('recovery_scope') not in SCOPES:
        raise ValueError('Missing recovery candidate or scope')
    arm, mode, scale = CANDIDATES[chosen]
    if (job['model'] != arm or (job['arm_settings']['precision'], job['arm_settings']['rate_scale'],
            job['arm_settings']['upper']) != (mode, scale, 25.)
            or (job['recovery_scope'] == 'software_validation') is not job['smoke']
            or set(job['seeds']) & set(job['recovery_generation_seeds'])):
        raise ValueError('Recovery scoring changed its mechanism, scope or independent seeds')


def configure(settings: dict, inputs: dict):
    modules = cohort.helpers()
    recalled = modules[-1]
    original_archive = recalled.base.archive
    cohort.ENTRY = ENTRY
    cohort.configure(modules, settings, inputs)
    old_hashes, old_execute = recalled.base.source_hashes, recalled.base.execute

    def hashes(v):
        return {**old_hashes(v), ENTRY: v.digest(ROOT / ENTRY), COHORT: v.digest(ROOT / COHORT)}

    def execute(jobs, output, budget):
        tagged = [dict(deepcopy(j), recovery_scope=settings['recovery_scope'],
            recovery_candidate=settings['recovery_candidate'], recovery_case=settings['recovery_case'],
            recovery_generation_seeds=settings['recovery_generation_seeds']) for j in jobs]
        for job in tagged:
            check_job(job)
        old_execute(tagged, output, budget)

    def archive(output, config, sources):
        original_archive(output, config, sources)
        write(output / 'recovery_protocol.json', dict(scope=settings['recovery_scope'], inputs_sha256=inputs,
            candidate=settings['recovery_candidate'], initialization=settings['recovery_initialization'],
            generating_parameters_used_for_initialization=False, human_warm_starts_used=False,
            data_use='Synthetic training choices only; all13variable coordinates unknown; no oral',
            search='Single recommended arm; original per-arm budgets; no budget extension'))

    recalled.base.source_hashes, recalled.base.execute, recalled.base.archive = hashes, execute, archive
    return recalled


def worker(file: Path) -> None:
    check_job(read(file))
    cohort.worker(file)


def search(prepared: Path, subject: int, replicate: int, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import propose_round, select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new recovery fitting directory')
    settings, inputs, neutral, _ = case_settings(prepared, subject, replicate, smoke)
    recalled = configure(settings, inputs)
    base, data = recalled.base, Path(settings['data'])
    arm, = settings['arms']
    key = subject, arm
    banks = {key: deepcopy(neutral['rows'][:6] if smoke else neutral['rows'])}
    required = {key: deepcopy(neutral['rows'][0])}
    if len(banks[key]) != (6 if smoke else 40) or required[key]['id'] != neutral['required_reference_id']:
        raise ValueError('Recovery starting bank changed')
    seen = {key: {p['id'] for p in banks[key]}}
    output.mkdir(parents=True)
    hashes = base.source_hashes(base.validation())
    with tempfile.TemporaryDirectory(prefix='online_recovery_fit_') as temporary:
        file = Path(temporary) / 'config.yaml'
        file.write_text(yaml.safe_dump(settings, sort_keys=False))
        base.archive(output, file, hashes)
    write(output / 'parameter_spaces.json', {arm: base.model_space(settings, arm)})
    write(output / 'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        initial_banks=[dict(subject=subject, model=arm, rows=banks[key])], data_use='Synthetic training choices only'))
    screened = base.stage(settings, data, output, banks, 'screen', smoke)
    elites = select_elites(screened[key], 1 if smoke else settings['elites'], exclude_key=io.H_PATH + 'max_capacity')
    proposals = propose_round(elites, base.model_space(settings, arm), seen[key],
        1 if smoke else settings['proposals_per_elite'],
        stable_seed(dict(base=settings['seed'], subject=subject, role='joint_refine')), .5, .25)
    if len(proposals) != (1 if smoke else settings['elites'] * settings['proposals_per_elite']):
        raise ValueError('Recovery refinement changed its fixed budget')
    seen[key].update(p['id'] for p in proposals)
    refined = base.stage(settings, data, output, {key: proposals + elites}, 'refine', smoke)
    leading = {key: sorted({r['id']: r for r in screened[key] + refined[key]}.values(), key=lambda r: (r['mean_nll'], r['id']))}
    banks = recalled.refit.exchange_banks(leading, required, 2 if smoke else settings['exchange_per_arm'])
    seen[key].update(p['id'] for p in banks[key])
    rows = base.stage(settings, data, output, banks, 'exchange', smoke)
    for stage, count in [('selection', settings['selection_candidates']), ('confirmation', settings['confirmation_candidates'])]:
        banks = {key: base.shortlist(rows[key], required[key], 2 if smoke else min(count, len(rows[key])))}
        rows = base.stage(settings, data, output, banks, stage, smoke)
    for path, sha in {**inputs, **{str(ROOT / k): v for k, v in hashes.items()}}.items():
        if digest(Path(path)) != sha:
            raise ValueError('Recovery fitting source changed')
    write(output / 'nominees.json', dict(status='frozen_before_truth_scoring', settings=settings, smoke=smoke,
        source_sha256=hashes, nominees=[dict(subject=subject, model=arm, candidate=rows[key][0],
        finalists=rows[key], unique_candidates=len(seen[key]), reference_id=required[key]['id'])]))
    write(output / 'completion.json', dict(status='complete', smoke=smoke, test_choices_used=False, oral_used=False,
        nomination_sha256=digest(output / 'nominees.json'), recovery_scope=settings['recovery_scope']))


def score_truth(prepared: Path, subject: int, replicate: int, fit: Path, output: Path, smoke: bool) -> None:
    """Score the generating point only after the corresponding estimates freeze."""
    if output.exists():
        raise FileExistsError('Use a new after-fit truth score directory')
    settings, inputs, _, _ = case_settings(prepared, subject, replicate, smoke)
    frozen, done = [read(fit / name) for name in ('nominees.json', 'completion.json')]
    if (done['status'] != 'complete' or done['smoke'] is not smoke or frozen['smoke'] is not smoke
            or frozen['status'] != 'frozen_before_truth_scoring' or frozen['settings'] != settings
            or done['nomination_sha256'] != digest(fit / 'nominees.json')):
        raise ValueError('Freeze matching recovery estimates before opening truth parameters')
    completed = read(prepared / 'completion.json')
    if digest(prepared / 'truths.json') != completed['truths_sha256']:
        raise ValueError('Generating parameters changed')
    truth, = [r for r in read(prepared / 'truths.json')['rows']
              if (r['subject'], r['replicate']) == (subject, replicate)]
    for path in (fit / 'completion.json', fit / 'nominees.json', prepared / 'truths.json'):
        inputs[str(path.resolve())] = digest(path)
    recalled = configure(settings, inputs)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='online_recovery_truth_') as temporary:
        file = Path(temporary) / 'config.yaml'
        file.write_text(yaml.safe_dump(settings, sort_keys=False))
        recalled.base.archive(output, file, recalled.base.source_hashes(recalled.base.validation()))
    write(output / 'protocol.json', dict(settings=settings, smoke=smoke, fit=str(fit.resolve()), inputs_sha256=inputs))
    recalled.base.stage(settings, Path(settings['data']), output,
        {(subject, truth['model']): [truth['candidate']]}, 'confirmation', smoke)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Truth-scoring source changed')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, inputs_sha256=inputs,
        fit_nomination_sha256=digest(fit / 'nominees.json'), truths_sha256=digest(prepared / 'truths.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'search', 'score-truth'))
    for name in ('config', 'prepared', 'fit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--subject', type=int)
    parser.add_argument('--replicate', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'prepare' and args.config and args.output and not args.smoke:
        prepare(args.config, args.output)
    elif (args.phase in ('search', 'score-truth') and args.prepared and args.output
          and args.subject is not None and args.replicate is not None):
        if args.phase == 'search':
            search(args.prepared, args.subject, args.replicate, args.output, args.smoke)
        elif args.fit:
            score_truth(args.prepared, args.subject, args.replicate, args.fit, args.output, args.smoke)
        else:
            parser.error('Truth scoring needs --fit')
    else:
        parser.error('Provide the selected phase inputs and a new output directory')
