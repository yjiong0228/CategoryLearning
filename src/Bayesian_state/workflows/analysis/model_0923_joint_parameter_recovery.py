"""Recover all variable coordinates from current-model synthetic behavior.

Prepare two seed-selected paths per declared development subject, then use the
existing five-stage POOL/SEARCH optimizer with independent neutral starts.
The generating model is SEPARATE+RETAIN+MIX with LOG25 precision and SEARCH.
All thirteen variable coordinates are unknown to the optimizer; transfer_scale
remains structurally fixed. Source parameters are stored outside fitting data.
The reference fit supplies numerical policy and implementation checks only.
After estimates freeze, score the generating point with the same confirmation
seed bank to diagnose search shortfalls separately from sequence uncertainty.

--phase prepare needs --config --subject --output.
--phase search needs --prepared --replicate --output [--smoke].
--phase score-truth adds --fit and accepts --smoke only for a smoke fit.
Smoke uses the existing six-start, five-stage eight-trial two-particle workflow
on one worker. Every phase writes to a new directory; oral reports are unused.
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
TRANSFER = 'engine.modules.hypo_transitions_mod.kwargs.transfer_scale'
CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'
USE = 'synthetic_joint_parameter_recovery'


def helpers():
    spec = importlib.util.spec_from_file_location('joint_recovery_cohort', ROOT / COHORT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_starts(space: dict, count: int, seed: int) -> list[dict]:
    """Create the anchor and all joint starts solely from the declared supports."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, initial_points
    if (not space or any(not levels or len(set(levels)) != len(levels) for levels in space.values())
            or sum(len(levels) > 1 for levels in space.values()) != 13
            or space.get(TRANSFER) != [.01] or count < len(space[CAPACITY])):
        raise ValueError('Require all thirteen variable coordinates and fixed transfer scale')
    anchor = {key: sorted(levels)[len(levels)//2] for key, levels in space.items()}
    # A single stratified draw is exactly the support midpoint and duplicates
    # its capacity ray. Build at least two draws, then retain the fixed budget.
    pool_size = max(count, len(space[CAPACITY]) + 2)
    points = initial_points(space, anchor, pool_size, seed, cover_key=CAPACITY)
    # initial_points covers every capacity at the neutral settings first.
    # Choose the capacity-midpoint member as the required common reference.
    ordered = ([anchor] + [p for p in points if p != anchor])[:count]
    if len(ordered) != count:
        raise ValueError('Neutral reference or stratified candidate count changed')
    return [candidate(point, 'independent_support_start') for point in ordered]


def selected_path_indices(seeds: np.ndarray, replicates: int) -> list[int]:
    if (seeds.ndim != 1 or len(set(seeds.tolist())) != len(seeds)
            or type(replicates) is not int or not 1 <= replicates <= len(seeds)):
        raise ValueError('Need distinct generation seeds and an available replicate count')
    return np.argsort(seeds)[:replicates].tolist()


def prepare(config: Path, subject: int, output: Path) -> None:
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
    from src.Bayesian_state.evaluation.model_0923_estimation_scope import validate_generation
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write
    from src.Bayesian_state.workflows.analysis.model_0923_h_objective_recovery import fixture
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError('Use a new synthetic-recovery preparation directory')
    spec = yaml.safe_load(config.read_text())
    if (spec['cell'] != 'MIX_LOG25' or spec['replicates'] != 2 or subject not in spec['sources']
            or spec['initialization'] != 'support_midpoint_and_independent_stratified_starts'
            or spec['path_selection'] != 'lowest_original_generation_seeds'
            or spec['parameter_estimation'] != 'all_thirteen_variable_coordinates'
            or not 1 <= spec['parallel_budget'] <= 14):
        raise ValueError('Require the declared bounded joint-recovery pilot')
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); value = digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Recovery preparation input changed')
        inputs[str(path)] = value
        return path

    bind(config); bind(__file__)
    policy = ROOT / spec['policy_reference_fit']
    policy_protocol, policy_frozen, policy_done = [load(bind(policy/name)) for name in
        ('protocol.json', 'nominees.json', 'completion.json')]
    if (policy_done['status'] != 'complete' or policy_done['smoke']
            or policy_done['nomination_sha256'] != digest(policy/'nominees.json')
            or policy_frozen['settings'] != policy_protocol['settings']
            or policy_frozen['settings']['belief_mechanism'] != 'SEPARATE'):
        raise ValueError('Require a completed original SEPARATE numerical policy')
    settings = deepcopy(policy_frozen['settings'])
    source = ROOT / spec['sources'][subject]
    done, protocol, frozen = [load(bind(source/name)) for name in ('completion.json','protocol.json','nominees.json')]
    if (done['status'] != 'complete' or done['smoke'] or protocol['smoke']
            or protocol['settings']['subject'] != subject or frozen['status'] != 'frozen_before_test'
            or frozen['test_choices_used'] or frozen['oral_used'] or done['oral_used']
            or done['nomination_sha256'] != digest(source/'nominees.json')
            or done['protocol_sha256'] != digest(source/'protocol.json')):
        raise ValueError('Require completed behavior-selected precision generation')
    for path, sha in done['inputs_sha256'].items():
        if digest(bind(path)) != sha:
            raise ValueError('Generating study source changed')
    cell = spec['cell']; truth = frozen['nominees'][cell]
    file = source / 'banks' / (cell+'_autonomous.npz')
    record = load(bind(file.with_suffix('.json')))
    if (digest(bind(file)) != record['predictions_sha256'] or record['candidate'] != truth
            or record['subject'] != subject or record['model'] != 'SEARCH'):
        raise ValueError('Generating path bank differs from its frozen point')
    with np.load(file) as z:
        generated = {k:z[k] for k in z.files}
    validate_generation(generated, record)
    jobs = [load(bind(p)) for p in sorted((source/'jobs').glob(f'autonomous_{cell}_*.json'))]
    if (not jobs or any(j['candidate'] != truth or j['execution_policy'] != 'MIX'
            or j['arm_settings']['precision'] != 'log' or not j['retain_precision'] for j in jobs)):
        raise ValueError('Require actual retained-precision MIX LOG25 generation')
    original = Path(jobs[0]['folder']); metadata = load(bind(original/'prepared.json'))
    cut = spec['expected_training_trials'][subject]
    if cut != metadata['training_trials'] or record['training_trials'] != cut:
        raise ValueError('Keep the original chronological training length')
    with np.load(bind(original/'training.npz')) as z:
        physical = {k:z[k] for k in ('stimulus','categories')}
    if digest(original/'training.npz') != metadata['array_sha256']['training.npz']:
        raise ValueError('Original physical training inputs changed')
    np.testing.assert_array_equal(generated['categories'][:cut], physical['categories'])
    model = helpers(); modules = model.helpers(); separate, online, _, recalled = modules
    space = online.fixed_space(recalled.base.model_space(settings, 'SEARCH'), settings['fixed_coordinates'])
    rate = 'engine.modules.beta_mod.kwargs.learning_rate'
    space[rate] = spec['precision_rates']
    if set(truth['hyperparams']) != set(space) or any(truth['hyperparams'][k] not in levels for k,levels in space.items()):
        raise ValueError('Generating point lies outside the joint fitting support')
    independent_starts(space, settings['initial_candidates'], spec['seed'])
    family = protocol['settings']['families']['LOG25']
    if family['precision'] != 'log' or family['upper'] != 25. or family['rate_scale'] != 1.:
        raise ValueError('Generating precision update changed')
    settings.update(protocol=spec['protocol'], parallel_budget=spec['parallel_budget'], subjects=[subject], scope=USE,
        recovery_initialization=spec['initialization'], recovery_generating_proposal='SEARCH',
        recovery_generation_seeds=generated['seeds'].tolist(),
        resource_predecessor=str(source.resolve()))
    settings['arms'] = {arm:dict(precision='log',upper=25.,rate_scale=1.,rates=spec['precision_rates']) for arm in ('POOL','SEARCH')}
    settings['particle_overrides'] = {}
    output.mkdir(parents=True)
    cases, truths = [], []
    for replicate, index in enumerate(selected_path_indices(generated['seeds'], spec['replicates'])):
        case = output / f'replicate_{replicate}'
        data = case / 'data'; folder = data / f'subject_{subject}'
        arrays = dict(physical, choices=generated['choices'][index,:cut].copy(), feedback=generated['feedback'][index,:cut].copy())
        fixture(original, folder, arrays)
        runtime = Path(jobs[0]['runtime'])
        write(data/'preparation_manifest.json', dict(status='complete', settings={'source_fit':str(runtime.parent.resolve())},
            data_role='synthetic_training_only', source_runtime=str(runtime.resolve()), subjects=[subject]))
        own = dict(deepcopy(settings), data=str(data.resolve()), seed=stable_seed(dict(
            protocol=spec['protocol'], base=spec['seed'], subject=subject, replicate=replicate, role='joint_recovery_search')))
        config_path = case / 'fit.yaml'; config_path.write_text(yaml.safe_dump(own,sort_keys=False))
        cases.append(dict(subject=subject,replicate=replicate,fit_config=str(config_path.resolve()),
                          fit_config_sha256=digest(config_path),training_trials=cut,
                          generation_seed=int(generated['seeds'][index]),source_runtime=str(runtime.resolve()),
                          prepared_sha256={name:digest(folder/name) for name in ('prepared.json','engine.json','training.npz','sealed_test.npz')}))
        truths.append(dict(subject=subject,replicate=replicate,model='SEARCH',cell=cell,candidate=truth,
                           generation_seed=int(generated['seeds'][index])))
    write(output/'truths.json',dict(rows=truths,role='Evaluation-only generating points; search reads fit_config and synthetic arrays'))
    write(output/'protocol.json',dict(settings=spec,subject=subject,source=str(source.resolve()),cases=cases,
        parameter_space=space,variable_coordinates=[k for k,v in space.items() if len(v)>1],
        initialization='Neutral support midpoint and independent stratified starts; human and generating nominees excluded from initialization',
        interpretation='Two sequences at each declared generating environment; pilot joint estimation, no population reliability claim'))
    for path,sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Generating input changed during preparation')
    write(output/'completion.json',dict(status='complete',subject=subject,cases=len(cases),inputs_sha256=inputs,
        protocol_sha256=digest(output/'protocol.json'),truths_sha256=digest(output/'truths.json')))


def case_settings(prepared: Path, replicate: int) -> tuple[dict, dict]:
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
    done, protocol = [load(prepared/name) for name in ('completion.json','protocol.json')]
    if done['status'] != 'complete' or done['protocol_sha256'] != digest(prepared/'protocol.json'):
        raise ValueError('Synthetic preparation changed')
    cases = [c for c in protocol['cases'] if c['replicate'] == replicate]
    if len(cases) != 1:
        raise ValueError('Unknown or duplicate recovery case')
    case = cases[0]; file = Path(case['fit_config'])
    if digest(file) != case['fit_config_sha256']:
        raise ValueError('Recovery fitting policy changed')
    settings = yaml.safe_load(file.read_text())
    inputs = {str((prepared/name).resolve()):digest(prepared/name) for name in ('completion.json','protocol.json')}
    inputs[str(file.resolve())] = digest(file)
    folder = Path(settings['data']) / f"subject_{case['subject']}"
    for name,sha in case['prepared_sha256'].items():
        if digest(folder/name) != sha:
            raise ValueError('Synthetic fitting arrays changed')
        inputs[str((folder/name).resolve())] = sha
    # Do not open truths.json or source nomination records during search.
    return settings, inputs


def configure(settings: dict, inputs: dict):
    cohort = helpers(); modules = cohort.helpers(); recalled = modules[-1]
    original_archive = recalled.base.archive
    cohort.configure(modules, settings, inputs)
    old_hashes, old_execute = recalled.base.source_hashes, recalled.base.execute

    def hashes(v):
        return {**old_hashes(v), ENTRY:v.digest(ROOT/ENTRY)}

    def starts(own, _unused_reference, smoke):
        from src.Bayesian_state.utils.seeding import stable_seed
        space = recalled.base.model_space(own,'SEARCH')
        if recalled.base.model_space(own,'POOL') != space:
            raise ValueError('Require common supports across proposal mechanisms')
        bank, required = {}, {}
        for subject in own['subjects']:
            count = 6 if smoke else own['initial_candidates']
            rows = independent_starts(space,count,stable_seed(dict(base=own['seed'],subject=subject,role='independent_joint_starts')))
            for arm in own['arms']:
                bank[subject,arm] = deepcopy(rows); required[subject,arm] = deepcopy(rows[0])
        return bank, required

    def execute(jobs, output, budget):
        forbidden = set(settings['recovery_generation_seeds'])
        if any(set(j['seeds']) & forbidden for j in jobs):
            raise ValueError('Scoring seeds overlap generating paths')
        old_execute(jobs,output,budget)

    def archive(output, config, source_hashes):
        original_archive(output,config,source_hashes)
        recalled.base.validation().write(output/'joint_recovery_protocol.json',dict(
            scope=USE,input_sha256=inputs,variable_coordinates=13,fixed_coordinates=settings['fixed_coordinates'],
            initialization=settings['recovery_initialization'],human_warm_starts_used=False,
            generating_parameters_used_for_initialization=False,selection='Synthetic training choices only; no oral input'))

    recalled.ENTRY = ENTRY
    recalled.starting_banks = starts
    recalled.base.source_hashes, recalled.base.execute, recalled.base.archive = hashes, execute, archive
    return recalled


def worker(file: Path) -> None:
    cohort = helpers()
    cohort.worker(file)


def search(prepared: Path, replicate: int, output: Path, smoke: bool = False) -> None:
    settings, inputs = case_settings(prepared,replicate)
    if smoke:
        settings['parallel_budget'] = 1
    module = configure(settings,inputs)
    with tempfile.TemporaryDirectory(prefix='joint_recovery_fit_') as temporary:
        config = Path(temporary)/'config.yaml';config.write_text(yaml.safe_dump(settings,sort_keys=False))
        module.search(config,output,smoke)


def score_truth(prepared: Path, replicate: int, fit: Path, output: Path, smoke: bool = False) -> None:
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import write
    if output.exists():
        raise FileExistsError('Use a new after-estimation truth-scoring directory')
    settings, inputs = case_settings(prepared,replicate)
    frozen, done = [load(fit/name) for name in ('nominees.json','completion.json')]
    if smoke:
        settings['parallel_budget'] = 1
    if (done['status'] != 'complete' or done['smoke'] != smoke or frozen['smoke'] != smoke
            or frozen['settings'] != settings or frozen['status'] != 'frozen_before_test'
            or done['nomination_sha256'] != digest(fit/'nominees.json')):
        raise ValueError('Generating point can be scored only after the matching estimates freeze')
    prepared_done = load(prepared/'completion.json')
    if digest(prepared/'truths.json') != prepared_done['truths_sha256']:
        raise ValueError('Evaluation-only generating parameters changed')
    truths = [r for r in load(prepared/'truths.json')['rows'] if r['replicate'] == replicate]
    if len(truths) != 1:
        raise ValueError('Missing unique generating point')
    truth = truths[0]
    for path in (fit/'nominees.json',fit/'completion.json',prepared/'truths.json'):
        inputs[str(path.resolve())] = digest(path)
    module = configure(settings,inputs)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='joint_recovery_truth_') as temporary:
        config=Path(temporary)/'config.yaml';config.write_text(yaml.safe_dump(settings,sort_keys=False))
        module.base.archive(output,config,module.base.source_hashes(module.base.validation()))
    write(output/'protocol.json',dict(settings=settings,fit=str(fit.resolve()),smoke=smoke,
        scope='Generating-point score after all estimates freeze',inputs_sha256=inputs))
    banks={(truth['subject'],arm):[truth['candidate']] for arm in ('POOL','SEARCH')}
    module.base.stage(settings,Path(settings['data']),output,banks,'confirmation',smoke)
    for path,sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Recovery source changed during truth scoring')
    write(output/'completion.json',dict(status='complete',smoke=smoke,inputs_sha256=inputs,
        fit_nomination_sha256=digest(fit/'nominees.json'),truths_sha256=digest(prepared/'truths.json')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','search','score-truth'))
    for name in ('config','prepared','fit','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--subject',type=int)
    parser.add_argument('--replicate',type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase=='prepare' and args.config and args.subject and args.output and not args.smoke:
        prepare(args.config,args.subject,args.output)
    elif args.phase=='search' and args.prepared and args.replicate is not None and args.output:
        search(args.prepared,args.replicate,args.output,args.smoke)
    elif args.phase=='score-truth' and args.prepared and args.replicate is not None and args.fit and args.output:
        score_truth(args.prepared,args.replicate,args.fit,args.output,args.smoke)
    else:
        parser.error('Provide the input paths required by the selected phase and a new output directory')
