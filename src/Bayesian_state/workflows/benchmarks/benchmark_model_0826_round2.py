"""Bounded round-2 ablations and isolated probability/diagnostic probes.

Baseline functions are loaded from explicitly supplied, hashed source snapshots.
The probability and summary probes are scoped to this process and are NOT
production inference modes. In particular, omitted summaries must never be
passed to the fitting/selection pipeline as if they were measured diagnostics.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from weakref import WeakKeyDictionary

from joblib import Parallel, delayed
import numpy as np

from ...hypothesis_space.observation_model.base_partition import BasePartition
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.geometry.stimuli import as_stimuli
from ...inference.backends import particle_filter as pf
from ...model.modules.hypothesis_transition import contracts, execution
from . import benchmark_model_0826_acceleration as benchmark


def load_function(path: Path, name: str, namespace: dict, owner: str | None = None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if owner is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner
    )
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(module, str(path), 'exec', dont_inherit=True, flags=__import__('__future__').annotations.compiler_flag), scope)
    return scope[name]


def summary_probe():
    """Remove two output-only aggregation blocks, not cognitive state updates."""
    tree = ast.parse(inspect.getsource(pf.run_state_model_particle_filter))
    function = tree.body[0]
    trial_loop = next(node for node in function.body if isinstance(node, ast.For)
                      and isinstance(node.target, ast.Name) and node.target.id == 'trial_index')
    def target_name(node):
        if not isinstance(node, ast.Assign):
            return None
        target = node.targets[0]
        return target.value.id if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) else getattr(target, 'id', None)
    names = [target_name(node) for node in trial_loop.body]
    ranges = [(names.index('predictive_transition_rate'), names.index('choice_index')),
              (names.index('filtered_execution_switch_event_probability'),
               next(i for i, node in enumerate(trial_loop.body)
                    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == 'particle_swap_counts'))]
    # Assertions intentionally fail if future refactoring changes these blocks.
    assert ranges[0][0] < ranges[0][1] < ranges[1][0] < ranges[1][1]
    trial_loop.body = [node for i, node in enumerate(trial_loop.body)
                       if not any(start <= i < stop for start, stop in ranges)]
    scope = dict(vars(pf))
    exec(compile(ast.fix_missing_locations(tree), '<summary-only-probe>', 'exec'), scope)
    return scope[function.name]


@contextmanager
def probability_probe():
    """Bounded exact memoization; mutable geometry is outside this probe's scope."""
    original = ContinuousPartition.get_category_probabilities
    caches = WeakKeyDictionary()
    stats = {'hits': 0, 'misses': 0, 'max_entries_per_partition': 0}
    def cached(self, hypo, data, beta, distance_mode=None, **kwargs):
        mode = self._resolve_distance_mode(distance_mode)
        if beta == 0 or kwargs or type(self) is not ContinuousPartition or not np.isfinite(beta):
            return original(self, hypo, data, beta, distance_mode, **kwargs)
        values = as_stimuli(data[0], self.n_dims)
        if values.shape[0] != 1:
            return original(self, hypo, data, beta, distance_mode, **kwargs)
        geometry = self.boundary_geometry if mode == 'boundary' else self.prototype_geometry
        context = (geometry.space, geometry.method,
                   getattr(geometry, 'tolerance', None),
                   getattr(geometry, 'projection_iterations', None),
                   getattr(geometry, 'dykstra_backend', None))
        key = (context, int(hypo), mode, float(beta), values.shape, values.tobytes())
        cache = caches.setdefault(self, OrderedDict())
        if key in cache:
            stats['hits'] += 1
            cache.move_to_end(key)
            return cache[key].copy()
        stats['misses'] += 1
        result = original(self, hypo, data, beta, distance_mode, **kwargs)
        cache[key] = result.copy()
        if len(cache) > 512:
            cache.popitem(last=False)
        stats['max_entries_per_partition'] = max(stats['max_entries_per_partition'], len(cache))
        return result
    with patch.object(ContinuousPartition, 'get_category_probabilities', cached):
        yield stats


@contextmanager
def variant(mode: str, snapshots: Path):
    with ExitStack() as stack:
        if mode in ('baseline', 'likelihood'):
            for cls, name, module, file in (
                (contracts.HypothesisSelection, 'from_active_sets', contracts, 'contracts.py.txt'),
                (contracts.TwoStepHypothesisTransitionMixin, '_validate_selection', contracts, 'contracts.py.txt'),
                (execution.WorkspaceTransitionExecutionMixin, 'select_hypotheses', execution, 'execution.py.txt'),
            ):
                old = load_function(snapshots / file, name, vars(module), cls.__name__)
                stack.enter_context(patch.object(cls, name, classmethod(old) if name == 'from_active_sets' else old))
            old_pf = load_function(snapshots / 'particle_filter.py.txt', 'run_state_model_particle_filter', vars(pf))
            stack.enter_context(patch.object(benchmark, 'run_state_model_particle_filter', old_pf))
        if mode == 'baseline':
            stack.enter_context(patch.object(ContinuousPartition, 'calc_likelihood', BasePartition.calc_likelihood))
        if mode == 'probability':
            stats = stack.enter_context(probability_probe())
        else:
            stats = {}
        if mode == 'summaries':
            stack.enter_context(patch.object(benchmark, 'run_state_model_particle_filter', summary_probe()))
        yield stats


def check(reference, arrays, summaries=False):
    if not summaries:
        benchmark.compare_arrays(reference, arrays)
        return len(arrays)
    # The probe deliberately omits some latent summaries; all other namespaces
    # (including predictive probabilities, state and resampling) must match.
    keys = {key for key in reference if not key.startswith('result/latent_summaries/')}
    for key in keys:
        np.testing.assert_array_equal(arrays[key], reference[key], err_msg=key)
    return len(keys)


def ablations(args):
    selected = [case for case in benchmark.cases() if case['name'] in args.cases]
    if {case['name'] for case in selected} != set(args.cases):
        raise ValueError('Unknown case name')
    modes = ('baseline', 'likelihood', 'production', 'probability', 'summaries')
    reports = []
    for case in selected:
        rows = {}
        for repeat in range(args.repeats):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                with variant(mode, args.baseline_source) as stats:
                    row, arrays = benchmark.run_case(case, 'default', 1)
                with np.load(args.reference / (case['name'] + '.npz'), allow_pickle=False) as ref:
                    row['compared_output_items'] = check(ref, arrays, mode == 'summaries')
                if mode not in rows:
                    rows[mode] = row
                    row['probe_stats'] = dict(stats)
                else:
                    rows[mode]['warm_seconds'].extend(row['warm_seconds'])
                rows[mode]['warm_median_seconds'] = float(np.median(rows[mode]['warm_seconds']))
        reports.append({'case': case['name'], 'modes': rows})
        print(case['name'], {mode: round(row['warm_median_seconds'], 4) for mode, row in rows.items()}, flush=True)
    return reports


def parallel_task(case, reference):
    row, arrays = benchmark.run_case(case, 'default', 1)
    with np.load(reference / (case['name'] + '.npz'), allow_pickle=False) as ref:
        benchmark.compare_arrays(ref, arrays)
    return row['warm_median_seconds']


def throughput(args):
    case = next(case for case in benchmark.cases() if case['name'] == 'c1_M3_chi0')
    rows = []
    for jobs in (1, 2, 4, 8):
        with Parallel(n_jobs=jobs) as pool:
            start = perf_counter()
            pool(delayed(parallel_task)(case, args.reference) for _ in range(jobs))
            warmup = perf_counter() - start
            seconds = []
            for _ in range(args.repeats):
                start = perf_counter()
                pool(delayed(parallel_task)(case, args.reference) for _ in range(8))
                seconds.append(perf_counter() - start)
        rows.append({'jobs': jobs, 'tasks': 8, 'pf_runs_per_task': 2,
                     'startup_warmup_seconds': warmup, 'wall_seconds': seconds,
                     'median_seconds': float(np.median(seconds))})
        print('parallel', rows[-1], flush=True)
    return rows


def state_checks(args):
    """Compare every particle's pre/post-update state, including RNG payloads."""
    import pandas as pd
    import yaml
    from ...model import StateModel
    frame = pd.read_csv('data/exp123/processed/Task2_processed.csv')
    rows = []
    for condition, subject, gate in ((1, 129, 0.), (1, 129, .35), (1, 129, 1.),
                                      (2, 229, 1.), (3, 301, 1.)):
        data = frame.loc[frame.iSub.eq(subject)].iloc[:16]
        config = yaml.safe_load(Path(f'configs/exp123/model_struct/pmh_model_cond{condition}_0826.yaml').read_text())
        config['modules']['hypo_transitions_mod']['kwargs']['persistent_execution']['enabled'] = True
        outputs = {}
        original = StateModel.complete_trial
        for mode in ('baseline', 'production'):
            states = []
            def capture(model, *positional, **keywords):
                before = pf._snapshot(model).payload
                result = original(model, *positional, **keywords)
                states.append({'before': before, 'after': pf._snapshot(model).payload})
                return result
            with variant(mode, args.baseline_source), patch.object(StateModel, 'complete_trial', capture):
                result = benchmark.run_state_model_particle_filter(
                    engine_config=config, subject_id=subject, condition=condition,
                    stimulus=data[[f'feature{i}' for i in range(1, 5)]].to_numpy(),
                    choices=data.choice.to_numpy(), feedback=data.feedback.to_numpy(),
                    particle_count=4, choice_readout_power=1., filter_seed=8326,
                    learning_update_probability=gate, resample_threshold_fraction=1.,
                    choice_transmission_audit=condition == 1,
                )
            outputs[mode] = benchmark.numerical_outputs(result)
            outputs[mode].update(benchmark.numerical_outputs(SimpleNamespace(particle_states=states)))
        benchmark.compare_arrays(outputs['baseline'], outputs['production'])
        row = {'condition': condition, 'subject': subject, 'gate': gate, 'trials': len(data),
               'particles': 4, 'seed': 8326, 'resample_threshold_fraction': 1.,
               'exact_items': len(outputs['production']), 'exact_match': True}
        rows.append(row)
        print('states', row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--state-check', action='store_true')
    parser.add_argument('--cases', nargs='+', default=['c1_M3_chi0', 'c2_M3_chi0', 'c3_M3_chi0', 'c1_long'])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.baseline_source / 'manifest.json').read_text())
    for name, expected in manifest.items():
        snapshot = args.baseline_source / (Path(name).name + '.txt')
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected, snapshot
    started = perf_counter()
    if args.parallel and args.state_check:
        parser.error('parallel and state-check are separate protocols')
    rows = state_checks(args) if args.state_check else throughput(args) if args.parallel else ablations(args)
    paths = [*Path('src/Bayesian_state').rglob('*.py'),
             Path('data/exp123/processed/Task2_processed.csv'),
             *Path('configs/exp123/model_struct').glob('*0826.yaml')]
    output = {'results': rows, 'wall_seconds': perf_counter() - started,
              'reference': str(args.reference), 'baseline_sha256': manifest,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output_dir / 'report.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
