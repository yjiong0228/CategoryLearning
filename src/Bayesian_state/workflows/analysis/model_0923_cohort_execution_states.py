"""Replay cohort SELECT nominees for external oral validation.

Use --phase replay --summary SUMMARY --output NEW [--workers N] [--smoke],
then --phase align --source REPLAY --oral-audit AUDIT --output NEW. The complete
cohort summary fixes every participant, parameter, filter seed and particle
count. Existing SELECT observers reproduce its saved choice probabilities.
The common oral readout uses states after choice and before current feedback;
only complete scientific replays can enter oral alignment. Short replay checks
retain both proposal arms and have software-validation scope.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = 'src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_states.py'
OBSERVER = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_refit_states.py'
COMPARISON = 'src/Bayesian_state/evaluation/model_0923_cohort_belief_comparison.py'
USE = 'development_cohort_dominant_execution_refit'
CELLS = ('SELECT_POOL', 'SELECT_SEARCH')
POLICY = 'existing_dominant_rule_only'


def helper(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Defer evaluation imports until source selection, so workers retain the
# archived cognitive package placed first on sys.path by the replay launcher.
cohort = helper('cohort_execution_state_reference', COHORT)
load, digest = cohort.load, cohort.digest
original_check_filter = cohort.check_filter


def bind_summary(source: Path, manifest: dict, mechanism: str, original) -> dict:
    """Add SELECT execution and analysis identity to the original cohort audit."""
    from src.Bayesian_state.evaluation import model_0923_cohort_execution_refit as scores
    if mechanism != 'SEPARATE':
        raise ValueError('SELECT states require SEPARATE evidence')
    inputs = original(source, manifest, mechanism)
    file = source / 'execution_manifest.json'; declaration = load(file)
    frozen = load(Path(manifest['fit']) / 'nominees.json')['settings']
    if (declaration.get('status') != 'complete' or declaration.get('execution_policy') != 'SELECT'
            or declaration.get('rule_execution_policy') != POLICY or declaration.get('scope') != USE
            or declaration.get('cohort_manifest_sha256') != digest(source / 'cohort_manifest.json')
            or frozen.get('execution_policy') != 'SELECT' or frozen.get('scope') != USE):
        raise ValueError('Cohort summary does not identify the completed SELECT fit')
    for path in (Path(scores.__file__), Path(scores.driver.__file__)):
        if declaration['inputs_sha256'].get(str(path.resolve())) != digest(path):
            raise ValueError('SELECT summary used a different implementation')
    additions = dict(declaration['inputs_sha256'], **{str(file.resolve()): digest(file)})
    for path, expected in additions.items():
        if (path in inputs and inputs[path] != expected) or digest(Path(path)) != expected:
            raise ValueError('SELECT summary inputs changed')
        inputs[path] = expected
    for role in ('fit', 'evaluation'):
        for path in (Path(manifest[role]) / 'jobs').glob('*.json'):
            job = load(path); target = Path(job['target'])
            sidecar = target.with_suffix('.execution.json')
            if str(sidecar.resolve()) not in inputs:
                raise ValueError('SELECT execution is absent from the completed summary audit')
            scores.execution.check_selection(job, load(target.with_suffix('.json')), load(sidecar))
    return inputs


@lru_cache(maxsize=1)
def comparison_helpers():
    from src.Bayesian_state.evaluation import model_0923_cohort_execution_refit as scores
    module = helper('src.Bayesian_state.evaluation._cohort_execution_state_comparison', COMPARISON)
    module.cohort = scores.cohort_summary()
    original = module.bind_summary
    module.bind_summary = lambda source, manifest, mechanism: bind_summary(source, manifest, mechanism, original)
    return module


def check_filter(planned: dict, actual: dict, point: dict, settings: dict, mechanism: str,
                 subject: int, arm: str, seed: int, particles: int, folder: Path, metadata: dict,
                 record: dict, precision: dict, belief: dict, references: dict,
                 execution: dict | None = None) -> None:
    if mechanism != 'SEPARATE' or actual.get('execution_policy') != 'SELECT':
        raise ValueError('Cohort SELECT replay lost its execution identity')
    original_check_filter(dict(planned, execution_policy='SELECT'), actual, point, settings, mechanism,
                          subject, arm, seed, particles, folder, metadata, record, precision, belief, references)
    if execution is None:
        execution = load(Path(actual['target']).with_suffix('.execution.json'))
    from src.Bayesian_state.evaluation.model_0923_dominant_execution_refit import check_selection
    check_selection(actual, record, execution)


cohort.comparison_helpers, cohort.check_filter, cohort.USE = comparison_helpers, check_filter, USE


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    if tuple(settings['arms']) != CELLS:
        raise ValueError('Keep both SELECT proposal arms')
    files, context = cohort.source_jobs(dict(settings, arms=['POOL', 'SEARCH']), source, smoke)
    summary = ROOT / settings['summary']
    context.update(execution_policy='SELECT', rule_execution_policy=POLICY,
        execution_fit=context['cohort_fit'], execution_summary=str(summary.resolve()),
        execution_manifest_sha256=digest(summary / 'execution_manifest.json'))
    context['selected_parameters'] = [dict(row, arm='SELECT_' + row['arm']) for row in context['selected_parameters']]
    module = observer()
    for relative in (ENTRY, COHORT, OBSERVER, module.REPLAY, module.helpers().EXECUTION):
        context['refit_replay_inputs'][str((ROOT / relative).resolve())] = digest(ROOT / relative)
    return files, context


def observer():
    module = helper('cohort_execution_observer', OBSERVER)
    module.USE, module.ENTRY, module.CELLS = USE, ENTRY, CELLS
    module.source_jobs, module.estimation_context = source_jobs, estimation_context
    return module


def worker(file: Path) -> None:
    job = load(file)
    if job.get('cohort_belief_mechanism') != 'SEPARATE' or job.get('belief_mechanism') != 'SEPARATE':
        raise ValueError('Cohort SELECT state worker lost its belief identity')
    observer().worker(file)


def run(summary: Path, output: Path, workers: int = 128, smoke: bool = False) -> None:
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError('Use one to 128 workers, bounded by ready tasks and available CPUs')
    manifest = load(summary / 'manifest.json')
    frozen = load(Path(manifest['fit']) / 'nominees.json')['settings']
    config = dict(source=manifest['evaluation'], summary=str(summary.resolve()), sample_role='development',
        subjects=frozen['subjects'], arms=list(CELLS), workers=workers,
        seeds_per_arm=frozen['budgets']['evaluate']['seeds'])
    with tempfile.TemporaryDirectory(prefix='cohort_select_states_') as temporary:
        path = Path(temporary) / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        observer().run(path, output, smoke)


def check_record(job: dict, record: dict) -> None:
    observer().check_record(job, record)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    metadata, _ = cohort.estimation_context(protocol)
    if (protocol.get('execution_policy') != 'SELECT' or protocol.get('rule_execution_policy') != POLICY
            or protocol.get('execution_fit') != protocol['cohort_fit']
            or protocol.get('execution_summary') != protocol['cohort_summary']
            or not protocol.get('execution_manifest_sha256')):
        raise ValueError('Oral evaluation requires a completed cohort SELECT execution summary')
    selected = protocol['selected_parameters']
    keys = {(row['subject'], row['arm']) for row in selected}
    if len(keys) != len(selected) or keys != {(subject, arm) for subject, _ in keys for arm in CELLS}:
        raise ValueError('Oral evaluation requires both SELECT proposal arms for every declared participant')
    for key in ('execution_policy', 'rule_execution_policy', 'execution_fit', 'execution_summary', 'execution_manifest_sha256'):
        metadata[key] = protocol[key]
    return metadata, (
        '<p>参数由SEPARATE＋SELECT机制下各被试原训练段的选择似然估计。'
        '每个粒子使用当前主导规则回答，其他在场规则继续更新信念和精度。'
        '状态逐值复现原评价的类别预测，保留原参数、粒子数、全部独立种子与试次边界。'
        '口述检验本题选择后、反馈前的局部分类依据；后段属于开发阶段的时间外推诊断。</p>')


def align(source: Path, oral_audit: Path, output: Path) -> None:
    observer().align(source, oral_audit, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('replay', 'align'), default='replay')
    for name in ('summary', 'source', 'oral-audit', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == 'replay' and args.summary and args.output:
        run(args.summary, args.output, args.workers, args.smoke)
    elif args.phase == 'align' and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source, args.oral_audit, args.output)
    else:
        parser.error('Replay needs --summary; align needs --source and --oral-audit; use a new --output')
