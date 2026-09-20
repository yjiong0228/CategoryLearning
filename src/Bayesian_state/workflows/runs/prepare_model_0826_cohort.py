"""Read-only reuse audit and a frozen launch plan; this module never runs PF.

Reuse means retaining a compatible bounded fit, including unresolved outcomes.
It does not promote supplemental small-bank checks to whole-fit acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import shlex

from ...optimization.adaptive_boundary import adaptive_support
from ...optimization.adaptive_fit import DEFAULT_CONFIG, load_fit_config, run_fit, select_subjects
from ...optimization.adaptive_runtime import (
    Scorer, digest, freeze_json, run_manifest, verify_manifest,
)
from ...optimization.diagnostics.decision_precision import mixture_nll
from ...optimization.parameter_space import load_model_parameter_space
from ...optimization.search.adaptive_proposals import point_id
from ...utils.paths import ROOT_DIR


# These standalone entries were added after the archived fits. None is imported
# by the shared fitting/model path; additions elsewhere require a new review.
REVIEWED_ADDITIONS = {
    'src/Bayesian_state/workflows/analysis/diagnose_model_0826_numerics.py',
    'src/Bayesian_state/workflows/analysis/probe_model_0826_frozen_boundaries.py',
    'src/Bayesian_state/workflows/runs/prepare_model_0826_cohort.py',
}


def policy_equal(first: dict, second: dict) -> bool:
    """Only the inert reporting label may differ; keep seeds and budgets exact."""
    a, b = deepcopy(first), deepcopy(second)
    a.pop('analysis_id', None)
    b.pop('analysis_id', None)
    return a == b


def split_subjects(specs: list[dict], reused: list[int]) -> list[int]:
    ids = [s['subject'] for s in specs]
    if len(ids) != len(set(ids)) or len(reused) != len(set(reused)) or not set(reused) <= set(ids):
        raise ValueError('Duplicate or out-of-cohort subject in reuse sources')
    return [sid for sid in ids if sid not in set(reused)]


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT_DIR))


def check_receipts(run: Path) -> tuple[int, int]:
    """Verify every unique PF against its completed receipt, without replay."""
    expected = {}
    receipts = sorted((run/'batches').rglob('scores.json'))
    for path in receipts:
        for rel, sha in json.loads(path.read_text())['cache_sha256'].items():
            if not (run/rel).resolve().is_relative_to((run/'cache').resolve()):
                raise ValueError('Receipt points outside its own cache')
            if rel in expected and expected[rel] != sha:
                raise ValueError('Receipts disagree about a reused PF')
            expected[rel] = sha
    actual = {str(p.relative_to(run)) for p in (run/'cache').glob('*/*/*.npz')}
    if not expected or actual != set(expected):
        raise ValueError('Incomplete or unreceipted PF cache')
    for rel, sha in expected.items():
        if digest(run/rel) != sha:
            raise ValueError(f'PF checksum differs: {run/rel}')
    return len(expected), len(receipts)


def audit_source(run: Path, current: dict, all_plan: dict, metadata: dict) -> dict:
    manifest_path = run/'manifest.json'
    old = json.loads(manifest_path.read_text())
    verify_manifest(old)
    if old['smoke'] or not policy_equal(old['config'], current['config']) or old['versions'] != current['versions']:
        raise ValueError('Source policy/environment differs or is smoke-only')
    additions = sorted(set(current['input_sha256']) - set(old['input_sha256']))
    added_python = [relative(Path(p)) for p in additions if p.endswith('.py')]
    if not set(added_python) <= REVIEWED_ADDITIONS:
        raise ValueError(f'Unreviewed source additions: {added_python}')
    completion = json.loads((run.parent/'finalization.json').read_text())
    if completion['status'] != 'complete':
        raise ValueError('Source full run did not complete validation')
    results = json.loads((run/'fit_results.json').read_text())
    specs = {s['subject']: s for s in current['subjects']}
    trials = {s['subject']: s['trials'] for s in all_plan['subjects']}
    if results['smoke_only'] or set(results['subjects']) != {str(s['subject']) for s in old['subjects']}:
        raise ValueError('Source subject inventory differs')
    for spec in old['subjects']:
        if specs.get(spec['subject']) != spec:
            raise ValueError('Subject/condition/engine differs')
    _, _, support = adaptive_support(load_model_parameter_space(current['config']['parameter_space']), current['config']['boundary']['extensions'])
    if json.loads((run/'effective_parameter_support.json').read_text()) != support:
        raise ValueError('Parameter support differs')
    pf_count, receipt_count = check_receipts(run)
    scorer = Scorer(run, old['config'], {})
    rows = []
    for sid_text, row in results['subjects'].items():
        sid = int(sid_text)
        if (row['subject'] != sid or row['condition'] != specs[sid]['condition'] or
                row['trial_count'] != trials[sid] or row['scored_trial_count'] != trials[sid]-1 or row['smoke_only']):
            raise ValueError('Subject result metadata differs from complete data')
        if row != json.loads((run/'subjects'/sid_text/'fit_result.json').read_text()):
            raise ValueError('Aggregate and subject result differ')
        cycle = row['audit_family'].split('/')[0]
        frozen = json.loads((run/'progress'/f'{cycle}_frozen_selection.json').read_text())
        if frozen['primary'][sid_text] != row['selected'] or row['selected'] != point_id(row['hyperparams']):
            raise ValueError('Frozen nominee changed')
        bank = {r['id']: r for r in row['candidate_bank']}
        if bank != {r['id']: r for r in frozen['banks'][sid_text]} or set(bank) != set(row['independent_audit']['scores']):
            raise ValueError('Frozen candidate bank changed')
        for pid, point in bank.items():
            if point_id(point['hyperparams']) != pid:
                raise ValueError('Candidate identity differs')
            arrays = scorer.arrays(sid, point, row['audit_budget'], row['audit_family'])
            mask = arrays['mask']
            if len(mask) != trials[sid] or mask[0] or not mask[1:].all():
                raise ValueError('Trial count or scoring mask differs')
            nll = mixture_nll(arrays['probabilities'], arrays['observed'], mask)
            if abs(nll-row['independent_audit']['scores'][pid]) > 1e-12:
                raise ValueError('Archived audit NLL differs')
        if abs(row['selected_mean_nll']-row['independent_audit']['scores'][row['selected']]) > 1e-12:
            raise ValueError('Selected audit score differs')
        rows.append({
            'subject': sid, 'condition': row['condition'], 'trials': row['trial_count'],
            'scored_trials': row['scored_trial_count'], 'reuse': 'compatible_bounded_fit',
            'status': row['status'], 'issues': row['issues'], 'selected': row['selected'],
            'audit_status': row['independent_audit']['status'], 'audit_budget': row['audit_budget'],
            'audit_regret_upper': row['independent_audit']['regret_upper'],
            'selected_mean_nll': row['selected_mean_nll'], 'near_candidates': row['near_candidates'],
            'search_plateau': row['search_trace'][-1]['plateau'],
            'challenge_status': row['challenge_diagnostic']['status'],
            'artificial_boundary_parameters': sorted({h['parameter'] for h in row['boundary']['hits'] if h['kind'] == 'artificial_boundary'}),
            'source_result': relative(run/'subjects'/sid_text/'fit_result.json'),
        })
    # Freeze all non-cache source artifacts as well; hash every PF via receipts.
    for path in run.rglob('*'):
        if path.is_file() and 'cache' not in path.relative_to(run).parts and path.name != '.run.lock':
            metadata[str(path.resolve())] = digest(path)
    metadata[str((run.parent/'finalization.json').resolve())] = digest(run.parent/'finalization.json')
    return {'run': relative(run), 'subjects': rows, 'pf_checksums_verified': pf_count,
            'receipts_verified': receipt_count, 'existing_source_data_config_hashes_unchanged': True,
            'environment_matches': True, 'support_matches': True, 'policy_matches_except_analysis_id': True,
            'reviewed_source_additions': added_python, 'historical_strict_resume_with_current_inventory': False,
            'reuse_scope': 'Retain original parameters, scores and issues; no replacement by supplemental winners.'}


def verify_prepared(directory: Path) -> dict:
    plan = json.loads((directory/'launch_plan.json').read_text())
    frozen = json.loads((directory/'frozen_manifest.json').read_text())
    verify_manifest(frozen)
    cfg_path = ROOT_DIR/plan['config']
    cfg = load_fit_config(cfg_path)
    specs = select_subjects(cfg, plan['remaining_subjects'], [1, 2, 3], False)
    if run_manifest(cfg, cfg_path.resolve(), specs, False) != frozen['remaining_run_manifest']:
        raise ValueError('Prepared source inventory, inputs, environment or roster changed')
    for source in plan['source_runs']:
        check_receipts(ROOT_DIR/source)
    return plan


def prepare(config_path: Path, source_runs: list[Path], output: Path, fit_output: Path) -> dict:
    config_path = config_path.resolve()
    if output.exists() or fit_output.exists() or output.resolve() == fit_output.resolve():
        raise ValueError('Use distinct new preparation and future-fit directories')
    # run_fit returns before making a Scorer or writing outputs in dry-run mode.
    all_plan = run_fit(config_path, None, conditions=[1, 2, 3], dry_run=True)
    cfg = all_plan['config']
    specs = select_subjects(cfg, None, [1, 2, 3], False)
    current = run_manifest(cfg, config_path, specs, False)
    metadata = {}
    sources = [audit_source(p.resolve(), current, all_plan, metadata) for p in source_runs]
    reused_rows = [r for source in sources for r in source['subjects']]
    remaining = split_subjects(specs, [r['subject'] for r in reused_rows])
    if not remaining:
        raise ValueError('No remaining subjects to plan')
    remaining_plan = run_fit(config_path, None, remaining, [1, 2, 3], dry_run=True)
    remaining_specs = select_subjects(cfg, remaining, [1, 2, 3], False)
    command = ['python', '-m', 'src.Bayesian_state.run_model_0826_fit', '--config', relative(config_path),
               '--subjects', *map(str, remaining), '--conditions', '1', '2', '3', '--output-dir', relative(fit_output)]
    plan = {'config': relative(config_path), 'source_runs': [s['run'] for s in sources],
            'reused_subjects': sorted(r['subject'] for r in reused_rows), 'remaining_subjects': remaining,
            'conditions': [1, 2, 3], 'command': command, 'future_output': relative(fit_output),
            'parallel_budget': cfg['parallel_budget'], 'numeric_threads_per_worker': 1,
            'status': 'prepared_not_started', 'new_pf_runs': 0,
            'interpretation': 'Bounded fits including unresolved outcomes; no whole-cohort acceptance or recovery claim.'}
    output.mkdir(parents=True, exist_ok=False)
    freeze_json(output/'reuse_audit.json', {'sources': sources, 'total_verified_pf': sum(s['pf_checksums_verified'] for s in sources)})
    freeze_json(output/'all_subjects_dry_run.json', all_plan)
    freeze_json(output/'remaining_dry_run.json', remaining_plan)
    freeze_json(output/'launch_plan.json', plan)
    reused = {r['subject']: r for r in reused_rows}
    with (output/'subject_manifest.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'condition', 'trials', 'scored_trials', 'action', 'source_result', 'original_status', 'audit_status', 'issues'])
        writer.writeheader()
        for s in all_plan['subjects']:
            prior = reused.get(s['subject'], {})
            writer.writerow({'subject': s['subject'], 'condition': s['condition'], 'trials': s['trials'], 'scored_trials': s['trials']-1,
                'action': 'reuse_with_original_flags' if prior else 'fit_pending', 'source_result': prior.get('source_result', ''),
                'original_status': prior.get('status', ''), 'audit_status': prior.get('audit_status', ''), 'issues': '|'.join(prior.get('issues', []))})
    script = '\n'.join([
        '#!/usr/bin/env bash', '# Run from the repository root; prepare/verify does not launch PF.', 'set -euo pipefail',
        'if (( $# > 1 )); then echo "Usage: bash run_remaining.sh [--dry-run|--resume]" >&2; exit 2; fi',
        'case "${1:-}" in ""|--dry-run|--resume) ;; *) echo "Only --dry-run or --resume is allowed" >&2; exit 2 ;; esac',
        'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export NUMBA_CACHE_DIR=/tmp/model0826_cohort_numba',
        shlex.join(['python', '-m', 'src.Bayesian_state.workflows.runs.prepare_model_0826_cohort', '--verify-plan', relative(output)]),
        'exec '+shlex.join(command)+' "$@"', ''])
    (output/'run_remaining.sh').write_text(script)
    for p in output.iterdir():
        if p.is_file(): metadata[str(p.resolve())] = digest(p)
    metadata.update(current['input_sha256'])
    freeze_json(output/'frozen_manifest.json', {'input_sha256': metadata,
        'remaining_run_manifest': run_manifest(cfg, config_path, remaining_specs, False)})
    counts = Counter(s['condition'] for s in remaining_specs)
    return {'prepared': relative(output), 'reused': len(reused), 'remaining': len(remaining),
            'remaining_per_condition': dict(counts), 'verified_pf': sum(s['pf_checksums_verified'] for s in sources), 'new_pf_runs': 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--source-runs', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--fit-output-dir', type=Path)
    parser.add_argument('--verify-plan', type=Path)
    args = parser.parse_args()
    if args.verify_plan:
        if any((args.source_runs, args.output_dir, args.fit_output_dir)):
            parser.error('--verify-plan cannot be combined with preparation outputs')
        plan = verify_prepared(args.verify_plan)
        result = {'verified': str(args.verify_plan), 'remaining': len(plan['remaining_subjects']), 'new_pf_runs': 0}
    else:
        if not all((args.source_runs, args.output_dir, args.fit_output_dir)):
            parser.error('Need --source-runs, --output-dir and --fit-output-dir')
        result = prepare(args.config, args.source_runs, args.output_dir, args.fit_output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
