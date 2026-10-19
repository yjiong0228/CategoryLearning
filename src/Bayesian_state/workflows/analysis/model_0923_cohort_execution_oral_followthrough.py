"""Finish planned oral comparisons after the cohort queue's marker mismatch.

Use --predecessor ORIGINAL_QUEUE --output NEW_DIRECTORY. The original queue
checks completion.json after an oral comparison that writes manifest.json.
Its participant branch stops there; other participant fits continue. Reuse
each completed midpoint POOL comparison and run only its four remaining oral
steps in new directories. Scientific fits, replay arrays, coding and scorers
remain unchanged. At most two single-thread report processes run together.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from ...evaluation.model_0923_h_validation import digest, load
from .model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[4]
STATES = 'src.Bayesian_state.workflows.analysis.model_0923_cohort_execution_states'
COMPARE = 'src.Bayesian_state.workflows.analysis.model_0923_execution_refit_oral'


def verify_report(folder: Path) -> dict:
    """Read the scorer's actual marker and check every declared artifact."""
    record = load(folder / 'manifest.json')
    if record.get('status') != 'complete' or not record.get('artifacts'):
        raise ValueError('Require a complete oral manifest with artifacts')
    for name, expected in record['artifacts'].items():
        if digest(folder / name) != expected:
            raise ValueError('Completed oral artifact changed')
    for key in ('input_sha256', 'source_sha256'):
        for name, expected in record.get(key, {}).items():
            if digest(ROOT / name) != expected:
                raise ValueError('Completed oral input or source changed')
    return record


def run(predecessor: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new oral followthrough directory')
    predecessor = predecessor.resolve(); output = output.resolve()
    plan = load(predecessor / 'plan.json')
    if [row['subject'] for row in plan['subjects']] != [207, 306]:
        raise ValueError('Require the existing 207/306 cohort queue')
    original = predecessor / 'run.py'
    if digest(original) != plan['inputs_sha256'][str(original)]:
        raise ValueError('Original queue source changed')
    original_text = original.read_text()
    if "comparison,'completion.json'" not in original_text:
        raise ValueError('Original queue does not contain the diagnosed marker mismatch')
    sources = [Path(__file__), original, predecessor / 'plan.json']
    sources += [Path(importlib.import_module(name).__file__) for name in (STATES, COMPARE)]
    bindings = {str(p.resolve()): digest(p) for p in sources}
    output.mkdir(parents=True)
    write(output / 'protocol.json', dict(status='running', subjects=[207, 306],
        predecessor=str(predecessor), inputs_sha256=bindings, max_report_processes=2,
        numeric_threads=1, scope='Existing planned oral scoring only; no fitting or state replay'))

    def verify_sources():
        if any(digest(Path(name)) != expected for name, expected in bindings.items()):
            raise ValueError('Bound followthrough source changed')

    def subject_run(item):
        subject = item['subject']; result_root = ROOT / 'results/model_0923'
        old_pool = result_root / f'cohort_execution_s{subject}_oral_midpoint_pool_comparison_20261020_v1'
        print(f'S{subject}: waiting for original midpoint POOL manifest', flush=True)
        while not (old_pool / 'manifest.json').exists():
            if (predecessor / 'failure.json').exists():
                raise RuntimeError(f'Original queue ended before S{subject} midpoint POOL completion')
            time.sleep(20)
        adopted = verify_report(old_pool)
        if (adopted.get('subject') != subject or adopted.get('arm') != 'POOL'
                or not adopted.get('original_scores_reproduced') or not adopted.get('revised_scores_reproduced')):
            raise ValueError('Original comparison identity differs')
        if (old_pool / 'completion.json').exists():
            raise ValueError('Original marker now exists; check ownership before resuming')
        states = Path(item['paths']['states']); done = load(states / 'completion.json')
        if done['status'] != 'complete' or done['smoke'] or done['exact_behavior_replays'] != done['jobs'] or done['jobs'] != 16:
            raise ValueError('Need all original scientific state replays')
        folder = output / f'subject_{subject}'; folder.mkdir()
        midpoint = Path(adopted['revised']); verify_report(midpoint)
        relative = folder / 'oral_relative'
        tasks = [('midpoint_search_comparison', COMPARE, [
            '--original', adopted['original'], '--revised', str(midpoint), '--subject', str(subject), '--arm', 'SEARCH'])]
        tasks += [('oral_relative', STATES, ['--phase', 'align', '--source', str(states),
            '--oral-audit', str(result_root / 'oral_relative_sensitivity_20261018_v1')])]
        for arm in ('POOL', 'SEARCH'):
            reference = result_root / f'cohort_belief_separate_s{subject}_oral_relative_20261019_v1'
            tasks.append((f'relative_{arm.lower()}_comparison', COMPARE, [
                '--original', str(reference), '--revised', str(relative), '--subject', str(subject), '--arm', arm]))
        records = [dict(stage='midpoint_pool_comparison', source=str(old_pool), manifest_sha256=digest(old_pool / 'manifest.json'))]
        write(folder / 'adopted.json', dict(status='complete', records=records,
            scientific_state_completion_sha256=digest(states / 'completion.json')))
        for tag, module, arguments in tasks:
            verify_sources(); destination = folder / tag
            print(f'S{subject}: starting {tag}', flush=True)
            with tempfile.TemporaryDirectory(prefix='cohort_oral_finish_') as cache:
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=cache, NUMBA_CACHE_DIR=cache)
                with (folder / f'{tag}.log').open('x') as log:
                    subprocess.run([sys.executable, '-m', module, *arguments, '--output', str(destination)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            verify_report(destination); verify_sources()
            records.append(dict(stage=tag, source=str(destination), manifest_sha256=digest(destination / 'manifest.json')))
            write(folder / f'{tag}_verified.json', records[-1])
            print(f'S{subject}: {tag} complete', flush=True)
        result = dict(status='complete', subject=subject, records=records)
        write(folder / 'completion.json', result)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(subject_run, plan['subjects']))
    verify_sources()
    write(output / 'completion.json', dict(status='complete', subjects=results, inputs_sha256=bindings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.predecessor, args.output)
