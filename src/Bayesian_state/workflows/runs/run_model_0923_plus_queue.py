"""Freeze the model, fit subjects in order, and automatically finish evaluation.

The default queue continues completed fits once at artificial support edges,
independently audits the frozen candidates, then generates individual reports.
Use --fit-only for the explicit fitting-only workflow. Existing run snapshots
retain their original behavior and can feed the separate continuation entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

from src.Bayesian_state.optimization.adaptive_runtime import freeze_json, verify_manifest
from src.Bayesian_state.optimization.model_0923_adaptive import load_config, make_contexts
from src.Bayesian_state.utils.paths import ROOT_DIR

DEFAULT_CONFIGS = [ROOT_DIR / 'configs/meg/model0923_adaptive_335.yaml',
                   ROOT_DIR / 'configs/exp123/specific_models/model0923_adaptive_307.yaml']


def prepare(root: Path, configs: list[Path]) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    (source / 'src').mkdir(parents=True)
    shutil.copy2(ROOT_DIR / 'src/__init__.py', source / 'src/__init__.py')
    shutil.copytree(ROOT_DIR / 'src/Bayesian_state', source / 'src/Bayesian_state',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.copytree(ROOT_DIR / 'configs', source / 'configs',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    jobs, copied = [], set()
    for path in configs:
        path = path.resolve()
        config = load_config(path)
        specs, contexts, _ = make_contexts(config, None, False)
        if len(specs) != 1:
            raise ValueError('Each sequential job must declare exactly one subject')
        sid = specs[0]['subject']
        for data in contexts[sid]['dataset_paths'].values():
            data = Path(data)
            if data.is_file() and data not in copied:
                target = source / data.relative_to(ROOT_DIR)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data, target)
                copied.add(data)
        relative = path.relative_to(ROOT_DIR)
        # Historical candidate banks remain read-only external inputs, with
        # absolute paths confined to this generated runtime snapshot.
        frozen_config = yaml.safe_load(path.read_text())
        frozen_config['warm_start']['fit_runs'] = config['warm_start']['fit_runs']
        (source / relative).write_text(yaml.safe_dump(frozen_config, sort_keys=False))
        jobs.append(dict(subject=sid, config=str(relative), output=str(root / f'subject_{sid}'),
                         valid_trials=len(contexts[sid]['arrays'].choices)))
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source.rglob('*') if p.is_file()}
    plan = dict(source=str(source), jobs=jobs,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT_DIR).strip(),
                prepared_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                snapshot_sha256=files,
                scope='Ordered Plus protocol runs; completed and unresolved fit states are both retained')
    freeze_json(root / 'queue_plan.json', plan)
    for path in source.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    return plan


def run_prepared(root: Path) -> dict:
    root = root.resolve()
    plan = json.loads((root / 'queue_plan.json').read_text())
    source = Path(plan['source'])
    for name, digest in plan['snapshot_sha256'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen source changed: {name}')
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       MPLCONFIGDIR=str(root / 'mpl_cache'), NUMBA_CACHE_DIR=str(root / 'numba_cache'))
    state = dict(status='running', queue_pid=os.getpid(), jobs=[])

    def publish_state() -> None:
        state['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        temporary = root / '.queue_status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(root / 'queue_status.json')

    with (root / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in plan['jobs']:
            output = Path(job['output'])
            row = dict(subject=job['subject'], output=str(output), status='running',
                       started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            state['jobs'].append(row)
            state['active_subject'] = job['subject']
            publish_state()
            if (output / 'fit_results.json').exists():
                verify_manifest(json.loads((output / 'manifest.json').read_text()))
                for receipt in (output / 'batches').rglob('scores.json'):
                    for name, digest in json.loads(receipt.read_text())['cache_sha256'].items():
                        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                            raise ValueError(f'Completed probability cache changed: {name}')
                result = json.loads((output / 'fit_results.json').read_text())['subjects'][str(job['subject'])]
                row.update(status='completed_existing', fit_status=result['status'], issues=result['issues'],
                           selected_mean_nll=result['selected_mean_nll'])
                publish_state()
                continue
            command = [sys.executable, '-m', 'src.Bayesian_state.run_model_0923_fit',
                       '--config', str(source / job['config']), '--output-dir', str(output)]
            if (output / 'manifest.json').exists():
                command.append('--resume')
            with (root / f'subject_{job["subject"]}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
                row['fit_pid'] = child.pid
                publish_state()
                returncode = child.wait()
            row.update(returncode=returncode, ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
            if returncode:
                row['status'] = 'failed'
                state['status'] = 'failed'
                publish_state()
                raise RuntimeError(f'Subject {job["subject"]} failed; later jobs remain pending')
            result = json.loads((output / 'fit_results.json').read_text())['subjects'][str(job['subject'])]
            row.update(status='completed', fit_status=result['status'], issues=result['issues'],
                       selected_mean_nll=result['selected_mean_nll'])
            publish_state()
        state['status'] = 'completed'
        state.pop('active_subject', None)
        publish_state()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--continuation-policy', type=Path,
                        default=ROOT_DIR / 'configs/shared/model0923_continuation.yaml')
    args = parser.parse_args()
    if args.prepare_only and args.run_prepared:
        parser.error('Choose prepare-only or run-prepared')
    if not args.run_prepared:
        plan = prepare(args.output_dir, args.configs)
        if args.prepare_only:
            print(json.dumps(dict(source=plan['source'], jobs=plan['jobs']), ensure_ascii=False, indent=2))
            return
    fitted = run_prepared(args.output_dir)
    if not args.fit_only:
        from .run_model_0923_complete_pipeline import prepare as prepare_complete, run_prepared as run_complete
        complete = args.output_dir.resolve() / 'complete_pipeline'
        if not complete.exists():
            prepare_complete(args.output_dir, complete, args.continuation_policy)
        fitted = dict(fit_queue=fitted, complete_pipeline=run_complete(complete))
    print(json.dumps(fitted, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
