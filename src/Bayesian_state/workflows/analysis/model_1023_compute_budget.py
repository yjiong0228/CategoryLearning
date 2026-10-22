"""Share a strict numerical-process budget across independent study queues.

File locks give each running numerical job one slot. The running diagnostic
retains a conservative reservation that decreases with completed jobs while
covering later stage peaks. Separate fit controllers keep their original seeds, candidates,
particle budgets and scientific code. An append-only event log records every
lease. The wrapper changes scheduling only; each numerical subprocess still
uses one numeric thread and the existing Model 1022 worker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import uuid

from . import model_1023_estimation_reliability as reliability

read, write, digest, ROOT = reliability.read, reliability.write, reliability.digest, reliability.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def maybe_read(path: Path) -> dict | None:
    try:
        return read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def create_budget(folder: Path, capacity: int, legacy: Path | None = None) -> None:
    if not 1 <= capacity <= 128:
        raise ValueError('Numerical capacity must be between one and 128')
    folder.mkdir(parents=True)
    if legacy is not None:
        cases = read(legacy/'cases.json')['cases']
        if read(legacy/'protocol.json')['phase'] != 'diagnostic' or len(cases) != 2:
            raise ValueError('Legacy reservation requires the two registered diagnostic cases')
        legacy = legacy.resolve()
    write(folder/'protocol.json', dict(capacity=capacity, legacy=str(legacy) if legacy else None,
        source_sha256={ENTRY: digest(ROOT/ENTRY)}, slots='One advisory file lock per numerical process'))
    for slot in range(capacity):
        (folder/f'slot_{slot:03d}.lock').touch(exist_ok=False)


class SharedBudget:
    def __init__(self, folder: Path, phase: str):
        self.folder, self.phase = folder, phase
        settings = read(folder/'protocol.json')
        self.capacity = settings['capacity']
        self.legacy = Path(settings['legacy']) if settings['legacy'] else None
        self.cases = read(self.legacy/'cases.json')['cases'] if self.legacy else []
        if self.legacy:
            from ...evaluation.model_1023_estimation_reliability import expected_stages
            self.fit_jobs = sum(count*seeds for c in self.cases for count, _, seeds in
                                expected_stages(c['settings'], False).values())
        self._completed, self._checked = 0, 0.0

    def completed_legacy_jobs(self) -> int:
        # Lagged completion counts only reserve too much capacity. Reserve the
        # next stage's full peak as well, because it can start between snapshots.
        now = time.monotonic()
        if now-self._checked >= 1:
            snapshots = sorted((self.legacy/'resource_snapshots').glob('*.json'))
            latest = maybe_read(snapshots[-1]) if snapshots else None
            if latest is not None:
                self._completed = max(self._completed, latest['completed'])
            self._checked = now
        return self._completed

    def reservation(self) -> int:
        if self.legacy is None:
            return 0
        finished_queue = maybe_read(self.legacy/'resource_status.json')
        if finished_queue is not None and finished_queue['active'] == 0:
            return 0
        if maybe_read(self.legacy/'completion.json') is not None:
            return 0
        audit_jobs = sum(c['settings']['audit_candidates']*c['settings']['budgets']['confirmation']['seeds'] for c in self.cases)
        truth_jobs = sum(c['settings']['budgets']['confirmation']['seeds'] for c in self.cases)
        completed = self.completed_legacy_jobs()
        if maybe_read(self.legacy/'audit_fit/completion.json') is not None:
            return min(truth_jobs, max(0, self.fit_jobs+audit_jobs+truth_jobs-completed))
        if maybe_read(self.legacy/'selected_nominees.json') is not None:
            return min(audit_jobs, max(truth_jobs, self.fit_jobs+audit_jobs-completed))
        if all(maybe_read(Path(c['output'])/'selection_completion.json') is not None for c in self.cases):
            confirmation_jobs = sum(c['settings']['confirmation_candidates']*c['settings']['budgets']['confirmation']['seeds'] for c in self.cases)
            return min(confirmation_jobs, max(audit_jobs, self.fit_jobs-completed))
        return self.capacity

    def event(self, row: dict) -> None:
        with (self.folder/'events.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if row['event'] == 'enter':
                row = dict(row, reservation=self.reservation())
            with (self.folder/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(row, phase=self.phase, unix_time=time.time()))+'\n')

    @contextmanager
    def lease(self, name: str):
        handle = None
        while handle is None:
            reserved = self.reservation()
            candidates = list(range(reserved, self.capacity))
            if candidates:
                offset = threading.get_native_id() % len(candidates)
                candidates = candidates[offset:]+candidates[:offset]
            for slot in candidates:
                own = (self.folder/f'slot_{slot:03d}.lock').open('r+')
                try:
                    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    own.close()
                    continue
                handle = own
                break
            if handle is None:
                time.sleep(.25)
        identifier = uuid.uuid4().hex
        self.event(dict(event='enter', job=identifier, name=name, slot=slot, reservation=reserved, pid=os.getpid()))
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.event(dict(event='exit', job=identifier, name=name, slot=slot, pid=os.getpid(), failed=failed))
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def install_budget(folder: Path, phase: str) -> None:
    budget = SharedBudget(folder, phase)
    original = reliability.NumericQueue.submit

    def submit(self, function, *args, **kwargs):
        name = args[0].get('name', function.__name__) if args and isinstance(args[0], dict) else function.__name__

        def guarded():
            with budget.lease(name):
                return function(*args, **kwargs)

        return original(self, guarded)

    reliability.NumericQueue.submit = submit


def summarize_budget(folder: Path) -> dict:
    settings = read(folder/'protocol.json')
    active, counts, failures, peak, combined = {}, {}, {}, 0, 0
    for line in (folder/'events.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['event'] == 'enter':
            if row['job'] in active or row['slot'] in {a['slot'] for a in active.values()}:
                raise ValueError('Duplicated numerical lease')
            active[row['job']] = row
            peak = max(peak, len(active))
            combined = max(combined, len(active)+row['reservation'])
            if len(active)+row['reservation'] > settings['capacity']:
                raise ValueError('Global numerical capacity exceeded')
        else:
            original = active.pop(row['job'])
            if original['slot'] != row['slot']:
                raise ValueError('Numerical lease changed slots')
            own = failures if row['failed'] else counts
            own[row['phase']] = own.get(row['phase'], 0)+1
    if active:
        raise ValueError('Some numerical leases are still running')
    return dict(status='complete', maximum=settings['capacity'], peak_new_processes=peak,
                peak_including_legacy_reservation=combined, completed_by_phase=counts,
                failed_by_phase=failures,
                events_sha256=digest(folder/'events.jsonl'))


def run(kind: str, phase: str, config: Path | None, study: Path | None, output: Path,
        budget: Path, smoke: bool) -> None:
    install_budget(budget, phase if kind == 'fit' else 'human_evaluation')
    if kind == 'fit':
        original_sources = reliability.refit.sources
        reliability.refit.sources = lambda v: dict(original_sources(v), **{ENTRY: digest(ROOT/ENTRY)})
        reliability.run(config, phase, output, smoke, False)
    else:
        from .model_1023_frozen_evaluate import run as evaluate
        evaluate(study, output, smoke, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('fit', 'evaluate'), required=True)
    parser.add_argument('--phase', choices=('diagnostic', 'validation', 'human'), default='human')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.kind == 'fit' and args.config is None or args.kind == 'evaluate' and args.study is None:
        parser.error('Fits need --config; evaluations need --study')
    run(args.kind, args.phase, args.config, args.study, args.output, args.budget, args.smoke)
