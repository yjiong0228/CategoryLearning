"""Worker-isolated fixed-cardinality H ablation on archived participant fits.

Reuse the existing intervention runner, proposal/evidence/belief operations and
complete sparse path recorder. Only named J/search variants use the new shared
single-proposal sampler. The frozen regular H source is never modified, allowing
independent training jobs to retain their exact scientific version.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[4]
HELPER = 'src/Bayesian_state/model/modules/hypothesis_transition/single_proposal_competition.py'
FROZEN_SOURCES: dict[str, str] = {}


def workflow():
    path = REPO / 'src/Bayesian_state/workflows/analysis/model_0923_mechanism_interventions.py'
    spec = importlib.util.spec_from_file_location('competition_interventions', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    import numpy as np
    w = workflow(); c = w.cohort(); job = c.load_json(file); d = c.load_diagnosis()
    for name, sha in job['competition_source_sha256'].items():
        if c.digest(REPO / name) != sha:
            raise ValueError('Capacity-competition source changed')
    fixed = job.get('variant') in job['settings']['fixed_size_variants']
    if fixed:
        intervention = job['settings']['variants'][job['variant']]
        if intervention['kind'] != 'J' or intervention['refresh'] != 'search':
            raise ValueError('Fixed-cardinality intervention requires proposal-triggered J selection')
        original_install = w.install_transition
        def install():
            original_install()
            helper = w.load_source('src.Bayesian_state.model.modules.hypothesis_transition.single_proposal_competition', HELPER)
            helper.install_worker_intervention(sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'])
        w.install_transition = install
    with d.single_threaded_processes():
        (w.prepare if job['phase'] == 'prepare' else w.trace)(job, c, d)
    if job['phase'] == 'trace':
        directory = Path(job['directory']) / job['variant'] / 'traces'
        # The existing runner names each chunk by its first path index.
        record = directory / f'C_{job["indices"][0]:03d}.json'
        # Every process checks its own seed slice, avoiding concurrent chunks.
        prepared = c.load_json(Path(job['directory']) / 'prepared.json')
        seeds = [prepared['seeds'][i] for i in job['indices']]
        with np.load(record.with_suffix('.npz')) as saved:
            if saved['seeds'].tolist() != seeds:
                raise ValueError('Completed trace seed slice differs')
            counts = (saved['rule'] >= 0).sum(axis=-1)
            if fixed and np.any(np.diff(counts, axis=0) < 0):
                raise ValueError('Fixed-cardinality pool unexpectedly shrank')
        audit = directory.parent / 'competition_checks'; audit.mkdir(exist_ok=True)
        c.write_json(audit / record.name, dict(
            sampler='single_proposal_fixed_cardinality' if fixed else 'original',
            fixed_pool_size_invariant_passed=bool(fixed), seeds=seeds,
            source_sha256=job['competition_source_sha256']))


def launch(job: dict, output: Path, c) -> None:
    job = dict(job, competition_source_sha256=FROZEN_SOURCES)
    file = output / 'jobs' / f"{job['record']['id']}_{job['phase']}_{job.get('variant','')}_{job.get('indices',[0])[0]:03d}.json"
    c.write_json(file, job)
    launcher = 'import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='h_competition_') as cache, file.with_suffix('.log').open('x') as log:
        env['NUMBA_CACHE_DIR'] = cache
        subprocess.run([sys.executable, '-c', launcher, str(Path(job['record']['runtime']).resolve()), str(Path(__file__).resolve()),
            '--worker', str(file.resolve())], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--subjects',type=int,nargs='+'); parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker); return
    if args.config is None or args.output is None:
        parser.error('config and output are required')
    w = workflow(); c = w.cohort(); settings = yaml.safe_load(args.config.read_text())
    if not settings['fixed_size_variants'] or not set(settings['fixed_size_variants']) <= set(settings['variants']):
        raise ValueError('Need explicitly named fixed-size variants')
    for name in (HELPER, str(Path(__file__).resolve().relative_to(REPO))):
        FROZEN_SOURCES[name] = c.digest(REPO / name)
    w.launch = launch
    only = args.subjects or ([settings['subjects'][0]] if args.smoke else None)
    w.run(args.config, args.output, args.smoke, only)
    c.write_json(args.output / 'competition_manifest.json', dict(status='complete', fixed_size_variants=settings['fixed_size_variants'],
        source_sha256=FROZEN_SOURCES, implementation='process-local sampler intervention; standard H source unchanged',
        cognitive_difference='J/search samples exactly min(capacity,pool size); other operations inherited'))


if __name__ == '__main__':
    main()
