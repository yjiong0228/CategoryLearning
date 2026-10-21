"""Check REFERENCE against independently saved Model 1021 worker definitions.

Use --config YAML --output NEW. For each development condition, execute the
previous cohort worker and the new reference worker with identical parameters,
two particles, one saved seed and the first eight valid trials. Compare all
saved probabilities exactly. These runs validate software, not scientific fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import yaml

from . import model_1021_simplification as driver


def run(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new validation directory')
    settings = yaml.safe_load(config.read_text())
    output.mkdir(parents=True)
    records = []
    for subject in settings['subjects']:
        reference = driver.ROOT / settings['warm_start_fits'][subject].replace('_fit_', '_evaluation_')
        matching = [driver.io.read(p) for p in sorted((reference / 'jobs').glob('*.json'))]
        job = next(j for j in matching if j['model'] == 'SEARCH' and j['phase'] == 'evaluate')
        job.update(smoke=True, phase='training', particles=2, seeds=job['seeds'][:1])
        targets = []
        for kind in ('original', 'reference'):
            own = deepcopy(job)
            own.update(name=f'S{subject}_{kind}', target=str((output / f'S{subject}_{kind}').resolve()))
            if kind == 'reference':
                own.update(model='REFERENCE', source_sha256=driver.source_hashes(driver.io))
                script = driver.ROOT / driver.ENTRY
            else:
                script = driver.ROOT / 'src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_refit.py'
            file = output / (own['name'] + '_job.json')
            driver.io.write(file, own)
            launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                        "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
            with tempfile.TemporaryDirectory(prefix='compact_reference_check_') as cache:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache)
                with file.with_suffix('.log').open('x') as stream:
                    subprocess.run([sys.executable, '-c', launcher, own['runtime'], str(script),
                        '--worker', str(file)], env=env, cwd=driver.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        check=True)
            targets.append(Path(own['target']).with_suffix('.npz'))
        with np.load(targets[0]) as a, np.load(targets[1]) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        records.append(dict(subject=subject, trials=8, particles=2, seed=job['seeds'][0],
                            exact=True, artifacts={str(p): driver.io.digest(p) for p in targets}))
        print(f'S{subject}: exact reference reproduction', flush=True)
    driver.io.write(output / 'completion.json', dict(status='complete', software_validation=True,
        cases=records, config_sha256=driver.io.digest(config), source_sha256=driver.source_hashes(driver.io)))


def run_capture(config: Path, output: Path) -> None:
    """Check captured and original probabilities on sixteen training trials."""
    from . import model_1021_simplification_capture as capture
    from ...optimization.search.adaptive_proposals import candidate
    if output.exists():
        raise FileExistsError('Use a new capture validation directory')
    settings = yaml.safe_load(config.read_text())
    output.mkdir(parents=True)
    records = []
    for subject in settings['subjects']:
        source = driver.ROOT / settings['data'] / f'subject_{subject}'
        folder = output / f'subject_{subject}'
        folder.mkdir()
        meta = driver.io.read(source / 'prepared.json')
        engine = driver.io.read(source / 'engine.json')
        driver.io.write(folder / 'engine.json', engine)
        hashes = {}
        with np.load(source / 'training.npz') as data:
            for name, part in [('training.npz', slice(0, 8)), ('sealed_test.npz', slice(8, 16))]:
                with (folder / name).open('xb') as stream:
                    np.savez_compressed(stream, **{k: data[k][part] for k in data.files})
                hashes[name] = driver.io.digest(folder / name)
        meta.update(trials=16, training_trials=8, test_trials=8, array_sha256=hashes,
                    engine_sha256=driver.io.digest(folder / 'engine.json'),
                    software_validation='Sixteen original training trials only')
        driver.io.write(folder / 'prepared.json', meta)
        reference = driver.ROOT / settings['warm_start_fits'][subject].replace('_fit_', '_evaluation_')
        jobs = [driver.io.read(p) for p in sorted((reference / 'jobs').glob('*.json'))]
        template = next(j for j in jobs if j['model'] == 'SEARCH' and j['phase'] == 'evaluate')
        for arm in ('REFERENCE', 'COMPACT'):
            point = dict(template['candidate']['hyperparams'], **driver.fixed_coordinates(arm, meta['condition']))
            targets = []
            for kind in ('plain', 'capture'):
                name = f'S{subject}_{arm}_{kind}'
                own = deepcopy(template)
                own.update(model=arm, candidate=candidate(point, 'software_validation'), smoke=False,
                    particles=2, seeds=template['seeds'][:1], folder=str(folder.resolve()), name=name,
                    source_sha256=driver.source_hashes(driver.io), target=str((output / name).resolve()))
                own['mechanism']['capacity'] = 'evidence' if arm == 'COMPACT' else 'evidence_mean'
                script = driver.ENTRY
                if kind == 'capture':
                    script = capture.ENTRY
                    own.update(decision_basis_target=str((output / (name + '_states')).resolve()),
                               decision_basis_sources=capture.sources())
                file = output / (name + '_job.json')
                driver.io.write(file, own)
                launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                            "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
                with tempfile.TemporaryDirectory(prefix='compact_capture_check_') as cache:
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=cache)
                    with file.with_suffix('.log').open('x') as stream:
                        subprocess.run([sys.executable, '-c', launcher, own['runtime'], str(driver.ROOT / script),
                            '--worker', str(file)], env=env, cwd=driver.ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                targets.append(Path(own['target']).with_suffix('.npz'))
            with np.load(targets[0]) as a, np.load(targets[1]) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            records.append(dict(subject=subject, model=arm, trials=16, particles=2, exact=True,
                                artifacts={str(p): driver.io.digest(p) for p in targets}))
            print(f'S{subject} {arm}: capture preserves predictions exactly', flush=True)
    driver.io.write(output / 'completion.json', dict(status='complete', software_validation=True,
        cases=records, config_sha256=driver.io.digest(config), capture_source_sha256=capture.sources()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    (run_capture if args.capture else run)(args.config, args.output)
