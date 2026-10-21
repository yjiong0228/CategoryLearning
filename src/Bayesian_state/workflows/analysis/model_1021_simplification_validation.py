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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
