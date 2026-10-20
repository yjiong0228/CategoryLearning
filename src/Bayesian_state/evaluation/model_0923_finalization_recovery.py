"""Summarize the single recommended arm's bounded ONLINE joint recovery.

Use --prepared --subject --replicate --fit --truth-score --output NEW [--smoke].
All confirmation alternatives are retained. Paired whole-filter Monte Carlo
errors diagnose whether the fitted point missed the known generating point;
they do not quantify population or parameter uncertainty. Parameter errors and
exact closed channels keep their specific task and generating-point scope.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_0923_cohort_belief_refit as execution
from .model_0923_finalization_parameter_support import point_channels
from .model_0923_joint_parameter_recovery import paired_scores, parameter_errors
from ..workflows.analysis import model_0923_finalization_recovery as driver

read, write, digest = driver.read, driver.write, driver.digest


def summarize(prepared: Path, subject: int, replicate: int, fit: Path,
              truth_score: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new ONLINE recovery summary directory')
    settings, inputs, _, case = driver.case_settings(prepared, subject, replicate, smoke)

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if ((expected is not None and sha != expected)
                or (str(path) in inputs and inputs[str(path)] != sha)):
            raise ValueError(f'Frozen recovery input changed: {path}')
        inputs[str(path)] = sha
        return path

    def load(path):
        return read(bind(path))

    bind(__file__); bind(driver.__file__)
    frozen, done, protocol, declaration = [load(fit / name) for name in
        ('nominees.json', 'completion.json', 'protocol.json', 'recovery_protocol.json')]
    scored, scoring = [load(truth_score / name) for name in ('completion.json', 'protocol.json')]
    preparation = load(prepared / 'completion.json')
    if (any(record['status'] != 'complete' for record in (done, scored, preparation))
            or any(record['smoke'] is not smoke for record in (done, frozen, protocol, scored, scoring, preparation))
            or frozen['status'] != 'frozen_before_truth_scoring'
            or any(record['settings'] != settings for record in (frozen, protocol, scoring))
            or done['test_choices_used'] or done['oral_used']
            or done['nomination_sha256'] != digest(fit / 'nominees.json')
            or scored['fit_nomination_sha256'] != digest(fit / 'nominees.json')
            or scored['truths_sha256'] != preparation['truths_sha256']
            or declaration['generating_parameters_used_for_initialization']
            or declaration['human_warm_starts_used']
            or Path(scoring['fit']).resolve() != fit.resolve()):
        raise ValueError('Require matching frozen estimates and after-estimation truth scores')
    for record in (preparation, scored):
        for path, sha in record['inputs_sha256'].items():
            bind(path, sha)
    for path, sha in frozen['source_sha256'].items():
        bind(driver.ROOT / path, sha)
    bind(prepared / 'truths.json', preparation['truths_sha256'])
    truth, = [r for r in load(prepared / 'truths.json')['rows']
              if (r['subject'], r['replicate']) == (subject, replicate)]
    arm, = settings['arms']
    nominee, = frozen['nominees']
    if (nominee['subject'], nominee['model'], truth['model']) != (subject, arm, arm):
        raise ValueError('Recovery changed the generating proposal arm')
    space = load(fit / 'parameter_spaces.json')
    expected = load(prepared / 'protocol.json')['spaces'][str(subject)]
    if set(space) != {arm} or space[arm] != expected:
        raise ValueError('Fitting support differs from the registered generating environment')
    jobs = []
    for directory in (fit, truth_score):
        for file in sorted((directory / 'jobs').glob('*.json')):
            driver.check_job(load(file))
            jobs.append(execution.audit_job(file, 'ONLINE', bind, smoke=smoke))
    stages = {}
    for directory, label in [(fit, 'estimate'), (truth_score, 'truth')]:
        record = load(directory / 'confirmation_completion.json')
        group, = record['groups']
        if record['status'] != 'complete' or (group['subject'], group['model']) != (subject, arm):
            raise ValueError('Confirmation bank changed its participant or arm')
        stages[label] = group['rows']
    if (stages['estimate'] != nominee['finalists'] or stages['estimate'][0] != nominee['candidate']
            or len(stages['truth']) != 1
            or stages['truth'][0]['hyperparams'] != truth['candidate']['hyperparams']):
        raise ValueError('Confirmation bank differs from its frozen fitted or generating point')

    def arrays(point):
        file = Path(point['score_file'])
        record = load(file)
        path = bind(file.with_suffix('.npz'), record['predictions_sha256'])
        with np.load(path) as saved:
            bank = {k: saved[k].copy() for k in saved.files}
        if (record['candidate']['hyperparams'] != point['hyperparams']
                or (record['subject'], record['model']) != (subject, arm)
                or record['particles'] != (2 if smoke else settings['budgets']['confirmation']['particles'])
                or len(bank['seeds']) != (2 if smoke else settings['budgets']['confirmation']['seeds'])
                or bank['observed_probability'].shape[1] != (8 if smoke else case['training_trials'])
                or set(bank['seeds']) & set(settings['recovery_generation_seeds'])):
            raise ValueError('Confirmation numerical budget or source identity changed')
        np.testing.assert_array_equal(bank['seeds'], record['seeds'])
        own = -np.log(bank['observed_probability'].mean(0)).mean()
        np.testing.assert_allclose([record['mean_nll'], point['mean_nll']], own, rtol=0, atol=1e-12)
        return bank

    true_arrays = arrays(stages['truth'][0])
    base = dict(subject=subject, replicate=replicate, model=arm,
                candidate=settings['recovery_candidate'], generating_seed=truth['generation_seed'])
    scores = [dict(base, point=row['id'], selected=row['id'] == nominee['candidate']['id'],
                   **paired_scores(arrays(row), true_arrays)) for row in stages['estimate']]
    errors = parameter_errors(truth['candidate']['hyperparams'], nominee['candidate']['hyperparams'], space[arm])
    closed = {}
    for role, point in [('truth', truth['candidate']), ('estimate', nominee['candidate'])]:
        own = deepcopy(frozen)
        own['nominees'] = [dict(subject=subject, model=arm, candidate=point)]
        closed[role] = point_channels(own, subject, arm)
    parameters = []
    for row in errors:
        key = row['parameter']
        parameters.append(dict(base, **row, configured_fixed=key in settings['fixed_coordinates'],
            truth_closed=key in closed['truth'], truth_closure=closed['truth'].get(key, ''),
            estimate_closed=key in closed['estimate'], estimate_closure=closed['estimate'].get(key, '')))
    output.mkdir(parents=True)
    pd.DataFrame(parameters).to_csv(output / 'parameter_recovery.csv', index=False)
    pd.DataFrame(scores).to_csv(output / 'confirmation_vs_truth.csv', index=False)
    pd.DataFrame(jobs).to_csv(output / 'executed_jobs.csv', index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Recovery input changed during summary')
    write(output / 'completion.json', dict(status='complete', subject=subject, replicate=replicate,
        smoke=smoke, inputs_sha256=inputs, parameters=14, fitted_variable_coordinates=13,
        candidate=settings['recovery_candidate'], belief_mechanism='ONLINE', precision_storage='RETAIN',
        interpretation='One of at most four synthetic sequences; no population identification claim',
        optimization='Truth scored only after estimates freeze; a better truth score retains search-coverage limitations',
        uncertainty='Paired whole-filter Monte Carlo error, conditional on one synthetic sequence',
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'fit', 'truth-score', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--replicate', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.prepared, args.subject, args.replicate, args.fit, args.truth_score, args.output, args.smoke)
