"""Evaluate ONLINE target states captured in the original confirmation filters.

Use --prepared --subject --replicate --fit --truth-score --parameter-summary
--output NEW [--smoke]. This reads captured states and the frozen synthetic
truth without executing filters. The selected and generating parameter points
use the same original confirmation seeds. Summaries and empirical intervals
reuse the shared target-state evaluation definitions. They describe one rule
within one synthetic sequence and exclude parameter uncertainty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_target_state_recovery import summaries
from ..workflows.analysis import model_0923_finalization_recovery as driver

read, write, digest = driver.read, driver.write, driver.digest
PHASES = ('predictive', 'filtered')


def summarize(prepared: Path, subject: int, replicate: int, fit: Path,
              truth_score: Path, parameter_summary: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new captured-state summary directory')
    settings, inputs, _, case = driver.case_settings(prepared, subject, replicate, smoke)

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if ((expected is not None and sha != expected)
                or (str(path) in inputs and inputs[str(path)] != sha)):
            raise ValueError(f'Captured-state input changed: {path}')
        inputs[str(path)] = sha
        return path

    def load(path):
        return read(bind(path))

    bind(__file__)
    parameter_done = load(parameter_summary / 'completion.json')
    if (parameter_done['status'] != 'complete' or parameter_done['smoke'] is not smoke
            or (parameter_done['subject'], parameter_done['replicate']) != (subject, replicate)
            or parameter_done['candidate'] != settings['recovery_candidate']
            or parameter_done['belief_mechanism'] != 'ONLINE'):
        raise ValueError('Require this case\'s completed ONLINE parameter recovery summary')
    for path, sha in parameter_done['inputs_sha256'].items():
        bind(path, sha)
    for directory in (fit, truth_score):
        if str((directory / 'completion.json').resolve()) not in parameter_done['inputs_sha256']:
            raise ValueError('Parameter summary uses different fit or truth scores')
    table = pd.read_csv(bind(parameter_summary / 'confirmation_vs_truth.csv',
                            parameter_done['artifacts_sha256']['confirmation_vs_truth.csv']))
    selected, = table[table.selected].to_dict('records')
    frozen = load(fit / 'nominees.json')
    fitted, = frozen['nominees']
    actual, = [row for row in load(prepared / 'truths.json')['rows']
               if (row['subject'], row['replicate']) == (subject, replicate)]
    trials = 8 if smoke else case['training_trials']
    count = 2 if smoke else settings['budgets']['confirmation']['seeds']
    particles = 2 if smoke else settings['budgets']['confirmation']['particles']
    with np.load(bind(case['states_truth'], case['states_truth_sha256'])) as saved:
        truth = {key: saved[key][:trials].copy() for key in ('present', 'contribution', 'beta')}
    scores, curves, individual, roles_seeds = [], [], [], []
    for role, directory, point in [('truth', truth_score, actual['candidate']),
                                    ('estimate', fit, fitted['candidate'])]:
        declaration = load(directory / 'state_capture_protocol.json')
        if declaration['target_rule'] != case['target_rule'] or declaration['additional_filters'] != 0:
            raise ValueError('Capture declaration differs from the registered target rule')
        for path, sha in declaration['inputs_sha256'].items():
            bind(path, sha)
        jobs = [load(path) for path in sorted((directory / 'jobs').glob('confirmation_*.json'))]
        chosen = [job for job in jobs if job['candidate']['id'] == point['id']]
        if len(chosen) != count:
            raise ValueError('Preserve all original confirmation filters at this point')
        loaded, probabilities, seeds = [], [], []
        for job in chosen:
            driver.check_job(job)
            target = Path(job['state_capture_target'])
            record = load(target.with_suffix('.json'))
            if (record['status'] != 'complete' or record['target_rule'] != case['target_rule']
                    or record['trials'] != trials or record['particles'] != particles
                    or record['candidate']['hyperparams'] != point['hyperparams']
                    or record['seeds'] != job['seeds'] or len(job['seeds']) != 1
                    or record['belief_mechanism'] != 'ONLINE' or record['precision_storage'] != 'RETAIN'
                    or record['additional_filters'] != 0
                    or Path(record['score_file']) != Path(job['target']).with_suffix('.npz')):
                raise ValueError('Captured state and original confirmation identity differ')
            bind(record['original_job'], record['original_job_sha256'])
            with np.load(bind(target.with_suffix('.npz'), record['array_sha256'])) as saved:
                np.testing.assert_array_equal(saved['seeds'], job['seeds'])
                arrays = {key: saved[key].copy() for key in saved.files if key != 'seeds'}
            if any(value.shape != (trials, particles) for value in arrays.values()):
                raise ValueError('Captured state budget differs from its original filter')
            with np.load(bind(record['score_file'], record['score_sha256'])) as saved:
                np.testing.assert_array_equal(saved['seeds'], job['seeds'])
                probabilities.append(saved['observed_probability'][0].copy())
            loaded.append(arrays)
            seeds.extend(job['seeds'])
        if len(set(seeds)) != count or set(seeds) & set(settings['recovery_generation_seeds']):
            raise ValueError('Capture uses duplicate or generation-overlapping filter seeds')
        roles_seeds.append(seeds)
        bank = {key: np.stack([row[key] for row in loaded]) for key in loaded[0]}
        nll = float(-np.log(np.clip(np.mean(probabilities, axis=0), 1e-12, 1)).mean())
        np.testing.assert_allclose(nll, selected[role + '_nll'], rtol=0, atol=1e-12)
        common = dict(subject=subject, replicate=replicate, role=role, candidate=point['id'],
                      generating_seed=case['generation_seed'], target_rule=case['target_rule'])
        for phase in PHASES:
            values, metrics = summaries(bank, truth, phase)
            scores.append(dict(common, phase=phase, filters=count, particles=particles, choice_nll=nll, **metrics))
            for i, seed in enumerate(seeds):
                _, own = summaries({key: value[i:i+1] for key, value in bank.items()}, truth, phase)
                individual.append(dict(common, phase=phase, seed=seed, **own))
            for t in range(trials):
                curves.append(dict(common, phase=phase, trial=t+1,
                    **{'truth_'+key: value[t] for key, value in truth.items()},
                    **{key: value[t] for key, value in values.items()}))
    np.testing.assert_array_equal(*roles_seeds)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Captured recovery input changed during summary')
    output.mkdir(parents=True)
    for name, rows in [('trials', curves), ('scores', scores), ('filter_scores', individual)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)
    write(output / 'completion.json', dict(status='complete', subject=subject, replicate=replicate, smoke=smoke,
        belief_mechanism='ONLINE', candidate=settings['recovery_candidate'], target_rule=case['target_rule'],
        original_filters_read=2*count, additional_filters=0, inputs_sha256=inputs,
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')},
        timing='Predictive and choice-filtered; both before current feedback',
        interval='5th to 95th empirical quantiles of equal-weight independent filters; parameter uncertainty excluded',
        precision='Conditional on target presence; undefined cases explicitly retained',
        scope='One target rule and one synthetic sequence; retrospective parameter estimates'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'fit', 'truth-score', 'parameter-summary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--replicate', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.prepared, args.subject, args.replicate, args.fit, args.truth_score,
              args.parameter_summary, args.output, args.smoke)
