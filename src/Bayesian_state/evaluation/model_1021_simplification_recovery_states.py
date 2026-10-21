"""Evaluate the target rule already captured in COMPACT recovery confirmations.

Use --prepared --fit --truth-score --output NEW. All estimates and generating-
point scores must be complete. Equal mixtures of original independent filters
reuse the shared membership, contribution and conditional-precision definitions.
The scope is one target rule in each synthetic sequence at fixed parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_target_state_recovery import summaries
from .model_0923_h_validation import nll
from ..workflows.analysis import model_1021_simplification_recovery as recovery

read, write, digest = recovery.read, recovery.write, recovery.digest
PHASES = ('predictive', 'filtered')


def summarize(prepared: Path, fit: Path, truth_score: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery-state summary directory')
    finished = read(fit/'completion.json')
    scored = read(truth_score/'completion.json')
    if (finished['status'] != 'complete' or scored['status'] != 'complete'
            or finished['smoke'] is not scored['smoke']
            or Path(scored['source_fit']).resolve() != fit.resolve()
            or Path(scored['prepared']).resolve() != prepared.resolve()
            or finished['nomination_sha256'] != digest(fit/'nominees.json')):
        raise ValueError('Require complete matching parameter recovery and truth scores')
    smoke = finished['smoke']
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Recovery-state input changed: '+str(path))
        inputs[str(path)] = actual
        return path

    for path in (prepared/'protocol.json', prepared/'completion.json', fit/'protocol.json',
                 fit/'completion.json', fit/'nominees.json', truth_score/'completion.json', Path(__file__),
                 Path(__file__).with_name('model_0923_target_state_recovery.py')):
        bind(path)
    prep_done = read(prepared/'completion.json')
    truths = {r['case_id']: r for r in read(bind(prepared/'truths.json', prep_done['truths_sha256']))['rows']}
    fitted = {r['case_id']: r for r in read(fit/'nominees.json')['nominees']}
    cases = recovery.checked_cases(prepared)
    cases = cases[:1] if smoke else cases
    curves, scores, filters = [], [], []
    for case in cases:
        case_id, subject = case['case_id'], case['subject']
        target_rule = case['settings']['recovery_target_rule']
        trials = min(8, case['trials']) if smoke else case['trials']
        count = 2 if smoke else case['settings']['budgets']['confirmation']['seeds']
        particles = 2 if smoke else case['settings']['budgets']['confirmation']['particles']
        actual = truths[case_id]
        with np.load(bind(actual['states_truth'], actual['states_truth_sha256'])) as saved:
            truth = {k: saved[k][:trials].copy() for k in ('present', 'contribution', 'beta')}
        roles = [('estimate', fit, fitted[case_id]['candidate']),
                 ('truth', truth_score, recovery.candidate(actual['point'], 'generating_point_after_freeze'))]
        role_seeds = []
        for role, directory, candidate in roles:
            loaded, probabilities, seeds = [], [], []
            selected = sorted((directory/case_id/'state_capture').glob(
                'confirmation_S'+str(subject)+'_COMPACT_'+candidate['id']+'_C*.json'))
            if len(selected) != count:
                raise ValueError('Missing original confirmation captures at this parameter point')
            for file in selected:
                record = read(bind(file))
                if (record['status'] != 'complete' or record['smoke'] is not smoke
                        or record['candidate']['hyperparams'] != candidate['hyperparams']
                        or record['target_rule'] != target_rule or record['trials'] != trials
                        or record['particles'] != particles or record['additional_filters'] != 0
                        or record['exact_original_filter_capture'] is not True or len(record['seeds']) != 1):
                    raise ValueError('Captured target state differs from this confirmation point')
                original = read(bind(record['original_job'], record['original_job_sha256']))
                for name, sha in record['capture_source_sha256'].items():
                    bind(recovery.ROOT/name, sha)
                if (original['seeds'] != record['seeds'] or original['target_rule'] != target_rule
                        or original['candidate']['hyperparams'] != candidate['hyperparams']
                        or Path(original['folder']).resolve() != Path(case['settings']['data']).resolve()/f'subject_{subject}'):
                    raise ValueError('Capture job uses a different synthetic case')
                with np.load(bind(file.with_suffix('.npz'), record['array_sha256'])) as saved:
                    np.testing.assert_array_equal(saved['seeds'], record['seeds'])
                    arrays = {k: saved[k].copy() for k in saved.files if k != 'seeds'}
                if set(arrays) != {p+'_'+k for p in PHASES for k in ('weights', 'present', 'contribution', 'active_beta')}:
                    raise ValueError('Incomplete target-state fields')
                if any(v.shape != (trials, particles) for v in arrays.values()):
                    raise ValueError('Original filter budget changed')
                with np.load(bind(record['score_file'], record['score_sha256'])) as saved:
                    np.testing.assert_array_equal(saved['seeds'], record['seeds'])
                    probabilities.append(saved['observed_probability'][0].copy())
                loaded.append(arrays)
                seeds.extend(record['seeds'])
            if len(set(seeds)) != count or case['generation_seed'] in seeds:
                raise ValueError('Confirmation filters duplicate or overlap the generating seed')
            role_seeds.append(seeds)
            bank = {k: np.stack([r[k] for r in loaded]) for k in loaded[0]}
            common = dict(case_id=case_id, subject=subject, replicate=case['replicate'], role=role,
                          target_rule=target_rule, candidate=candidate['id'], choice_nll=nll(np.asarray(probabilities)))
            for phase in PHASES:
                values, metrics = summaries(bank, truth, phase)
                scores.append(dict(common, phase=phase, filters=count, particles=particles, **metrics))
                for i, seed in enumerate(seeds):
                    _, own = summaries({k: v[i:i+1] for k, v in bank.items()}, truth, phase)
                    filters.append(dict(common, phase=phase, seed=seed, **own))
                for t in range(trials):
                    curves.append(dict(common, phase=phase, trial=t+1,
                        **{'truth_'+k: v[t] for k, v in truth.items()}, **{k: v[t] for k, v in values.items()}))
        np.testing.assert_array_equal(*role_seeds)
    output.mkdir(parents=True)
    for name, rows in [('trials', curves), ('scores', scores), ('filter_scores', filters)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'), index=False)
    write(output/'completion.json', dict(status='complete', smoke=smoke, cases=len(cases),
        additional_filters=0, inputs_sha256=inputs,
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')},
        scope='One recorded target rule per sequence; retrospective parameter estimates; limited generating points',
        interval='5th to 95th empirical weighted quantiles; fixed parameters, equal mixture of independent filters',
        precision='Conditional on target presence; undefined estimates retained explicitly'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'fit', 'truth-score', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.prepared, args.fit, args.truth_score, args.output)
