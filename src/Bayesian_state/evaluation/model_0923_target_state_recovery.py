"""Summarize recorded target-rule recovery, preserving both inference timings.

Use --source COMPLETED_REPLAYS --output NEW_DIRECTORY. Equal-weight independent
filters approximate each trial's latent-state distribution. Five and ninety-five
percent empirical weighted quantiles describe that distribution, conditional on
one fixed parameter point. They exclude parameter uncertainty. Precision is
summarized conditional on target presence; absent-rule zeros are never treated as
learned precision. Error summaries are descriptive on the simulated sequence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows.analysis.model_0923_target_state_recovery import PHASES, digest, read, verify, write


def weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict:
    """Return trialwise means and inverse empirical CDF quantiles.

    Inputs are filter × trial × particle. Zero total mass produces NaNs, which
    mark an undefined conditional distribution and remain explicit in the CSV.
    """
    if (values.ndim != 3 or values.shape != weights.shape
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError('Require finite matching filter/trial/particle arrays')
    x = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    w = weights.transpose(1, 0, 2).reshape(weights.shape[1], -1)
    mass = w.sum(axis=1)
    mean = np.divide((x*w).sum(axis=1), mass, out=np.full(len(x), np.nan), where=mass > 0)
    quantiles = np.full((len(x), 3), np.nan)
    for t in np.flatnonzero(mass > 0):
        keep = w[t] > 0
        order = np.argsort(x[t, keep], kind='stable')
        sorted_values = x[t, keep][order]
        cdf = np.cumsum(w[t, keep][order])
        cdf /= cdf[-1]
        quantiles[t] = sorted_values[np.searchsorted(cdf, [.05, .5, .95], side='left')]
    return dict(mean=mean, q05=quantiles[:, 0], q50=quantiles[:, 1], q95=quantiles[:, 2])


def summaries(arrays: dict, truth: dict, phase: str) -> tuple[dict, dict]:
    """Score membership, raw contribution and precision as distinct quantities."""
    w = arrays[phase+'_weights']
    active = arrays[phase+'_present']
    u = arrays[phase+'_contribution']
    beta = arrays[phase+'_active_beta']
    if (active.shape != w.shape or u.shape != w.shape or beta.shape != w.shape
            or not np.isin(active, [0, 1]).all() or np.any(u < 0) or np.any(u > 1 + 1e-12)
            or np.any(beta < 0) or np.any(beta[~active.astype(bool)] != 0)):
        raise ValueError('Invalid recorded target state')
    np.testing.assert_allclose(w.sum(axis=2), 1, atol=2e-12, rtol=0)
    presence = weighted_summary(active, w)['mean']
    contribution = weighted_summary(u, w)
    precision = weighted_summary(beta, w*active)
    a = np.asarray(truth['present'], dtype=bool)
    if (a.shape != presence.shape or not np.isin(truth['present'], [0, 1]).all()
            or np.any(np.asarray(truth['beta'])[~a] != 0)):
        raise ValueError('Synthetic truth shape or inactive precision changed')
    beta_mask = a & np.isfinite(precision['mean'])
    curves = dict(presence_probability=presence,
        **{'contribution_'+k: v for k, v in contribution.items()},
        **{'conditional_beta_'+k: v for k, v in precision.items()})
    beta_rmse = float(np.sqrt(np.mean((precision['mean'][beta_mask]-truth['beta'][beta_mask])**2))) if beta_mask.any() else None
    metrics = dict(trials=len(a), truth_present_trials=int(a.sum()),
        membership_brier=float(np.mean((presence-a)**2)),
        contribution_rmse=float(np.sqrt(np.mean((contribution['mean']-truth['contribution'])**2))),
        contribution_interval_fraction=float(np.mean((truth['contribution'] >= contribution['q05']) &
                                                     (truth['contribution'] <= contribution['q95']))),
        contribution_interval_mean_width=float(np.mean(contribution['q95']-contribution['q05'])),
        conditional_beta_rmse=beta_rmse, conditional_beta_scored_trials=int(beta_mask.sum()),
        conditional_beta_undefined_when_truth_present=int((a & ~beta_mask).sum()))
    return curves, metrics


def summarize(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new target-state summary directory')
    done, protocol = read(source/'completion.json'), read(source/'protocol.json')
    if (done['status'] != 'complete' or done['protocol_sha256'] != digest(source/'protocol.json')
            or done['jobs'] != len(protocol['jobs']) or done['smoke'] != protocol['smoke']):
        raise ValueError('Require complete matching state replays')
    verify(protocol['bindings'])
    inputs = dict(protocol['bindings'])
    for path in (Path(__file__), source/'completion.json', source/'protocol.json'):
        inputs[str(path.resolve())] = digest(path)
    rows, scores, seed_scores = [], [], []
    labels = list(dict.fromkeys(j['label'] for j in protocol['jobs']))
    with np.load(source/'truth.npz') as truth_bank:
        for label in labels:
            jobs = [j for j in protocol['jobs'] if j['label'] == label]
            loaded, probabilities, seeds = [], [], []
            replicate = jobs[0]['replicate']
            truth = {k: truth_bank[f'r{replicate}_{k}'] for k in ('present', 'contribution', 'beta')}
            for job in jobs:
                file = Path(job['target']).with_suffix('.npz')
                record = read(file.with_suffix('.json'))
                if not record['exact_behavior_replay'] or record['array_sha256'] != digest(file):
                    raise ValueError('Replay record or arrays changed')
                for path in (file, file.with_suffix('.json'), Path(job['score_target']).with_suffix('.npz')):
                    inputs[str(path.resolve())] = digest(path)
                with np.load(file) as z:
                    loaded.append({k: z[k] for k in z.files if k != 'seeds'})
                    seeds.extend(z['seeds'].tolist())
                with np.load(job['reference']) as ref, np.load(Path(job['score_target']).with_suffix('.npz')) as actual:
                    np.testing.assert_array_equal(actual['seeds'], record['seeds'])
                    for key in ('observed_probability', 'correct_probability'):
                        np.testing.assert_array_equal(actual[key], ref[key][:, :job['trials']])
                    probabilities.append(actual['observed_probability'][0])
            if len(set(seeds)) != len(seeds):
                raise ValueError('Duplicate filter seeds')
            arrays = {k: np.stack([z[k] for z in loaded]) for k in loaded[0]}
            nll = float(-np.log(np.clip(np.mean(probabilities, axis=0), 1e-12, 1)).mean())
            for phase in PHASES:
                curves, metrics = summaries(arrays, truth, phase)
                scores.append(dict(label=label, replicate=replicate, phase=phase, filters=len(seeds),
                    particles=arrays[phase+'_weights'].shape[-1], choice_nll=nll, **metrics))
                for i, seed in enumerate(seeds):
                    _, per_filter = summaries({k: v[i:i+1] for k, v in arrays.items()}, truth, phase)
                    seed_scores.append(dict(label=label, replicate=replicate, phase=phase, seed=seed, **per_filter))
                for t in range(len(truth['present'])):
                    rows.append(dict(label=label, replicate=replicate, phase=phase, trial=t+1,
                        **{'truth_'+k: v[t] for k, v in truth.items()}, **{k: v[t] for k, v in curves.items()}))
    output.mkdir(parents=True)
    for name, records in (('trials', rows), ('scores', scores), ('filter_scores', seed_scores)):
        pd.DataFrame(records).to_csv(output/(name+'.csv'), index=False)
    verify(inputs)
    write(output/'completion.json', dict(status='complete', smoke=done['smoke'], source=str(source.resolve()),
        inputs_sha256=inputs, groups=labels, trials_rows=len(rows),
        artifacts_sha256={name:digest(output/name) for name in ('trials.csv','scores.csv','filter_scores.csv')},
        interval='5th to 95th empirical quantiles; equal mixture of normalized independent filters',
        precision='Conditional on target presence; exclude undefined conditional estimates explicitly',
        interpretation='Within-sequence descriptive errors; intervals omit parameter uncertainty; no oral or held-out validation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summarize(args.source, args.output)
