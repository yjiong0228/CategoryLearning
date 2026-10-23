"""Describe early and late training behavior for every frozen candidate.

Run --generation COMPLETED_BALANCED_GENERATION --output NEW. Reuse all saved
training paths and confirmation filters, without fitting or generating new
paths. Whole training, halves and quarters are descriptive windows; their
boundaries are fractions of the original training length. This exploratory
diagnostic does not select parameters and never opens sealed_test.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_1022_belief_balanced_selection as selection
from .model_1022_belief_summary import nll, jackknife_se
from ...metrics.autonomous_fit import behavior_features, energy_u

ROOT, read, write, digest = selection.ROOT, selection.read, selection.write, selection.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def training_windows(length: int) -> list[tuple[str, int, int]]:
    """Return fixed descriptive windows, all inside the original training data."""
    if length < 4:
        raise ValueError('Need at least four training trials')
    result = [('all', 0, length)]
    for divisions, prefix in ((2, 'half'), (4, 'quarter')):
        result.extend((f'{prefix}_{i+1}', i*length//divisions, (i+1)*length//divisions)
                      for i in range(divisions))
    return result


def describe(paths: np.ndarray, categories: np.ndarray, choices: np.ndarray,
             probabilities: np.ndarray, features: dict, n_categories: int) -> list[dict]:
    """Retain path/filter repeats when measuring learning within each window."""
    length = len(categories)
    if (choices.shape != (length,) or paths.ndim != 2 or paths.shape[1] != length
            or probabilities.ndim != 2 or probabilities.shape[1] != length
            or len(probabilities) < 2 or not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0) or np.any(probabilities > 1)):
        raise ValueError('Training choices, paths and probabilities must align')
    rows = []
    for name, start, stop in training_windows(length):
        kwargs = dict(n_categories=n_categories, **features)
        empirical = behavior_features(choices[None, start:stop], categories[start:stop], **kwargs)[0]
        simulated = behavior_features(paths[:, start:stop], categories[start:stop], **kwargs)
        energy = energy_u(simulated, empirical)
        p = probabilities[:, start:stop]
        deleted = np.array([nll(np.delete(p, i, axis=0)) for i in range(len(p))])
        accuracy = (paths[:, start:stop] == categories[None, start:stop]).mean(axis=1)
        rows.append(dict(window=name, start=start, stop=stop, trials=stop-start,
                         observed_accuracy=float((choices[start:stop] == categories[start:stop]).mean()),
                         generated_accuracy=float(accuracy.mean()),
                         generated_accuracy_mcse=float(accuracy.std(ddof=1)/np.sqrt(len(accuracy))),
                         nll=nll(p), nll_mcse=jackknife_se(deleted),
                         energy=energy['value'], energy_mcse=energy['mc_se'],
                         paths=len(paths), filters=len(p)))
    return rows


def run(generated: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new training-window diagnostic directory')
    protocol = read(generated / 'protocol.json')
    policy = protocol['policy']['selection_policy']
    scores, _, inputs = selection.inspect_generation(policy, generated, False)
    cases = read(generated / 'cases.json')['cases']
    rows = []
    for case in cases:
        folder = Path(case['folder'])
        meta = read(folder / 'prepared.json')
        with np.load(folder / 'training.npz') as z:
            categories, choices = z['categories'], z['choices']
        point = case['candidate']
        score_file = Path(point['score_file'])
        score = read(score_file)
        if (score['subject'] != case['subject'] or score['condition'] != meta['condition']
                or score['candidate']['id'] != point['id']
                or score['candidate']['hyperparams'] != point['hyperparams']
                or digest(score_file.with_suffix('.npz')) != score['predictions_sha256']):
            raise ValueError('Confirmation identity or predictions changed')
        with np.load(score_file.with_suffix('.npz')) as z:
            probabilities = z['observed_probability']
        paths = []
        for target in case['targets']:
            with np.load(Path(target).with_suffix('.npz')) as z:
                paths.append(z['choices'])
        paths = np.concatenate(paths)
        described = describe(paths, categories, choices, probabilities, policy['features'],
                             2 if meta['condition'] == 1 else 4)
        whole = described[0]
        saved = next(s for s in scores if (s['subject'], s['arm'], s['candidate']) ==
                     (case['subject'], case['arm'], point['id']))
        np.testing.assert_allclose(whole['nll'], point['mean_nll'], atol=1e-14, rtol=0)
        np.testing.assert_allclose(whole['energy'], saved['training_energy'], atol=1e-14, rtol=0)
        rows.extend(dict(subject=case['subject'], condition=meta['condition'], arm=case['arm'],
                         candidate=point['id'], original_nominee=case['original_nominee'], **row)
                    for row in described)
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Training diagnostic input changed: ' + file)
    output.mkdir(parents=True)
    sources = dict(selection.sources(selection.pilot.driver.io), **{ENTRY: digest(ROOT / ENTRY)})
    for name in sources:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'windows.csv', index=False)
    write(output / 'completion.json', dict(status='complete', cases=len(cases),
        windows_per_case=7, generation=str(generated.resolve()), inputs_sha256=inputs,
        source_sha256=sources, summary_sha256=digest(output / 'windows.csv'),
        later_choices_read=False, oral_used=False, nominees_changed=False,
        interpretation='Exploratory training description; candidates were already fitted to all training trials',
        uncertainty='MCSE uses complete simulated paths or complete filter repeats'))
    print(table.query("subject == 306 and arm == 'SLOT' and window == 'half_2'").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.generation, args.output)
