"""Audit data identity and save the complete exploratory execution-study screen.

Run --config YAML --data PREPARED --output NEW. Behavioral ranking uses only
each participant's chronological training prefix. The full candidate table,
both dataset hashes, and oral-coding differences are retained for review.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import yaml

from .model_0923_h_validation import training_boundary
from .model_1022_belief_summary import read, digest
from .model_1023_execution_study import write


def run(config: Path, data: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new cohort audit destination')
    policy = yaml.safe_load(config.read_text())
    current = Path('data/exp123/processed')
    archived = Path(policy['preparation']['source_fit']) / 'source/data/exp123/processed'
    audits = []
    for name in ('Task2_processed.csv', 'Task1b_errorsummary_24.csv', 'Task1b_errorsummary_72.csv'):
        a, b = [pd.read_csv(p / name) for p in (archived, current)]
        if a.shape != b.shape or list(a) != list(b):
            raise ValueError('Archived and current data dimensions differ')
        changed = {}
        for col in a:
            different = ~(a[col].eq(b[col]) | (a[col].isna() & b[col].isna()))
            if different.any():
                changed[col] = int(different.sum())
        if set(changed) - {f'feature{i}_use' for i in range(1, 5)}:
            raise ValueError('Behavior or perception data changed; preparation needs review')
        audits.append(dict(name=name, archived_sha256=digest(archived / name),
            current_sha256=digest(current / name), changed_cells=changed,
            behavioral_and_design_columns_identical=True))
    frame = pd.read_csv(current / 'Task2_processed.csv')
    if 'miss' in frame or (frame['choice'] <= 0).any():
        raise ValueError('This screen requires the existing fully answered exp123 dataset')
    width = policy['behavior_diagnostics']['low_accuracy_window']
    stride = policy['behavior_diagnostics']['low_accuracy_stride']
    chance = policy['behavior_diagnostics']['chance']
    windows, included = [], []
    for sid, rows in frame.groupby('iSub', sort=True):
        conditions = rows['condition'].unique()
        if len(conditions) != 1:
            raise ValueError('Participant spans multiple conditions')
        condition = int(conditions[0])
        if len(rows) < policy['preparation']['minimum_valid_trials']:
            continue
        cut = training_boundary(len(rows), policy['preparation']['training_fraction'])
        correct = rows['choice'].eq(rows['category']).to_numpy()
        for start in range(0, cut - width + 1, stride):
            successes = int(correct[start:start + width].sum())
            window = rows.iloc[start:start+width]
            labels = range(1, 3 if condition == 1 else 5)
            categories = np.array([window.category.eq(c).sum() for c in labels])
            responses = np.array([window.choice.eq(c).sum() for c in labels])
            marginal_chance = float(categories @ responses / width**2)
            windows.append(dict(subject=int(sid), condition=condition, training_trials=cut,
                first_trial=start + 1, last_trial=start + width, successes=successes, trials=width,
                accuracy=successes / width, chance=chance[condition],
                marginal_chance=marginal_chance, accuracy_minus_marginal_chance=successes/width-marginal_chance,
                **{f'category_{c}_count': int(value) for c, value in zip(labels, categories)},
                **{f'response_{c}_count': int(value) for c, value in zip(labels, responses)},
                raw_one_sided_p=binomtest(successes, width, chance[condition], alternative='less').pvalue))
        if int(sid) in [s for s, c in policy['subjects']]:
            folder = data / f'subject_{sid}'
            meta = read(folder / 'prepared.json')
            if meta['training_trials'] != cut or meta['condition'] != condition:
                raise ValueError('Cohort and prepared split differ')
            for name in ('training.npz', 'sealed_test.npz'):
                if digest(folder / name) != meta['array_sha256'][name]:
                    raise ValueError('Prepared data hash mismatch')
            with np.load(folder / 'training.npz') as a, np.load(folder / 'sealed_test.npz') as b:
                for key, col in [('choices', 'choice'), ('categories', 'category'), ('feedback', 'feedback')]:
                    np.testing.assert_array_equal(np.concatenate([a[key], b[key]]), rows[col].to_numpy())
            included.append(dict(subject=int(sid), condition=condition, trials=len(rows),
                training_trials=cut, test_trials=len(rows) - cut, accuracy=float(correct.mean()),
                prepared_sha256=digest(folder / 'prepared.json')))
    ranking = pd.DataFrame(windows).sort_values(['raw_one_sided_p', 'subject', 'first_trial'])
    if (len(ranking) != policy['cohort_selection']['training_screen_windows']
            or int(ranking.iloc[0]['subject']) != policy['cohort_selection']['training_screen_case']):
        raise ValueError('Screen differs from the registered exploratory cohort')
    output.mkdir(parents=True)
    ranking.to_csv(output / 'all_training_windows.csv', index=False)
    ranking.sort_values(['accuracy_minus_marginal_chance', 'subject', 'first_trial']).to_csv(
        output / 'marginal_baseline_ranking.csv', index=False)
    pd.DataFrame(included).to_csv(output / 'included_participants.csv', index=False)
    write(output / 'completion.json', dict(status='complete', policy=policy, configuration_sha256=digest(config),
        source_sha256=digest(Path(__file__)), data_audit=audits, windows=len(windows),
        selected=int(ranking.iloc[0]['subject']), selection_uses_later_trials=False,
        interpretation='exploratory case selection; raw p values have no familywise significance interpretation',
        marginal_diagnostic='all original windows; category/response independence baseline; '
            'descriptive supplementary ranking; original cohort unchanged',
        oral_validation_source='current corrected coding; separate from behavioral estimation',
        artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))
    print(ranking.head(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'data', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.data, args.output)
