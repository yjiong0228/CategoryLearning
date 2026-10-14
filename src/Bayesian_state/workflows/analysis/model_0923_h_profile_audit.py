"""Independently rescore training-profile minima against frozen H parameters.

This checks the bounded optimizer after profile selection using fresh common
filter seeds. It preserves the original nominees and held-out comparisons.
All scores read training arrays only; the selected alternative is a diagnostic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import REPO, read, write, digest, execute
from .model_0923_h_profiles import base_job, numerical_scores
from .model_0923_h_training_audit import combine
from ...evaluation.model_0923_h_validation import jackknife_se


def select_alternative(frame: pd.DataFrame, anchor: str) -> str:
    """Choose one minimum, preferring the frozen anchor in an exact tie."""
    if anchor not in set(frame.candidate) or not np.isfinite(frame.mean_nll).all():
        raise ValueError('Need finite training scores including the frozen anchor')
    unique = frame[['candidate', 'mean_nll']].drop_duplicates()
    if unique.candidate.duplicated().any():
        raise ValueError('Repeated candidate has inconsistent training scores')
    best = unique[unique.mean_nll == unique.mean_nll.min()]
    return anchor if anchor in set(best.candidate) else sorted(best.candidate)[0]


def run(profiles: Path, config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new profile-audit destination')
    protocol = read(profiles / 'protocol.json')
    if protocol['role'] != 'conditional_training_profiles' or read(profiles / 'completion.json')['status'] != 'complete':
        raise ValueError('Need completed training-only profiles')
    fit = Path(protocol['fit'])
    if digest(fit / 'nominees.json') != protocol['nomination_sha256']:
        raise ValueError('Frozen nominees changed')
    frozen = read(fit / 'nominees.json')
    for name, sha in frozen['source_sha256'].items():
        if digest(REPO / name) != sha:
            raise ValueError('Scientific implementation changed')
    settings = yaml.safe_load(config.read_text())
    if smoke:
        settings.update(particles=2, seeds=2, parallel_budget=1)
    data = pd.read_csv(profiles / 'profiles.csv')
    selected = [r for r in frozen['nominees'] if r['model'] in set(data.model)]
    if smoke:
        selected = selected[:1]
    output.mkdir(parents=True)
    jobs, chunks, groups, pairs = [], [], {}, []
    for row in selected:
        anchor = row['candidate']['id']
        group = data[(data.subject == row['subject']) & (data.model == row['model'])]
        alternative = select_alternative(group, anchor)
        pairs.append(dict(subject=row['subject'], model=row['model'], anchor=anchor, alternative=alternative))
        for candidate_id in dict.fromkeys([anchor, alternative]):
            original = read(profiles / f'subject_{row["subject"]}' / row['model'] / (candidate_id + '.json'))
            job = base_job(frozen, settings, row, 'independent_profile_audit')
            job['source_sha256'][str(Path(__file__).resolve().relative_to(REPO))] = digest(Path(__file__))
            job.update(candidate=original['candidate'], name=f'S{row["subject"]}_{row["model"]}_{candidate_id}',
                       target=str((output / f'subject_{row["subject"]}' / row['model'] / candidate_id).resolve()))
            jobs.append(job)
            groups[job['name']] = []
            for index, seed in enumerate(job['seeds']):
                chunk = deepcopy(job)
                target = Path(job['target'])
                chunk.update(name=job['name'] + f'_C{index:03d}', seeds=[seed],
                             target=str(target.parent / 'chunks' / (target.name + f'_C{index:03d}')))
                groups[job['name']].append(chunk)
                chunks.append(chunk)
    write(output / 'protocol.json', dict(profiles=str(profiles.resolve()),
        profile_scores_sha256=digest(profiles / 'profiles.csv'), nomination_sha256=protocol['nomination_sha256'],
        settings=settings, smoke=smoke, selection=pairs, source_sha256=digest(Path(__file__)),
        role='training_only_optimizer_diagnostic_after_primary_test; nominees_unchanged'))
    lengths = {j['folder']: read(Path(j['folder']) / 'prepared.json')['training_trials'] for j in jobs}
    chunks.sort(key=lambda job: -lengths[job['folder']])
    execute(chunks, output, settings['parallel_budget'])
    for job in jobs:
        combine(job, groups[job['name']])
    rows = []
    for pair in pairs:
        folder = output / f'subject_{pair["subject"]}' / pair['model']
        old_file = folder / (pair['anchor'] + '.json')
        new_file = folder / (pair['alternative'] + '.json')
        old, new = read(old_file), read(new_file)
        a, old_jack = numerical_scores(old_file)
        b, new_jack = numerical_scores(new_file)
        np.testing.assert_array_equal(a, b)
        difference = new['mean_nll'] - old['mean_nll']
        se = jackknife_se(new_jack - old_jack)
        rows.append(dict(pair, training_trials=old['trials'], anchor_nll=old['mean_nll'], alternative_nll=new['mean_nll'],
                         delta_nll=difference, delta_mc_se=se, total_delta=difference * old['trials'],
                         total_mc_se=se * old['trials'], lower_by_two_mc_se=difference < -2 * se))
    pd.DataFrame(rows).to_csv(output / 'contrasts.csv', index=False)
    write(output / 'completion.json', dict(status='complete', smoke=smoke, pairs=len(pairs),
        candidates=len(jobs), seed_jobs=len(chunks), nominees_changed=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profiles', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.profiles, args.config, args.output, args.smoke)


if __name__ == '__main__':
    main()
