"""Training-only H refinement with independent numerical confirmation.

Retain eight archived finalists, propose an equal number of new points for
each subject/model, screen with common filter seeds, audit a short list with
fresh seeds, then confirm the selected challenger against the original fit
with a third seed bank. Confirmation accepts a challenger only when its paired
training-score improvement exceeds two Monte Carlo standard errors. This is
a numerical decision rule, not a significance test or convergence proof.

These participants have already supplied inspected test results. All parameter
operations here read their training prefix only; subsequent evaluation remains
development evaluation. Original fits and predictions are immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import REPO, read, write, digest, model_space, score_jobs, execute, pooled_nll
from .model_0923_h_training_audit import combine
from ...evaluation.model_0923_h_validation import jackknife_scores, jackknife_se
from ...optimization.search.adaptive_proposals import ray_proposals, propose_round, select_elites, point_id
from ...utils.seeding import stable_seed


def proposal_bank(row: dict, space: dict, settings: dict, seed: int) -> list[dict]:
    """Cover parameter axes and joint moves around diverse archived finalists."""
    originals = deepcopy(row['finalists'][:settings['archived_finalists']])
    if len(originals) != settings['archived_finalists'] or row['candidate']['id'] not in {r['id'] for r in originals}:
        raise ValueError('Need the original nominee within the complete finalist bank')
    for item in originals:
        if point_id(item['hyperparams']) != item['id'] or set(item['hyperparams']) != set(space):
            raise ValueError('Archived point differs from its parameter support')
        if any(value not in space[key] for key, value in item['hyperparams'].items()):
            raise ValueError('Archived coordinate outside original support')
    elites = select_elites(originals, settings['elites'])
    seen = {r['id'] for r in originals}
    rays = ray_proposals(elites, space, seen, settings['rays_per_elite'], seed, 'training_refinement_ray')
    seen.update(r['id'] for r in rays)
    joint = propose_round(elites, space, seen, settings['joint_per_elite'], seed + 1, 0., 0.)
    expected = settings['elites'] * (settings['rays_per_elite'] + settings['joint_per_elite'])
    if len(rays) + len(joint) != expected:
        raise ValueError('Proposal bank did not meet the equal new-point budget')
    return originals + rays + joint


def audit_bank(rows: list[dict], anchor: dict, count: int) -> list[dict]:
    """Always retain the original fit, then the best distinct screening points."""
    ordered = sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
    selected = [anchor] + [r for r in ordered if r['id'] != anchor['id']][:count - 1]
    if len(selected) != count:
        raise ValueError('Not enough distinct screened points for the audit')
    return selected


def paired_comparison(anchor: dict, challenger: dict, multiplier: float) -> dict:
    def arrays(row):
        file = Path(row['score_file']); record = read(file)
        if digest(file.with_suffix('.npz')) != record['predictions_sha256']:
            raise ValueError('Saved training predictions changed')
        with np.load(file.with_suffix('.npz')) as saved:
            return saved['seeds'], saved['observed_probability']
    seeds, p0 = arrays(anchor); other, p1 = arrays(challenger)
    np.testing.assert_array_equal(seeds, other)
    if p0.shape != p1.shape:
        raise ValueError('Training predictions do not align')
    delta = pooled_nll(p1) - pooled_nll(p0)
    se = jackknife_se(jackknife_scores(p1) - jackknife_scores(p0))
    return dict(delta_mean_nll=delta, paired_mc_se=se, delta_total_nll=delta * p0.shape[1],
        paired_total_mc_se=se * p0.shape[1], accepted=bool(delta < -multiplier * se), trials=p0.shape[1],
        seeds=len(seeds), anchor=anchor['id'], challenger=challenger['id'])


def score_stage(frozen: dict, settings: dict, banks: dict, stage: str, output: Path, smoke: bool) -> dict:
    directory = output / stage; directory.mkdir()
    scoring = deepcopy(frozen['settings'])
    scoring.update(protocol=settings['protocol'], seed=settings['seed'])
    scoring['budgets'][stage] = settings['budgets'][stage]
    jobs = score_jobs(Path(frozen['data']), directory, scoring, banks, stage, False)
    groups, chunks = {}, []
    for job in jobs:
        job['source_sha256'][str(Path(__file__).resolve().relative_to(REPO))] = digest(Path(__file__))
        if smoke:
            job.update(particles=2, smoke=True, seeds=job['seeds'][:2])
        groups[job['name']] = []
        width = settings['seed_chunk'] if not smoke else 2
        for first in range(0, len(job['seeds']), width):
            chunk = deepcopy(job); target = Path(job['target'])
            chunk.update(name=job['name'] + f'_C{first:03d}', seeds=job['seeds'][first:first+width],
                target=str(target.parent / 'chunks' / (target.name + f'_C{first:03d}')))
            groups[job['name']].append(chunk); chunks.append(chunk)
    lengths = {j['folder']: read(Path(j['folder']) / 'prepared.json')['training_trials'] for j in jobs}
    chunks.sort(key=lambda j: -lengths[j['folder']])
    start = time.monotonic(); execute(chunks, directory, 1 if smoke else settings['parallel_budget'])
    scored = {key: [] for key in banks}
    for job in jobs:
        row = combine(job, groups[job['name']])
        subject = read(Path(job['folder']) / 'prepared.json')['subject']
        scored[(subject, job['model'])].append(row)
    for rows in scored.values():
        rows.sort(key=lambda r: (r['mean_nll'], r['id']))
    write(directory / 'completion.json', dict(status='complete', points=len(jobs), seed_jobs=len(chunks),
        seconds=time.monotonic()-start, groups=[dict(subject=k[0], model=k[1], rows=rows) for k, rows in scored.items()]))
    return scored


def run(fit: Path, config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new training-refinement destination')
    frozen = read(fit / 'nominees.json'); settings = yaml.safe_load(config.read_text())
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test':
        raise ValueError('Need frozen scientific starting fits')
    for name, sha in frozen['source_sha256'].items():
        if digest(REPO / name) != sha:
            raise ValueError('Frozen scientific implementation changed')
    selected = [r for r in frozen['nominees'] if r['model'] in settings['models']]
    if {r['model'] for r in selected} != set(settings['models']):
        raise ValueError('Missing requested starting models')
    if smoke:
        selected = selected[:1]
        settings.update(archived_finalists=2, elites=1, rays_per_elite=1, joint_per_elite=1, audit_candidates=2)
    banks, originals = {}, {}
    for row in selected:
        key = row['subject'], row['model']; originals[key] = row
        space, _ = model_space(frozen['settings'], row['model'])
        seed = stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], role='proposals', subject=row['subject'], model=row['model']))
        banks[key] = proposal_bank(row, space, settings, seed)
    output.mkdir(parents=True)
    write(output / 'protocol.json', dict(parent_fit=str(fit.resolve()), parent_nomination_sha256=digest(fit / 'nominees.json'),
        settings=settings, smoke=smoke, source_sha256=digest(Path(__file__)),
        data_boundary='training.npz only; no sealed_test arrays in search or confirmation',
        data_use_status='previously inspected test participants are development participants',
        acceptance='independently confirmed training NLL improvement exceeds two paired numerical MCSE',
        banks=[dict(subject=k[0], model=k[1], candidates=rows) for k, rows in banks.items()]))
    screened = score_stage(frozen, settings, banks, 'screen', output, smoke)
    banks = {key: audit_bank(rows, originals[key]['candidate'], settings['audit_candidates']) for key, rows in screened.items()}
    write(output / 'audit_selection.json', dict(groups=[dict(subject=k[0], model=k[1], candidates=rows) for k, rows in banks.items()]))
    audited = score_stage(frozen, settings, banks, 'audit', output, smoke)
    banks, challengers = {}, {}
    for key, rows in audited.items():
        anchor = originals[key]['candidate']; best = rows[0]
        challengers[key] = best['id']
        banks[key] = [anchor] + ([best] if best['id'] != anchor['id'] else [])
    write(output / 'confirmation_selection.json', dict(groups=[dict(subject=k[0], model=k[1], candidates=rows) for k, rows in banks.items()]))
    confirmed = score_stage(frozen, settings, banks, 'confirm', output, smoke)
    nominees, decisions = [], []
    for key, rows in confirmed.items():
        index = {r['id']: r for r in rows}; anchor = index[originals[key]['candidate']['id']]
        challenger = index[challengers[key]]
        decision = paired_comparison(anchor, challenger, settings['mcse_multiplier'])
        decisions.append(dict(subject=key[0], model=key[1], **decision))
        nominee = challenger if decision['accepted'] else anchor
        nominees.append(dict(originals[key], candidate=nominee, finalists=audited[key], previous_candidate_id=anchor['id'],
            scientific_status='bounded_training_refinement_with_independent_numerical_confirmation'))
    pd.DataFrame(decisions).to_csv(output / 'decisions.csv', index=False)
    write(output / 'nominees.json', dict(status='frozen_for_development_evaluation', settings=frozen['settings'],
        source_sha256=frozen['source_sha256'], smoke=smoke, data=frozen['data'], nominees=nominees,
        parent_fit=str(fit.resolve()), refinement_settings=settings, data_use_status='development'))
    write(output / 'completion.json', dict(status='complete', smoke=smoke, groups=len(nominees),
        accepted=sum(r['accepted'] for r in decisions), test_arrays_opened=False))
    print('Training-only refinement and independent numerical confirmation complete', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.fit, args.config, args.output, args.smoke)


if __name__ == '__main__':
    main()
