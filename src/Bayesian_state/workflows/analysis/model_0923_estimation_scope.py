"""Compare prefix and full-sequence selection on an existing parameter bank.

Run --config CONFIG --output NEW_DIRECTORY. The POOL/SEARCH mechanisms score
the same union of previously searched points. Each filter predicts the entire
recorded sequence; prefix and full scores are computed from those same arrays.
Screening, selection and confirmation use disjoint seeds. All outcomes are
development diagnostics; full-sequence selection is retrospective. Oral reports
are excluded. --smoke checks one subject and two old nominees on eight trials.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

from . import model_0923_recalled_search_refit as recalled
from ...evaluation.model_0923_h_validation import jackknife_scores, jackknife_se, nll
from ...optimization.search.adaptive_proposals import candidate, point_id

ROOT = recalled.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
METRICS = 'src/Bayesian_state/evaluation/model_0923_h_validation.py'
OBJECTIVES = ('prefix', 'full')


def score_scopes(probability: np.ndarray, cut: int) -> dict:
    """Pool independent filter probabilities before taking each trial's log."""
    p = np.asarray(probability, float)
    if (p.ndim != 2 or len(p) < 2 or not 0 < cut < p.shape[1]
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError('Need multiple valid filter sequences and an internal prefix boundary')
    result = {}
    for name, values in [('prefix', p[:, :cut]), ('full', p), ('later', p[:, cut:])]:
        result[name + '_nll'] = nll(values)
        result[name + '_nll_mc_se'] = jackknife_se(jackknife_scores(values))
    return result


def common_bank(stage_groups: list[dict], subjects: list[int], arms: list[str]) -> dict:
    """Give every arm the entire per-subject union; no score-based exclusion."""
    union = {subject: {} for subject in subjects}
    for group in stage_groups:
        subject = group['subject']
        if subject not in union or group['model'] not in arms:
            continue
        for row in group['rows']:
            if row['id'] != point_id(row['hyperparams']):
                raise ValueError('Existing candidate identity changed')
            item = candidate(row['hyperparams'], 'existing_prefix_search_union')
            if item['id'] in union[subject] and item['hyperparams'] != union[subject][item['id']]['hyperparams']:
                raise ValueError('Candidate identifier collision')
            union[subject][item['id']] = item
    if any(not rows for rows in union.values()):
        raise ValueError('Every requested subject needs an existing candidate bank')
    return {(subject, arm): deepcopy([rows[key] for key in sorted(rows)])
            for subject, rows in union.items() for arm in arms}


def promoted_banks(scored: dict, required: dict, count: int) -> dict:
    """Preserve top points for each objective and arm, plus old nominees."""
    if count < 1:
        raise ValueError('Need a positive shortlist length')
    union = {}
    for (subject, arm), rows in scored.items():
        selected = union.setdefault(subject, {})
        for row in required[subject]:
            selected[row['id']] = candidate(row['hyperparams'], 'previous_nominee')
        for objective in OBJECTIVES:
            for row in sorted(rows, key=lambda r: (r[objective + '_nll'], r['id']))[:count]:
                selected.setdefault(row['id'], candidate(row['hyperparams'], 'scope_shortlist_union'))
    return {key: deepcopy([union[key[0]][name] for name in sorted(union[key[0]])]) for key in scored}


def smoke_data(source: Path, output: Path, subject: int, v) -> Path:
    """Make an explicitly marked eight-trial fixture from prepared behavior."""
    target = output / 'smoke_data'
    folder = target / f'subject_{subject}'
    folder.mkdir(parents=True)
    original = source / f'subject_{subject}'
    meta = v.read(original / 'prepared.json')
    with np.load(original / 'training.npz') as arrays:
        for name, first, last in [('training.npz', 0, 4), ('sealed_test.npz', 4, 8)]:
            with (folder / name).open('xb') as stream:
                np.savez_compressed(stream, **{key: arrays[key][first:last] for key in arrays.files})
    shutil.copy2(original / 'engine.json', folder / 'engine.json')
    meta.update(trials=8, training_trials=4, test_trials=4, split='eight_trial_smoke_fixture', smoke=True,
                source_prepared_sha256=v.digest(original / 'prepared.json'),
                array_sha256={name: v.digest(folder / name) for name in ('training.npz', 'sealed_test.npz')})
    v.write(folder / 'prepared.json', meta)
    preparation = v.read(source / 'preparation_manifest.json')
    v.write(target / 'preparation_manifest.json', dict(preparation, smoke=True, subjects=[subject]))
    return target


def stage(settings: dict, data: Path, output: Path, banks: dict, name: str, smoke: bool, hashes: dict) -> dict:
    """Run full sequences through the existing archived scientific worker."""
    v = recalled.base.validation()
    jobs = v.score_jobs(data, output, recalled.base.settings_for_scoring(settings), banks, name, smoke)
    groups, chunks = {}, []
    for job in jobs:
        job.update(phase='evaluate', source_sha256=hashes, arm_settings=settings['arms'][job['model']])
        groups[job['name']] = []
        width = len(job['seeds']) if name == 'screen' else 1
        for index in range(0, len(job['seeds']), width):
            target = Path(job['target'])
            chunk = dict(job, seeds=job['seeds'][index:index+width], name=job['name']+f'_C{index:03d}',
                         target=str(target.parent / 'chunks' / (target.name+f'_C{index:03d}')))
            groups[job['name']].append(chunk)
            chunks.append(chunk)
    start = time.monotonic()
    recalled.execute(chunks, output, 1 if smoke else settings['parallel_budget'])
    scored = {key: [] for key in banks}
    for job in jobs:
        record = v.combine_evaluation_chunks(job, groups[job['name']], False)
        with np.load(Path(job['target']).with_suffix('.npz')) as saved:
            scores = score_scopes(saved['observed_probability'], record['training_trials'])
        np.testing.assert_allclose(scores['full_nll'], record['mean_nll'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(scores['prefix_nll'], record['training_nll'], atol=1e-12, rtol=0)
        scored[record['subject'], record['model']].append(dict(job['candidate'], **scores,
            score_file=str(Path(job['target']).with_suffix('.json')), training_trials=record['training_trials'],
            trials=record['trials'], particles=record['particles'], seeds=record['seeds']))
    v.write(output / (name + '_completion.json'), dict(status='complete', stage=name, seconds=time.monotonic()-start,
        points=len(jobs), seed_jobs=len(chunks), groups=[dict(subject=key[0], model=key[1], rows=rows)
            for key, rows in scored.items()], data_use='development; both objectives scored on identical filter arrays'))
    return scored


def run(config: Path, output: Path, smoke: bool) -> None:
    settings_override = yaml.safe_load(config.read_text())
    v = recalled.base.validation()
    fit = ROOT / settings_override['fit']
    frozen = v.read(fit / 'nominees.json')
    if (output.exists() or frozen['smoke'] or frozen['status'] != 'frozen_before_test'
            or settings_override['sample_role'] != 'development'):
        raise ValueError('Need a new destination and a completed development fit')
    for name, expected in frozen['source_sha256'].items():
        if v.digest(ROOT / name) != expected:
            raise ValueError(f'Original scientific implementation changed: {name}')
    settings = dict(deepcopy(frozen['settings']), **settings_override)
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    if list(settings['arms']) != ['POOL', 'SEARCH']:
        raise ValueError('Require the two original mechanisms')
    input_hashes = {str(fit / 'nominees.json'): v.digest(fit / 'nominees.json')}
    groups = []
    for name in settings['bank_stages']:
        file = fit / (name + '_completion.json')
        value = v.read(file)
        if value['status'] != 'complete':
            raise ValueError('Candidate source stage is incomplete')
        input_hashes[str(file)] = v.digest(file)
        groups.extend(value['groups'])
    banks = common_bank(groups, subjects, list(settings['arms']))
    required = {subject: [row['candidate'] for row in frozen['nominees'] if row['subject'] == subject]
                for subject in subjects}
    for subject in subjects:
        if len(required[subject]) != 2:
            raise ValueError('Require both previous parameter nominees')
        for arm in settings['arms']:
            space = recalled.base.model_space(settings, arm)
            for row in banks[subject, arm]:
                if set(row['hyperparams']) != set(space) or any(row['hyperparams'][key] not in levels for key, levels in space.items()):
                    raise ValueError('Existing candidate falls outside its original support')
            if not {r['id'] for r in required[subject]} <= {r['id'] for r in banks[subject, arm]}:
                raise ValueError('Candidate union must include both original nominees')
    data = Path(frozen['data'])
    input_hashes[str(data / 'preparation_manifest.json')] = v.digest(data / 'preparation_manifest.json')
    for subject in subjects:
        folder = data / f'subject_{subject}'
        meta = v.read(folder / 'prepared.json')
        for name in ('training.npz', 'sealed_test.npz', 'prepared.json', 'engine.json'):
            input_hashes[str(folder / name)] = v.digest(folder / name)
        for name, sha in meta['array_sha256'].items():
            if v.digest(folder / name) != sha:raise ValueError('Prepared behavior changed')
    output.mkdir(parents=True)
    if smoke:
        data = smoke_data(data, output, subjects[0], v)
        banks = {key: [candidate(row['hyperparams'], 'smoke_previous_nominee') for row in required[key[0]]]
                 for key in banks}
        settings['parallel_budget'] = 1
    recalled.configure()
    hashes = recalled.base.source_hashes(v)
    hashes.update({name: v.digest(ROOT / name) for name in (ENTRY, METRICS)})
    recalled.base.archive(output, config, hashes)
    v.write(output / 'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        input_sha256=input_hashes, fit=str(fit.resolve()), data=str(data.resolve()),
        nomination_sha256=v.digest(fit / 'nominees.json'),
        data_use='development scope comparison; complete choices used; oral excluded; no prospective test claim',
        candidate_policy='same existing parameter union for both arms; shared top-prefix/full shortlists',
        python=sys.version, versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    v.write(output / 'candidate_bank.json', dict(groups=[dict(subject=key[0], model=key[1], rows=rows)
        for key, rows in banks.items()], candidate_counts={str(subject): len(banks[subject, 'POOL']) for subject in subjects}))
    print(f'Existing candidate counts: {[len(banks[subject, "POOL"]) for subject in subjects]}; identical in both arms', flush=True)
    rows = stage(settings, data, output, banks, 'screen', smoke, hashes)
    for name, count in [('selection', settings['screen_top_per_objective']),
                        ('confirmation', settings['selection_top_per_objective'])]:
        banks = promoted_banks(rows, required, 1 if smoke else count)
        rows = stage(settings, data, output, banks, name, smoke, hashes)
    nominees = []
    for (subject, arm), candidates in rows.items():
        for objective in OBJECTIVES:
            ranked = sorted(candidates, key=lambda row: (row[objective + '_nll'], row['id']))
            nominees.append(dict(subject=subject, model=arm, objective=objective, candidate=ranked[0], finalists=ranked,
                                 status='bounded_existing_bank_selection', data_use='development_retrospective_comparison'))
    for path, expected in input_hashes.items():
        if v.digest(Path(path)) != expected:raise ValueError(f'Comparison input changed: {path}')
    for path, expected in hashes.items():
        if v.digest(ROOT / path) != expected:raise ValueError(f'Comparison source changed: {path}')
    v.write(output / 'nominees.json', dict(status='frozen_for_retrospective_diagnostics', smoke=smoke,
        settings=settings, source_sha256=hashes, data=str(data.resolve()), nominees=nominees,
        original_nomination_sha256=v.digest(fit / 'nominees.json'),
        boundary='Both selection scopes are development diagnostics; preserve original chronological evaluation separately'))
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, nominees=len(nominees),
        nomination_sha256=v.digest(output / 'nominees.json'), oral_used=False, complete_behavior_used=True))
    print('Prefix/full scope nominees saved for retrospective diagnostics', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)


if __name__ == '__main__':
    main()
