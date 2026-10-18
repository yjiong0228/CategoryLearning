"""Generate autonomous diagnostics for prefix/full scope-selected parameters.

Use --scope SCOPE_FIT --reference ORIGINAL_EVALUATION --output NEW_DIRECTORY.
Parameters and mechanism implementations stay frozen. Original, prefix-selected
and full-selected points use the original autonomous random seeds. Identical
points within a mechanism share one path bank; original banks are referenced
without copying. New paths use the shared archived autonomous learning engine.
All comparisons are development diagnostics. --smoke uses two paths of sixteen
trials per distinct point and one worker, and checks original-point predictions.
--selection-stage compares the completed selection stage's full-sequence winners
with the originals while confirmation continues. Its provenance remains explicitly
provisional. --parallel-budget bounds this independent generation queue.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from . import model_0923_recalled_search_refit as recalled
from ...optimization.search.adaptive_proposals import candidate, point_id

ROOT = recalled.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def point_bindings(selected: list[dict], original: list[dict],
                   objectives: tuple[str, ...] = ('prefix', 'full')) -> tuple[list[dict], dict]:
    """Share matching points within an arm while keeping objective identities."""
    if not objectives or len(set(objectives)) != len(objectives) or not set(objectives) <= {'prefix', 'full'}:
        raise ValueError('Need unique recognized selection objectives')
    subjects = {row['subject'] for row in selected}
    bindings, points, seen = [], {}, set()
    items = selected + [dict(row, objective='original') for row in original if row['subject'] in subjects]
    for row in items:
        subject, model, objective = row['subject'], row['model'], row['objective']
        key = subject, model, objective
        if model not in ('POOL', 'SEARCH') or objective not in (*objectives, 'original') or key in seen:
            raise ValueError('Need unique original/prefix/full bindings for both mechanisms')
        seen.add(key)
        point = row['candidate']
        if point['id'] != point_id(point['hyperparams']):
            raise ValueError('Parameter identity does not match the candidate')
        identity = subject, model, point['id']
        points.setdefault(identity, candidate(point['hyperparams'], 'frozen_scope_diagnostic'))
        bindings.append(dict(subject=subject, model=model, objective=objective, candidate_id=point['id']))
    expected = {(subject, model, objective) for subject in subjects
                for model in ('POOL', 'SEARCH') for objective in (*objectives, 'original')}
    if not subjects or seen != expected:
        raise ValueError('Incomplete original/prefix/full parameter comparison')
    return bindings, points


def provisional_full_nominees(stage: dict, subjects: list[int]) -> list[dict]:
    """Select the full-NLL minimum only after every common-bank score is ready."""
    if stage['status'] != 'complete' or stage['stage'] != 'selection':
        raise ValueError('Require a completed selection stage')
    groups, budgets, seed_jobs = {}, {}, 0
    for group in stage['groups']:
        key = group['subject'], group['model']
        if key in groups or not group['rows']:
            raise ValueError('Duplicate or empty selection group')
        rows = {}
        for row in group['rows']:
            if (row['id'] != point_id(row['hyperparams']) or row['id'] in rows
                    or not np.isfinite(row['full_nll']) or row['full_nll'] < 0
                    or len(row['seeds']) < 2 or len(set(row['seeds'])) != len(row['seeds'])):
                raise ValueError('Invalid selection candidate or independent seeds')
            current = row['particles'], tuple(row['seeds'])
            if group['subject'] in budgets and current != budgets[group['subject']]:
                raise ValueError('Selection candidates must share the numerical budget')
            budgets[group['subject']] = current; rows[row['id']] = row; seed_jobs += len(row['seeds'])
        groups[key] = rows
    expected = {(subject, arm) for subject in subjects for arm in ('POOL', 'SEARCH')}
    if (not subjects or len(set(subjects)) != len(subjects) or set(groups) != expected
            or stage['points'] != sum(map(len, groups.values()))
            or stage['seed_jobs'] != seed_jobs):
        raise ValueError('Incomplete selection checkpoint')
    for subject in subjects:
        if set(groups[subject, 'POOL']) != set(groups[subject, 'SEARCH']):
            raise ValueError('Both mechanisms require the same selection bank')
    return [dict(subject=subject, model=arm, objective='full',
                 candidate=min(rows.values(), key=lambda row:(row['full_nll'], row['id'])))
            for (subject, arm), rows in sorted(groups.items())]


def run(scope: Path, reference: Path, output: Path, smoke: bool,
        selection_stage: bool = False, parallel_budget: int | None = None) -> None:
    v = recalled.base.validation()
    protocol = v.read(scope / 'protocol.json')
    checkpoint = scope / ('selection_completion.json' if selection_stage else 'nominees.json')
    if selection_stage:
        nominees = provisional_full_nominees(v.read(checkpoint), protocol['settings']['subjects'])
        frozen = dict(protocol, nominees=nominees, original_nomination_sha256=protocol['nomination_sha256'])
        binding_hash = dict(selection_checkpoint_sha256=v.digest(checkpoint))
    else:
        frozen = v.read(checkpoint)
        binding_hash = dict(nomination_sha256=v.digest(checkpoint))
    original_fit = Path(protocol['fit'])
    original = v.read(original_fit / 'nominees.json')
    if output.exists():
        raise FileExistsError('Use a new autonomous scope destination')
    if (frozen['smoke'] and not smoke) or (not selection_stage and (
            frozen['status'] != 'frozen_for_retrospective_diagnostics'
            or v.read(scope / 'completion.json')['nomination_sha256'] != v.digest(checkpoint))):
        raise ValueError('Need completed scope selection; smoke fits require smoke diagnostics')
    if (original['smoke'] or v.digest(original_fit / 'nominees.json') != frozen['original_nomination_sha256']
            or v.read(reference / 'protocol.json')['nomination_sha256'] != frozen['original_nomination_sha256']
            or v.read(reference / 'completion.json')['status'] != 'complete'):
        raise ValueError('Original autonomous reference differs from the parent fit')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(ROOT / name) != sha:
            raise ValueError(f'Frozen scope implementation changed: {name}')
    settings = deepcopy(frozen['settings'])
    if parallel_budget is not None and not 1 <= parallel_budget <= settings['parallel_budget']:
        raise ValueError('Generation budget must stay within the declared CPU limit')
    settings['parallel_budget'] = 1 if smoke else parallel_budget or settings['parallel_budget']
    bindings, points = point_bindings(frozen['nominees'], original['nominees'],
                                      ('full',) if selection_stage else ('prefix', 'full'))
    references = {(row['subject'], row['model']): row for row in
                  v.read(reference / 'autonomous/evaluation.json')['rows']}
    hashes = dict(frozen['source_sha256'], **{ENTRY: v.digest(ROOT / ENTRY)})
    input_files = [checkpoint, scope / 'protocol.json',
        original_fit / 'nominees.json', reference / 'protocol.json', reference / 'completion.json',
        reference / 'autonomous/evaluation.json']
    if not selection_stage: input_files.append(scope / 'completion.json')
    inputs = {str(file): v.digest(file) for file in input_files}
    if selection_stage:
        from .model_0923_estimation_scope import score_scopes
        for row in frozen['nominees']:
            point = row['candidate']; file = Path(point['score_file']); record = v.read(file)
            array = file.with_suffix('.npz')
            if (v.digest(array) != record['predictions_sha256']
                    or record['candidate']['hyperparams'] != point['hyperparams']
                    or (record['subject'], record['model']) != (row['subject'], row['model'])
                    or record['particles'] != point['particles'] or record['seeds'] != point['seeds']):
                raise ValueError('Selected candidate differs from its completed score arrays')
            with np.load(array) as saved:
                np.testing.assert_array_equal(saved['seeds'], point['seeds'])
                scores = score_scopes(saved['observed_probability'], record['training_trials'])
            np.testing.assert_allclose(scores['full_nll'], point['full_nll'], atol=1e-12, rtol=0)
            inputs[str(file)] = v.digest(file); inputs[str(array)] = v.digest(array)
    jobs, plans, paths_by_subject = [], [], {}
    for (subject, model, identity), point in points.items():
        old = references[subject, model]
        old_file = reference / 'autonomous/evaluate' / f'subject_{subject}' / model / (old['candidate']['id'] + '.npz')
        if v.digest(old_file) != old['predictions_sha256']:
            raise ValueError('Original autonomous paths changed')
        inputs[str(old_file)] = old['predictions_sha256']
        with np.load(old_file) as saved:
            seeds = saved['seeds'][:2 if smoke else settings['autonomous_paths']].tolist()
        count = 2 if smoke else settings['autonomous_paths']
        if len(seeds) != count or len(set(seeds)) != count:
            raise ValueError('Original paths do not provide the requested common seed bank')
        if subject in paths_by_subject:
            np.testing.assert_array_equal(seeds, paths_by_subject[subject])
        paths_by_subject[subject] = seeds
        same = point['hyperparams'] == old['candidate']['hyperparams']
        if same and not smoke:
            if old['paths'] != count:
                raise ValueError('Reference reuse requires the complete original path bank')
            plans.append(dict(subject=subject, model=model, candidate_id=identity, original_reference=str(old_file),
                              target=str(old_file.with_suffix('')), reused_reference=True, record=old))
            continue
        templates = sorted((reference / 'jobs').glob(f'AUTO_evaluate_S{subject}_{model}_*.json'))
        if not templates:
            raise ValueError('Missing archived autonomous worker template')
        template = templates[0]
        inputs[str(template)] = v.digest(template)
        job = v.read(template)
        if (job['candidate']['hyperparams'] != old['candidate']['hyperparams']
                or job['arm_settings'] != settings['arms'][model] or job['mechanism'] != settings['mechanism']):
            raise ValueError('Original worker settings do not match the fixed scope mechanisms')
        if not smoke and Path(job['folder']).resolve() != (Path(frozen['data']) / f'subject_{subject}').resolve():
            raise ValueError('Scope and generation use different prepared task inputs')
        target = output / 'generation' / f'subject_{subject}' / model / identity
        job.update(candidate=point, seeds=seeds, smoke=smoke, phase='autonomous', source_sha256=hashes,
                   name=f'scope_AUTO_S{subject}_{model}_{identity}', target=str(target.resolve()))
        chunks = []
        width = count if smoke else settings['autonomous_chunk']
        for index in range(0, count, width):
            chunk = dict(job, seeds=seeds[index:index+width], name=job['name']+f'_C{index:03d}',
                         target=str(target.parent / 'chunks' / (target.name+f'_C{index:03d}')))
            chunks.append(chunk)
            jobs.append(chunk)
        plans.append(dict(subject=subject, model=model, candidate_id=identity, original_reference=str(old_file),
                          target=str(target.resolve()), reused_reference=False, job=job, chunks=chunks,
                          exact_reference_check=bool(smoke and same)))
    output.mkdir(parents=True)
    for name in hashes:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    v.write(output / 'protocol.json', dict(scope=str(scope.resolve()), reference=str(reference.resolve()),
        settings=settings, smoke=smoke, source_sha256=hashes, input_sha256=inputs, bindings=bindings,
        **binding_hash, selection_phase='selection' if selection_stage else 'confirmation',
        diagnostic_status='provisional_selection_stage' if selection_stage else 'confirmed_scope_comparison',
        data_use='development autonomous diagnostics at frozen points; complete behavior used in full selection; oral excluded',
        random_streams='same complete trajectory seeds as the original autonomous evaluation',
        path_sharing='same subject, mechanism and parameter identity share one bank; objectives are labels of selection'))
    print(f'{len(bindings)} selection bindings, {len(plans)} distinct points, {len(jobs)} new path chunks', flush=True)
    if jobs:
        recalled.execute(jobs, output, settings['parallel_budget'])
    records, exact = [], 0
    for plan in plans:
        row = plan['record'] if plan['reused_reference'] else v.combine_evaluation_chunks(plan['job'], plan['chunks'], True)
        target = Path(plan['target']).with_suffix('.npz')
        if v.digest(target) != row['predictions_sha256']:
            raise ValueError('Generated bank changed')
        if plan.get('exact_reference_check'):
            with np.load(target) as new, np.load(plan['original_reference']) as old:
                np.testing.assert_array_equal(new['seeds'], old['seeds'][:len(new['seeds'])])
                for field in new.files:
                    if field in ('seeds', 'categories'):
                        expected = old[field][:len(new[field])]
                    else:
                        expected = old[field][:len(new['seeds']), :new[field].shape[1]]
                    np.testing.assert_array_equal(new[field], expected, err_msg=f'Original autonomous prefix changed: {field}')
            exact += 1
        records.append(dict(row, target=plan['target'], candidate_id=plan['candidate_id'],
                            reused_reference=plan['reused_reference']))
    if smoke and exact != len(paths_by_subject)*2:
        raise ValueError('Smoke run must verify the original point for each mechanism')
    for name, sha in inputs.items():
        if v.digest(Path(name)) != sha:raise ValueError(f'Diagnostic input changed: {name}')
    for name, sha in hashes.items():
        if v.digest(ROOT / name) != sha:raise ValueError(f'Diagnostic source changed: {name}')
    v.write(output / 'generation.json', dict(status='complete', smoke=smoke, bindings=bindings, rows=records,
        selection_phase='selection' if selection_stage else 'confirmation',
        exact_original_prefix_checks=exact, distinct_points=len(records), new_chunks=len(jobs),
        distinct_paths=sum(row['paths'] for row in records)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scope', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--selection-stage', action='store_true')
    parser.add_argument('--parallel-budget', type=int)
    args = parser.parse_args()
    run(args.scope, args.reference, args.output, args.smoke, args.selection_stage, args.parallel_budget)


if __name__ == '__main__':
    main()
