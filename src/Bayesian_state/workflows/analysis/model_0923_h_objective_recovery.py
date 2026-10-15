"""Prepare full-coordinate synthetic recovery for the frozen objective comparison.

Generation uses six pre-existing ECM_S points and two fresh paths per point.
Truth parameters are stored outside prepared fitting data. Every active
coordinate is estimated by the unmodified objective-comparison search.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from . import model_0923_h_validation as v
from .model_0923_h_objective_comparison import validate_settings
from ...utils.seeding import stable_seed


def save_arrays(folder: Path, name: str, arrays: dict) -> str:
    file = folder/name
    with file.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return v.digest(file)


def fixture(original: Path, destination: Path, arrays: dict) -> dict:
    """Reuse physical/perceptual metadata while replacing only synthetic behavior."""
    destination.mkdir(parents=True)
    meta = deepcopy(v.read(original/'prepared.json'))
    v.write(destination/'engine.json', v.read(original/'engine.json'))
    meta.update(engine_sha256=v.digest(destination/'engine.json'), training_trials=len(arrays['choices']),
        test_trials=0, trials=len(arrays['choices']), data_role='synthetic_training_only',
        array_sha256={'training.npz': save_arrays(destination, 'training.npz', arrays),
            'sealed_test.npz': save_arrays(destination, 'sealed_test.npz', {k: x[:0] for k, x in arrays.items()})})
    v.write(destination/'prepared.json', meta)
    return meta


def prepare(config: Path, output: Path) -> None:
    settings = yaml.safe_load(config.read_text()); validate_settings(settings)
    if output.exists():
        raise FileExistsError('Use a new recovery destination')
    spec = settings['recovery']; source = v.REPO/spec['source_fit']; frozen = v.read(source/'nominees.json')
    if spec['estimate_all_coordinates'] is not True:
        raise ValueError('This recovery must estimate every active coordinate')
    data = Path(frozen['data']); space, _ = v.model_space(settings, 'NLL')
    rows = [next(r for r in frozen['nominees'] if r['subject'] == subject and r['model'] == spec['model'])
            for subject in spec['subjects']]
    for row in rows:
        point = row['candidate']['hyperparams']
        if set(point) != set(space) or any(point[k] not in values for k, values in space.items()):
            raise ValueError('Recovery truth falls outside the shared fitting support')
    output.mkdir(parents=True)
    v.write(output/'protocol.json', dict(settings=settings, source_fit=str(source.resolve()),
        source_nomination_sha256=v.digest(source/'nominees.json'), coordinates=list(space),
        generation_role='fixed_schedule_full_coordinate_parameter_recovery',
        source_sha256=v.digest(Path(__file__))))
    shutil.copy2(__file__, output/'preparation_source.py')
    jobs = []
    for row in rows:
        original = data/f'subject_{row["subject"]}'
        meta = v.read(original/'prepared.json')
        if v.digest(original/'training.npz') != meta['array_sha256']['training.npz']:
            raise ValueError('Original training input changed')
        with np.load(original/'training.npz') as saved:
            arrays = {k: saved[k] for k in saved.files}
        folder = output/'generation_data'/f'subject_{row["subject"]}'
        fixture(original, folder, arrays)
        jobs.append(dict(phase='autonomous', name=f'recovery_generate_{row["subject"]}',
            folder=str(folder.resolve()), model=spec['model'], candidate=row['candidate'],
            mechanism=settings['models']['NLL'], runtime=str((v.REPO/frozen['settings'].get('source_fit',
                v.read(data/'preparation_manifest.json')['settings']['source_fit'])/'source').resolve()),
            seeds=[stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], role='synthetic_truth_generation',
                subject=row['subject'], replicate=i)) for i in range(spec['replicates'])],
            source_sha256=v.source_hashes(), target=str((output/'generated'/f'S{row["subject"]}').resolve())))
    v.execute(jobs, output, min(settings['parallel_budget'], len(jobs)))
    truths = []; cases = []; prepared = output/'prepared_data'; prepared.mkdir()
    for row, job in zip(rows, jobs):
        original = data/f'subject_{row["subject"]}'
        with np.load(original/'training.npz') as saved:
            physical = {key: saved[key] for key in ('stimulus', 'categories')}
        with np.load(Path(job['target']).with_suffix('.npz')) as saved:
            choices, feedback = saved['choices'], saved['feedback']
        for repeat in range(spec['replicates']):
            case = 900001+len(cases)
            fixture(original, prepared/f'subject_{case}', dict(physical, choices=choices[repeat], feedback=feedback[repeat]))
            cases.append(case)
            truths.append(dict(case_id=case, original_subject=row['subject'], replicate=repeat,
                generation_seed=job['seeds'][repeat], candidate_id=row['candidate']['id'],
                hyperparams=row['candidate']['hyperparams']))
    prep = v.read(data/'preparation_manifest.json')
    v.write(prepared/'preparation_manifest.json', dict(status='complete', settings=prep['settings'],
        subjects=cases, smoke=False, data_role='synthetic_parameter_recovery',
        original_preparation_sha256=v.digest(data/'preparation_manifest.json')))
    v.write(output/'truths.json', dict(rows=truths, coordinates=list(space), fitted_coordinate_count=len(space)))
    fit_settings = deepcopy(settings); fit_settings['data_use_status'] = 'synthetic_parameter_recovery'
    with (output/'fit_config.yaml').open('x') as handle:
        yaml.safe_dump(fit_settings, handle, sort_keys=False)
    v.write(output/'completion.json', dict(status='complete', synthetic_sequences=len(cases),
        truths=len(rows), estimated_coordinates=len(space), nuisance_parameters_fixed=False))


def summarize(prepared: Path, fit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery-summary destination')
    truth = {r['case_id']: r for r in v.read(prepared/'truths.json')['rows']}
    frozen = v.read(fit/'nominees.json'); space, _ = v.model_space(frozen['settings'], 'NLL')
    rows = []
    for estimate in frozen['nominees']:
        target = truth[estimate['subject']]
        for parameter, values in space.items():
            actual = target['hyperparams'][parameter]; fitted = estimate['candidate']['hyperparams'][parameter]
            span = max(values)-min(values)
            rows.append(dict(case_id=estimate['subject'], original_subject=target['original_subject'],
                replicate=target['replicate'], model=estimate['model'], parameter=parameter,
                truth=actual, estimate=fitted, error=fitted-actual,
                normalized_absolute_error=abs(fitted-actual)/span if span else 0.,
                grid_absolute_error=abs(values.index(fitted)-values.index(actual)), exact=fitted==actual,
                estimate_at_boundary=fitted in (min(values), max(values)), truth_at_boundary=actual in (min(values), max(values))))
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby(['model','parameter'], sort=False).agg(cases=('case_id','size'),
        normalized_mae=('normalized_absolute_error','mean'), grid_mae=('grid_absolute_error','mean'),
        exact_fraction=('exact','mean'), boundary_fraction=('estimate_at_boundary','mean')).reset_index()
    output.mkdir(parents=True)
    frame.to_csv(output/'parameters.csv',index=False); aggregate.to_csv(output/'aggregate.csv',index=False)
    v.write(output/'completion.json',dict(status='complete',cases=len(truth),coordinates=len(space),
        source_truths_sha256=v.digest(prepared/'truths.json'),nomination_sha256=v.digest(fit/'nominees.json'),
        scientific_status='bounded_joint_recovery_pilot; six_truth_points_with_two_replicates_each; descriptive_not_population_reliability'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','summarize'],required=True)
    for name in ('config','prepared','fit','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.config,args.output)
    else:summarize(args.prepared,args.fit,args.output)


if __name__=='__main__':main()
