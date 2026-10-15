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
from .model_0923_h_objective_comparison import validate_settings, training_jobs, execute
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


def recovery_aggregate(frame: pd.DataFrame, space: dict) -> pd.DataFrame:
    """Describe error, ordering and variability across independent sequences.

    Rank correlation uses the six generating points after averaging their two
    estimates. It is descriptive; constant estimates yield an undefined rank
    correlation. Replicate gaps combine sequence and search variability.
    """
    from scipy.stats import spearmanr
    aggregate = frame.groupby(['model','parameter'], sort=False).agg(cases=('case_id','size'),
        distinct_truth_values=('truth','nunique'),truth_min=('truth','min'),truth_max=('truth','max'),
        normalized_mae=('normalized_absolute_error','mean'), grid_mae=('grid_absolute_error','mean'),
        exact_fraction=('exact','mean'), boundary_fraction=('estimate_at_boundary','mean'),
        truth_boundary_fraction=('truth_at_boundary','mean')).reset_index()
    descriptions=[]
    for (model,parameter),group in frame.groupby(['model','parameter'],sort=False):
        points=group.groupby('original_subject').agg(truth=('truth','first'),estimate=('estimate','mean'))
        if not (group.groupby('original_subject').truth.nunique()==1).all():
            raise ValueError('Generating point differs across its replicates')
        estimates=group.pivot(index='original_subject',columns='replicate',values='estimate')
        if list(estimates.columns)!=[0,1] or estimates.isna().any().any():
            raise ValueError('Each generating point needs its two independent sequences')
        span=max(space[parameter])-min(space[parameter])
        rank=(float(spearmanr(points.truth,points.estimate).statistic)
              if points.truth.nunique()>1 and points.estimate.nunique()>1 else np.nan)
        descriptions.append(dict(model=model,parameter=parameter,generating_points=len(points),
            spearman_mean_estimate=rank,
            normalized_replicate_gap=float(abs(estimates[1]-estimates[0]).mean()/span) if span else 0.))
    return aggregate.merge(pd.DataFrame(descriptions),on=['model','parameter'],validate='one_to_one')


def recovery_inputs(prepared: Path, fit: Path) -> tuple[dict,dict]:
    """Require a complete paired fit of these exact synthetic cases."""
    truths=v.read(prepared/'truths.json')['rows'];truth={r['case_id']:r for r in truths}
    frozen=v.read(fit/'nominees.json')
    if frozen['smoke'] or frozen['status']!='frozen_before_test':
        raise ValueError('Recovery estimates must be frozen before accessing generating points')
    if Path(frozen['data']).resolve()!=(prepared/'prepared_data').resolve():
        raise ValueError('Recovery fit uses another synthetic dataset')
    expected={(case,arm) for case in truth for arm in ('NLL','GEN')}
    actual=[(r['subject'],r['model']) for r in frozen['nominees']]
    if len(truth)!=len(truths) or len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Recovery requires both objective arms for every unique synthetic case')
    return frozen,truth


def truth_chunks(jobs: list[dict], generation_chunk: int = 128) -> list[dict]:
    """Split independent simulations without changing seeds or state updates."""
    if generation_chunk<3:raise ValueError('Generation chunks need at least three paths')
    chunks=[]
    for job in jobs:
        n=len(job['seeds'])
        if n<(3 if job['model']=='GEN' else 1):raise ValueError('Insufficient independent simulations')
        count=min(max(1,int(np.ceil(n/generation_chunk))),n//3) if job['model']=='GEN' else n
        for index,indices in enumerate(np.array_split(np.arange(n),count)):
            item=deepcopy(job);target=Path(job['target'])
            item.update(seeds=[job['seeds'][i] for i in indices],parent_name=job['name'],
                name=job['name']+f'_C{index:03d}',target=str(target.parent/'chunks'/(target.name+f'_C{index:03d}')))
            chunks.append(item)
    return chunks


def combine_truth_chunks(job: dict, chunks: list[dict]) -> None:
    """Pool complete independent simulations before computing either objective."""
    from ...metrics.autonomous_fit import energy_u
    arrays=[];records=[]
    for chunk in chunks:
        file=Path(chunk['target']).with_suffix('.npz');record=v.read(file.with_suffix('.json'))
        if v.digest(file)!=record['predictions_sha256']:raise ValueError('Truth-score chunk changed')
        if any(pool['num_threads']!=1 for pool in record['numeric_pools']):raise ValueError('Numeric thread budget exceeded')
        records.append(record)
        with np.load(file) as saved:arrays.append({k:saved[k] for k in saved.files})
    shared={'categories','observed_features'} if job['model']=='GEN' else set()
    combined={}
    for key in arrays[0]:
        if key in shared:
            for other in arrays[1:]:np.testing.assert_array_equal(arrays[0][key],other[key])
            combined[key]=arrays[0][key]
        else:combined[key]=np.concatenate([part[key] for part in arrays],axis=0)
    np.testing.assert_array_equal(combined['seeds'],job['seeds'])
    target=Path(job['target']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:np.savez_compressed(handle,**combined)
    record=deepcopy(records[0])
    record.update(predictions_sha256=v.digest(target.with_suffix('.npz')),
        cpu_seconds=sum(r['cpu_seconds'] for r in records),seconds=sum(r['seconds'] for r in records),
        chunk_seconds_definition='sum of individual chunk wall times',chunk_files=[c['target'] for c in chunks])
    if job['model']=='GEN':
        score=energy_u(combined['features'],combined['observed_features'])
        record.update(paths=len(job['seeds']),state_updates=int(combined['choices'].size),
            objective_score=score['value'],objective_mc_se=score['mc_se'])
    else:record.update(seeds=job['seeds'],mean_nll=v.pooled_nll(combined['observed_probability']))
    v.write(target.with_suffix('.json'),record)


def summarize(prepared: Path, fit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery-summary destination')
    frozen,truth=recovery_inputs(prepared,fit)
    space, _ = v.model_space(frozen['settings'], 'NLL')
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
    aggregate = recovery_aggregate(frame,space)
    output.mkdir(parents=True)
    shutil.copy2(__file__,output/'analysis_source.py')
    frame.to_csv(output/'parameters.csv',index=False); aggregate.to_csv(output/'aggregate.csv',index=False)
    v.write(output/'completion.json',dict(status='complete',cases=len(truth),coordinates=len(space),
        source_sha256=v.digest(Path(__file__)),
        source_truths_sha256=v.digest(prepared/'truths.json'),nomination_sha256=v.digest(fit/'nominees.json'),
        scientific_status='bounded_joint_recovery_pilot; six_truth_points_with_two_replicates_each; descriptive_not_population_reliability'))


def score_truths(prepared: Path, fit: Path, output: Path) -> None:
    """Diagnose bounded search after estimates freeze; never alter nominees.

    Each true point receives the same confirmation seeds as its corresponding
    estimate. A better true-point score flags search limitations. A worse score
    alone cannot establish identifiability from one finite synthetic sequence.
    """
    from ...optimization.search.adaptive_proposals import candidate
    from ...metrics.autonomous_fit import energy_u
    from ...evaluation.model_0923_h_validation import nll, jackknife_scores, jackknife_se
    from ...evaluation.model_0923_h_objective_comparison import feature_jackknife
    if output.exists():
        raise FileExistsError('Use a new recovery-truth diagnostic destination')
    frozen,truth=recovery_inputs(prepared,fit)
    banks = {(case, arm): [candidate(row['hyperparams'], 'postfit_recovery_truth')]
             for case, row in truth.items() for arm in ('NLL', 'GEN')}
    output.mkdir(parents=True)
    shutil.copy2(__file__,output/'diagnostic_source.py')
    v.write(output/'protocol.json',dict(nomination_sha256=v.digest(fit/'nominees.json'),
        truths_sha256=v.digest(prepared/'truths.json'),source_sha256=v.digest(Path(__file__)),
        parallel_units='one complete NLL filter seed or up to 128 GEN paths; pooled before scoring',
        role='postfit_truth_score_diagnostic; estimates_and_search_budget_unchanged'))
    jobs = training_jobs(v,Path(frozen['data']),output,frozen['settings'],banks,'confirmation',False)
    chunks=truth_chunks(jobs)
    execute(chunks,output,frozen['settings']['parallel_budget'])
    for job in jobs:combine_truth_chunks(job,[c for c in chunks if c['parent_name']==job['name']])
    rows=[]
    for nominee in frozen['nominees']:
        case,arm=nominee['subject'],nominee['model']
        true_id=banks[(case,arm)][0]['id'];estimate_id=nominee['candidate']['id']
        arrays=[]
        for folder,point in [(output,true_id),(fit,estimate_id)]:
            file=folder/'confirmation'/f'subject_{case}'/arm/(point+'.npz')
            record=v.read(file.with_suffix('.json'))
            if v.digest(file)!=record['predictions_sha256']:raise ValueError('Recovery score changed')
            with np.load(file) as saved:arrays.append({k:saved[k] for k in saved.files})
        actual,estimate=arrays
        np.testing.assert_array_equal(actual['seeds'],estimate['seeds'])
        if arm=='NLL':
            left,right=nll(estimate['observed_probability']),nll(actual['observed_probability'])
            se=jackknife_se(jackknife_scores(estimate['observed_probability'])-jackknife_scores(actual['observed_probability']))
        else:
            np.testing.assert_array_equal(actual['observed_features'],estimate['observed_features'])
            target=actual['observed_features']
            left=energy_u(estimate['features'],target)['value'];right=energy_u(actual['features'],target)['value']
            se=jackknife_se(feature_jackknife(estimate['features'],target)-feature_jackknife(actual['features'],target))
        rows.append(dict(case_id=case,original_subject=truth[case]['original_subject'],replicate=truth[case]['replicate'],
            model=arm,estimated_point_score=left,generating_point_score=right,
            estimate_minus_truth=left-right,paired_mc_se=se))
    pd.DataFrame(rows).to_csv(output/'truth_scores.csv',index=False)
    v.write(output/'completion.json',dict(status='complete',cases=len(truth),scores=len(rows),
        numerical_uncertainty='paired confirmation-seed or complete-path jackknife',
        interpretation='A positive estimate-minus-truth gap diagnoses bounded-search loss; finite-sequence variation also affects parameter recovery'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','summarize','score-truths'],required=True)
    for name in ('config','prepared','fit','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.config,args.output)
    elif args.phase=='summarize':summarize(args.prepared,args.fit,args.output)
    else:score_truths(args.prepared,args.fit,args.output)


if __name__=='__main__':main()
