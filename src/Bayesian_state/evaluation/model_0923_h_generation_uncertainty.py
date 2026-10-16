"""Whole-path Monte Carlo error for frozen H generation comparisons.

Delete one complete autonomous path at a time, preserving dependence across
trials and blocks. Common trajectory seeds pair model or parameter comparisons.
Pair-distance sums evaluate every deletion in quadratic time, preserving the
empirical-distribution Energy and CRPS definitions used by the full scores.
The resulting standard errors measure finite simulation, not population or
parameter uncertainty. Existing predictions and summaries remain unchanged.

Use --summary SUMMARY --output NEW_OUTPUT for fitted predictions, optionally
adding --finalists FINALISTS. The reference follows the summary manifest or
--reference-model. Use --screen SCREEN --scores SCORES --output
NEW_OUTPUT for fixed-parameter intervention paths and their existing scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .model_0923_h_validation import load,digest,block_paths,distribution_scores,jackknife_se


def path_jackknife(samples: np.ndarray, observed: np.ndarray) -> dict[str,np.ndarray]:
    """Delete each path by removing its target term and both pair-distance rows."""
    x,y=np.asarray(samples,float),np.asarray(observed,float)
    if (x.ndim!=2 or len(x)<2 or x.shape[1]<1 or y.shape!=(x.shape[1],)
        or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Need multiple finite paths and aligned observed blocks')
    remaining=len(x)-1;result={}
    for key,metric,normalizer in [('energy','euclidean',np.sqrt(x.shape[1])),('crps','cityblock',x.shape[1])]:
        target=np.linalg.norm(x-y,axis=1) if key=='energy' else np.abs(x-y).sum(axis=1)
        pairs=cdist(x,x,metric=metric);row_sums=pairs.sum(axis=1)
        # The diagonal is zero; deleting a path removes its row and column.
        result[key]=((target.sum()-target)/remaining-
            .5*(row_sums.sum()-2*row_sums)/remaining**2)/normalizer
    return result


def run(summary: Path, finalists: Path | None, output: Path, reference_model: str | None = None) -> None:
    if output.exists():raise FileExistsError('Use a new simulation-uncertainty destination')
    manifest=load(summary/'summary_manifest.json');fit=Path(manifest['fit']);frozen=load(fit/'nominees.json')
    generated=Path(manifest['autonomous']);conditional=Path(manifest['conditional'])
    reference_model=reference_model or manifest.get('reference_model','B')
    if finalists is not None:
        if load(finalists/'protocol.json')['nomination_sha256']!=manifest['nomination_sha256']:
            raise ValueError('Finalists and primary predictions use different nominees')
        if load(finalists/'completion.json')['status']!='complete':raise ValueError('Finalist generation is incomplete')
    targets={};cuts={};primary={};candidate_results={}
    for row in load(conditional/'evaluation.json')['rows']:
        if row['model']!=reference_model:continue
        with np.load(conditional/'evaluate'/f'subject_{row["subject"]}'/reference_model/(row['candidate']['id']+'.npz')) as z:
            targets[row['subject']]=(z['choices']==z['categories']).astype(float)
        cuts[row['subject']]=row['training_trials']
    def read_generated(source: Path, row: dict) -> dict:
        file=source/'evaluate'/f'subject_{row["subject"]}'/row['model']/(row['candidate']['id']+'.npz')
        if digest(file)!=row['predictions_sha256']:raise ValueError('Generated paths changed')
        with np.load(file) as z:seeds=z['seeds'];realized=(z['choices']==z['categories'][None,:]).astype(float)
        subject=row['subject'];first=cuts[subject];last=len(targets[subject])
        samples=block_paths(realized,first,last);observed=block_paths(targets[subject][None,:],first,last)[0]
        return dict(seeds=seeds,metrics=distribution_scores(samples,observed),jackknife=path_jackknife(samples,observed))
    for row in load(generated/'evaluation.json')['rows']:
        primary[(row['subject'],row['model'])]=read_generated(generated,row)
    if finalists is not None:
        for row in load(finalists/'generation.json')['rows']:
            candidate_results[(row['subject'],row['model'],row['candidate']['id'])]=read_generated(finalists,row)
    primary_rows=[];candidate_rows=[]
    for (subject,model),row in primary.items():
        baseline=primary[(subject,reference_model)];np.testing.assert_array_equal(row['seeds'],baseline['seeds'])
        result=dict(subject=subject,model=model,reference=reference_model,paths=len(row['seeds']))
        for metric in ['energy','crps']:
            result.update({metric:row['metrics'][metric],metric+'_mc_se':jackknife_se(row['jackknife'][metric]),
                'delta_'+metric:row['metrics'][metric]-baseline['metrics'][metric],
                'delta_'+metric+'_mc_se':jackknife_se(row['jackknife'][metric]-baseline['jackknife'][metric])})
        primary_rows.append(result)
    anchors={(r['subject'],r['model']):r['candidate']['id'] for r in frozen['nominees']}
    for (subject,model,candidate),row in candidate_results.items():
        anchor=candidate_results[(subject,model,anchors[(subject,model)])]
        np.testing.assert_array_equal(row['seeds'],anchor['seeds'])
        result=dict(subject=subject,model=model,candidate=candidate,reference_candidate=anchors[(subject,model)],paths=len(row['seeds']))
        for metric in ['energy','crps']:
            result.update({metric:row['metrics'][metric],metric+'_mc_se':jackknife_se(row['jackknife'][metric]),
                'delta_'+metric:row['metrics'][metric]-anchor['metrics'][metric],
                'delta_'+metric+'_mc_se':jackknife_se(row['jackknife'][metric]-anchor['jackknife'][metric])})
        candidate_rows.append(result)
    output.mkdir(parents=True);p=pd.DataFrame(primary_rows);c=pd.DataFrame(candidate_rows)
    # Independently computed metrics must reproduce the published summaries.
    expected=pd.read_csv(summary/'scores.csv');expected=expected[expected.phase=='test'].set_index(['subject','model'])
    np.testing.assert_allclose(p.set_index(['subject','model']).energy.reindex(expected.index),expected.energy,atol=1e-12,rtol=0)
    if finalists is not None:
        original=pd.read_csv(finalists/'candidate_scores.csv');original=original[original.phase=='test'].set_index(['subject','model','candidate'])
        np.testing.assert_allclose(c.set_index(['subject','model','candidate']).energy.reindex(original.index),original.energy,atol=1e-12,rtol=0)
        c.to_csv(output/'finalists.csv',index=False)
    p.to_csv(output/'primary.csv',index=False)
    aggregate=[]
    for model,group in p.groupby('model',sort=False):
        aggregate.append(dict(model=model,delta_energy=group.delta_energy.mean(),
            delta_energy_mc_se=float(np.linalg.norm(group.delta_energy_mc_se)/len(group)),
            delta_crps=group.delta_crps.mean(),delta_crps_mc_se=float(np.linalg.norm(group.delta_crps_mc_se)/len(group))))
    pd.DataFrame(aggregate).to_csv(output/'aggregate.csv',index=False)
    import json
    with (output/'completion.json').open('x') as f:json.dump(dict(status='complete',source_sha256=digest(Path(__file__)),
        summary=str(summary.resolve()),reference_model=reference_model,finalists=str(finalists.resolve()) if finalists is not None else None,nomination_sha256=manifest['nomination_sha256'],
        interpretation='whole-path jackknife Monte Carlo standard errors; parameter points fixed; participant population uncertainty excluded'),f,indent=2)


def screen_comparisons(source: Path, scores: Path, output: Path) -> None:
    """Pair complete intervention paths for every baseline/configured contrast."""
    import json
    if output.exists():raise FileExistsError('Use a new screen-uncertainty destination')
    manifest=load(source/'simulation_manifest.json');scored=load(scores/'scores_manifest.json')
    if manifest['status']!='complete' or manifest['smoke'] or scored['status']!='complete':
        raise ValueError('Need a complete scientific screen and scores')
    if Path(scored['source']).resolve()!=source.resolve() or digest(source/'simulation_manifest.json')!=scored['manifest_sha256']:
        raise ValueError('Scores belong to a different screen')
    expected=pd.read_csv(scores/'scores.csv');expected=expected[expected.seed_group=='all'].set_index(['subject','model'])
    variants=list(manifest['settings']['variants'])
    pairs=list(dict.fromkeys([(name,'B') for name in variants if name!='B']+
        [tuple(pair) for pair in manifest['settings'].get('comparisons',[])]))
    rows=[];observed_hashes={}
    for record in manifest['records']:
        directory=source/record['id'];frame=pd.read_csv(directory/'observed_trials.csv')
        observed_hashes[record['id']]=digest(directory/'observed_trials.csv')
        observed=(frame.choice==frame.category).to_numpy(dtype=float)
        target=block_paths(observed[None,:],0,len(observed))[0];group={}
        for model in variants:
            file=directory/model/'correct_paths.npz'
            if digest(file)!=scored['artifacts_sha256'][str(file.relative_to(source))]:
                raise ValueError('Scored intervention paths changed')
            with np.load(file) as z:
                seeds=z['seeds'];samples=block_paths(z['realized'].T,0,len(observed))
            if len(seeds)!=len(samples) or len(set(seeds))!=len(seeds):
                raise ValueError('Need one distinct seed per complete path')
            metrics=distribution_scores(samples,target)
            for metric in ('energy','crps'):
                np.testing.assert_allclose(metrics[metric],expected.loc[(record['subject'],model),metric],atol=1e-12,rtol=0)
            group[model]=dict(seeds=seeds,metrics=metrics,jackknife=path_jackknife(samples,target))
        for model,reference in pairs:
            left,right=group[model],group[reference]
            np.testing.assert_array_equal(left['seeds'],right['seeds'])
            row=dict(subject=record['subject'],model=model,reference=reference,paths=len(left['seeds']))
            for metric in ('energy','crps'):
                row['delta_'+metric]=left['metrics'][metric]-right['metrics'][metric]
                row['delta_'+metric+'_mc_se']=jackknife_se(left['jackknife'][metric]-right['jackknife'][metric])
            rows.append(row)
    frame=pd.DataFrame(rows);aggregate=[]
    for (model,reference),group in frame.groupby(['model','reference'],sort=False):
        row=dict(model=model,reference=reference,participants=len(group))
        for metric in ('energy','crps'):
            row['delta_'+metric]=float(group['delta_'+metric].mean())
            row['delta_'+metric+'_mc_se']=float(np.linalg.norm(group['delta_'+metric+'_mc_se'])/len(group))
        aggregate.append(row)
    output.mkdir(parents=True)
    frame.to_csv(output/'participants.csv',index=False);pd.DataFrame(aggregate).to_csv(output/'aggregate.csv',index=False)
    with (output/'completion.json').open('x') as f:
        json.dump(dict(status='complete',source=str(source.resolve()),scores=str(scores.resolve()),
            source_sha256=digest(Path(__file__)),scores_sha256=digest(scores/'scores.csv'),
            observed_sha256=observed_hashes,contrasts=pairs,
            interpretation='paired whole-path Monte Carlo error; historical full-sequence fitted parameters fixed; development comparison'),f,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['summary','finalists','screen','scores']:p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-model')
    args=p.parse_args()
    if args.screen and args.scores and not args.summary and not args.finalists:
        screen_comparisons(args.screen,args.scores,args.output)
    elif args.summary and not args.screen and not args.scores:
        run(args.summary,args.finalists,args.output,args.reference_model)
    else:p.error('Supply --summary [--finalists], or --screen and --scores')


if __name__=='__main__':main()
