"""Whole-path Monte Carlo error for frozen H generation comparisons.

Delete one complete autonomous path at a time, preserving dependence across
trials and blocks. Common trajectory seeds pair model or parameter comparisons.
The resulting standard errors measure finite simulation, not population or
parameter uncertainty. Existing predictions and summaries remain unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import load,digest,block_paths,distribution_scores,jackknife_se


def path_jackknife(samples: np.ndarray, observed: np.ndarray) -> dict[str,np.ndarray]:
    if len(samples)<2:raise ValueError('Need multiple complete generated paths')
    scores=[distribution_scores(np.delete(samples,index,axis=0),observed) for index in range(len(samples))]
    return {key:np.array([row[key] for row in scores]) for key in ['energy','crps']}


def run(summary: Path, finalists: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new simulation-uncertainty destination')
    manifest=load(summary/'summary_manifest.json');fit=Path(manifest['fit']);frozen=load(fit/'nominees.json')
    generated=Path(manifest['autonomous']);conditional=Path(manifest['conditional'])
    if load(finalists/'protocol.json')['nomination_sha256']!=manifest['nomination_sha256']:
        raise ValueError('Finalists and primary predictions use different nominees')
    if load(finalists/'completion.json')['status']!='complete':raise ValueError('Finalist generation is incomplete')
    targets={};cuts={};primary={};candidate_results={}
    for row in load(conditional/'evaluation.json')['rows']:
        if row['model']!='B':continue
        with np.load(conditional/'evaluate'/f'subject_{row["subject"]}'/'B'/(row['candidate']['id']+'.npz')) as z:
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
    for row in load(finalists/'generation.json')['rows']:
        candidate_results[(row['subject'],row['model'],row['candidate']['id'])]=read_generated(finalists,row)
    primary_rows=[];candidate_rows=[]
    for (subject,model),row in primary.items():
        baseline=primary[(subject,'B')];np.testing.assert_array_equal(row['seeds'],baseline['seeds'])
        result=dict(subject=subject,model=model,reference='B',paths=len(row['seeds']))
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
    original=pd.read_csv(finalists/'candidate_scores.csv');original=original[original.phase=='test'].set_index(['subject','model','candidate'])
    np.testing.assert_allclose(c.set_index(['subject','model','candidate']).energy.reindex(original.index),original.energy,atol=1e-12,rtol=0)
    p.to_csv(output/'primary.csv',index=False);c.to_csv(output/'finalists.csv',index=False)
    aggregate=[]
    for model,group in p.groupby('model',sort=False):
        aggregate.append(dict(model=model,delta_energy=group.delta_energy.mean(),
            delta_energy_mc_se=float(np.linalg.norm(group.delta_energy_mc_se)/len(group)),
            delta_crps=group.delta_crps.mean(),delta_crps_mc_se=float(np.linalg.norm(group.delta_crps_mc_se)/len(group))))
    pd.DataFrame(aggregate).to_csv(output/'aggregate.csv',index=False)
    import json
    with (output/'completion.json').open('x') as f:json.dump(dict(status='complete',source_sha256=digest(Path(__file__)),
        summary=str(summary.resolve()),finalists=str(finalists.resolve()),nomination_sha256=manifest['nomination_sha256'],
        interpretation='whole-path jackknife Monte Carlo standard errors; parameter points fixed; participant population uncertainty excluded'),f,indent=2)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['summary','finalists','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run(args.summary,args.finalists,args.output)


if __name__=='__main__':main()
