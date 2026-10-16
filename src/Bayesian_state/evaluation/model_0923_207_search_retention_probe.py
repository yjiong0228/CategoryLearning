"""Paired scores for the fixed 207 search/retention factorial diagnostic.

Recomputes energy on the entire path bank, preserving cross-chunk path pairs.
The intervention changes one or two fitted coordinates without refitting the
remaining coordinates. Score intervals describe simulation error only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd

from .model_0923_h_validation import digest, load, nll, jackknife_scores, jackknife_se, block_paths
from .model_0923_h_objective_comparison import feature_jackknife
from ..metrics.autonomous_fit import energy_u


def checked(path: Path, expected: str) -> dict:
    if digest(path)!=expected:raise ValueError('Probe source changed')
    with np.load(path) as saved:return {name:saved[name] for name in saved.files}


def factorial_effects(values: dict[str,float], deleted: dict[str,np.ndarray]) -> list[dict]:
    coefficients={'search_average':{'baseline':-.5,'search':.5,'retention':-.5,'both':.5},
                  'retention_average':{'baseline':-.5,'search':-.5,'retention':.5,'both':.5},
                  'interaction':{'baseline':1.,'search':-1.,'retention':-1.,'both':1.}}
    return [dict(effect=effect,value=sum(c*values[k] for k,c in weights.items()),
                 paired_mc_se=jackknife_se(sum(c*deleted[k] for k,c in weights.items())))
            for effect,weights in coefficients.items()]


def summarize(source: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new probe summary directory')
    protocol=load(source/'protocol.json');completion=load(source/'completion.json')
    assert not protocol['smoke'] and completion['status']=='complete'
    assert digest(source/'protocol.json')==completion['protocol_sha256']
    inputs=dict(protocol['reference_sha256'])
    for path,sha in inputs.items():assert digest(Path(path))==sha
    original=[]
    for job in protocol['baseline_filter_jobs']:
        target=Path(job['target']);row=load(target.with_suffix('.json'))
        original.append(checked(target.with_suffix('.npz'),row['predictions_sha256']))
    file=Path(protocol['baseline_generation']);row=load(file.with_suffix('.json'))
    bank={'baseline':{'NLL':{'observed_probability':np.concatenate([a['observed_probability'] for a in original]),
            'seeds':np.concatenate([a['seeds'] for a in original])},'GEN':checked(file,row['predictions_sha256'])}}
    for variant in protocol['points']:
        bank[variant]={}
        for arm in ('NLL','GEN'):
            jobs=sorted([j for j in protocol['jobs'] if Path(j['target']).parent.parent.name==variant and j['model']==arm],key=lambda j:j['name'])
            arrays=[]
            for job in jobs:
                target=Path(job['target']);row=load(target.with_suffix('.json'))
                inputs[str(target.with_suffix('.npz'))]=row['predictions_sha256']
                a=checked(target.with_suffix('.npz'),row['predictions_sha256'])
                np.testing.assert_array_equal(a['seeds'],job['seeds']);arrays.append(a)
            shared={'observed_features','categories'} if arm=='GEN' else set()
            combined={}
            for name in arrays[0]:
                if name in shared:
                    for other in arrays[1:]:np.testing.assert_array_equal(other[name],arrays[0][name])
                    combined[name]=arrays[0][name]
                else:combined[name]=np.concatenate([a[name] for a in arrays])
            np.testing.assert_array_equal(combined['seeds'],bank['baseline'][arm]['seeds'])
            bank[variant][arm]=combined
    scores,curves,paths,effects=[],[],[],[];values={arm:{} for arm in ('NLL','GEN')};deleted={arm:{} for arm in values}
    for variant,arms in bank.items():
        p=arms['NLL']['observed_probability'];g=arms['GEN']
        assert p.shape[0]==8 and len(g['choices'])==1024
        values['NLL'][variant]=nll(p);deleted['NLL'][variant]=jackknife_scores(p)
        values['GEN'][variant]=energy_u(g['features'],g['observed_features'])['value']
        deleted['GEN'][variant]=feature_jackknife(g['features'],g['observed_features'])
        accuracy=(g['choices']==g['categories'][None,:]).astype(float)
        blocks=block_paths(accuracy,0,accuracy.shape[1]);low,high=np.quantile(blocks,[.1,.9],axis=0)
        for b in range(blocks.shape[1]):
            curves.append(dict(variant=variant,block=b+1,mean=blocks[:,b].mean(),q10=low[b],q90=high[b],paths=len(blocks)))
        for i,seed in enumerate(g['seeds']):
            paths.append(dict(variant=variant,seed=int(seed),late_accuracy=accuracy[i,-128:].mean(),whole_accuracy=accuracy[i].mean()))
    for arm in values:
        for variant,value in values[arm].items():
            scores.append(dict(variant=variant,objective=arm,score=value,mc_se=jackknife_se(deleted[arm][variant]),
                delta_baseline=value-values[arm]['baseline'],
                paired_delta_mc_se=jackknife_se(deleted[arm][variant]-deleted[arm]['baseline'])))
        effects.extend(dict(objective=arm,**r) for r in factorial_effects(values[arm],deleted[arm]))
    output.mkdir(parents=True);shutil.copy2(__file__,output/'analysis_source.py')
    for name,rows in [('scores',scores),('curves',curves),('paths',paths),('effects',effects)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    with (output/'completion.json').open('x') as handle:
        json.dump(dict(status='complete',inputs_sha256=inputs,source_sha256=digest(Path(__file__)),
            protocol_sha256=digest(source/'protocol.json'),settings=protocol['settings'],
            interpretation='Fixed-parameter diagnostic; no refitting or held-out mechanism selection',
            variants=4,paths_per_variant=1024,filter_repeats=8,particles=256),handle,indent=2)
    print(pd.DataFrame(scores).to_string(index=False))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();summarize(args.source,args.output)


if __name__=='__main__':main()
