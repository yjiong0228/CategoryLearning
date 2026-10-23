"""Decompose expected-accuracy changes using existing paired dynamic paths.

Run --source COMPLETE_DYNAMIC_SUMMARY --output NEW. At fixed parameters,
the difference between Q2 and Q1 expected accuracy equals the immediate
readout change on Q2 states plus the change in raw predictions across state
distributions. Prefix comparisons also include different inferred start states.
This descriptive identity does not decompose Energy scores or identify separate
psychological causes. No filters, trajectories or parameter estimates are run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_1023_response_dynamics import read,write,digest,verify,clustered_mean_se


def accuracy_components(original: np.ndarray,squared: np.ndarray,categories: np.ndarray) -> dict:
    """Compare the two readouts on Q2 states, then their raw state predictions."""
    a,b=np.asarray(original,dtype=float),np.asarray(squared,dtype=float)
    labels=np.asarray(categories)
    if a.ndim!=3 or a.shape!=b.shape or labels.shape!=(a.shape[1],):
        raise ValueError('Require matching path/trial/category arrays')
    if not np.issubdtype(labels.dtype,np.integer) or np.any((labels<1)|(labels>a.shape[-1])):
        raise ValueError('Task categories must be one-based integer labels')
    for p in (a,b):
        if not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid probabilities')
        np.testing.assert_allclose(p.sum(-1),1.,atol=1e-12,rtol=0)
    # Invert q_k=p_k^2/sum(p^2). This also preserves exact zero probabilities.
    raw=np.sqrt(b);raw/=raw.sum(-1,keepdims=True)
    forward=raw**2;forward/=forward.sum(-1,keepdims=True)
    np.testing.assert_allclose(forward,b,atol=1e-12,rtol=0)
    take=lambda p:np.take_along_axis(p,labels[None,:,None]-1,axis=-1)[...,0]
    p1,p2,q2=take(a),take(raw),take(b)
    result=dict(original_expected=p1,squared_raw_expected=p2,squared_expected=q2,
        readout_change=q2-p2,state_prediction_change=p2-p1,total_change=q2-p1)
    np.testing.assert_allclose(result['total_change'],result['readout_change']+result['state_prediction_change'],
                               atol=1e-12,rtol=0)
    return result


def summarize(source: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new probability decomposition directory')
    done=read(source/'completion.json')
    if done['status']!='complete' or done['participants']!=6 or done['candidates']!=18:
        raise ValueError('Require the complete registered dynamic comparison')
    verify(done['inputs_sha256']);inputs={}

    def bind(file,expected=None):
        p=Path(file).resolve();sha=digest(p)
        if expected is not None and sha!=expected:raise ValueError('Dynamic input changed: '+str(p))
        inputs[str(p)]=sha;return p

    bind(source/'completion.json');bind(Path(__file__))
    protocol=read(bind(Path(done['study'])/'protocol.json'))
    width=protocol['settings']['block_width'];late=protocol['settings']['late_trials']
    rows,paths,curves=[],[],[];prefix_checks=0
    for case in protocol['cases']:
        own=Path(case['evaluation'])
        behavior,=read(bind(own/'conditional/evaluation.json'))['rows'];cut=behavior['training_trials']
        tag=dict(subject=case['subject'],condition=case['condition'],candidate=case['candidate'],
            selected='selected' in case['roles'],roles='|'.join(case['roles']),cohort=case['cohort'])
        for start in ('task_start','prefix'):
            pair=[]
            for exponent in (1,2):
                name=f"arrays/{case['case_id']}_{start}_Q{exponent}.npz"
                with np.load(bind(source/name,done['artifacts_sha256'][name])) as z:
                    keys=['probabilities','categories','choices','seeds' if start=='task_start' else 'branch_seeds']
                    if start=='prefix':
                        keys.append('branch_filter_seed')
                        if exponent==2:keys.append('diagnostic_cognitive_probabilities')
                    pair.append({k:z[k].copy() for k in keys})
            a,b=pair;seedkey='seeds' if start=='task_start' else 'branch_seeds'
            np.testing.assert_array_equal(a['categories'],b['categories'])
            np.testing.assert_array_equal(a[seedkey],b[seedkey])
            labels=a[seedkey] if start=='task_start' else a['branch_filter_seed']
            if start=='prefix':
                np.testing.assert_array_equal(labels,b['branch_filter_seed'])
                raw=np.sqrt(b['probabilities']);raw/=raw.sum(-1,keepdims=True)
                np.testing.assert_allclose(raw,b['diagnostic_cognitive_probabilities'],atol=1e-12,rtol=0)
                prefix_checks+=raw.shape[0]*raw.shape[1]
            values=accuracy_components(a['probabilities'],b['probabilities'],a['categories'])
            length=len(a['categories']);offset=0 if start=='task_start' else cut
            windows={'whole':(0,length),'last128':(max(0,length-late),length)}
            if start=='task_start':windows.update(first128=(0,min(late,length)),training=(0,cut),test=(cut,length))
            for window,(first,last) in windows.items():
                means={key:v[:,first:last].mean(1) for key,v in values.items()}
                row=dict(tag,start=start,window=window,first_trial=offset+first+1,last_trial=offset+last,
                         paths=len(labels),clusters=len(np.unique(labels)))
                for key,v in means.items():row.update({key:v.mean(),key+'_mc_se':clustered_mean_se(v,labels)})
                sampled=(b['choices'][:,first:last]==b['categories'][None,first:last]).mean(1)-\
                        (a['choices'][:,first:last]==a['categories'][None,first:last]).mean(1)
                row['sampled_accuracy_change']=sampled.mean()
                rows.append(row)
                paths.extend(dict(tag,start=start,window=window,seed=int(seed),cluster=int(label),
                    **{key:float(v[i]) for key,v in means.items()}) for i,(seed,label) in enumerate(zip(a[seedkey],labels)))
            for first in range(0,length,width):
                last=min(first+width,length)
                curves.append(dict(tag,start=start,trial=offset+last,
                    **{key:float(v[:,first:last].mean()) for key,v in values.items()}))
    output.mkdir(parents=True)
    for name,data in [('windows',rows),('path_windows',paths),('curves',curves)]:
        pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    verify(inputs)
    write(output/'completion.json',dict(status='complete',source=str(source.resolve()),candidates=18,
        participants=6,numeric_jobs=0,independent_raw_probability_checks=prefix_checks,inputs_sha256=inputs,
        identity='Q2 minus Q1 = readout change on Q2 states + raw prediction change across state distributions',
        scope='Descriptive expected-accuracy decomposition at fixed parameters; prefix includes inferred-start differences',
        uncertainty='Paired trajectory or prefix-filter cluster Monte Carlo error',
        artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.source,args.output)
