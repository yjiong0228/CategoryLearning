"""Describe choice sharpness on saved continuation states, without new paths.

Use --summary COMPLETE_SUMMARY --output NEW --power POWER. This exploratory
diagnostic compares original probabilities, normalized powers and uniform-tie
argmax on exactly the same generated states. It measures immediate prediction
headroom only: feedback, learning and parameter likelihoods are not recomputed.
The transformation is not evidence that a new dynamic model has been validated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from ..workflows.analysis.model_1023_prefix_forecast import read,write,digest,verify


def readout_probabilities(probabilities: np.ndarray, power: float) -> tuple[np.ndarray,np.ndarray]:
    """Apply declared sharpness and a uniform tie convention to valid rows."""
    p=np.asarray(probabilities,float)
    if not np.isfinite(power) or power<=0 or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Need finite probabilities and positive sharpness')
    # All-zero rows encode an absent target rule and remain zero.
    active=p.sum(-1)>0
    np.testing.assert_allclose(p.sum(-1)[active],1.,atol=1e-12,rtol=0)
    maximum=p.max(-1,keepdims=True)
    relative=np.divide(p,maximum,out=np.zeros_like(p),where=maximum>0)
    powered=relative**power
    sharp=np.divide(powered,powered.sum(-1,keepdims=True),out=np.zeros_like(p),where=powered.sum(-1,keepdims=True)>0)
    ties=np.isclose(p,p.max(-1,keepdims=True),atol=1e-14,rtol=0)&active[...,None]
    hard=np.divide(ties,ties.sum(-1,keepdims=True),out=np.zeros_like(p),where=ties.sum(-1,keepdims=True)>0)
    return sharp,hard


def run(summary: Path,output: Path,power: float) -> None:
    if output.exists():raise FileExistsError('Use a new fixed-state probe directory')
    done=read(summary/'completion.json')
    if done['status']!='complete':raise ValueError('Need completed forecasts')
    verify({str(summary/k):v for k,v in done['artifacts_sha256'].items()})
    meta=pd.read_csv(summary/'scores.csv').query('start=="filtered_prefix"')
    rows=[]
    for row in meta.itertuples():
        with np.load(summary/'arrays'/f'S{row.subject}_TRANSFER_{row.candidate}.npz') as z:
            categories=z['categories'][-128:]
            for basis,key in [('full_mixture','probabilities'),('target_if_present','diagnostic_target_probabilities')]:
                p=z[key][:,-128:]
                mask=p.sum(-1)>0
                sharp,hard=readout_probabilities(p,power)
                for label,probability in [('original',p),('power',sharp),('argmax',hard)]:
                    correct=np.take_along_axis(probability,(categories-1)[None,:,None],axis=2)[...,0]
                    rows.append(dict(subject=row.subject,candidate=row.candidate,roles=row.roles,basis=basis,
                        transformation=label,power=power if label=='power' else np.nan,
                        expected_accuracy=float(correct[mask].mean()) if mask.any() else np.nan,
                        included_path_trials=int(mask.sum()),total_path_trials=int(mask.size),observed_accuracy=row.late_observed))
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'fixed_state_readout.csv',index=False)
    write(output/'completion.json',dict(status='complete',summary=str(summary.resolve()),power=power,
        input_summary_sha256=digest(summary/'completion.json'),source_sha256=digest(Path(__file__)),
        outputs_sha256={'fixed_state_readout.csv':digest(output/'fixed_state_readout.csv')},
        scope='Exploratory instantaneous prediction at saved states; no generated choices, new filters or refits',
        limitation='Changed future feedback and state evolution are not simulated; argmax is a diagnostic limit'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--power',type=float,required=True)
    args=parser.parse_args();run(args.summary,args.output,args.power)
