"""Behavior-first tests of the three learning narratives in the confirmed abstract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .bottleneck_analysis import ROOT, sha256
from src.Bayesian_state.hypothesis_space.analysis.oral_evidence import partition_inventory

CONFIG = {'minimum_segment': 64, 'shape_margin': 6., 'early_trials': 128,
          'event_window': 32, 'high_belief': .5, 'high_duration': 16,
          'low_belief': .25, 'low_duration': 8, 'dpi': 450}


def behavior_shapes(correct: np.ndarray, minimum: int = 64,
                    keep: np.ndarray | None = None) -> tuple[dict, dict]:
    """Bernoulli constant, nondecreasing logit trend, and unknown upward step.

    The optional mask holds original time coordinates fixed for block-deletion
    sensitivity. These are descriptive shape comparisons, not cognition models.
    """
    y = np.asarray(correct, dtype=float)
    if y.ndim != 1 or not np.isin(y, [0., 1.]).all() or len(y) < minimum*2:
        raise ValueError('Need a valid binary series and two complete segments')
    mask = np.ones(len(y), bool) if keep is None else np.asarray(keep, bool)
    if mask.shape != y.shape:
        raise ValueError('Shape mismatch')
    x = np.linspace(-.5, .5, len(y))
    yy, xx = y[mask], x[mask]
    n = len(yy)
    if n < 2*minimum:
        raise ValueError('Need two complete observed segments after masking')
    def objective(v):
        z = v[0] + v[1]*xx
        return float(np.sum(np.logaddexp(0., z) - yy*z))
    runs = [minimize(objective, start, method='L-BFGS-B', bounds=[(-12.,12.),(0.,24.)])
            for start in ([0.,1.],[1.,4.],[-1.,8.])]
    best = min(runs, key=lambda r: r.fun)
    if not np.isfinite(best.fun) or not any(r.success for r in runs):
        raise ValueError('Behavioral trend optimization failed')
    successes = np.r_[0., np.cumsum(y*mask)]
    counts = np.r_[0, np.cumsum(mask)]
    splits = np.arange(minimum, len(y)-minimum+1)
    left_n, right_n = counts[splits], n-counts[splits]
    good = (left_n >= minimum) & (right_n >= minimum)
    if not good.any():
        raise ValueError('No sufficiently observed split')
    splits, left_n, right_n = splits[good],left_n[good],right_n[good]
    left_y, right_y = successes[splits],successes[-1]-successes[splits]
    left = np.clip(left_y/left_n, 1e-9, 1-1e-9)
    right = np.clip(right_y/right_n, 1e-9, 1-1e-9)
    loss = -left_y*np.log(left)-(left_n-left_y)*np.log1p(-left)
    loss += -right_y*np.log(right)-(right_n-right_y)*np.log1p(-right)
    upward = right >= left
    if not upward.any():
        split, step_loss, lo, hi = np.nan, np.inf, np.nan, np.nan
        step = np.full(len(y),np.nan)
    else:
        j = int(np.argmin(np.where(upward,loss,np.inf)))
        split,step_loss,lo,hi=int(splits[j]),float(loss[j]),float(left[j]),float(right[j])
        step=np.where(np.arange(len(y))<split,lo,hi)
    mean=float(np.clip(yy.mean(),1e-9,1-1e-9))
    constant_loss=-np.sum(yy*np.log(mean)+(1-yy)*np.log1p(-mean))
    bics={'constant':2*constant_loss+np.log(n),'trend':2*best.fun+2*np.log(n),
          'step':2*step_loss+3*np.log(n)}
    row={f'bic_{k}':float(v) for k,v in bics.items()}
    row.update(delta_bic=float(bics['trend']-bics['step']),split=split,
               step_before=lo,step_after=hi,preferred=min(bics,key=bics.get),
               trend_intercept=float(best.x[0]),trend_slope=float(best.x[1]))
    return row, {'trend':expit(best.x[0]+best.x[1]*x),'step':step,
                 'constant':np.full(len(y),mean)}


def support_reversals(q: np.ndarray, high_duration: int = 16, low_duration: int = 8) -> list[tuple[int,int]]:
    """Count a sustained loss once per rebuilt high-support episode (one-based)."""
    high_run=low_run=0
    supported=False
    origin=None
    events=[]
    for i,value in enumerate(np.asarray(q,float)):
        if not np.isfinite(value):
            high_run=low_run=0
            continue
        high_run=high_run+1 if value>.5 else 0
        low_run=low_run+1 if value<.25 else 0
        if not supported and high_run>=high_duration:
            supported=True
            origin=i-high_duration+2
        if supported and low_run>=low_duration:
            events.append((int(origin),i-low_duration+2))
            supported=False
    return events


def select_examples(summary: pd.DataFrame) -> dict[str,int]:
    eligible=summary.loc[summary.criterion.notna()]
    rapid=int(eligible.sort_values(['criterion','subject']).iloc[0].subject)
    gradual=eligible.loc[(eligible.delta_bic<=-CONFIG['shape_margin']) & (eligible.preferred!='constant')]
    abrupt=eligible.loc[(eligible.delta_bic>=CONFIG['shape_margin']) & (eligible.preferred=='step')]
    if gradual.empty or abrupt.empty:
        raise ValueError('Observed cohort does not contain all three illustration candidates')
    return {'rapid':rapid,
            'gradual':int(gradual.sort_values(['criterion','subject']).iloc[-1].subject),
            'abrupt':int(abrupt.sort_values(['criterion','subject']).iloc[-1].subject)}


def build(source: Path, states: Path, output: Path) -> None:
    source,states=source.resolve(),states.resolve()
    output.mkdir(parents=True,exist_ok=False)
    d=pd.read_csv(source/'trials.csv')
    people=pd.read_csv(source/'subjects.csv').set_index('subject')
    inventory,_=partition_inventory()
    summaries=[];trials=[];jumps=[];reversals=[];sensitivity=[];shape_checks=[]
    inputs=[source/'trials.csv',source/'subjects.csv',Path(__file__),Path(__file__).with_name('abstract_story_design.md')]
    for sid,g in d.groupby('iSub',sort=True):
        g=g.copy().reset_index(drop=True);meta=people.loc[sid];condition=int(meta.condition)
        shape,curves=behavior_shapes(g.correct.to_numpy())
        for name,curve in curves.items():g['behavior_'+name]=curve
        runs=[];alt=[]
        for variant,count,destination in [('selected',8,runs),('alternative',4,alt)]:
            for repeat in range(count):
                path=states/f'S{sid}'/f'{variant}_{repeat:02}.npz';inputs.append(path)
                destination.append(dict(np.load(path,allow_pickle=False)))
        dims=inventory.loc[inventory.n_categories.eq(2 if condition==1 else 4)].sort_values('hypothesis_index').active_dimensions.to_numpy()
        q=np.stack([run['marginal_prior'] for run in runs]);np.testing.assert_allclose(q.sum(2),1,atol=1e-9)
        g['rule_dimensions']=q.mean(0)@dims
        g['belief_reallocation']=np.r_[np.nan, .5*np.abs(np.diff(q,axis=1)).sum(2).mean(0)]
        g['global_range_w32']=g.global_range.rolling(32,min_periods=32).mean()
        g['reallocation_w32']=g.belief_reallocation.rolling(32,min_periods=32).mean()
        g['rule_dimensions_w32']=g.rule_dimensions.rolling(32,min_periods=32).mean()
        early=g.iloc[:CONFIG['early_trials']]
        events=support_reversals(g.belief.to_numpy())
        for origin,loss in events:reversals.append({'subject':sid,'support_start':origin,'loss_start':loss})
        for hd in (8,16,32):
            for ld in (4,8,16):
                sensitivity.append({'subject':sid,'high_duration':hd,'low_duration':ld,
                                    'reversals':len(support_reversals(g.belief.to_numpy(),hd,ld))})
        risk=(g.belief>.5)&g.feedback.eq(0)&g.iSession.eq(g.iSession.shift(-1))
        stay=g.belief.shift(-1)>.5
        stable_n=int(risk.sum())
        summaries.append({'subject':sid,**meta.to_dict(),**shape,
            'early_global_range':float(early.global_range.mean()),
            'early_rule_dimensions':float(early.rule_dimensions.mean()),
            'late_rule_dimensions':float(g.rule_dimensions.iloc[-128:].mean()),
            'early_reallocation':float(early.belief_reallocation.mean()),
            'mean_reallocation':float(g.belief_reallocation.mean()),
            'support_reversals':len(events),'reversals_per_100':100*len(events)/len(g),
            'error_retention_n':stable_n,
            'error_retention':float(stay[risk].mean()) if stable_n else np.nan})
        # Event windows are saved for every participant, but main breakthrough
        # panels use only behavior-defined step cases; no outcome-chosen window.
        if np.isfinite(shape['split']):
            split=int(shape['split'])
            for window in (32,64):
                for variant,rr in [('selected',runs),('alternative',alt)]:
                    for repeat,run in enumerate(rr):
                        target=int(meta.target)
                        vals={'available':run['marginal_active_probability'][:,target],
                              'belief':run['marginal_prior'][:,target],
                              'predicted':run['pred_category_probs'][np.arange(len(g)),g.category.to_numpy(int)-1]}
                        if 'marginal_executed_probability' in run:
                            vals['executed']=run['marginal_executed_probability'][:,target]
                        for name,val in vals.items():
                            jumps.append({'subject':sid,'split':split,'window':window,'variant':variant,
                                'repeat':repeat,'measure':name,'before':float(np.mean(val[split-window:split])),
                                'after':float(np.mean(val[split:split+window]))})
        # Delete blocks from the behavioral likelihood without compressing time.
        for block_size in (16,32):
            for start in range(0,len(g),block_size):
                mask=np.ones(len(g),bool);mask[start:start+block_size]=False
                sh,_=behavior_shapes(g.correct.to_numpy(),keep=mask)
                shape_checks.append({'subject':sid,'block_size':block_size,'deleted_start':start+1,
                                     'deleted_end':min(start+block_size,len(g)),**sh})
        trials.append(g)
    summary=pd.DataFrame(summaries)
    examples=select_examples(summary)
    tables={'subjects':summary,'trials':pd.concat(trials,ignore_index=True),
            'jump_windows':pd.DataFrame(jumps),'support_reversals':pd.DataFrame(reversals),
            'reversal_sensitivity':pd.DataFrame(sensitivity),
            'behavior_shape_sensitivity':pd.DataFrame(shape_checks),'rule_inventory':inventory}
    for name,table in tables.items():table.to_csv(output/f'{name}.csv',index=False)
    manifest={'config':CONFIG,'examples':examples,'participants':len(people),'trials':len(d),
              'example_selection':'Behavior-only: earliest criterion; latest criterion among trend-preferring; latest criterion among step-preferring. No latent-state selection.',
              'scope':'Retrospective hypothesis-driven comparison. No cognitive refits or new simulations. No causal compensation claim.',
              'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in dict.fromkeys(inputs)}}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(summary[['subject','criterion','delta_bic','early_global_range','early_rule_dimensions','support_reversals','error_retention']].round(3).to_string(index=False))
    print('Behavior-selected examples:',examples)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--states',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.states,args.output)


if __name__=='__main__':main()
