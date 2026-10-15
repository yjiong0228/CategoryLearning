"""Score the complete adaptive first-passing-time distribution on a fixed horizon.

T*=min(T,H+1) is observed exactly for every simulated path: an event after H,
including a path still censored at H, belongs to the single H+1 category.
CRPS in block units penalizes predictions that are too early or too late.
This supplements frozen comparisons; it never selects or changes parameters.
Run from the repository root with --source ADAPTIVE_OUTPUT --output NEW_OUTPUT.
Observed first-passing times must be available; early-censored observations
need a separately specified survival score.
"""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd
from .model_0923_h_validation import load, digest, distribution_scores, jackknife_se
from src.Bayesian_state.simulation.exp123_adaptive_task import block_performance, stimulus_labels


def capped_first_pass_times(blocks: np.ndarray, passed: np.ndarray, horizon: int) -> np.ndarray:
    """Retain non-passing paths as the H+1 category of min(T,H+1)."""
    lengths = np.asarray(blocks)
    events = np.asarray(passed)
    if (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1
            or lengths.ndim != 1 or not len(lengths) or events.shape != lengths.shape
            or events.dtype.kind != 'b' or lengths.dtype.kind not in 'iuf'
            or not np.isfinite(lengths).all()
            or np.any(lengths != np.floor(lengths)) or np.any(lengths < 1)
            or np.any(lengths > horizon)):
        raise ValueError('Need integer event/censoring times within a positive common horizon')
    if np.any(lengths[~events] != horizon):
        raise ValueError('Non-passing simulated paths must reach the common horizon')
    return np.where(events, lengths, horizon + 1).astype(float)


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new stopping-score destination')
    protocol=load(source/'protocol.json');done=load(source/'completion.json')
    assert done['status']=='complete' and not done['smoke']
    task_source=Path('src/Bayesian_state/simulation/exp123_adaptive_task.py')
    if digest(task_source)!=protocol['source_sha256'][str(task_source)]:
        raise ValueError('Task criterion implementation changed since simulation')
    fit=Path(protocol['fit']);nominees=load(fit/'nominees.json')
    assert digest(fit/'nominees.json')==protocol['nomination_sha256']
    settings=protocol['settings'];horizon=settings['max_blocks'];threshold=settings['threshold']
    paths=pd.read_csv(source/'paths.csv').query("policy == 'adaptive'")
    included=set(protocol.get('included_subjects',[r['subject'] for r in nominees['nominees']]))
    expected={(r['subject'],r['model']) for r in nominees['nominees'] if r['subject'] in included}
    assert set(zip(paths.subject,paths.model))==expected
    observed={};observed_rows=[];hashes={str(source/'paths.csv'):digest(source/'paths.csv')}
    for subject in sorted(paths.subject.unique()):
        folder=Path(nominees['data'])/f'subject_{subject}'
        meta=load(folder/'prepared.json');parts=[]
        for name in ['training.npz','sealed_test.npz']:
            file=folder/name;assert digest(file)==meta['array_sha256'][name]
            hashes[str(file)]=digest(file)
            with np.load(file) as saved:parts.append({k:saved[k] for k in saved.files})
        arrays={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
        categories,ambiguous=stimulus_labels(arrays['stimulus'],meta['condition'])
        np.testing.assert_array_equal(categories,arrays['categories'])
        assert len(categories)%64==0
        passing=[]
        for first in range(0,len(categories),64):
            accuracy,_=block_performance(categories[first:first+64],arrays['choices'][first:first+64],
                ambiguous[first:first+64],2 if meta['condition']==1 else 4)
            if np.all(accuracy>=threshold):passing.append(first//64+1)
        if not passing:
            raise ValueError(f'Observed time is censored for {subject}; point CRPS is not defined')
        target=min(passing[0],horizon+1)
        observed[subject]=target
        observed_rows.append(dict(subject=subject,condition=meta['condition'],recorded_blocks=len(categories)//64,
            first_passing_block=passing[0],all_passing_blocks=str(passing),target_capped_time=target,
            first_pass_matches_recorded_end=passing[0]==len(categories)//64))
    scored=[];numeric={}
    for (subject,model),group in paths.groupby(['subject','model'],sort=False):
        group=group.sort_values('seed');values=[]
        assert not group.seed.duplicated().any()
        for row in group.itertuples():
            file=Path(row.path)
            assert digest(file)==row.path_sha256;hashes[str(file)]=row.path_sha256
            with np.load(file) as saved:block_passed=saved['block_passed']
            assert len(block_passed)==row.blocks
            if row.passed:
                assert block_passed.sum()==1 and block_passed[-1] and row.blocks<=horizon
                values.append(row.blocks)
            else:
                assert not block_passed.any() and row.blocks==horizon
                values.append(horizon+1)
        samples=capped_first_pass_times(group.blocks.to_numpy(),group.passed.to_numpy(),horizon)
        np.testing.assert_array_equal(samples,np.array(values,dtype=float))
        target=observed[subject]
        metrics=distribution_scores(samples[:,None],np.array([target]))
        score=lambda x: float(np.abs(x-target).mean()-.5*np.abs(x[:,None]-x[None,:]).mean())
        np.testing.assert_allclose(metrics['crps'],score(samples),atol=1e-12,rtol=0)
        jk=np.array([score(np.delete(samples,i)) for i in range(len(samples))])
        numeric[(subject,model)]=dict(seeds=group.seed.to_numpy(),score=metrics['crps'],jk=jk)
        scored.append(dict(subject=subject,model=model,paths=len(samples),horizon=horizon,
            observed_first_passing_block=target,crps_blocks=metrics['crps'],crps_mc_se=jackknife_se(jk),
            mean_capped_time=float(samples.mean()),median_capped_time=float(np.median(samples)),
            interval80_coverage=metrics['interval_coverage'],interval80_width_blocks=metrics['interval_width'],
            censored_fraction=float((samples==horizon+1).mean()),
            before_observed_fraction=float((samples<target).mean()),at_observed_fraction=float((samples==target).mean()),
            by_observed_fraction=float((samples<=target).mean())))
    comparisons=[]
    for (subject,model),value in numeric.items():
        reference=numeric[(subject,'B')]
        np.testing.assert_array_equal(value['seeds'],reference['seeds'])
        comparisons.append(dict(subject=subject,model=model,reference='B',
            delta_crps_blocks=value['score']-reference['score'],
            delta_crps_mc_se=jackknife_se(value['jk']-reference['jk'])))
    c=pd.DataFrame(comparisons);s=pd.DataFrame(scored);aggregate=[]
    for model,group in c.groupby('model',sort=False):
        subset=s[s.model==model]
        aggregate.append(dict(model=model,participants=len(group),crps_blocks=float(subset.crps_blocks.mean()),
            delta_crps_blocks=float(group.delta_crps_blocks.mean()),
            delta_crps_mc_se=float(np.linalg.norm(group.delta_crps_mc_se)/len(group)),
            interval80_coverage=float(subset.interval80_coverage.mean()),
            censored_fraction=float(subset.censored_fraction.mean())))
    output.mkdir(parents=True);shutil.copy2(__file__,output/'analysis_source.py')
    for name,rows in [('participants',scored),('observed',observed_rows),('contrasts',comparisons),('aggregate',aggregate)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    with (output/'completion.json').open('x') as f:
        json.dump(dict(status='complete',source=str(source.resolve()),source_sha256=digest(Path(__file__)),
            dependency_sha256={str(task_source):digest(task_source),
                'src/Bayesian_state/evaluation/model_0923_h_validation.py':
                digest(Path('src/Bayesian_state/evaluation/model_0923_h_validation.py'))},
            nomination_sha256=protocol['nomination_sha256'],data_use_status=protocol['data_use_status'],
            included_subjects=sorted(included),excluded_subjects=protocol.get('excluded_subjects',[]),
            horizon_blocks=horizon,censored_category=horizon+1,inputs_sha256=hashes,
            score='empirical CRPS for T*=min(first_passing_block,H+1), block units; all censored mass retained',
            uncertainty='paired whole-path jackknife; fixed parameters; population and parameter uncertainty excluded',
            scope='Supplementary diagnostic; frozen primary comparisons and selected parameters unchanged'),f,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
