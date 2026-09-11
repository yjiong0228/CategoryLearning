"""Replay four archived finalists and retain probabilities for paired PF audit.

No parameter search: reuse recorded final-rescore seed family and full sequence.
Bootstrap resamples paired numerical PF repeats, not participants or trials.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import yaml
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.utils.subjects import deep_update


def run(hyper_path: Path, final_path: Path, output: Path, jobs: int) -> None:
    output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(hyper_path.read_text())
    # The constructor creates its output directory; keep all new artifacts isolated.
    cfg['output_dir']=str((output/'optimizer_context').resolve())
    optimizer=HyperCDOptimizer(cfg,hyper_path.resolve())
    rows=[json.loads(s) for s in final_path.read_text().splitlines()]
    probabilities=[];observed=None;mask=None
    for ci,row in enumerate(rows):
        sid=next(iter(row['subject_metrics']));sid=int(sid)
        stage_cfg=deep_update(optimizer.base_sim_config,cfg['final_rescore']['simulation_overrides'])
        sub,eng,pm,sm,loss,loss_delta,window,_=optimizer._resolve_sim_components(stage_cfg,sid,[sid])
        point_cfg,point_eng=optimizer._apply_hyperparams(row['hyperparams'],sub,eng)
        runner,paths=optimizer._build_runner(point_cfg,point_eng)
        runs,condition,point_seed,score_context=optimizer._simulate_runs_for_point(
            stage_name='final_rescore',runner=runner,dataset_paths=paths,subject_id=sid,
            point=row['hyperparams'],simulation_repeats=int(point_cfg['simulation_repeats']),
            window_size=window,stop_at=float(point_cfg.get('stop_at',1)),max_trials=None,
            keep_logs=False,prediction_mode=pm,selection_prediction_mode=sm,loss_metric=loss,
            loss_delta=loss_delta,hyper_candidate_seed=int(row['hyper_candidate_seed']),n_jobs=jobs,
            evaluation_protocol=point_cfg.get('evaluation_protocol'),force_common_random_numbers=True,
            seed_family=row['seed_family'])
        assert point_seed==row['subject_metrics'][str(sid)]['simulation_point_seed']
        pp=[]
        for result in runs:
            metrics=result.metrics_by_mode[sm]
            p=np.asarray(metrics['pred_category_probs'],dtype=float)
            y=np.asarray(metrics['observed_choice_index'],dtype=int)
            m=np.asarray(metrics['valid_trial_mask'],dtype=bool)
            if observed is None:observed=y;mask=m
            np.testing.assert_array_equal(y,observed);np.testing.assert_array_equal(m,mask)
            pp.append(p[np.arange(len(y)),y])
        pp=np.array(pp)
        assert pp.shape==(16,256) and mask.sum()==255 and np.isfinite(pp).all()
        score=float(-np.log(np.clip(pp.mean(axis=0)[mask],1e-12,1)).mean())
        np.savez_compressed(output/f'candidate_{ci+1}_probabilities.npz',observed_choice_probability=pp,observed=observed,score_mask=mask)
        record={'candidate':ci+1,'original_mean_nll':row['aggregated_error'],'replayed_mean_nll':score,'difference':score-row['aggregated_error'],'simulation_point_seed':point_seed}
        with (output/'replay_scores.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        np.testing.assert_allclose(score,row['aggregated_error'],atol=1e-10,rtol=0)
        probabilities.append(pp)
    p=np.array(probabilities);loss=-np.log(np.clip(p.mean(axis=1)[:,mask],1e-12,1)).sum(axis=1)
    rng=np.random.default_rng(20260910)
    weights=rng.multinomial(16,np.full(16,1/16),size=4000)/16
    boot_mean=np.einsum('kb,cbt->ckt',weights,p)
    boot_loss=-np.log(np.clip(boot_mean[:,:,mask],1e-12,1)).sum(axis=2)
    best=int(loss.argmin());diff=boot_loss-boot_loss[best]
    interval=np.quantile(diff,[.025,.975],axis=1)
    summary={'interpretation':'Paired PF-repeat bootstrap conditional on original final-scoring seeds and fixed candidates; not parameter intervals, participant uncertainty, or independent validation.',
        'bootstrap_replicates':4000,'bootstrap_seed':20260910,'score_trials':int(mask.sum()),'best_candidate':best+1,
        'total_nll':loss.tolist(),'delta_total_nll':(loss-loss[best]).tolist(),
        'paired_mc_interval95_lower':interval[0].tolist(),'paired_mc_interval95_upper':interval[1].tolist(),
        'bootstrap_selected_frequency':[(boot_loss.argmin(axis=0)==i).mean() for i in range(4)],
        'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [hyper_path,final_path]},'python':platform.python_version()}
    (output/'paired_bootstrap.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(output/'paired_bootstrap.npz',total_nll=boot_loss,delta_from_selected=diff)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hyper-config',type=Path,required=True);p.add_argument('--finalists',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',type=int,default=16)
    a=p.parse_args();run(a.hyper_config,a.finalists,a.output,a.jobs)
