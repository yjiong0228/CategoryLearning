"""Audit saved full-v3 fits, replay fixed parameters and export individual reports."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from zoneinfo import ZoneInfo
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import yaml

from src.Bayesian_state.evaluation.model_0923_postfit import (
    audit_search, postfit_seed, resolve_subject,
    rule_catalog, run_particle_task, score_probabilities, summarize_subject,
)
from src.Bayesian_state.evaluation.model_0923_fit_source import (
    original_replay_spec, validate_fit_source,
)
from src.Bayesian_state.evaluation.autonomous_trajectories import (
    generate_autonomous_ensemble, load_autonomous_evaluation_spec,
)
from src.Bayesian_state.utils.parallel import single_threaded_processes, parallel_job_count
from src.Bayesian_state.utils.paths import ROOT_DIR


def resolve(base: Path, value: str | Path) -> Path:
    path=Path(value)
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(config_path: Path, output: Path | None = None, smoke: bool = False,
        plot_subject: Callable | None = None, plot_cohort: Callable | None = None,
        *, settings_override: dict | None = None, figure_output: Path | None = None) -> dict:
    settings=deepcopy(settings_override) if settings_override is not None else yaml.safe_load(config_path.read_text())
    fit=resolve(config_path.parent,settings['fit_run'])
    output=output or resolve(config_path.parent,settings['output_dir'])
    figures=figure_output or ((output/'figures') if smoke else resolve(config_path.parent,settings['figure_output_dir']))
    source_manifest=validate_fit_source(fit)
    output.mkdir(parents=True,exist_ok=False)
    figures.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    analysis_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    core_paths = [*sorted((ROOT_DIR/'src/Bayesian_state').rglob('*.py')), config_path]
    for plotter in (plot_subject, plot_cohort):
        if plotter is not None:
            core_paths.append(Path(inspect.getsourcefile(plotter)))
    source_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in core_paths}
    subjects=settings['subjects'][:1] if smoke else settings['subjects']
    repeats=2 if smoke else int(settings['pf_repeats'])
    particles=4 if smoke else int(settings['particle_count'])
    max_trials=16 if smoke else None
    workers=1 if smoke else int(settings['parallel_budget'])
    rollouts=2 if smoke else int(settings['autonomous_rollouts'])
    window=4 if smoke else int(settings['window_size'])
    settings.update(subjects=subjects,pf_repeats=repeats,particle_count=particles,
                    parallel_budget=workers,autonomous_rollouts=rollouts,window_size=window,max_trials=max_trials)
    (output/'effective_config.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    cache=output/'resources_cache'
    if (fit/'resources_cache').is_dir():shutil.copytree(fit/'resources_cache',cache)
    audit=audit_search(fit,subjects,output)
    inputs={sid:resolve_subject(fit,sid,cache,max_trials) for sid in subjects}
    for value in inputs.values():rule_catalog(value)  # Warm immutable shared rule resources before workers start.
    tasks=[]
    by_subject={sid:[] for sid in subjects}
    calibration={sid:[] for sid in subjects}
    for sid in subjects:
        directory=output/f'subject_{sid}'
        (directory/'seeds').mkdir(parents=True)
        for repeat in range(repeats):
            path=directory/'seeds'/f'pf_{repeat:02d}.npz'
            by_subject[sid].append(str(path))
            tasks.append(dict(fit_run=str(fit),subject=sid,cache=str(cache),max_trials=max_trials,
                              particles=particles,seed=postfit_seed(settings['analysis_seed'],sid,repeat),output=str(path)))
        path=directory/'original_seed_replay.npz'
        if not smoke:
            replay=original_replay_spec(fit,sid)
            tasks.append(dict(fit_run=str(fit),subject=sid,cache=str(cache),particles=replay['particles'],
                              seed=replay['seed'],record_states=False,output=str(path)))
        else:
            tasks.append(dict(fit_run=str(fit),subject=sid,cache=str(cache),particles=particles,
                              max_trials=max_trials,seed=postfit_seed(settings['analysis_seed'],sid,0),
                              record_states=False,output=str(path)))
        if sid in settings['calibration_subjects'] and (not smoke or settings_override is not None):
            for repeat in range(settings['calibration_repeats']):
                path=directory/'seeds'/f'calibration_{repeat:02d}.npz'
                calibration[sid].append(str(path))
                tasks.append(dict(fit_run=str(fit),subject=sid,cache=str(cache),max_trials=max_trials,
                                  particles=settings['calibration_particle_count'],
                                  seed=postfit_seed(settings['analysis_seed'],sid,repeat),record_states=False,output=str(path)))
    def progress(phase,completed=0,total=0):
        write_json(output/'progress.json',{'phase':phase,'completed':completed,'total':total,
                                         'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
        print(phase,completed,total,flush=True)
    progress('fixed_parameter_pf',0,len(tasks))
    with single_threaded_processes():
        generated=Parallel(n_jobs=parallel_job_count(workers,len(tasks)),batch_size=1,return_as='generator_unordered')(
            delayed(run_particle_task)(task) for task in tasks)
        for index,path in enumerate(generated):
            if index%16==0 or index+1==len(tasks):progress('fixed_parameter_pf',index+1,len(tasks))
    original_checks=[]
    if not smoke:
        for sid in subjects:
            replay_spec=original_replay_spec(fit,sid)
            expected=replay_spec['expected_mean_choice_nll']
            with np.load(output/f'subject_{sid}'/'original_seed_replay.npz') as replay_data:
                actual=float(replay_data['mean_choice_nll'])
            if not np.isclose(actual,expected,rtol=0,atol=1e-12):
                raise ValueError(f'Original final seed replay disagrees for S{sid}: {actual} vs {expected}')
            original_checks.append({'subject':sid,'original':expected,'replayed':actual,'absolute_difference':abs(actual-expected),
                                    'particles':replay_spec['particles'],'seed':replay_spec['seed']})
    write_json(output/'original_replay_validation.json',{'checks':original_checks,'tolerance':1e-12,'smoke':smoke})
    summaries=[]
    figure_paths={}
    for index,sid in enumerate(subjects):
        progress('individual_report',index,len(subjects))
        directory=output/f'subject_{sid}'
        value=inputs[sid]
        summary,states=summarize_subject(value,by_subject[sid],directory,window)
        simulation={'subjects':[sid],'engine_config':deepcopy(value['engine']),
                    'dataset':{key:str(path) for key,path in value['paths'].items()},
                    'max_trials':max_trials,'stop_at':1.,'window_size':window,
                    'prediction_mode':'prior_t','selection_prediction_mode':'prior_t',
                    'loss_metric':'choice_nll','simulation_repeats':1,'repeat_aggregation':'mean_probability',
                    'evaluation_protocol':{'mode':'all'}}
        config=directory/'fixed_simulation.yaml'
        config.write_text(yaml.safe_dump(simulation,sort_keys=False))
        spec=load_autonomous_evaluation_spec(config,subject_id=sid)
        ensemble=generate_autonomous_ensemble(spec,rollout_count=rollouts,analysis_seed=settings['analysis_seed'],n_jobs=workers)
        with (directory/'autonomous_rollouts.npz').open('xb') as stream:
            np.savez_compressed(stream,choices=ensemble.choices,feedback=ensemble.feedback,
                                expected_correct_probability=ensemble.expected_correct_probability,seeds=ensemble.trajectory_seeds)
        summary['observed_exact_accuracy']=float(np.mean(value['arrays'].feedback[1:]==1))
        autonomous_accuracy=np.mean(ensemble.feedback[:,1:]==1,axis=1)
        summary['autonomous_accuracy_q10'],summary['autonomous_accuracy_median'],summary['autonomous_accuracy_q90']=np.quantile(autonomous_accuracy,[.1,.5,.9]).tolist()
        summary['observed_accuracy_in_autonomous_10_90']=bool(summary['autonomous_accuracy_q10']<=summary['observed_exact_accuracy']<=summary['autonomous_accuracy_q90'])
        if calibration[sid]:
            def probabilities_from(path):
                with np.load(path) as saved:
                    return saved['probabilities']
            high=np.mean([probabilities_from(path) for path in calibration[sid]],axis=0)
            low=np.mean([probabilities_from(path) for path in by_subject[sid][:len(calibration[sid])]],axis=0)
            summary['calibration_high_minus_base_nll']=score_probabilities(high,value['arrays'].choices)['mean_choice_nll']-score_probabilities(low,value['arrays'].choices)['mean_choice_nll']
            summary['calibration_mean_abs_probability_difference']=float(np.mean(np.abs(high-low)))
            pd.DataFrame({'valid_trial':np.arange(1,len(high)+1),
                          'mean_abs_probability_difference':np.mean(np.abs(high-low),axis=1),
                          'max_abs_probability_difference':np.max(np.abs(high-low),axis=1)}).to_csv(directory/'budget_comparison.csv',index=False)
        summaries.append(summary)
        figure_dir=figures/f'subject_{sid}'
        figure_dir.mkdir()
        if plot_subject is not None:
            plot_subject(sid,value['condition'],value['arrays'],states,ensemble.feedback,
                         value['engine']['modules']['hypo_transitions_mod']['kwargs']['max_capacity'],window,figure_dir)
        figure_paths[sid]=figure_dir
        pd.DataFrame([summary]).to_csv(directory/'summary.csv',index=False)
    summary=pd.DataFrame(summaries).merge(audit,on=['subject','condition'],validate='one_to_one')
    summary.to_csv(output/'subjects.csv',index=False)
    if plot_cohort is not None:
        plot_cohort(summary,pd.read_csv(output/'parameters.csv'),figures)
    report_html(output,figures,summary,figure_paths,repeats,particles,rollouts,plot_subject is not None)
    elapsed=perf_counter()-started
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest
           for path,digest in source_hashes.items()):
        raise ValueError('Analysis source changed during computation')
    manifest={'status':'completed','scope':'first_stage_fixed_parameter_postfit','source_fit':str(fit),
              'source_fit_commit':source_manifest.get('git_commit'), 'fit_source_validation':source_manifest,
              'analysis_commit_at_start':analysis_commit,
              'analysis_source_sha256':source_hashes,
              'settings':settings,'elapsed_seconds':elapsed,'modeled_trials':int(summary.modeled_trials.sum()),
              'original_seed_replays':original_checks,'versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','matplotlib','joblib','PyYAML')},
              'figure_directory':str(figures),'state_timing':{'predictive':'pre-choice, pre-feedback','filtered':'after current choice, before current feedback','pairing_post_feedback':'after current choice and feedback'},
              'uncertainty':'Pointwise 10–90% latent-state mixture at fixed selected parameters; autonomous trajectory variation at those parameters.',
              'remaining':['parameter profiles and uncertainty intervals','parameter/state/model recovery','held-out refitting and prediction','mechanism comparison and oral validation']}
    write_json(output/'manifest.json',manifest)
    progress('completed',len(subjects),len(subjects))
    return manifest


def report_html(output,figures,summary,figure_paths,repeats,particles,rollouts,include_figures=True):
    def relative(path):return Path(os.path.relpath(path,output)).as_posix()
    def picture(path):
        return f'<img src="{html.escape(relative(path))}" loading="lazy">' if include_figures else ''
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 0923：12人拟合后分析</title>',
           '<style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:17px/1.7 sans-serif;color:#25313c}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:6px 9px;border-bottom:1px solid #ccd}h2{margin-top:40px}a{color:#236b9b}</style>',
           '<h1>Model 0923：固定参数的第一阶段后分析</h1>',
           f'<p>本报告对每人固定已入选的13项行为参数，以{particles}粒子和{repeats}组新种子重新推断完整有效序列，并生成{rollouts}条自主行为轨迹。参数来自整段行为，预测分数属于拟合内描述。自主轨迹沿用观察到的刺激顺序和有效长度，各自的反馈由其回答及任务规则产生。</p>',
           '<p>状态图的规则信念与作答贡献在当前选择之后、当前反馈之前汇总。候选数量、有效参与数量和作答加权精度的阴影为固定参数条件下的逐题10%–90%隐藏状态范围；自主行为阴影为模拟轨迹的逐点范围。参数统计区间、跨人群不确定性和整体轨迹覆盖率另行评价。</p>',
           '<h2>搜索与参数审查</h2>',picture(figures/'parameter_audit.png'),
           '<p>数值在网格端点时记录为数学边界或搜索范围端点。†表示该参数在入选组合下的影响通道关闭，个体解释保留“条件性不可辨识”。轮数上限、局部最优与范围问题通过后续定向检查评价。</p>',
           '<h2>行为与数值复现</h2>',picture(figures/'cohort_behavior.png'),
           '<p>原始最终评分与新种子评分使用相同128粒子预算和先平均概率再计分的规则。首条有效试次用于初始化，后续试次计分。原最终种子重放另逐人核对保存的单次NLL；重复数减半与三个条件的256粒子检查用于评估数值敏感性。</p>',
           summary.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
           '<p><a href="parameters.csv">参数与可辨识性表</a> · <a href="restarts.csv">停止记录</a> · <a href="finalists.csv">入围候选评分</a> · <a href="subjects.csv">逐人指标</a></p>']
    for row in summary.to_dict('records'):
        sid=row['subject'];directory=figure_paths[sid]
        parts.extend([f'<h2>S{sid}：条件{row["condition"]}</h2>',
                      f'<p>共{row["modeled_trials"]}个有效试次。新种子平均选择NLL为{row["mean_choice_nll"]:.4f}；相对本条件均匀选择基线改善{row["nll_gain_over_uniform"]:.4f}。平均候选数{row["mean_pre_choice_candidates"]:.2f}，平均有效参与数{row["mean_pre_choice_participation"]:.2f}，作答加权精度{row["mean_pre_choice_beta"]:.2f}。这些量描述当前任务内、固定参数条件下的模型推断。</p>',
                      picture(directory/'behavior.png'),picture(directory/'representations.png'),
                      f'<p><a href="subject_{sid}/trials.csv">逐题数据与状态区间</a> · <a href="subject_{sid}/rule_states.csv">完整规则状态</a> · <a href="subject_{sid}/rule_catalog.csv">规则目录及特征</a></p>'])
    parts.append('<h2>后续验收</h2><p>这12人继续作为开发样本。参数区间与恢复、连续路径及转变时点恢复、留出数据重新拟合、机制比较与口述验证属于后续阶段。当前报告呈现固定参数下的在线状态分布；连续路径与跨任务稳定特质需要相应证据。</p></html>')
    (output/'report.html').write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/exp123/specific_models/model_0923_selected12_postfit.yaml'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config=resolve(ROOT_DIR,args.config)
    output=resolve(ROOT_DIR,args.output_dir) if args.output_dir else None
    if args.smoke and output is None:parser.error('--smoke requires a new --output-dir')
    with single_threaded_processes():
        result=run(config,output,args.smoke)
    print(json.dumps({'status':result['status'],'seconds':result['elapsed_seconds']},ensure_ascii=False))


if __name__=='__main__':main()
