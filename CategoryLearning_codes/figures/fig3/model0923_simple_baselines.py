"""Compare full and single-rule mechanisms under the same particle inference.

Figure contract: quantify the predictive value of complex rule search after
giving simpler mechanisms the same conditional inference. Python quantitative
grid, 300 dpi PNG for screen review. All scored trials and all seeds are used;
seed dispersion measures numerical variation, with fixed fitted parameters.
See src/Bayesian_state/docs/model_architecture/model_0923_simple_baselines.md.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_postfit import score_probabilities


MODELS=("full","fixed_rule","random_rule")
LABELS=dict(full="Full 0923",fixed_rule="Fixed rule",random_rule="Random switch")
CHINESE=dict(full="完整0923",fixed_rule="固定单规则",random_rule="随机切换单规则")
COLORS=dict(full="#3578A7",fixed_rule="#CC813F",random_rule="#8662A6")


def read_runs(folder: Path) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    probabilities,active,records=[],[],[]
    for sub in sorted(folder.glob("repeat_*")):
        with np.load(sub/"seed_00.npz",allow_pickle=False) as saved:
            probabilities.append(saved["probabilities"])
            active.append(saved["active"])
        records.append(json.loads((sub/"scores.json").read_text()))
    if not probabilities:
        raise ValueError(f"Missing final runs: {folder}")
    return np.stack(probabilities),np.stack(active),records


def trial_nll(p: np.ndarray, choices: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(p)),choices-1],1e-12,1))


def windows(values: np.ndarray, size: int) -> tuple[np.ndarray,np.ndarray]:
    """Non-overlapping scored windows including the final incomplete window."""
    spans=[np.arange(i,min(i+size,len(values))) for i in range(1,len(values),size)]
    return np.array([s.mean()+1 for s in spans]),np.stack([values[s].mean(axis=0) for s in spans])


def build(source: Path, figures: Path) -> None:
    protocol=json.loads((source/"protocol.json").read_text())
    validation=json.loads((source/"validation.json").read_text())
    if protocol["smoke"] or validation["status"]!="passed":
        raise ValueError("Require completed non-smoke analysis")
    if figures.exists() or (source/"index.html").exists():
        raise FileExistsError("Use new figure and report destinations")
    figures.mkdir(parents=True)
    settings=protocol["settings"]
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
        "font.size":10,"axes.spines.right":False,"axes.spines.top":False,
        "legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42})
    fig,axes=plt.subplots(len(settings["subjects"]),2,figsize=(11,7),squeeze=False,layout="constrained",
                           gridspec_kw={"width_ratios":[1.5,1]})
    state_fig,state_axes=plt.subplots(len(settings["subjects"]),2,figsize=(11,7),squeeze=False,layout="constrained")
    rows,trial_rows,seed_rows,parameter_rows=[],[],[],[]
    input_paths=[source/"protocol.json",source/"validation.json",Path(__file__)]
    for r,item in enumerate(settings["subjects"]):
        sid=item["subject"]
        folder=source/f"subject_{sid}"
        original=Path(item["evaluation_dir"])/f"subject_{sid}"
        frame=pd.read_csv(original/"trials.csv")
        n=len(frame)
        choices=frame.choice.to_numpy(int)
        categories=frame.category.to_numpy(int)
        if frame.scored.iloc[0] or not frame.scored.iloc[1:].all():
            raise ValueError("Unexpected scoring mask")
        with np.load(original/"pooled_states.npz") as saved:
            full_prob=saved["probabilities"]
            full_active=saved["predictive_active"]
            reference=int(saved["predictive_choice_weights"][-settings["late_window"]:].mean(axis=0).argmax())
        history=Path(settings["history_control_dir"])/f"subject_{sid}"
        with np.load(history/"comparison.npz") as saved:
            full_uniform=saved["B_probabilities"]
        x,observed=windows((frame.feedback.to_numpy()==1).astype(float),settings["window_size"])
        left,right=axes[r]
        left.plot(x,observed,color="#343A40",linestyle=":",linewidth=1.8,label="Observed")
        left.set_title(f"{chr(97+2*r)}  S{sid}: conditional prediction",loc="left",fontweight="bold")
        right.set_title(f"{chr(98+2*r)}  Cumulative NLL difference",loc="left",fontweight="bold")
        right.axhline(0,color=COLORS["full"],linewidth=1,label="Full reference")
        for m,variant in enumerate(MODELS):
            point={}
            if variant=="full":
                a_prob=full_prob
                b_prob=full_uniform
                state_a=full_active
                with np.load(history/"comparison.npz") as saved:
                    state_b=saved["active"]
                saved_seeds=[]
                for p in sorted((original/"seeds").glob("pf_*.npz")):
                    with np.load(p) as saved:
                        saved_seeds.append(saved["probabilities"])
                    input_paths.append(p)
                all_prob=np.stack(saved_seeds)
                np.testing.assert_allclose(all_prob.mean(axis=0),a_prob,atol=1e-12,rtol=0)
                audit_nll=None
                audit_correct=None
                audit_sd=None
                audit_initial_coverage=None
                audit_final_coverage=None
                n_parameters=13
                repetitions=[]
            else:
                all_prob,active,repetitions=read_runs(folder/"evaluation"/variant)
                b_all,b_active,_=read_runs(folder/"uniform"/variant)
                a_prob,b_prob=all_prob.mean(axis=0),b_all.mean(axis=0)
                state_a,state_b=active.mean(axis=0),b_active.mean(axis=0)
                audit,_,audit_records=read_runs(folder/"audit"/variant)
                audit_mean=audit.mean(axis=0)
                audit_nll=score_probabilities(audit_mean,choices)["mean_choice_nll"]
                audit_sd=float(np.std([score_probabilities(p,choices)["mean_choice_nll"] for p in audit],ddof=1))
                audit_correct=float(audit_mean[np.arange(n),categories-1][-settings["late_window"]:].mean())
                audit_initial_coverage=float(np.mean([v["repeats"][0]["initial_supported_rules"] for v in audit_records]))
                audit_final_coverage=float(np.mean([v["repeats"][0]["final_supported_rules"] for v in audit_records]))
                point=json.loads((folder/f"{variant}_selected.json").read_text())["point"]
                input_paths.append(folder/f"{variant}_selected.json")
                n_parameters=3 if variant=="fixed_rule" else 4
                parameter_rows.append(dict(subject=sid,model=variant,**point))
                for path in (folder/"evaluation"/variant).rglob("*.npz"):
                    input_paths.append(path)
                for phase in ("uniform","audit"):
                    input_paths.extend((folder/phase/variant).rglob("*.npz"))
                for phase in ("evaluation","uniform","audit"):
                    input_paths.extend((folder/phase/variant).rglob("scores.json"))
            if len(all_prob)!=settings["evaluation"]["repeats"]:
                raise ValueError("All models must have the same final repeat count")
            a_score=score_probabilities(a_prob,choices)
            b_score=score_probabilities(b_prob,choices)
            a_correct=a_prob[np.arange(n),categories-1]
            b_correct=b_prob[np.arange(n),categories-1]
            seed_nll=np.array([score_probabilities(p,choices)["mean_choice_nll"] for p in all_prob])
            for i,value in enumerate(seed_nll):
                seed_rows.append(dict(subject=sid,model=variant,repeat=i,mean_choice_nll=float(value)))
            row=dict(subject=sid,model=variant,parameter_count=n_parameters,**a_score,
                B_mean_choice_nll=b_score["mean_choice_nll"],seed_nll_sd=float(seed_nll.std(ddof=1)),
                seed_nll_min=float(seed_nll.min()),seed_nll_max=float(seed_nll.max()),
                early_correct=float(a_correct[1:65].mean()),late_correct=float(a_correct[-64:].mean()),
                B_late_correct=float(b_correct[-64:].mean()),observed_late=float((frame.feedback.iloc[-64:]==1).mean()),
                audit_mean_choice_nll=audit_nll,audit_late_correct=audit_correct,
                audit_seed_nll_sd=audit_sd,audit_initial_rule_coverage=audit_initial_coverage,
                audit_final_rule_coverage=audit_final_coverage,
                audit_minus_main_nll=None if audit_nll is None else audit_nll-a_score["mean_choice_nll"],
                numerical_check="reference" if audit_nll is None else
                    "within_tolerance" if abs(audit_nll-a_score["mean_choice_nll"])<=settings["evaluation"]["numerical_nll_tolerance"]
                    else "budget_sensitive",
                reference_rule=reference,reference_late_active=float(state_a[-64:,reference].mean()),
                B_reference_late_active=float(state_b[-64:,reference].mean()),
                initial_rule_coverage=None if not repetitions else float(np.mean([v["repeats"][0]["initial_supported_rules"] for v in repetitions])),
                final_rule_coverage=None if not repetitions else float(np.mean([v["repeats"][0]["final_supported_rules"] for v in repetitions])))
            rows.append(row)
            nll=trial_nll(a_prob,choices)
            difference=nll-trial_nll(full_prob,choices)
            x,y=windows(a_correct,settings["window_size"])
            left.plot(x,y,color=COLORS[variant],linewidth=1.8,label=LABELS[variant])
            if variant!="full":
                right.plot(np.arange(2,n+1),np.cumsum(difference[1:]),color=COLORS[variant],label=LABELS[variant])
            ax=state_axes[r,0]
            _,y=windows(state_a[:,reference],settings["window_size"])
            ax.plot(x,y,color=COLORS[variant],linewidth=1.8,label=LABELS[variant])
            ax=state_axes[r,1]
            ax.plot([m-.12,m+.12],[row["B_late_correct"],row["late_correct"]],color=COLORS[variant],linewidth=1)
            ax.scatter(m-.12,row["B_late_correct"],marker="o",s=48,facecolor="white",edgecolor=COLORS[variant],label="Uniform" if m==0 else None)
            ax.scatter(m+.12,row["late_correct"],marker="o",s=48,color=COLORS[variant],label="Conditional" if m==0 else None)
            trial_rows.extend(dict(subject=sid,model=variant,valid_trial=i+1,scored=i>0,
                observed_correct=int(frame.feedback.iloc[i]==1),A_correct=float(a_correct[i]),B_correct=float(b_correct[i]),
                choice_nll=float(nll[i]),nll_minus_full=float(difference[i]),
                reference_rule=reference,reference_active=float(state_a[i,reference]),
                B_reference_active=float(state_b[i,reference])) for i in range(n))
        left.axhline(.25,color="#A0A0A0",linewidth=.7,linestyle=":")
        left.set(xlabel="Valid trial",ylabel="Correct response probability",ylim=(-.02,1.03))
        right.set(xlabel="Valid trial",ylabel="Baseline minus full (nats)")
        if r==0:
            left.legend(fontsize=8.5,ncol=2,loc="upper left")
            right.legend(fontsize=8.5)
        state_axes[r,0].set(title=f"S{sid}: reference rule R{reference} present",xlabel="Valid trial",ylabel="Posterior probability",ylim=(-.02,1.03))
        state_axes[r,1].set(title=f"S{sid}: last 64 trials",ylabel="Correct response probability",ylim=(-.02,1.03),
                            xticks=np.arange(3),xticklabels=[LABELS[k] for k in MODELS])
        state_axes[r,1].legend(fontsize=9)
        if r==0:
            state_axes[r,0].legend(fontsize=8.5)
        input_paths.extend([original/"trials.csv",original/"pooled_states.npz",history/"comparison.npz"])
    fig.savefig(figures/"conditional_baselines.png",dpi=300,facecolor="white")
    state_fig.savefig(figures/"rule_and_conditioning.png",dpi=300,facecolor="white")
    plt.close(fig)
    plt.close(state_fig)
    summary=pd.DataFrame(rows)
    summary.to_csv(source/"comparison.csv",index=False)
    pd.DataFrame(parameter_rows).to_csv(source/"baseline_parameters.csv",index=False)
    pd.DataFrame(seed_rows).to_csv(source/"seed_scores.csv",index=False)
    pd.DataFrame(trial_rows).to_csv(figures/"trial_plot_source.csv",index=False)
    render_report(source,figures,summary,parameter_rows,settings)
    contract=dict(backend="python",archetype="quantitative grid",export="300 dpi PNG, screen review",
        conclusion="Compare fit to actual choices and learning curves under shared conditional inference",
        trial_policy="Exclude initialization from scores and windows; retain every subsequent valid trial",
        windows=settings["window_size"],replicates="16 particle-filter seeds per model, 128 particles per seed",
        audit="4 independent seeds with 512 particles for each selected baseline",
        split="Full-sequence in-sample parameters; independent numerical seeds, not held-out behavior",
        reference_rule="Largest full-model late mean response weight; selected only for description",
        tests="No population significance test; seed spread is Monte Carlo variation",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figures.iterdir() if p.is_file()})
    (figures/"figure_contract.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n")
    print(summary.to_string(index=False))


def render_report(source,figures,summary,parameters,settings):
    columns=["subject","model","parameter_count","mean_choice_nll","B_mean_choice_nll","late_correct","B_late_correct","audit_mean_choice_nll","numerical_check"]
    shown=summary[columns].copy()
    shown["model"]=shown.model.map(CHINESE)
    shown.columns=["被试","模型","参数数","A选择NLL ↓","B选择NLL ↓","A后期正确概率","B后期正确概率","512粒子NLL ↓","数值检查"]
    table=shown.to_html(index=False,float_format=lambda x:f"{x:.4f}",border=0)
    pictures="".join(f'<img src="{html.escape(os.path.relpath(figures/name,source))}" alt="{caption}"><p>{caption}</p>' for name,caption in (
        ("conditional_baselines.png","上、下行分别为335和307。左列比较相同粒子推断下的正确概率；右列为累计选择NLL相对完整模型的差，正值表示基线对真实选择的预测较差。"),
        ("rule_and_conditioning.png","左列是A中参考规则进入候选集合的概率；右列连接同一参数下B等权对照与A条件预测的后期表现。参考规则按完整模型后期权重选择。")))
    lines=[]
    for sid in summary.subject.unique():
        group=summary.query("subject==@sid")
        full=group.query("model=='full'").iloc[0]
        for _,row in group.query("model!='full'").iterrows():
            delta=row.mean_choice_nll-full.mean_choice_nll
            lines.append(f"<li>{sid}，{CHINESE[row.model]}：A选择NLL为{row.mean_choice_nll:.4f}，相对完整模型差值{delta:+.4f}；"
                f"后期正确概率{row.late_correct:.1%}。每次滤波512粒子时的NLL为{row.audit_mean_choice_nll:.4f}，数值检查：{row.numerical_check}。</li>")
    numeric=summary[["subject","model","seed_nll_sd","initial_rule_coverage","final_rule_coverage",
                     "audit_seed_nll_sd","audit_initial_rule_coverage","audit_final_rule_coverage"]].copy()
    numeric["model"]=numeric.model.map(CHINESE)
    numeric.columns=["被试","模型","128粒子逐种子NLL标准差","128粒子初始规则数","128粒子最终规则数",
                     "512粒子逐种子NLL标准差","512粒子初始规则数","512粒子最终规则数"]
    page=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>0923：简单机制的条件预测</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui;color:#20262c}}img{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#f3f5f7}}</style>
<h1>让简单机制接受同样的粒子推断</h1>
<p>固定单规则基线每条路径初始抽取一条规则，之后保持不变；随机切换基线以恒定概率换为当前未使用的一条规则。
两者都使用同一152条规则目录、被试知觉参数、条件3配对反馈、分层遗忘和反馈梯度精度更新。
切换时配对信念随原规则的质量转移，精度重置为拟合初值。两个基线的候选数始终为1。</p>
<p>固定单规则估计β初值、精度学习率和证据保持率，共3项参数；随机切换另估计切换概率，共4项。
完整模型沿用13项参数的已有拟合。所有模型在本题真实选择到来之前计分，随后使用该选择更新粒子权重，必要时重采样。</p>
<h2>结果</h2>{table}<ul>{''.join(lines)}</ul>
<p>NLL为真实选择的平均负对数似然，越低越好；四分类均匀猜测为ln(4)≈1.3863。
正确概率列为最后64题的平均正确类别概率。A使用条件推断，B沿相同真实历史推进并保持路径等权。</p>
<p>固定单规则模型中，每条路径从头到尾持有同一规则；不同路径持有不同规则。
过去的真实选择使外层推断逐渐提高相容路径的权重。因此，汇总后的规则权重和正确概率可以随试次提高。
路径内部的配对信念与精度也继续从反馈学习。观察到A的上升，需要结合这些来源解释。</p>
<p>这两个基线同时简化了搜索、候选集合和规则读出。比较结果衡量整组复杂机制的增量预测收益，
对作答惯性参数的单独作用仍需控制其他模块的消融。本次口述数据未用于参数选择，也未重新评分。</p>
{pictures}
<h2>拟合与数值检查</h2>
<p>每个被试、每种基线预先限定24个参数点，包含完整模型的共同参数作为起点，其余采用确定种子的Sobol分散抽样。
先用64粒子×2种子筛选，再将最好的4点用128粒子×4个新种子确认；确认后固定参数。
最终评估使用与完整模型相同的128粒子×16种子；这些种子未参与基线参数选择。
另用512粒子×4个独立种子检查数值敏感性，两种配置的NLL差超过0.02时标为budget_sensitive。
每次滤波的粒子数增加，独立重复数减少；两种配置的粒子数乘重复数均为2048。
两种配置使用不同随机种子，差值反映综合数值敏感性。种子独立性用于检查蒙特卡洛误差。</p>
{numeric.to_html(index=False,float_format=lambda x:f"{x:.4f}",border=0,na_rep="—")}
<p>规则数为每次滤波开始或结束时有正权重的不同规则数，再对种子取均值。
逐种子NLL标准差只描述数值波动；汇总分数先平均概率再取负对数，因此与逐种子NLL的均值不同。</p>
<p>各模型的参数均基于完整行为序列，因此本次是样本内、有预算上限的机制比较。
基线搜索尚未做全空间收敛验收，完整模型原拟合也保留unresolved / boundary_review_required状态。
本次没有行为留出测试、模型恢复、重新拟合完整模型或重新计算口述。</p>
<p>固定规则的重采样可能删除以后有用的规则。报告保存每次运行的初始和最终规则覆盖数、重采样次数和512粒子检查，
解释该基线时应结合这些数值诊断。较低的A分数表明对观察历史的条件预测较好，自主学习能力需要另行检验。</p>
<h2>参数</h2>{pd.DataFrame(parameters).to_html(index=False,border=0)}
<h2>复现</h2>
<p>执行沿用拟合时的冻结数值源码，额外加载新基线配置验证器和固定工作空间的子类声明；新增文件哈希单独记录。
完整模型第一个种子在335的748题、307的704题均与原预测逐项一致，允许误差1e-12。
两种基线复用已有规则切换、反馈学习和粒子滤波算法。</p>
<p><a href="comparison.csv">指标表</a> · <a href="baseline_parameters.csv">参数表</a> ·
<a href="seed_scores.csv">逐种子NLL</a> · <a href="protocol.json">预算与源码记录</a> ·
<a href="validation.json">验证与输入哈希</a> · <a href="run_index.json">全部候选评分</a></p></html>"""
    with (source/"index.html").open("x",encoding="utf-8") as stream:
        stream.write(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--figure-output-dir",type=Path,required=True)
    args=parser.parse_args()
    build(args.analysis_dir,args.figure_output_dir)


if __name__=="__main__":
    main()
