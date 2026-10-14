"""Display rule discovery, residence and late readout in saved B/C paths.

Figure contract: locate which stages constrain task-compatible rule use in
each selected subject. Quantitative grid: acquisition/current availability is
the main panel; residence retention and conditional readout identify possible
downstream limitations. Python, 300 dpi PNG, 12 x 7 inch screen review. All
recorded paths contribute; residence cohorts exclude insufficient follow-up
at each horizon. Paths/episodes are simulation units, not human replicates.
Conditional late means use only target-active path-trials, with denominators
in the source table. No statistical tests or parameter confidence intervals.
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
import yaml

from src.Bayesian_state.evaluation.model_0923_path_diagnosis import residence_retention, trace_tables


COLORS = {"B": "#8061A1", "C": "#C67E3D"}


def draw(source: Path, output: Path) -> None:
    validation = json.loads((source / "validation.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    settings = yaml.safe_load((source / "effective_config.yaml").read_text())
    if validation["status"] != "passed" or manifest["smoke"]:
        raise ValueError("Complete validated traces are required")
    if output.exists() or (source / "index.html").exists():
        raise FileExistsError("Use new figure and report destinations")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(settings["subjects"]), 3, figsize=(12, 7),
                             gridspec_kw={"width_ratios": [1.7, 1, 1]}, layout="constrained", squeeze=False)
    all_curves, all_batches, all_feedback, all_exits, all_retention, rows, source_files = [], [], [], [], [], [], []
    catalog_descriptions = []
    for i, item in enumerate(settings["subjects"]):
        sid = item["subject"]
        directory = source / f"subject_{sid}"
        summary = json.loads((directory / "summary.json").read_text())
        reference = summary["reference_rule"]
        curves, batches, feedback, exits = trace_tables(directory, reference, settings["late_window"])
        catalog = pd.read_csv(directory / "rule_catalog.csv")
        useful = catalog.loc[catalog.physical_argmax_accuracy >= settings["geometric_accuracy_threshold"]]
        names = "、".join(f"R{int(r.rule)}（{r.physical_argmax_accuracy:.1%}）" for r in useful.itertuples()) or "无"
        catalog_descriptions.append(f"S{sid}达到{settings['geometric_accuracy_threshold']:.0%}阈值的规则为：{names}；A后期参考规则为R{reference}。")
        episodes = pd.read_csv(directory / "episodes.csv")
        visits = episodes[episodes.rule == reference]
        for frame, collector in ((curves, all_curves), (batches, all_batches), (feedback, all_feedback), (exits, all_exits)):
            frame.insert(0, "subject", sid)
            collector.append(frame)
        ax, survival, readout = axes[i]
        ax.set_title(f"{chr(97+3*i)}  S{sid}: R{reference} availability", loc="left", fontweight="bold")
        survival.set_title(f"{chr(98+3*i)}  Residence retention", loc="left", fontweight="bold")
        readout.set_title(f"{chr(99+3*i)}  Late, while R{reference} active", loc="left", fontweight="bold")
        for j, kind in enumerate(("B", "C")):
            group = batches[batches.condition == kind]
            count = int(group.paths.sum())
            curve = curves[curves.condition == kind].copy()
            for key in ("ever", "active", "contribution"):
                curve[key] *= curve.paths
            curve = curve.groupby("trial")[["ever", "active", "contribution", "paths"]].sum()
            for key, style, label in (("ever", "-", "ever entered"), ("active", "--", "currently active")):
                ax.plot(curve.index, curve[key]/curve.paths, color=COLORS[kind], linestyle=style,
                        label=f"{kind}: {label} (n={count})", linewidth=1.7)
            local = visits[visits.condition == kind]
            retention = residence_retention(local, int(group.trials.iloc[0]))
            retention.insert(0, "condition", kind)
            retention.insert(0, "subject", sid)
            all_retention.append(retention)
            survival.plot(retention.horizon, retention.fraction, marker="o", color=COLORS[kind],
                          label=f"{kind}: {len(local)} residences", markersize=4)
            denom = int(group.late_active_trials.sum())
            keys = ("prior", "contribution", "rule_correct")
            means = [group["late_"+key+"_sum"].sum()/denom if denom else np.nan for key in keys]
            readout.plot(np.arange(3)+(j-.5)*.15, means, "o", color=COLORS[kind], label=f"{kind}: {denom} path-trials")
            removed = exits[exits.condition == kind] if len(exits) else exits
            fields = dict(subject=sid, condition=kind, reference=reference, paths=count,
                ever_fraction=group.ever_paths.sum()/count,
                initial_fraction=group.initial_paths.sum()/count,
                proposals_per_path=group.all_proposals.sum()/count,
                target_proposals=int(group.proposed.sum()),
                late_active_fraction=denom/(count*min(settings["late_window"], int(group.trials.iloc[0])-1)),
                residences=len(local), exits=int((local.exit_cause == "autonomous_exit").sum()),
                capacity_drops=int((local.exit_cause == "capacity").sum()), censored=int(local.right_censored.sum()),
                median_completed_residence=local.loc[~local.right_censored, "duration"].median(),
                retention_8=retention.loc[retention.horizon == 8, "fraction"].iloc[0],
                late_prior_given_active=means[0], late_contribution_given_active=means[1],
                late_rule_correct_given_active=means[2], late_active_path_trials=denom)
            fields["exited_with_posterior_ge_half"] = int((removed.posterior >= .5).sum()) if len(removed) else 0
            fields["exited_with_contribution_ge_half"] = int((removed.contribution >= .5).sum()) if len(removed) else 0
            fields["exited_after_full_correct"] = int((removed.previous_feedback == 1).sum()) if len(removed) else 0
            for key in ("beta_pre", "pairing_pre_true"):
                fields["late_"+key+"_given_active"] = group["late_"+key+"_sum"].sum()/denom if denom else np.nan
            rows.append(fields)
        for axis in (ax, survival, readout):
            axis.set_ylim(-.03, 1.03)
            axis.set_yticks([0, .25, .5, .75, 1])
            axis.grid(axis="y", color="#ECECEC", linewidth=.6)
        ax.set_xlabel("Valid trial")
        ax.set_ylabel("Fraction of paths")
        ax.legend(fontsize=7.5, loc="upper left")
        survival.set_xscale("log", base=2)
        survival.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        survival.set_xlabel("Feedback observations in residence")
        survival.set_ylabel("Fraction retained among eligible entries")
        survival.legend(fontsize=7.5, loc="lower left")
        readout.set_xticks([0, 1, 2], ["Belief", "Response\nweight", "Rule correct\nprobability"])
        readout.tick_params(axis="x", labelsize=8)
        readout.set_ylabel("Mean conditional on presence")
        readout.legend(fontsize=7.5, loc="lower left")
        source_files.extend([directory / "episodes.csv", directory / "summary.json", directory / "rule_catalog.csv"])
        source_files.extend(sorted((directory / "traces").glob("*.npz")))
    fig.savefig(output / "path_diagnosis.png", dpi=300, facecolor="white")
    plt.close(fig)
    summary = pd.DataFrame(rows)
    for name, frame in (("summary", summary), ("curves", pd.concat(all_curves)),
                        ("batches", pd.concat(all_batches)), ("feedback", pd.concat(all_feedback)), ("exits", pd.concat(all_exits)),
                        ("retention", pd.concat(all_retention))):
        frame.to_csv(output / f"{name}_plot_source.csv", index=False)
    table = summary.copy()
    for key in table:
        if pd.api.types.is_float_dtype(table[key]):
            table[key] = table[key].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    image_path = html.escape(os.path.relpath(output / "path_diagnosis.png", source))
    descriptions = []
    for row in rows:
        descriptions.append(f"<p><b>S{row['subject']}，{row['condition']}</b>：{row['paths']}条路径中，"
            f"{row['ever_fraction']:.1%}曾考虑R{row['reference']}；最后64题的包含率为{row['late_active_fraction']:.1%}。"
            f"每条路径平均提出{row['proposals_per_path']:.1f}次新规则。参考规则共出现{row['residences']}段驻留，"
            f"其中{row['exits']}次自主退出、{row['capacity_drops']}次容量移除，{row['censored']}段延续至记录结束。"
            f"具有足够后续记录的驻留中，{row['retention_8']:.1%}持续至少8题。"
            f"移除事件中，{row['exited_with_posterior_ge_half']}次的上一题反馈后信念至少为0.5，"
            f"{row['exited_with_contribution_ge_half']}次的上一题作答贡献至少为0.5。"
            f"后期参考规则仍在候选集合时，平均信念为{row['late_prior_given_active']:.3f}，"
            f"作答贡献为{row['late_contribution_given_active']:.3f}，"
            f"该规则的本题正确概率为{row['late_rule_correct_given_active']:.3f}。</p>")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Model 0923：规则路径诊断</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 24px;font:17px/1.7 system-ui;color:#25313c}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border-bottom:1px solid #ddd}}.table{{overflow-x:auto}}.note{{background:#f1f3f5;padding:16px}}</style>
<h1>Model 0923：规则怎样进入、保留并影响回答</h1>
<p>固定335和307原有参数，补录已保存路径的内部事件。B为真实回答与反馈驱动的等权路径，每人4个保存种子×128条路径；
C为原有64条自主路径。全部自主回答与反馈逐项重现，B逐路径正确概率与原值一致。原拟合的未解决及边界审查状态保留。</p>
<p>按完整物理刺激序列和真实类别事后评价规则目录。{' '.join(catalog_descriptions)}
规则筛选仅用于诊断，真实类别没有进入模型的反馈前状态更新。
几何评分使用各规则类别概率的最大值；并列最大类别平分正确计分。逐规则结果保存在各被试rule_catalog.csv。</p>
<img src="{image_path}" alt="两名被试规则发现、驻留与作答贡献">
{''.join(descriptions)}
<h2>如何理解图中的三个环节</h2>
<p>左图实线表示截至该题曾进入候选集合的路径比例，虚线表示本题仍在集合中的比例。
中图按驻留长度检查保留率，每个长度只计入尚有足够后续记录的进入事件；末尾记录不足的驻留保留删失标记。
右图只汇总最后64题中参考规则仍在集合的路径试次，分母写在图例与源表中；它描述候选可用后的内部状态。</p>
<p>信念为当前选择前的规则概率，作答贡献加入主导惯性。规则正确概率由该条路径的实际知觉与精度计算。
完整路径文件还保存反馈后的信念、精度和配对分布。feedback_plot_source.csv把反馈引起的支持变化
与遗忘后的信念作比较，并另外保存相对原信念的净变化。</p>
<h2>数据表</h2><div class="table">{table.to_html(index=False, escape=True)}</div>
<p><a href="validation.json">重放与数值校验</a> · <a href="manifest.json">来源与计算记录</a> ·
<a href="subject_335/episodes.csv">335完整驻留表</a> · <a href="subject_307/episodes.csv">307完整驻留表</a></p>
<p class="note">这是一项固定参数、样本内的路径诊断。路径、驻留与路径试次属于模拟记录，两个被试属于开发案例；
本次没有群体推断、参数恢复或重新拟合。B的四个种子按文件顺序预先选取，C保留全部64条原轨迹。
前8题单被试检查先通过后执行完整重放。驻留从初始化题开始计数，行为评分仍沿用首题初始化约定。
本次结果定位机制中的困难环节，因果归因和修订效果需要后续有针对性的对照。</p></html>"""
    with (source / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    source_files += [source / "manifest.json", source / "validation.json", Path(__file__),
                     Path("src/Bayesian_state/evaluation/model_0923_path_diagnosis.py")]
    contract = dict(backend="python", archetype="quantitative grid", export="300 dpi PNG; screen review",
        conclusion="Separate candidate discovery, residence and conditional readout in saved model paths",
        exclusions="Retention horizon k excludes entries with fewer than k remaining trials; late readout conditions on rule presence",
        units="Two development subjects; 512 B paths and 64 C paths per subject; episodes/path-trials are dependent within paths",
        uncertainty="Fixed selected parameters; descriptive simulation variation; no tests or confidence intervals",
        publication_format_exceptions="PNG only per repository policy; 12x7 inch screen layout, not final journal typography",
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output / "figure_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--figure-output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(args.analysis_dir, args.figure_output_dir)


if __name__ == "__main__":
    main()
