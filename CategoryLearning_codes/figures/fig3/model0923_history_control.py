"""Plot the fixed-parameter A/B/C history control and build its review report.

Figure contract: compare learning with observed-choice particle conditioning (A),
uniform paths receiving observed choices/feedback (B), and autonomous learning
(C); then show where A/B reference-rule states diverge. Quantitative grid with
learning curves as the main evidence. Python only, 300 dpi PNG, screen review.
All scored trials enter non-overlapping windows; every B/C path enters the late
distribution. Seeds and trajectories describe simulation variation, not people.
See src/Bayesian_state/docs/model_architecture/model_0923_history_control.md.
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


COLORS = dict(A="#3578A7", B="#8662A6", C="#CC813F", observed="#31363B")


def window_table(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    """Retain all scored trials, including a final incomplete window."""
    scored = frame.loc[frame.scored].copy()
    if len(scored) != len(frame)-1 or bool(frame.scored.iloc[0]):
        raise ValueError("Only the initialization trial may be unscored")
    rows = []
    for start in range(0, len(scored), size):
        section = scored.iloc[start:start+size]
        row = section.select_dtypes(include="number").mean().to_dict()
        row.update(first_valid_trial=int(section.valid_trial.iloc[0]),
                   last_valid_trial=int(section.valid_trial.iloc[-1]), trials=len(section))
        rows.append(row)
    return pd.DataFrame(rows)


def probability_axis(ax, *, xlabel=True):
    ax.set_ylim(-.02, 1.03)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(axis="y", color="#E9E9E9", linewidth=.6)
    if xlabel:
        ax.set_xlabel("Valid trial")


def draw(source: Path, output: Path) -> None:
    manifest = json.loads((source / "manifest.json").read_text())
    validation = json.loads((source / "validation.json").read_text())
    if validation["status"] != "passed" or manifest["smoke"]:
        raise ValueError("A complete validated analysis is required for the review figures")
    if output.exists() or (source / "index.html").exists():
        raise FileExistsError("Use new figure and report destinations")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["DejaVu Sans"],
                         "font.size":10, "axes.titlesize":11, "axes.spines.top":False,
                         "axes.spines.right":False, "legend.frameon":False,
                         "svg.fonttype":"none", "pdf.fonttype":42})
    subjects = manifest["subjects"]
    settings = manifest["settings"]
    overview, axes = plt.subplots(len(subjects), 2, figsize=(11, 3.5*len(subjects)),
                                  gridspec_kw={"width_ratios":[2.6, 1]}, squeeze=False,
                                  layout="constrained")
    states, state_axes = plt.subplots(4, len(subjects), figsize=(11, 9), sharex="col",
                                      squeeze=False, layout="constrained")
    sources, distributions = [], []
    for i, subject in enumerate(subjects):
        sid = subject["subject"]
        folder = source / f"subject_{sid}"
        frame = pd.read_csv(folder / "trials.csv")
        windows = window_table(frame, settings["window_size"])
        windows.insert(0, "subject", sid)
        sources.append(windows)
        x = windows.valid_trial
        ax, hist = axes[i]
        ax.set_title(f"{'ac'[i]}  S{sid} | inertia = {subject['inertia']:g}", loc="left", fontweight="bold")
        for key, column, label, linestyle in (
            ("observed", "observed_correct", "Observed", ":"),
            ("A", "A_correct_probability", "A: conditional", "-"),
            ("B", "B_correct_probability", "B: observed history, uniform paths", "-"),
            ("C", "C_mean_correct_probability", "C: autonomous", "--")):
            ax.plot(x, windows[column], color=COLORS[key], label=label,
                    linestyle=linestyle, linewidth=1.8)
        ax.axhline(.25, color="#9A9A9A", linewidth=.7, linestyle=":")
        ax.axvspan(len(frame)-settings["late_window"]+.5, len(frame)+.5,
                   color="#000000", alpha=.035, linewidth=0)
        probability_axis(ax)
        ax.set_ylabel("Correct response probability")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8.5, ncol=2)
        with np.load(folder / "comparison.npz", allow_pickle=False) as arrays:
            late = slice(-settings["late_window"], None)
            for key, array in (("B", "B_path_correct"), ("C", "C_path_correct")):
                values = arrays[array][:, late].mean(axis=1)
                hist.hist(values, bins=np.linspace(0, 1, 21), weights=np.ones(len(values))/len(values),
                          histtype="step", linewidth=1.7, color=COLORS[key], label=f"{key}: {len(values):,} paths",
                          linestyle="--" if key=="C" else "-")
                distributions.extend(dict(subject=sid, mode=key, path=j, late_expected=float(value))
                                     for j, value in enumerate(values))
        hist.set_title(f"{'bd'[i]}  Last {settings['late_window']} trials", loc="left", fontweight="bold")
        hist.set_xlim(0, 1)
        hist.set_xticks([0, .5, 1])
        hist.set_xlabel("Mean correct probability per path")
        hist.set_ylabel("Fraction of paths")
        hist.legend(fontsize=8.5)
        for j, (field, label) in enumerate((
            ("active", "Reference rule present"), ("belief", "Rule belief"),
            ("choice_weights", "Response weight"), ("dominant", "Dominant-rule probability"))):
            ax = state_axes[j, i]
            for key, style in (("A", "-"), ("B", "--")):
                ax.plot(x, windows[f"{key}_reference_{field}"], label=key, color=COLORS[key],
                        linestyle=style, linewidth=1.8)
            probability_axis(ax, xlabel=j==3)
            ax.set_ylabel(label)
            ax.set_title(f"{chr(97+j*len(subjects)+i)}  S{sid} | R{subject['reference_rule']}",
                         loc="left", fontweight="bold")
            if j==0:
                ax.legend(loc="upper left", ncol=2)
    overview.savefig(output / "history_comparison.png", dpi=300, facecolor="white")
    states.savefig(output / "reference_rule_comparison.png", dpi=300, facecolor="white")
    plt.close(overview)
    plt.close(states)
    pd.concat(sources, ignore_index=True).to_csv(output / "window_plot_source.csv", index=False)
    distribution = pd.DataFrame(distributions)
    distribution.to_csv(output / "late_path_plot_source.csv", index=False)
    table = pd.read_csv(source / "comparison.csv").query("period == 'late'").copy()
    report(source, output, manifest, table, distribution)
    files = [source / "manifest.json", source / "comparison.csv", Path(__file__)]
    files += [source / f"subject_{s['subject']}" / name for s in subjects
              for name in ("trials.csv", "comparison.npz", "summary.json")]
    contract = dict(backend="python", archetype="quantitative grid", purpose="fixed-parameter history-control diagnosis",
        export=dict(format="PNG", dpi=300, use="screen review"),
        trials="All scored trials; initialization trial excluded; last short window retained",
        center="A/B mean correct-category probability; C mean across 64 autonomous paths",
        distribution="Fractions in bins of width 0.05; each path represented once using its last-64 mean",
        repeats="B 16 seeds x 128 paths; C 64 saved paths; two selected cases, no population inference",
        state_timing="Before the current choice and feedback; A weights use earlier choices only",
        reference_rule="Highest A mean response weight over the last 64 trials; descriptive selection",
        parameter_uncertainty="Not estimated; parameters fixed at saved fitted values; in-sample diagnostic",
        statistical_test=None, missing_observations_excluded=0,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output / "figure_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2)+"\n")


def report(source: Path, output: Path, manifest: dict, table: pd.DataFrame, distribution: pd.DataFrame):
    """Use measured values in a short, standalone Chinese review page."""
    rows, details = [], []
    for subject in manifest["subjects"]:
        sid = subject["subject"]
        row = table.loc[table.subject==sid].iloc[0]
        pct = lambda value: f"{100*float(value):.1f}%"
        rows.append("<tr>"+"".join(f"<td>{value}</td>" for value in (
            sid, f"{subject['inertia']:g}", pct(row.observed_accuracy), pct(row.A_expected),
            pct(row.B_expected), pct(row.C_expected_mean),
            f"{100*row.A_minus_B:.1f}", f"{100*row.B_minus_C_mean:.1f}"))+"</tr>")
        strong_b = distribution.query("subject==@sid and mode=='B'").late_expected.ge(.8).mean()
        strong_c = distribution.query("subject==@sid and mode=='C'").late_expected.ge(.8).mean()
        details.append(f"""<h2>被试 {sid}</h2>
<p>最后64题中，B的16个种子均值范围为 {pct(row.B_seed_mean_min)}–{pct(row.B_seed_mean_max)}，
种子间标准差为 {100*row.B_seed_mean_sd:.2f} 个百分点。这是模拟重复的差异，参数固定为原拟合值。</p>
<p>按每条路径最后64题的平均正确概率衡量，达到0.8的路径比例：B为 {pct(strong_b)}，C为 {pct(strong_c)}。
原图b对应的C路径实际正确率中位数为 {pct(row.C_realized_median)}；本页主图统一展示平均正确概率，C为 {pct(row.C_expected_mean)}。</p>
<p>A后期作答权重最高的参考规则为R{subject['reference_rule']}。
其进入候选集合的概率从A的 {pct(subject['A_late_reference']['active'])} 变为B的 {pct(subject['B_late_reference']['active'])}；
作答权重从 {pct(subject['A_late_reference']['choice_weights'])} 变为 {pct(subject['B_late_reference']['choice_weights'])}；
主导规则概率从 {pct(subject['A_late_reference']['dominant'])} 变为 {pct(subject['B_late_reference']['dominant'])}。
这些数值是跨路径的汇总，参考规则按A的后期权重选择。</p>
<p>来源拟合状态：{html.escape(subject['fit_status'])}；待解决项：{html.escape(', '.join(subject['fit_issues']))}。</p>""")
    images = {name:html.escape(os.path.relpath(output/name, source)) for name in
              ("history_comparison.png", "reference_rule_comparison.png")}
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Model 0923：335 / 307 真实历史对照</title>
<style>body{{max-width:1150px;margin:36px auto;padding:0 24px;font:17px/1.7 system-ui,sans-serif;color:#20262c}}
h1{{font-size:29px}}h2{{font-size:23px;margin-top:28px}}table{{border-collapse:collapse;width:100%;font-size:15px}}
td,th{{padding:9px;border-bottom:1px solid #d9dce0;text-align:right}}th{{background:#f2f4f6}}
img{{width:100%;height:auto}}.note{{background:#f4f5f7;padding:14px 20px}}code{{font-size:14px}}</style>
<h1>Model 0923：真实历史能否带来后期学习？</h1>
<p>固定335和307的已有拟合参数，比较三种信息条件。新增计算为B：每人16个种子，每个种子128条路径；
A沿用已保存的16次粒子滤波，C沿用64条自主轨迹。335有748个有效试次，307有704个有效试次，均为条件3。</p>
<table><tr><th>条件</th><th>用于反馈学习的回答和反馈</th><th>按真实选择调整路径权重</th></tr>
<tr><td>A：conditional prediction</td><td>被试真实记录</td><td>逐题加权，必要时重采样</td></tr>
<tr><td>B：真实历史、等权路径</td><td>被试真实记录</td><td>路径始终等权，关闭重采样</td></tr>
<tr><td>C：自主生成</td><td>模型生成回答及其任务反馈</td><td>每条自主轨迹独立推进</td></tr></table>
<p>B保留候选搜索、退出、信念、配对与精度等认知更新。预测在本题选择和反馈到来之前读取。
作答权重仍由各路径的规则信念和惯性计算。B关闭的是研究者用于推断潜在路径的粒子加权。
口述数据未参与这次对照。</p>
<h2>最后64题的比较</h2>
<table><tr><th>被试</th><th>惯性λ</th><th>真实正确率</th><th>A平均正确概率</th><th>B平均正确概率</th>
<th>C平均正确概率</th><th>A−B（百分点）</th><th>B−C（百分点）</th></tr>{''.join(rows)}</table>
<p>A、B、C均按试次和路径取平均正确概率。真实正确率按反馈为1计算，条件3的0.5反馈不计为完全正确。</p>
<img src="{images['history_comparison.png']}" alt="两名被试的A/B/C学习曲线及后期路径分布">
<p>左列按连续32个有效计分试次分组求均值，最后不足32题的窗口保留；第一题为初始化题，排除计分。
浅灰区域标出最后64题。右列展示每条路径最后64题的平均正确概率分布；每组内柱高之和为1，箱宽0.05。</p>
{''.join(details)}
<h2>参考规则的内部状态</h2>
<img src="{images['reference_rule_comparison.png']}" alt="A和B的参考规则候选、信念、作答权重和主导概率">
<p>四行分别为参考规则在候选集合中的概率、规则信念、作答权重及主导规则概率。
A用当时的粒子权重汇总；B用等权路径汇总。各条曲线均采用与学习曲线相同的32题分组。
规则编号沿用原目录的零基索引。</p>
<h2>怎样解释这个对照</h2>
<p>A与B的差异衡量：在同一被试历史下，按真实选择更新路径分布、并重采样，会怎样改变模型预测。
B与C的差异衡量：沿着真实回答和反馈推进，与自主产生回答和反馈相比，会怎样改变预测。
两组差异共同描述当前参数下的运行行为，不能直接分摊到某个认知模块。</p>
<p>A是条件于过去选择的逐题预测。A在本题预测之前已利用过去的真实选择筛选潜在路径；
因此A表现良好与大量独立路径自主学会之间仍需分别验证。所有参数来自完整被试序列，本次属于样本内机制诊断。</p>
<p>335的λ为0，每条路径中的规则信念和作答权重相等。若B和C仍然偏低，后续应继续检查候选规则出现、保留、
证据积累和精度变化。307的λ为2，对该被试区分惯性与其他模块的作用，还需专门改变惯性的对照。</p>
<h2>复现与验证</h2>
<p>执行使用拟合时已有的冻结源码；科学输入和复用的A/C产物均检查SHA-256。
两名被试原A的第一个种子各复现前8题，逐项差异为0。
B检查逐题等权、ESS为128、无重采样、预测概率归一化、路径均值与汇总一致，以及λ为0时信念与权重一致。
全部参数保持原值，新增参数拟合为0次，新增自主模拟为0次。</p>
<p><a href="comparison.csv">分阶段数值表</a> · <a href="validation.json">运行验证</a> ·
<a href="manifest.json">来源、配置与哈希</a> ·
<a href="{html.escape(os.path.relpath(output/'window_plot_source.csv', source))}">曲线绘图数据</a> ·
<a href="{html.escape(os.path.relpath(output/'late_path_plot_source.csv', source))}">路径分布数据</a></p>
</html>"""
    with (source / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--figure-output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(args.analysis_dir, args.figure_output_dir)


if __name__ == "__main__":
    main()
