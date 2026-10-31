"""Audit a saved fit and export its descriptive report and PNG diagnostics.

Figure contract: a quantitative grid with the learning curve as the main panel;
show observed/model agreement, fixed-parameter states and independent searches.
All trials contribute. Block means are descriptive, with no confidence interval
or significance test. Export: Python/matplotlib, 180 x 185 mm, PNG at 300 dpi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_csv_history
from .model import Parameters, SUSTAIN
from .sequence import run_sequence


def block_summary(trials: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """Aggregate every trial, keeping session boundaries and partial last blocks."""
    if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    rows = []
    for session, group in trials.groupby("iSession", sort=False):
        for start in range(0, len(group), int(block_size)):
            block = group.iloc[start:start + int(block_size)]
            rows.append(dict(session=int(session), first_trial=int(block.global_trial.iloc[0]),
                             last_trial=int(block.global_trial.iloc[-1]), n=len(block),
                             trial_center=float(block.global_trial.mean()),
                             observed_accuracy=float(block.correct.mean()),
                             model_expected_accuracy=float(block.p_correct.mean()),
                             choice_nll=float(block.choice_nll.sum())))
    return pd.DataFrame(rows)


def _plot(blocks: pd.DataFrame, states: pd.DataFrame, progress: pd.DataFrame,
          sessions: list[int], title: str, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(180 / 25.4, 185 / 25.4), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, .95])
    curve = fig.add_subplot(grid[0, :])
    attention = fig.add_subplot(grid[1, 0])
    clusters = fig.add_subplot(grid[1, 1])
    search = fig.add_subplot(grid[2, :])
    curve.plot(blocks.trial_center, blocks.observed_accuracy, "o-", color="#333333",
               markersize=3, linewidth=1.2, label="Observed choices")
    curve.plot(blocks.trial_center, blocks.model_expected_accuracy, "s-", color="#377DA8",
               markersize=3, linewidth=1.2, label="Fitted P(correct)")
    curve.axhline(.5, color=".7", linestyle=":", linewidth=.8)
    curve.set(ylabel="Accuracy / probability", ylim=(0, 1.04), xlabel="Trial across sessions")
    curve.legend(loc="lower right", ncol=2, fontsize=7)
    colors = ["#377DA8", "#B88745", "#7F6E9A", "#57978A"]
    for i, column in enumerate(c for c in states if c.startswith("attention_")):
        attention.plot(states.global_trial, states[column], color=colors[i % len(colors)],
                       linewidth=1, label=f"F{i + 1}")
    attention.set(ylabel="Relative dimension weight", ylim=(0, 1.02), xlabel="Trial")
    attention.legend(ncol=2, loc="upper right", fontsize=6)
    clusters.step(states.global_trial, states.clusters_after, where="post", color="#577186", linewidth=1.1)
    clusters.set(ylabel="Clusters after feedback", xlabel="Trial", ylim=(0, states.clusters_after.max() + 1))
    for ax in (curve, attention, clusters):
        ax.set_xlim(.5, len(states) + .5)
        for boundary in sessions:
            ax.axvline(boundary + .5, color=".65", linestyle="--", linewidth=.7)
    for i, (restart, group) in enumerate(progress.groupby("restart", sort=False)):
        search.plot(group.generation, group.best_nll, color=colors[i % len(colors)], linewidth=1.1,
                    label=f"Search {restart}")
    search.set(xlabel="Generation", ylabel="Best choice NLL")
    search.legend(loc="upper right", ncol=3, fontsize=7)
    for ax, letter, label in ((curve,"a","Observed and fitted learning"),
                               (attention,"b","Attention before feedback"),
                               (clusters,"c","Cluster recruitment"),
                               (search,"d","Independent optimizer runs")):
        ax.set_title(f"{letter}  {label}", loc="left", fontweight="bold", pad=7)
    fig.suptitle(title + "\nFull-record fit; session boundaries marked by dashed lines", fontsize=9)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def make_report(run_dir: Path, outdir: Path) -> dict:
    """Recompute saved probabilities and summarize the unchanged run outputs."""
    if outdir.exists():
        raise FileExistsError(f"Report directory already exists: {outdir}")
    summary = json.loads((run_dir / "summary.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())
    traces = json.loads((run_dir / "state_trace.json").read_text())
    config = provenance["config"]
    diagnostics = config.get("diagnostics", {})
    data = load_csv_history(config["data"])
    if data.source_sha256 != provenance["data"]["sha256"]:
        raise ValueError("Source data differ from the fitted run")
    # Replay uses the same core that was hashed at fitting time.
    for name in ("model.py", "sequence.py"):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != provenance["source_sha256"][name]:
            raise ValueError(f"Model source changed since fitting: {name}")
    trials = pd.read_csv(run_dir / "trials.csv")
    if len(trials) != len(data.targets) or len(traces) != len(trials):
        raise ValueError("Saved trial/state lengths differ from the configured history")
    if not np.array_equal(trials.sustain_source_row, data.frame.sustain_source_row):
        raise ValueError("Saved row identities differ from the input history")
    for column in config["data"]["order_columns"]:
        if not np.array_equal(trials[column], data.frame[column]):
            raise ValueError("Saved trial order differs from the input history")
    if not np.array_equal(trials.choice, [data.category_values[i] for i in data.choices]):
        raise ValueError("Saved choices differ from the input history")
    if not np.array_equal(trials.target, [data.category_values[i] for i in data.targets]):
        raise ValueError("Saved targets differ from the input history")
    model = SUSTAIN(data.dimensions, len(data.category_values), Parameters(**summary["parameters"]),
                    **summary["model_options"])
    replay = run_sequence(model, data.stimuli, data.targets, choices=data.choices, trace=True)
    probabilities = trials[[f"p_category_{value}" for value in data.category_values]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(probabilities, replay.probabilities, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay.negative_log_likelihood, summary["negative_log_likelihood"], atol=1e-10)
    if traces != replay.traces:
        raise ValueError("Saved state traces differ from deterministic replay")
    n = len(trials)
    trials["global_trial"] = np.arange(1, n + 1)
    trials["p_correct"] = probabilities[np.arange(n), data.targets]
    trials["correct"] = data.choices == data.targets
    if "iSession" not in trials:
        trials["iSession"] = 1
    lambdas = np.array([item["lambdas_before"] for item in traces])
    log_attention = summary["parameters"]["r"] * np.log(lambdas)
    log_attention -= np.logaddexp.reduce(log_attention, axis=1)[:, None]
    states = pd.DataFrame({"global_trial": np.arange(1, n + 1),
                           "clusters_after": [len(item["post_feedback"]["clusters"]) for item in traces]})
    for i in range(lambdas.shape[1]):
        states[f"lambda_{i + 1}"] = lambdas[:, i]
        states[f"attention_{i + 1}"] = np.exp(log_attention[:, i])
    blocks = block_summary(trials, diagnostics.get("block_size", 32))
    assert blocks.n.sum() == n
    progress = pd.DataFrame([json.loads(line) for line in (run_dir / "optimization_progress.jsonl").read_text().splitlines()])
    generation = progress.loc[progress.event.eq("generation"), [
        "restart", "seed", "generation", "best_nll", "population_nll_std", "elapsed_seconds"]].copy()
    optimization = summary["optimization"]
    restarts = pd.DataFrame([{**{k: entry[k] for k in ("restart", "seed", "best_nll", "success", "iterations", "evaluations")},
                              **entry["parameters"]} for entry in optimization["restarts"]])
    observed_frequency = np.bincount(data.choices, minlength=len(data.category_values)) / n
    bias_nll = -float(np.log(observed_frequency[data.choices]).sum())
    boundary_fraction = diagnostics.get("boundary_fraction", .01)
    boundary_parameters = []
    for name, (lower, upper) in optimization["bounds"].items():
        fraction = (summary["parameters"][name] - lower) / (upper - lower)
        if min(fraction, 1 - fraction) <= boundary_fraction:
            boundary_parameters.append(name)
    sessions = trials.groupby("iSession", sort=False).agg(n=("correct", "size"),
        observed_accuracy=("correct", "mean"), model_expected_accuracy=("p_correct", "mean"),
        choice_nll=("choice_nll", "sum"))
    metrics = dict(
        n_trials=n, observed_accuracy=float(trials.correct.mean()),
        model_expected_accuracy=float(trials.p_correct.mean()), choice_nll=summary["negative_log_likelihood"],
        uniform_choice_nll=float(n * np.log(len(data.category_values))), constant_choice_bias_nll=bias_nll,
        mean_assigned_choice_probability_geometric=float(np.exp(-summary["negative_log_likelihood"] / n)),
        best_two_nll_gap=optimization["best_two_nll_gap"],
        restart_agreement_tolerance=diagnostics.get("best_two_nll_tolerance", .5),
        best_two_agree=(optimization["best_two_nll_gap"] is not None and
                       optimization["best_two_nll_gap"] <= diagnostics.get("best_two_nll_tolerance", .5)),
        converged_restarts=int(restarts.success.sum()), final_clusters=summary["final_clusters"],
        near_boundary_parameters=boundary_parameters,
        replay_max_probability_difference=float(np.max(np.abs(probabilities - replay.probabilities))),
        replay_state_traces_equal=True, optimization_elapsed_seconds=float(progress.elapsed_seconds.max()),
        total_evaluations=optimization["evaluations"], full_record_fit=config["fit"]["train_trials"] == n,
    )
    outdir.mkdir(parents=True, exist_ok=False)
    trials.to_csv(outdir / "trial_diagnostics.csv", index=False)
    blocks.to_csv(outdir / "block_summary.csv", index=False)
    states.to_csv(outdir / "state_summary.csv", index=False)
    restarts.to_csv(outdir / "restart_summary.csv", index=False)
    generation.to_csv(outdir / "convergence.csv", index=False)
    sessions.to_csv(outdir / "session_summary.csv")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    subject = config["data"]["filters"].get("iSub", "selected")
    boundaries = trials.groupby("iSession", sort=False).global_trial.max().tolist()[:-1]
    _plot(blocks, states, generation, boundaries, f"SUSTAIN | Subject {subject} | {n} trials", outdir / "fit_diagnostics.png")
    p = summary["parameters"]
    restart_lines = ["| 搜索 | 种子 | NLL | 收敛 | 代数 | r | beta | d | eta |",
                     "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in restarts.itertuples():
        restart_lines.append(f"| {row.restart} | {row.seed} | {row.best_nll:.6f} | {'是' if row.success else '否'} | {row.iterations} | {row.r:.5f} | {row.beta:.5f} | {row.d:.5f} | {row.eta:.6f} |")
    session_lines = ["| 会话 | 题数 | 实际正确率 | 模型平均 P(正确) | 选择 NLL |", "| --- | --- | --- | --- | --- |"]
    for session, row in sessions.iterrows():
        session_lines.append(f"| {session} | {int(row.n)} | {row.observed_accuracy:.2%} | {row.model_expected_accuracy:.2%} | {row.choice_nll:.6f} |")
    scope = "全部记录用于参数选择，本次结果为样本内描述；留出预测和参数恢复尚待验证。" if metrics["full_record_fit"] else "参数仅用配置中的训练前缀选择，后续试次按持续反馈协议评价。"
    report = f"""# SUSTAIN 被试 {subject} 拟合

选中参数的选择负对数似然（NLL）为 **{metrics['choice_nll']:.6f}**。
NLL 把每题实际选择的模型概率取负对数后相加，数值较小表示对该记录赋予较高概率。
{len(restarts)} 次独立搜索中，{metrics['converged_restarts']} 次满足优化器收敛条件；
最好两次相差 {metrics['best_two_nll_gap']:.8f}，本轮预先设置的重复搜索容忍为 {metrics['restart_agreement_tolerance']} NLL。
{scope}

## 数据与模型

保留 {n} 题，按原记录顺序跨会话延续状态。类别目录为 {data.category_values}，每题选择与对错反馈共同确定反馈类别。
四个连续刺激维度按任务量程编码；使用 `prediction_error` 簇招募和 `first` 并列簇规则。
参数在完整学习过程中固定，簇、注意调节与连接权重逐题更新。口述资料未参与本轮拟合。

{chr(10).join(session_lines)}

## 参数与搜索

选中参数：r = {p['r']:.8f}，beta = {p['beta']:.8f}，d = {p['d']:.8f}，eta = {p['eta']:.8f}。
r 控制维度注意贡献的差异，beta 控制簇竞争，d 控制反应一致性，eta 控制学习更新步幅。
最终形成 {summary['final_clusters']} 个簇。按搜索区间宽度的 {boundary_fraction:.0%} 判定，选中解接近边界的参数为：
{', '.join(boundary_parameters) if boundary_parameters else '无'}。

{chr(10).join(restart_lines)}

共完成 {optimization['evaluations']} 次目标函数计算。每次搜索最多 {optimization['maxiter']} 代，
种群含 {optimization['popsize'] * 4} 个候选；使用 {optimization['workers']} 个进程，每个进程一个数值线程。
搜索依次执行，种群评价在单一进程并行层完成。优化耗时约 {metrics['optimization_elapsed_seconds']:.1f} 秒。
优化器以种群 NLL 标准差不超过 `atol + tol × |种群平均 NLL|` 作为停止条件，
本轮 tol = {optimization['tol']}、atol = {optimization['atol']}。
不同局部解及其参数全部保留。重复搜索一致性衡量本次优化的稳定程度，心理参数可辨识性仍需恢复或剖面分析。

## 行为拟合

实际正确率为 {metrics['observed_accuracy']:.2%}，模型平均 P(正确) 为 {metrics['model_expected_accuracy']:.2%}。
均匀选择基线的 NLL 为 {metrics['uniform_choice_nll']:.6f}；用本记录估计固定反应偏好的基线为 {bias_nll:.6f}。
这些数值均在同一记录内计算。模型对实际选择赋予概率的几何均值为
{metrics['mean_assigned_choice_probability_geometric']:.4f}，定义为 exp(−NLL / 题数)。

![拟合曲线、注意权重、簇数与搜索过程](fit_diagnostics.png)

图 a 按会话内连续 {diagnostics.get('block_size',32)} 题汇总实际正确率和模型 P(正确)，全部试次均保留，
最后不足一个区块的试次单独汇总。虚线标记会话边界。图 b 为反馈前的相对维度权重，
由 lambda 的 r 次幂归一化得到，每题四个权重之和为 1；F1–F4 对应配置中的四个刺激列。
图 c 为反馈后的簇数。图 d 保留各搜索每代的最好 NLL。
这是一名被试的重复试次描述，三个搜索种子属于计算重复；图中没有置信区间或显著性检验。

## 验证与文件

重新读取同一输入并用选中参数逐题重放，保存概率与重放概率的最大绝对差为
{metrics['replay_max_probability_difference']:.3g}；全部反馈前后状态记录一致。
输入文件哈希、配置、软件版本和拟合时的源码哈希保存在上级目录的 `provenance.json`。
`restart_summary.csv` 保留三次搜索的参数与分数；`convergence.csv` 保留逐代进度；
`trial_diagnostics.csv`、`block_summary.csv`、`state_summary.csv` 和 `session_summary.csv` 为图表来源。
报告输出与原拟合目录分开保存，原数据及原拟合结果保持不变。
"""
    (outdir / "report.md").write_text(report, encoding="utf-8")
    qa = dict(backend="python", archetype="quantitative_grid", dimensions_mm=[180,185],
              export="PNG_300dpi", n_subjects=1, n_trials=n, excluded_trials=0,
              interval="none", test="none", data_use="full_record_fit" if metrics["full_record_fit"] else "prefix_fit",
              report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              replay=metrics["replay_state_traces_equal"],
              warnings_accepted=["PNG only follows repository export policy; no vector exports requested"])
    (outdir / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_report(args.run_dir, args.outdir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
