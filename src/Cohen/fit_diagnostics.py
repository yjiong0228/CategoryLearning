"""Full-trajectory numerical checks and in-sample fit diagnostics for exp123."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .continuous import ContinuousFeatureMixture
from .data import prepare_exp123
from .exp123 import parameter_record


def summarize_study(root: Path, plan: dict) -> dict:
    """Compare independent fits with successively finer complete replays.

    Candidate ranking uses the finest grid. Numerical adequacy of the fitted
    grid additionally requires every successive comparison to pass both declared
    tolerances; a failed check remains visible in the report and summary.
    """
    config = plan["config"]
    sid = config["subjects"][0]
    data = pd.read_csv(config["data"])
    frame, X, y, choices, feedback, condition = prepare_exp123(data[data.iSub == sid])
    rows, paths, traces = [], {}, {}
    for seed in plan["seeds"]:
        folder = root / f"seed_{seed}" / f"subject_{sid}"
        params = json.loads((folder / "parameters.json").read_text())["initial"]
        predictions = pd.read_csv(folder / "trial_predictions.csv")
        progress = pd.read_json(folder / "optimization_progress.jsonl", lines=True)
        traces[seed] = progress
        previous_p = predictions.filter(regex=r"^p_category\d+$").to_numpy()
        previous_nll = float(-predictions.log_likelihood.sum())
        row = {"seed": seed, "beta": params["beta"], "eta": params["eta"],
               f"nll_order{params['quadrature_order']}": previous_nll,
               "fit_elapsed_seconds": float(progress.elapsed_seconds.iloc[-1]),
               "n_evaluations": int(progress.n_evaluations.iloc[-1]),
               "all_quadrature_checks_pass": True}
        ga = progress[progress.phase == "ga"]
        row["last20_ga_improvement"] = float(ga.best_nll.iloc[max(0, len(ga) - 21)] - ga.best_nll.iloc[-1])
        row["anneal_improvement"] = float(ga.best_nll.iloc[-1] - progress.best_nll.iloc[-1])
        previous_order = params["quadrature_order"]
        with threadpool_limits(limits=1):
            for order in plan["check_orders"]:
                model = ContinuousFeatureMixture(**{**params, "quadrature_order": order})
                result = model.run_session(X, choices, feedback, condition, y)
                P = result["probabilities"]
                if not np.isfinite(P).all() or not np.allclose(P.sum(axis=1), 1, atol=1e-12):
                    raise FloatingPointError("invalid replay probabilities")
                nll = float(-result["log_likelihood"].sum())
                delta_nll = abs(nll - previous_nll)
                delta_p = float(np.max(np.abs(P - previous_p)))
                row[f"nll_order{order}"] = nll
                row[f"abs_nll_change_{previous_order}_to_{order}"] = delta_nll
                row[f"max_probability_change_{previous_order}_to_{order}"] = delta_p
                row["all_quadrature_checks_pass"] &= (delta_nll <= plan["nll_tolerance"]
                                                       and delta_p <= plan["probability_tolerance"])
                previous_nll, previous_p, previous_order = nll, P, order
        paths[seed] = (params, result, model)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(f"nll_order{plan['check_orders'][-1]}")
    dest = root / "diagnostics"
    dest.mkdir(exist_ok=False)
    table.to_csv(dest / "seed_comparison.csv", index=False)
    best_seed = int(table.seed.iloc[0])
    params, result, final_model = paths[best_seed]
    order = plan["check_orders"][-1]
    best = frame.copy()
    best["sequence_trial"] = np.arange(1, len(best) + 1)
    for index in range(final_model.n_categories):
        best[f"p_category{index + 1}"] = result["probabilities"][:, index]
    for key in ("p_correct", "predicted_category", "log_likelihood", "expected_feedback"):
        best[key] = result[key]
    best["scope"] = "train"
    best.to_csv(dest / "best_trial_predictions.csv", index=False)
    (dest / "best_parameters.json").write_text(json.dumps({
        "seed": best_seed, "selection": plan["selection"],
        "initial": {**params, "quadrature_order": order}, "final": parameter_record(final_model),
        "feature_subsets_zero_based": final_model.subsets}, indent=2), encoding="utf-8")
    np.savez_compressed(dest / "best_states.npz",
        alpha_before_feedback=result["alpha_before_feedback"],
        bias_before_feedback=result["bias_before_feedback"],
        sequence_trial=best.sequence_trial.to_numpy(), iSession=frame.iSession.to_numpy(),
        iTrial=frame.iTrial.to_numpy())
    P = result["probabilities"]
    observed_correct = choices == y
    block_rows = []
    for start in range(0, len(X), 64):
        stop = min(start + 64, len(X))
        block_rows.append({"first_trial": start + 1, "last_trial": stop, "n_trials": stop - start,
                           "observed_accuracy": float(observed_correct[start:stop].mean()),
                           "model_mean_p_correct": float(result["p_correct"][start:stop].mean()),
                           "choice_nll": float(-result["log_likelihood"][start:stop].sum())})
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(dest / "blocks64.csv", index=False)
    nll = float(-result["log_likelihood"].sum())
    chance_nll = len(X) * np.log(final_model.n_categories)
    metrics = {"subject": sid, "condition": condition, "n_trials": len(X), "best_seed": best_seed,
               "beta": params["beta"], "eta": params["eta"], "replay_order": order,
               "nll": nll, "uniform_choice_nll": float(chance_nll),
               "bits_per_trial_gain_over_uniform": float((chance_nll - nll) / len(X) / np.log(2)),
               "observed_accuracy": float(observed_correct.mean()),
               "model_mean_p_correct": float(result["p_correct"].mean()),
               "last64_observed_accuracy": float(observed_correct[-64:].mean()),
               "last64_model_mean_p_correct": float(result["p_correct"][-64:].mean()),
               "choice_agreement_argmax": float((result["predicted_category"] == choices).mean()),
               "multiclass_choice_brier": float(np.mean(np.sum((P - np.eye(P.shape[1])[choices - 1])**2, axis=1))),
               "best_quadrature_checks_pass": bool(table.all_quadrature_checks_pass.iloc[0]),
               "all_seeds_quadrature_checks_pass": bool(table.all_quadrature_checks_pass.all()),
               "seed_nll_range": float(table[f"nll_order{order}"].max() - table[f"nll_order{order}"].min()),
               "n_fit_parameters": final_model.n_categories * (final_model.n_features - 1) + 2}
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_diagnostics(dest, frame, observed_correct, result, traces, best_seed)
    _write_report(root, plan, table, blocks, metrics)
    return metrics


def _plot_diagnostics(dest: Path, frame: pd.DataFrame, correct: np.ndarray,
                      result: dict, traces: dict, best_seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    x = np.arange(1, len(frame) + 1)
    axes[0].plot(x, pd.Series(correct.astype(float)).rolling(32, min_periods=32).mean(),
                 color="black", label="Observed accuracy (trailing 32 trials)")
    axes[0].plot(x, pd.Series(result["p_correct"]).rolling(32, min_periods=32).mean(),
                 color="#2166ac", label="Fitted probability of correct choice (trailing 32)")
    for boundary in np.flatnonzero(np.diff(frame.iSession.to_numpy())) + 1:
        axes[0].axvline(boundary + .5, color="gray", ls="--", lw=1)
    axes[0].set(xlabel="Sequence trial", ylabel="Correct choice", ylim=(0, 1.03),
                title=f"Subject {frame.iSub.iloc[0]}: complete-record fit, selected seed {best_seed}")
    axes[0].legend(fontsize=8, loc="lower right")
    for seed, trace in traces.items():
        axes[1].plot(trace.n_evaluations, trace.best_nll, label=str(seed))
    axes[1].set(xlabel="Objective evaluations", ylabel="Best choice NLL", title="Independent optimization starts")
    axes[1].legend(title="Seed", fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(dest / "fit_diagnostics.png", dpi=180)
    plt.close(fig)


def _write_report(root: Path, plan: dict, table: pd.DataFrame, blocks: pd.DataFrame, m: dict) -> None:
    config = plan["config"]
    order = m["replay_order"]
    lines = [f"# Cohen 连续模型：被试 {m['subject']} 完整序列拟合", "",
        f"本次使用 condition {m['condition']} 的全部 {m['n_trials']} 题，按 session 和试次排序，跨 session 延续学习。",
        "实际选择用于计算似然，实际反馈用于更新学习状态。每题的选择概率在该题反馈前计算。",
        "全部试次参与参数选择，因此下述表现属于样本内拟合；留出预测与参数恢复尚待检验。", "",
        "## 拟合设置", "",
        f"采用 4 维连续刺激的全部 15 个乘积特征，学习更新为 `{config['model'].get('update_rule', 'exact_gradient')}`。",
        f"估计初始单位长度权重、beta 和 eta，共 {m['n_fit_parameters']} 个自由参数；类别偏置初始为零，随后随反馈更新。",
        f"独立起点为 {plan['seeds']}。每个起点的配置为 `{config['fit']}`。",
        "GA 对候选参数并行评分，随机数和候选生成顺序保持固定；退火顺序执行。每个计算进程使用一个数值线程。", "",
        "## 拟合结果", "",
        f"按每维 {order} 节点完整重放的 NLL 选择种子 {m['best_seed']}。NLL 为实际作答负对数概率之和，越小表示拟合越好。",
        f"选中结果：NLL = {m['nll']:.6f}，beta = {m['beta']:.6f}，eta = {m['eta']:.6f}。",
        f"均匀随机选择的 NLL 为 {m['uniform_choice_nll']:.6f}；每题对数概率改善为 {m['bits_per_trial_gain_over_uniform']:.4f} bits。",
        "均匀随机基线用于量化相对改善；模型间比较仍需使用相同数据边界并考虑模型复杂度。", "",
        f"观测正确率为 {m['observed_accuracy']:.2%}，模型平均正确选择概率为 {m['model_mean_p_correct']:.2%}。",
        f"末 64 题分别为 {m['last64_observed_accuracy']:.2%} 与 {m['last64_model_mean_p_correct']:.2%}。",
        f"最大概率类别与被试作答的一致率为 {m['choice_agreement_argmax']:.2%}；它与模型平均正确选择概率含义不同。", "",
        "| 连续题号 | 观测正确率 | 模型平均正确选择概率 | 选择 NLL |",
        "| --- | ---: | ---: | ---: |"]
    for row in blocks.itertuples():
        lines.append(f"| {row.first_trial}–{row.last_trial} | {row.observed_accuracy:.2%} | {row.model_mean_p_correct:.2%} | {row.choice_nll:.3f} |")
    lines += ["", "## 优化与积分诊断", "",
        "| 种子 | beta | eta | 高精度 NLL | 最后 20 代改善 | 退火改善 | 积分检查通过 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in table.to_dict("records"):
        lines.append(f"| {row['seed']} | {row['beta']:.5f} | {row['eta']:.5f} | {row[f'nll_order{order}']:.6f} | {row['last20_ga_improvement']:.4f} | {row['anneal_improvement']:.4f} | {row['all_quadrature_checks_pass']} |")
    lines += ["", f"独立起点间最终 NLL 极差为 {m['seed_nll_range']:.6f}。该范围反映本次搜索对初始化的敏感性。",
        "最后阶段仍有改善或不同起点差异较大时，当前最优值应作为已搜索到的候选解；全局最优和参数可辨识性尚待检验。",
        f"拟合使用每维 {config['model'].get('quadrature_order', 8)} 节点；随后在 {plan['check_orders']} 节点下分别从拟合初始状态重放全部试次。",
        f"每次相邻精度比较的通过标准为绝对 NLL 变化 ≤ {plan['nll_tolerance']} 且最大逐题类别概率变化 ≤ {plan['probability_tolerance']}。",
        f"所有起点通过：{m['all_seeds_quadrature_checks_pass']}；选中起点通过：{m['best_quadrature_checks_pass']}。",
        "详细数值保存于 [seed_comparison.csv](diagnostics/seed_comparison.csv)。提高积分精度后的轨迹已保存，参数沿用搜索结果。", "",
        "![拟合与优化诊断](diagnostics/fit_diagnostics.png)", "", "## 文件与复现", "",
        "每个 `seed_*/summary.json` 记录数据哈希、代码哈希、Git 状态、依赖版本和拟合范围；`config.yaml` 保存该起点配置。",
        "`study.json` 记录预设起点、积分检查阈值、选择标准和计算完成状态。",
        "`diagnostics/best_parameters.json`、`best_trial_predictions.csv` 和 `best_states.npz` 保存选中结果的高精度重放。",
        "状态是固定拟合参数、给定此前实际反馈的反馈前状态；参数不确定性尚未积分。",
        "本次结果评估连续扩展模型在这一被试上的行为拟合。原文二元模型的复现范围及学习约定见 `src/Cohen/README.md`。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
