"""Audit saved Model 0923 readout states and render PNG diagnostics.

Figure contract: show how fitted inertia separates rule support from response
contribution, with dominant-rule uncertainty and observed-history predictions.
Quantitative grid; Python/matplotlib only; screen-review figures, 300 dpi PNG.
All valid trials and rules enter the calculations. Six rules are displayed
separately in heatmaps; the remaining rules are summed into an explicit row.
No fitting, state replay, parameter changes, or group inference is performed.
See MODEL0923_READOUT_REVIEW.md for definitions and invocation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


BLUE, ORANGE, GRAY = "#3578A7", "#CB7932", "#60656B"
COLORS = [BLUE, ORANGE, "#7868A6", "#438B83"]
RULE_FIELDS = ("belief", "choice_weights", "dominant", "active")


@dataclass
class SavedSubject:
    subject: int
    condition: int
    dataset: str
    directory: Path
    trials: pd.DataFrame
    states: dict[str, np.ndarray]
    catalog: pd.DataFrame
    inertia: float
    capacity: float
    summary: dict
    settings: dict
    audit: dict
    files: list[Path]


def validate_states(states: dict[str, np.ndarray], trials: pd.DataFrame,
                    inertia: float, capacity: float, rule_count: int) -> None:
    """Check saved timing alignment, probability mass, and pathwise bounds."""
    n = len(trials)
    if n < 2 or not np.array_equal(trials.valid_trial, np.arange(1, n + 1)):
        raise ValueError("Expected contiguous saved valid-trial order")
    scored = trials.scored.to_numpy()
    if scored.dtype != bool or scored[0] or not scored[1:].all():
        raise ValueError("Scoring mask must exclude only the initialization trial")
    if not np.isfinite(inertia) or inertia < 0:
        raise ValueError("Invalid inertia")
    for field in RULE_FIELDS:
        a = states["predictive_" + field]
        if a.shape != (n, rule_count) or not np.isfinite(a).all():
            raise ValueError("Saved rule states are not aligned and finite")
        if np.any(a < -1e-12) or np.any(a > 1 + 1e-12):
            raise ValueError("Rule probabilities are outside [0, 1]")
        if field != "active":
            np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-10, rtol=0)
            if np.any(a > states["predictive_active"] + 1e-10):
                raise ValueError("Rule mass exceeds candidate inclusion probability")
    if inertia == 0:
        np.testing.assert_allclose(states["predictive_belief"],
                                   states["predictive_choice_weights"], atol=1e-12, rtol=0)
    candidates = states["predictive_candidate_count"]
    participation = states["predictive_effective_participation"]
    for a in (candidates, participation):
        if a.shape != (n, 4) or not np.isfinite(a).all():
            raise ValueError("Expected scalar mean and three path quantiles")
        if np.any(a < 1 - 1e-10) or np.any(a > capacity + 1e-10):
            raise ValueError("Scalar state violates capacity bounds")
        if np.any(np.diff(a[:, 1:], axis=1) < -1e-10):
            raise ValueError("Path quantiles are not ordered")
    if np.any(participation > candidates + 1e-10):
        raise ValueError("Effective participation exceeds candidate count")
    np.testing.assert_allclose(states["predictive_active"].sum(axis=1),
                               candidates[:, 0], atol=1e-10, rtol=0)
    p = states["probabilities"]
    if p.shape[0] != n or p.shape[1] not in (2, 4) or not np.isfinite(p).all():
        raise ValueError("Invalid choice-probability matrix")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Choice probabilities are outside [0, 1]")
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(p, trials[[f"probability_choice_{i+1}"
                              for i in range(p.shape[1])]], atol=1e-12, rtol=0)
    for name, probability_name in (("choice", "observed_choice_probability"),
                                   ("category", "predicted_correct_probability")):
        labels = trials[name].to_numpy(dtype=float)
        if np.any(labels != np.floor(labels)) or np.any((labels < 1) | (labels > p.shape[1])):
            raise ValueError("Category/choice indices are invalid")
        np.testing.assert_allclose(p[np.arange(n), labels.astype(int)-1],
                                   trials[probability_name], atol=1e-12, rtol=0)


def load_subject(directory: Path) -> SavedSubject:
    root = directory.parent
    subject = int(directory.name.removeprefix("subject_"))
    names = ["pooled_states.npz", "trials.csv", "rule_catalog.csv", "summary.csv",
             "fixed_simulation.yaml"]
    files = [directory / name for name in names]
    files += [root / name for name in ("parameters.csv", "effective_config.yaml", "search_summary.csv")]
    with np.load(files[0], allow_pickle=False) as saved:
        states = {key: saved[key] for key in saved.files}
    trials, catalog = pd.read_csv(files[1]), pd.read_csv(files[2])
    if not np.array_equal(catalog.rule, np.arange(len(catalog))):
        raise ValueError("Catalogue indices must match the saved array columns")
    summary = pd.read_csv(files[3]).iloc[0].to_dict()
    fixed = yaml.safe_load(files[4].read_text())
    provenance = fixed["engine_config"]["provenance"]
    if provenance["specification_version"] != "0923-full-v3":
        raise ValueError("Readout audit requires full-v3 states")
    params = pd.read_csv(root / "parameters.csv")
    params = params[params.subject == subject].set_index("parameter")
    inertia = float(params.loc["execution_inertia", "estimate"])
    capacity = float(params.loc["max_capacity", "estimate"])
    audit = pd.read_csv(root / "search_summary.csv")
    audit = audit[audit.subject == subject].iloc[0].dropna().to_dict()
    settings = yaml.safe_load((root / "effective_config.yaml").read_text())
    validate_states(states, trials, inertia, capacity, len(catalog))
    if int(summary["subject"]) != subject or not (trials.iSub == subject).all():
        raise ValueError("Subject identity mismatch")
    condition = int(summary["condition"])
    if condition != int(provenance["condition"]) or not (trials.condition == condition).all():
        raise ValueError("Condition mismatch")
    score = -np.log(np.clip(trials.loc[trials.scored, "observed_choice_probability"], 1e-12, 1)).mean()
    np.testing.assert_allclose(score, summary["mean_choice_nll"], atol=1e-12, rtol=0)
    return SavedSubject(subject, condition, provenance.get("dataset", "exp123"), directory,
                        trials, states, catalog, inertia, capacity, summary, settings, audit, files)


def diagnostic_metrics(s: SavedSubject) -> tuple[pd.DataFrame, dict]:
    """TV is computed after path averaging; it is not mean pathwise TV."""
    b, w, d = [s.states["predictive_" + key] for key in RULE_FIELDS[:3]]
    tv = .5 * np.abs(w - b).sum(axis=1)
    frame = s.trials[["valid_trial", "scored", "choice", "category", "feedback",
                      "observed_choice_probability", "predicted_correct_probability",
                      "observed_correct"]].copy()
    frame["marginal_total_variation"] = tv
    frame["marginal_belief_argmax"] = b.argmax(axis=1)
    frame["marginal_weight_argmax"] = w.argmax(axis=1)
    frame["dominant_probability_mode"] = d.argmax(axis=1)
    frame["dominant_mode_probability"] = d.max(axis=1)
    for key in ("candidate_count", "effective_participation"):
        frame["mean_path_" + key] = s.states["predictive_" + key][:, 0]
    # All modeled trials describe state trajectories; NLL retains the source mask.
    row = dict(subject=s.subject, condition=s.condition, dataset=s.dataset,
               trials=len(frame), inertia=s.inertia, multiplier=float(np.exp(s.inertia)),
               mean_marginal_tv=float(tv.mean()), max_marginal_tv=float(tv.max()),
               mean_candidates=float(frame.mean_path_candidate_count.mean()),
               mean_effective_participation=float(frame.mean_path_effective_participation.mean()),
               mean_dominant_mode_probability=float(d.max(axis=1).mean()),
               marginal_argmax_disagreement=float(np.mean(b.argmax(axis=1) != w.argmax(axis=1))),
               mean_choice_nll=float(s.summary["mean_choice_nll"]),
               observed_accuracy=float(s.summary["observed_exact_accuracy"]),
               autonomous_accuracy_median=float(s.summary["autonomous_accuracy_median"]),
               fit_status=s.audit.get("fit_acceptance_status", "initial_search"))
    return frame, row


def focus_trial(s: SavedSubject, metrics: pd.DataFrame) -> int:
    """Select the greatest marginal separation, or strongest dominance at lambda=0."""
    if s.inertia == 0:
        return int(metrics.dominant_mode_probability.to_numpy().argmax())
    return int(metrics.marginal_total_variation.to_numpy().argmax())


def group_rules(matrix: np.ndarray, selected: list[int]) -> np.ndarray:
    """Retain every rule's mass, including all undisplayed rules in one row."""
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError("Selected rules must be unique and nonempty")
    other = np.ones(matrix.shape[1], dtype=bool)
    other[selected] = False
    return np.column_stack([matrix[:, selected], matrix[:, other].sum(axis=1)])


def chosen_rules(s: SavedSubject, focus: int, count: int = 6) -> list[int]:
    b = s.states["predictive_belief"]
    w = s.states["predictive_choice_weights"]
    priority = [int(w[focus].argmax()), int(b[focus].argmax())]
    priority += np.argsort(-w.mean(axis=0), kind="stable").tolist()
    return list(dict.fromkeys(priority))[:count]


def style() -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trajectory_figure(s: SavedSubject, metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    t = metrics.valid_trial.to_numpy()
    focus = focus_trial(s, metrics)
    selected = chosen_rules(s, focus)
    labels = [f"R{h}" for h in selected] + ["其余规则合计"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), layout="constrained", sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.3, 1.3, .9, .85, .95]})
    source = []
    for ax, field, title in zip(axes[:3], RULE_FIELDS[:3],
                               ("a  规则信念：历史证据的相对支持",
                                "b  作答权重：本题类别预测的混合权重",
                                "c  主导规则概率：多少推断质量落在该规则主导的路径上")):
        grouped = group_rules(s.states["predictive_" + field], selected)
        image = ax.imshow(grouped.T, origin="upper", aspect="auto", cmap="Blues",
                          interpolation="nearest", vmin=0, vmax=1,
                          extent=(.5, len(t)+.5, len(labels)-.5, -.5))
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_title(title, loc="left", fontsize=10)
        fig.colorbar(image, ax=ax, pad=.01, ticks=[0, .5, 1], label="概率 / 权重")
        for i, label in enumerate(labels):
            source.append(pd.DataFrame({"valid_trial": t, "field": field,
                                        "rule_group": label, "value": grouped[:, i]}))
    for key, label, color in (("candidate_count", "候选数量", BLUE),
                              ("effective_participation", "有效参与数量", ORANGE)):
        mean, low, median, high = s.states["predictive_" + key].T
        axes[3].fill_between(t, low, high, color=color, alpha=.12)
        axes[3].plot(t, mean, color=color, lw=1, label=label)
    axes[3].set(title="d  每条路径内计算后汇总：均值与 10–90% 范围", ylabel="规则数", ylim=(.8, None))
    axes[3].legend(ncol=2, fontsize=8)
    axes[4].plot(t, metrics.marginal_total_variation * 100, color=ORANGE, lw=1)
    axes[4].axvline(focus+1, color=GRAY, ls="--", lw=.8)
    axes[4].set(title="e  信念与权重的分离：汇总分布的总变差距离", ylabel="百分点", ylim=(0, 100))
    for key, label, color in (("observed_correct", "被试实际正确率", GRAY),
                              ("predicted_correct_probability", "模型给正确类别的概率", BLUE)):
        curve = metrics[key].rolling(32, min_periods=32).mean()
        axes[5].plot(t, curve, color=color, lw=1.2, label=label)
    axes[5].legend(ncol=2, fontsize=8)
    axes[5].set(title="f  行为背景：过去 32 个有效试次的均值", ylabel="正确率 / 概率",
                ylim=(0, 1), xlabel="有效试次（保持原始顺序）")
    for ax in axes:
        ax.set_xlim(.5, len(t)+.5)
    fig.suptitle(f"S{s.subject} · {s.dataset} · 条件 {s.condition} · λ = {s.inertia:g}"
                 f"（惯性倍率 {np.exp(s.inertia):.2f}）\n"
                 "作答前推断；固定已拟合参数；热图各列总和均为 1", fontsize=12)
    save(fig, path)
    return pd.concat(source, ignore_index=True)


def focus_figure(s: SavedSubject, metrics: pd.DataFrame, path: Path) -> pd.DataFrame:
    anchor = focus_trial(s, metrics)
    selected = chosen_rules(s, anchor, count=4)
    b, w, d = [s.states["predictive_" + key] for key in RULE_FIELDS[:3]]
    start, stop = max(0, anchor-24), min(len(b), anchor+25)
    t = np.arange(start+1, stop+1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    for i, h in enumerate(selected[:2]):
        ax = axes[i, 0]
        ax.plot(t, b[start:stop, h], color=BLUE, ls="--", label="规则信念")
        ax.plot(t, w[start:stop, h], color=ORANGE, label="作答权重")
        ax.plot(t, d[start:stop, h], color=GRAY, ls=":", label="成为主导的概率")
        ax.axvline(anchor+1, color=GRAY, lw=.7, alpha=.5)
        ax.set(title=f"{'a' if i == 0 else 'c'}  R{h} 的局部轨迹", xlabel="有效试次",
               ylabel="概率 / 权重", ylim=(0, 1.03))
        ax.legend(fontsize=8, loc="best")
    rows = []
    labels = [f"R{h}" for h in selected] + ["其余合计"]
    for i, (matrix, field, label, color) in enumerate(zip((b, w, d), RULE_FIELDS[:3],
                                                        ("规则信念", "作答权重", "主导概率"),
                                                        (BLUE, ORANGE, GRAY))):
        values = group_rules(matrix, selected)[anchor]
        axes[0, 1].bar(np.arange(len(labels)) + (i-1)*.25, values, width=.24,
                       color=color, label=label)
        for group, value in zip(labels, values):
            rows.append({"valid_trial": anchor+1, "field": field, "rule_group": group,
                         "value": float(value)})
    axes[0, 1].set(xticks=np.arange(len(labels)), xticklabels=labels, ylim=(0, 1.04),
                   title=f"b  第 {anchor+1} 题的三种量", ylabel="概率 / 权重")
    axes[0, 1].legend(fontsize=8)
    p = s.states["probabilities"][anchor]
    choice, category = int(s.trials.choice.iloc[anchor]), int(s.trials.category.iloc[anchor])
    axes[1, 1].bar(np.arange(1, len(p)+1), p, color=BLUE, alpha=.8)
    for c, value in enumerate(p, 1):
        axes[1, 1].text(c, value+.02, f"{value:.1%}", ha="center", fontsize=9)
    axes[1, 1].set(xticks=np.arange(1, len(p)+1), ylim=(0, 1.04), ylabel="最终类别概率",
                   title=f"d  第 {anchor+1} 题：实际选 {choice}，正确类别 {category}", xlabel="类别")
    selection = "主导概率最集中的试次" if s.inertia == 0 else "信念与权重分离最大的试次"
    fig.suptitle(f"S{s.subject} · λ = {s.inertia:g} · {selection}\n"
                 "局部窗口用于检查机制；全序列见配套轨迹图", fontsize=12)
    save(fig, path)
    return pd.DataFrame(rows)


def cohort_figure(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), layout="constrained", sharey=True)
    y = np.arange(len(summary))
    for ax, key, title, color in zip(axes, ("inertia", "mean_marginal_tv", "mean_effective_participation"),
                                    ("a  拟合惯性 λ", "b  平均信念–权重分离", "c  平均有效参与数量"),
                                    (GRAY, ORANGE, BLUE)):
        values = summary[key] * (100 if key == "mean_marginal_tv" else 1)
        ax.barh(y, values, color=color, alpha=.85, height=.65)
        ax.set_title(title, fontsize=10, loc="left")
        for j, value in enumerate(values):
            ax.text(value + max(float(values.max()), 1)*.025, j, f"{value:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, max(float(values.max()), 1)*1.23)
        ax.set_xlabel("百分点" if key == "mean_marginal_tv" else "λ" if key == "inertia" else "规则数")
    labels = [f"S{r.subject} · {'MEG' if 'meg' in r.dataset else 'exp'} · C{r.condition}"
              for r in summary.itertuples()]
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("13 人选择读出诊断：335、307 使用续跑结果，其余为初始搜索结果\n"
                 "逐人描述；条件、任务与拟合阶段分别保留", fontsize=12)
    save(fig, path)


def relative_link(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def write_report(subjects: list[SavedSubject], summary: pd.DataFrame, output: Path,
                 figures: Path, metrics: dict[int, pd.DataFrame]) -> None:
    def image_tag(name: str) -> str:
        link = html.escape(relative_link(figures / name, output), quote=True)
        return f'<a href="{link}"><img loading="lazy" src="{link}" alt="{html.escape(name)}"></a>'
    display = summary[["subject", "dataset", "condition", "inertia", "mean_marginal_tv",
                       "mean_effective_participation", "mean_choice_nll", "fit_status"]].copy()
    display["mean_marginal_tv"] *= 100
    display.columns = ["被试", "数据", "条件", "惯性 λ", "平均分离（百分点）", "有效参与数量", "选择 NLL", "拟合状态"]
    parts = ["""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Model 0923 第7节选择读出诊断</title><style>
body{max-width:1120px;margin:36px auto;padding:0 24px;font:16px/1.8 system-ui,sans-serif;color:#22313a}
img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}
td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}summary{cursor:pointer;font-weight:600}
details{border-top:1px solid #ddd;margin-top:28px;padding-top:12px}a{color:#286b98}code{background:#f2f4f5}
.note{background:#f2f6f8;padding:14px 18px}h1{font-size:28px}h2{font-size:22px}</style>
<h1>Model 0923：规则信念怎样转为作答权重</h1>
<p>本报告复用 2026-10-13 保存的拟合后分析。共 13 人：exp123 原有 12 人中，307 采用最新续跑结果；另加入 MEG 335 的最新续跑结果。保留所有有效试次与目录规则。</p>
<p>当前结果显示，惯性为 0 的人保持信念与作答权重一致；高惯性拟合可以使低信念规则继续主导作答。后者是本次机制审查的重点。轨迹描述固定参数下的模型推断，机制的预测收益还需通过受约束模型比较确定。</p>
<h2>第7节的计算顺序</h2>
<p>本题作答前，π 是此前反馈与候选管理得到的规则信念。上一题主导规则的信念乘以 exp(λ)，其他规则保留原值；归一化后得到作答权重 w。每条规则对各类别的预测为 p(c|h)，最终类别概率为 Σh w(h)p(c|h)。主导规则取 w 最大者，在实际选择之前确定；本题反馈随后更新信念与精度。</p>
<p>主导规则记录的是最大混合权重。每个类别收到的具体贡献为 w(h)p(c|h)，因此最大权重规则与对某个特定答案贡献最大的规则可能不同。模型的全部活跃规则都参与预测。</p>
<p>若上一主导规则 A 仍在集合内，另一规则 B 取代它需要 π(B)/π(A) &gt; exp(λ)，并超过其他候选。λ=2 对应 7.39 倍；λ=8 对应 2980.96 倍。并列时沿用 A。候选退出也会终止 A 的惯性加成。</p>
<div class="note">所有状态图使用 <b>predictive：本题实际选择前</b>的粒子权重，历史选择和历史反馈已经进入推断。参数由全序列拟合，因此行为曲线属于样本内诊断。各次粒子滤波重复等权汇总；主导规则概率描述路径间的不确定性。参数不确定性尚未纳入。</div>
<h2>先看逐人对照</h2>""", image_tag("cohort.png"), display.to_html(index=False, float_format=lambda v: f"{v:.3f}"),
        """<p>总变差距离 TV = ½Σh|平均作答权重−平均信念|，0 表示相同，1 表示完全分离。它是路径汇总后的差异；方向相反的路径差异可能抵消。表中用百分点表示。有效参与数量先在每条路径内按 1/Σh w(h)² 计算，再加权平均；它与候选总数分别展示。</p>
<p>选择 NLL 为实际答案的平均负对数概率，沿用原计分掩码并排除初始化题。二分类的均匀基线为 ln(2)，四分类为 ln(4)。不同条件的原始 NLL 分别解读。初始搜索与续跑的拟合状态分别保留，当前表不构成人群推断。</p>
<h2>逐人查看信念、权重和主导规则</h2>
<p>热图各行使用相同规则与顺序，六条规则单列，其余规则的质量全部计入最后一行。R 后面的编号为各自目录的原始零基索引。下方目录保留规则族与结构参数；维度编号按原任务编码读取。</p>
<p>轨迹图包含全部试次。局部图选择分离最大的试次；λ=0 时选择主导规则概率最集中的试次。局部图用于机制检查，不能据此推断典型发生频率。行为面板是包含当前试次的过去32题均值，前31题没有完整窗口。</p>"""]
    ordered = sorted(subjects, key=lambda s: ([335, 307, 104, 122].index(s.subject)
                                             if s.subject in [335, 307, 104, 122] else s.subject))
    for s in ordered:
        m = metrics[s.subject]
        anchor = focus_trial(s, m)
        selected = chosen_rules(s, anchor)
        values = pd.DataFrame({"规则": [f"R{h}" for h in selected],
            "规则信念": s.states["predictive_belief"][anchor, selected],
            "作答权重": s.states["predictive_choice_weights"][anchor, selected],
            "主导概率": s.states["predictive_dominant"][anchor, selected]})
        catalog = s.catalog.iloc[selected].copy()
        source_link = html.escape(relative_link(s.directory, output), quote=True)
        expanded = " open" if s.subject in (335, 307, 104, 122) else ""
        parts.append(f'<details{expanded}><summary>S{s.subject} · {s.dataset} · 条件{s.condition} · λ={s.inertia:g}</summary>')
        parts.append(f'<p>共 {len(m)} 个有效试次；原评估使用 {s.settings["particle_count"]} 粒子 × '
                     f'{s.settings["pf_repeats"]} 次独立重复。<a href="{source_link}">原始评估目录</a>。</p>')
        parts.append(image_tag(f"subject_{s.subject}.png"))
        if s.subject in (335, 307, 104, 122):
            parts.append(image_tag(f"subject_{s.subject}_focus.png"))
        parts.append(f'<p>第 {anchor+1} 个有效试次的数值（路径加权均值）：</p>')
        parts.append(values.to_html(index=False, float_format=lambda v: f"{v:.4f}"))
        parts.append('<p>所列规则结构；parameters 中的维度编号从0开始：</p>' + catalog.to_html(index=False))
        parts.append('</details>')
    parts.append("""<h2>对机制合理性的当前判断</h2>
<p>分开表示证据支持与作答贡献，使模型能够描述有时间差的策略改变。λ=0 提供了精确的退化情形，本次数据检查确认这时两者相同。307 的 λ=2 产生了可见但相对温和的分离；335 的稳定主导规则在 λ=0 时出现，显示稳定性也可以来自信念积累和候选状态。</p>
<p>初始搜索中104、122、221、314的 λ=8 均达到原搜索上界。此时更换主导规则所需的信念优势很大，值得检查退出机制是否承担了主要切换作用，以及惯性是否在补偿记忆或候选管理参数。上界估计与局部极端轨迹提供了检查线索，参数可识别性需通过剖面或恢复分析评估。</p>
<p>实际选择发生前，模型根据最大权重记录主导规则。一个由其他规则贡献较多的实际答案也可能出现；这次实际答案本身不会直接重新指定该路径的主导规则。这是该惯性定义需要评估的一项认知假设。</p>
<p>后续模型比较可以固定 λ=0 并重新优化其余参数，按相同预算比较留出或序列预测表现；进一步比较沿用上一完整权重向量的连续惯性方案。重新拟合比较与固定状态下只修改读出的即时效应回答不同问题。本次只使用已有结果，未执行这些比较。</p>
<p>335 与307沿真实历史的选择 NLL 分别约0.740和0.812；原评估的自主模拟正确率中位数分别约27.5%和40.6%，被试分别约71.2%和66.0%。这一差距是整个学习过程的待解释现象；当前分析尚未定位它由哪个模块造成。</p>
<p>保存的汇总状态缺少每条路径内“规则×类别”的联合贡献和主导规则相邻试次联合分布。当前图可以核对权重分离与主导身份的不确定性；逐规则解释某个具体答案、计算真实路径的切换率或驻留时间，需要补充联合记录。</p>
<p><a href="summary.csv">逐人指标</a> · <a href="trial_metrics.csv">逐题指标</a> · <a href="manifest.json">来源与版本</a> · <a href="validation.json">数据检查</a></p></html>""")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", action="append", type=Path, required=True,
                        help="Saved evaluation root; later roots supersede duplicate subjects")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-output-dir", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_dir, args.figure_output_dir):
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
    discovered, superseded = {}, []
    for root in args.evaluation_dir:
        directories = sorted(root.glob("subject_*/pooled_states.npz"))
        if not directories:
            raise ValueError(f"No saved subject states in {root}")
        for file in directories:
            subject = int(file.parent.name.removeprefix("subject_"))
            if subject in discovered:
                superseded.append({"subject": subject, "previous": str(discovered[subject]),
                                   "selected": str(file.parent), "reason": "explicit_later_input_precedence"})
            discovered[subject] = file.parent
    subjects = [load_subject(path) for _, path in sorted(discovered.items())]
    # Report prose describes this specific review cohort; fail on accidental scope drift.
    if set(discovered) != {102, 104, 118, 122, 206, 215, 221, 222, 307, 314, 315, 328, 335}:
        raise ValueError("This review requires the documented thirteen-subject cohort")
    args.output_dir.mkdir(parents=True)
    args.figure_output_dir.mkdir(parents=True)
    style()
    metrics, rows, sources = {}, [], []
    for s in subjects:
        frame, row = diagnostic_metrics(s)
        metrics[s.subject] = frame
        rows.append(row)
        curves = trajectory_figure(s, frame, args.figure_output_dir / f"subject_{s.subject}.png")
        curves.insert(0, "subject", s.subject)
        sources.append(curves)
        if s.subject in (335, 307, 104, 122):
            focus_figure(s, frame, args.figure_output_dir / f"subject_{s.subject}_focus.png").to_csv(
                args.output_dir / f"subject_{s.subject}_focus.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat([frame.assign(subject=sid) for sid, frame in metrics.items()], ignore_index=True).to_csv(
        args.output_dir / "trial_metrics.csv", index=False)
    pd.concat(sources, ignore_index=True).to_csv(args.output_dir / "heatmap_source.csv", index=False)
    cohort_figure(summary, args.figure_output_dir / "cohort.png")
    source_files = sorted({path for s in subjects for path in s.files})
    source_files += [Path(__file__).relative_to(Path.cwd()),
                     Path("src/Bayesian_state/docs/model_architecture/model_0923.pdf"),
                     Path("src/Bayesian_state/model/modules/hypothesis_transition/mechanistic_rule_search.py"),
                     Path("src/Bayesian_state/inference/backends/particle_filter.py"),
                     Path("src/Bayesian_state/evaluation/model_0923_postfit.py")]
    manifest = dict(protocol="model0923_readout_review_v1", phase="predictive",
                    weighting="equal_seed_mixture_of_pre_choice_particle_weighted_states",
                    selected_sources={s.subject: str(s.directory) for s in subjects}, superseded=superseded,
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, matplotlib=matplotlib.__version__, pyyaml=yaml.__version__),
                    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files],
                    figure_contract=dict(claim="Inertia can separate rule support and response contribution",
                        archetype="quantitative grid", backend="python", format="PNG", dpi=300,
                        purpose="screen_review_not_submission", rule_aggregation="six_named_rules_plus_all_others",
                        state_uncertainty="fixed_parameter_path_quantiles_not_parameter_confidence_intervals",
                        statistical_tests="none", fitting="none", trial_exclusions="none_for_states_first_trial_for_NLL"))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    validation = dict(status="passed", subjects=len(subjects), modeled_trials=sum(len(s.trials) for s in subjects),
                      checks=["full_v3", "subject_condition_trial_alignment", "finite_normalized_probabilities",
                              "lambda_zero_identity", "candidate_and_participation_bounds", "ordered_path_quantiles",
                              "saved_probability_and_NLL_reproduction", "all_rule_mass_retained"],
                      exclusions=superseded, model_runs=0, parameter_changes=0)
    (args.output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2)+"\n")
    write_report(subjects, summary, args.output_dir, args.figure_output_dir, metrics)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
