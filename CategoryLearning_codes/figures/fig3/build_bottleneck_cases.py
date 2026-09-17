"""Build a descriptive Fig3 pilot from existing S129/S229 fits (PNG only).

Run from the repository root with a new --output directory. No fitting,
simulation, clustering or population inference is performed.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3.bottleneck_analysis import (
    ROOT, contiguous_runs, read_case, save_case_sources, screen_episodes,
    sha256, stage_profiles,
)


BLUE, ORANGE, GREEN, PURPLE, GRAY = "#417D9C", "#B77A3E", "#43856B", "#867093", "#91989F"


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def probability_axis(ax: Any, start: int, stop: int) -> None:
    ax.set(xlim=(start, stop), ylim=(-0.025, 1.045), yticks=[0, .5, 1])
    ax.grid(axis="y", color="#E9ECEE", linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def render_timeline(cases: list[dict[str, Any]], output: Path, config: dict[str, Any]) -> None:
    fig, axes = plt.subplots(5, len(cases), figsize=(183 / 25.4, 220 / 25.4), squeeze=False)
    fig.subplots_adjust(left=.09, right=.985, bottom=.10, top=.88, hspace=.63, wspace=.27)
    fig.suptitle("From considering a rule to using it", x=.09, y=.984, ha="left", fontsize=12, weight="bold")
    fig.text(.09, .949, "Two individual records  |  Saved fits  |  Descriptive pilot", fontsize=8, color="#535A60")
    titles = ["Learning performance", "Is the target rule considered?", "How much support does it receive?",
              "Does it guide the selected rule?", "What does the participant report?"]
    for column, case in enumerate(cases):
        d, meta = case["table"], case["metadata"]
        trial, n = d.trial.to_numpy(), len(d)
        x = axes[0, column].get_position().x0
        fig.text(x, .914, f"S{meta['subject']}  /  Task {meta['task']}  /  {n} trials", fontsize=9, weight="bold")
        for row in range(5):
            ax = axes[row, column]
            probability_axis(ax, 1, n)
            ax.set_title(f"{chr(97 + row * len(cases) + column)}   {titles[row]}", loc="left", pad=7)
            for boundary in np.linspace(0, n, config["stage_count"] + 1)[1:-1]:
                ax.axvline(boundary + .5, color="#E0E4E7", lw=.6, ls=":")
            if row < 4:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Trial")
        ax = axes[0, column]
        ax.scatter(trial, d.observed_accuracy, s=2, color="#CED3D6", alpha=.45, linewidths=0, rasterized=True)
        ax.plot(trial, d.observed_accuracy_rolling, color="#333B41", lw=1.05, label="Observed")
        ax.plot(trial, d.model_correct_probability_rolling, color=BLUE, lw=1, ls="--", label="Model")
        ax.legend(loc="upper left", ncol=2, handlelength=1.7)
        ax.set_ylabel("Accuracy")
        ax = axes[1, column]
        ax.plot(trial, d.target_active, color=BLUE, lw=.95)
        ax.set_ylabel("Probability A")
        ax = axes[2, column]
        ax.plot(trial, d.target_support_display, color=ORANGE, lw=1, label="Support if active, C")
        ax.plot(trial, d.target_mass, color=BLUE, lw=.85, ls="--", alpha=.85, label="Overall mass, Q")
        ax.legend(loc="upper left", fontsize=5.7, handlelength=1.6)
        ax.set_ylabel("Relative support")
        ax = axes[3, column]
        if meta["persistent_execution"]:
            ax.plot(trial, d.target_execution, color=GREEN, lw=.95)
        else:
            ax.set_facecolor("#F5F6F7")
            ax.text(.5, .54, "Mixture readout\nNo single executed rule", ha="center", va="center", transform=ax.transAxes, color="#697078", fontsize=8)
        ax.set_ylabel("Probability E")
        ax = axes[4, column]
        ax.plot(trial, d.oral_target_state, color=GRAY, lw=.8, alpha=.8, label="Latest by category")
        valid = d.oral_report_valid.to_numpy()
        ax.scatter(trial[valid], d.loc[valid, "oral_target_current_report"], s=5, color=PURPLE, alpha=.65, linewidths=0, label="Current report", rasterized=True)
        ax.legend(loc="upper left", fontsize=5.7, handlelength=1.6)
        ax.set_ylabel("Encoded target weight")
    fig.text(.09, .049, "A: probability of considering the target. C: support conditional on considering it. Q = A × C.", fontsize=6.3)
    fig.text(.09, .031, f"Behavior lines: {config['rolling_window']}-trial trailing means. C is shown only where A ≥ {config['conditional_display_min_active']:.2f}.", fontsize=6.3)
    fig.text(.09, .014, "16 PF repeats per person; no uncertainty interval. Reports follow choice and precede feedback. Full-sequence fits.", fontsize=6.1, color="#5A6269")
    fig.savefig(output / "belief_bottleneck_timelines.png", dpi=config["dpi"])
    plt.close(fig)


def render_profiles(profiles: pd.DataFrame, cases: list[dict[str, Any]], output: Path, config: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(183 / 25.4, 98 / 25.4), squeeze=False)
    fig.subplots_adjust(left=.16, right=.985, bottom=.27, top=.73, wspace=.67)
    fig.suptitle("Learning profiles across four stages", x=.07, y=.97, ha="left", fontsize=12, weight="bold")
    fig.text(.07, .865, "Each stage contains one quarter of that person's trials. These are descriptive profiles, not groups.", fontsize=7.1)
    cmap = LinearSegmentedColormap.from_list("profile_blue", ["#F5F8FA", "#8EB5C9", "#32657E"])
    cmap.set_bad("#E4E7E9")
    columns = ["target_active_mean", "target_support_pooled", "target_mass_mean", "target_execution_mean", "observed_accuracy", "oral_target_state_mean_at_new_reports"]
    labels = ["A", "C*", "Q", "E", "Accuracy", "Oral†"]
    for ax, case in zip(axes[0], cases):
        meta = case["metadata"]
        subset = profiles.loc[profiles.subject.eq(meta["subject"])]
        values = subset[columns].to_numpy(dtype=float)
        ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="none")
        ax.set_title(f"S{meta['subject']} / Task {meta['task']}", loc="left", weight="bold", pad=10)
        ax.set_xticks(np.arange(len(columns)), labels=labels, fontsize=6)
        ax.set_yticks(np.arange(len(subset)), labels=[f"{int(r.first_trial)}–{int(r.last_trial)}" for r in subset.itertuples()], fontsize=6.5)
        ax.set_ylabel("Trial interval")
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (row, col), value in np.ndenumerate(values):
            ax.text(col, row, "N/A" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=6.7,
                    color="white" if np.isfinite(value) and value > .65 else "#243640")
    fig.text(.07, .18, "A: considered. C*: conditional support. Q: overall belief mass. E: executed rule.", fontsize=6.2)
    fig.text(.07, .13, "C* = summed target mass / summed active probability within each stage.", fontsize=6.2)
    fig.text(.07, .08, "† Latest-by-category target weight at valid new reports; it includes reports from earlier trials.", fontsize=6.2)
    fig.text(.07, .03, "All scales: 0–1. Gray: structurally not applicable. Tasks, rule catalogues and record lengths differ.", fontsize=6.2, color="#5A6269")
    fig.savefig(output / "stage_learning_profiles.png", dpi=config["dpi"])
    plt.close(fig)


def render_zoom(cases: list[dict[str, Any]], episodes: pd.DataFrame, output: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(3, len(cases), figsize=(183 / 25.4, 150 / 25.4), squeeze=False)
    fig.subplots_adjust(left=.09, right=.985, top=.82, bottom=.14, hspace=.48, wspace=.28)
    fig.suptitle("Inspecting two candidate transitions", x=.09, y=.975, ha="left", fontsize=12, weight="bold")
    fig.text(.09, .92, "Windows are selected by stated model thresholds, not by agreement with verbal reports.", fontsize=7)
    selected = []
    for column, case in enumerate(cases):
        d, meta = case["table"], case["metadata"]
        subset = episodes.loc[episodes.subject.eq(meta["subject"]) & episodes.screen.eq("supported_low_execution_marginal_screen")]
        if len(subset):
            first, last = int(subset.iloc[0].first_trial), int(subset.iloc[0].last_trial)
            reason, title = "first_supported_low_execution_marginal_screen", "Support and execution separate"
        else:
            mask = (d.target_active >= config["screening"]["high_active"]) & (d.target_support_if_active >= config["screening"]["high_support"])
            runs = contiguous_runs(mask.to_numpy(), config["screening"]["minimum_run"])
            if not runs:
                raise ValueError("No predeclared zoom candidate; revise the figure plan explicitly")
            first, last = runs[0][0] + 1, runs[0][0] + config["screening"]["minimum_run"]
            reason, title = "first_sustained_high_availability_and_support", "Target-rule support becomes sustained"
        lo, hi = max(1, first - 16), min(len(d), last + 16)
        block = d.loc[d.trial.between(lo, hi)]
        block.to_csv(output / f"subject_{meta['subject']}_zoom_source.csv", index=False)
        selected.append({"subject": meta["subject"], "reason": reason, "screen_start": first, "screen_end": last, "window_start": lo, "window_end": hi})
        for row in range(3):
            ax = axes[row, column]
            probability_axis(ax, lo, hi)
            ax.axvspan(first - .5, last + .5, color="#F1E8D7", alpha=.65, zorder=0)
            if row < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Trial")
        ax = axes[0, column]
        ax.set_title(f"S{meta['subject']}  |  {title}", loc="left", fontsize=7.2, pad=10)
        ax.plot(block.trial, block.target_active, color=BLUE, label="Considered A", lw=1.2)
        ax.plot(block.trial, block.target_support_display, color=ORANGE, label="Support C", lw=1.2)
        if meta["persistent_execution"]:
            ax.plot(block.trial, block.target_execution, color=GREEN, label="Executed E", lw=1.2)
        ax.legend(loc="lower left", fontsize=5.7)
        ax.set_ylabel("Model state")
        ax = axes[1, column]
        ax.scatter(block.trial, block.observed_accuracy, s=12, color="#3E484F", label="Observed response")
        ax.plot(block.trial, block.model_correct_probability, color=BLUE, lw=1, ls="--", label="Model P(correct)")
        ax.set_yticks([0, 1], labels=["Error", "Correct"])
        ax.legend(loc="center left", fontsize=5.7)
        ax = axes[2, column]
        ax.step(block.trial, block.oral_target_state, where="post", color=GRAY, lw=1, label="Latest by category")
        valid = block.oral_report_valid
        ax.scatter(block.loc[valid, "trial"], block.loc[valid, "oral_target_current_report"], s=12, color=PURPLE, label="Current report", linewidths=0)
        ax.set_ylabel("Report target weight")
        ax.legend(loc="center left", fontsize=5.7)
    fig.text(.09, .065, "Shading marks an exploratory marginal screen, not a verified cognitive transition or a discovered subtype.", fontsize=6.3)
    fig.text(.09, .027, "All observations within each window are shown. Full trial records and original report text are retained in CSV.", fontsize=6.3)
    fig.savefig(output / "candidate_transition_closeups.png", dpi=config["dpi"])
    plt.close(fig)
    return selected


def write_report(cases: list[dict[str, Any]], profiles: pd.DataFrame, episodes: pd.DataFrame,
                 sensitivity: pd.DataFrame, zooms: list[dict[str, Any]], output: Path, config: dict[str, Any]) -> None:
    lines = ["# 首轮瓶颈分析：现有个案能支持到哪一步？", "",
             "本轮读取既有拟合和口述编码。没有新拟合、模拟、群体分型或显著性检验。", "",
             "## 查看图与数据", "",
             "- [完整时间轴](belief_bottleneck_timelines.png)：考虑概率、条件支持、执行概率与口述。",
             "- [四阶段画像](stage_learning_profiles.png)：连续描述，不是人群分类。",
             "- [候选变化片段](candidate_transition_closeups.png)：按模型阈值选窗，保留口述不一致之处。",
             "- [阶段指标](stage_profiles.csv)、[候选区间](candidate_episodes.csv)、[阈值敏感性](threshold_sensitivity.csv)。", "",
             "## 阶段结果", "",
             "A 是研究者推断的目标规则当前活跃概率；C 是在目标规则活跃的可能状态中的相对支持；Q=A×C。E 仅适用于持续执行结构。", "",
             "| 被试 | 试次 | 考虑概率 A | 条件支持 C | 执行概率 E | 实际正确率 | 有效新报告 |",
             "|---|---|---|---|---|---|---|"]
    for row in profiles.itertuples():
        e = "不适用" if not row.execution_applicable else f"{row.target_execution_mean:.3f}"
        lines.append(f"| S{row.subject} | {row.first_trial}–{row.last_trial} | {row.target_active_mean:.3f} | {row.target_support_pooled:.3f} | {e} | {row.observed_accuracy:.1%} | {row.oral_reports} |")
    lines += ["", "## 可以怎样解读", "",
              "### S129：低目标支持主要伴随较低的当前考虑概率", "",
              "前三个阶段 A 较低，而条件支持 C 并非始终接近零；末阶段两者同时升高，实际正确率也上升。这支持把‘是否在考虑’与‘考虑后有多少支持’分开描述。低 A 不证明过去从未想到，也不能据此断言人的搜索能力不足。", "",
              "口述提供了额外检查：t186–187 已报告‘脖子长’，t188 报告‘脖子短’后，按类别累积的目标编码已接近 1；模型 A 在随后试次继续上升，到 t198 才超过 0.75。两种量的定义和时点不同，不能把这一差别当成已标定的发现时间误差；但它提示不能声称当前模型精确同步解码每次想法变化。", "",
              "### S229：存在短暂的支持—执行分离候选，证据仍不完整", ""]
    gaps = episodes.loc[episodes.screen.eq("supported_low_execution_marginal_screen")]
    for row in gaps.itertuples():
        lines.append(f"S{row.subject} t{row.first_trial}–{row.last_trial}：平均 A={row.mean_active:.3f}，合并条件支持 C={row.pooled_support:.3f}，执行目标规则的概率 E={row.mean_execution:.3f}，实际正确率 {row.observed_accuracy:.0%}（{row.n_trials} 试次）。这是模型边际量分离的候选区间，不是已验证的‘知道却不会用’。")
    lines += ["", "这段中口述也已改变，按类别累积的目标权重接近零。例如同属 choice 2 的报告从 t955 的‘头长，腿短。’变为 t962 的‘头短，尾巴长。’；同属 choice 1 的报告从 t959 的‘腿短，头短。’变为 t964 的‘头短，尾巴短。’。这不是直接比较不同类别的报告。报告更直接约束正在表达的规则，无法证明后台仍保留多少信念。因此它能帮助检查使用变化，却不能独立证实模型估计的高条件支持。个体知觉、逐规则 β 和竞争路径的解释还未排除。", ""]
    strict = sensitivity.loc[
        sensitivity.high_active.eq(max(config["sensitivity_high_active"]))
        & sensitivity.high_support.eq(config["screening"]["high_support"])
        & sensitivity.execution_applicable
    ]
    for row in strict.itertuples():
        lines.append(f"S{row.subject} 将高活跃阈值提高至 {row.high_active:.2f}、条件支持阈值保持 {row.high_support:.2f} 时，分离筛选留下 {int(row.low_execution_trials)} 试次、{int(row.low_execution_runs)} 段持续候选。阈值敏感性需要结合连续量及联合状态恢复解释。")
    low_support = episodes.loc[episodes.screen.eq("available_support_at_or_below_equal_candidate_weight")]
    lines += ["", "### 检查‘仍在考虑，但支持较低’的区间", "",
              f"在 A≥{config['screening']['high_active']:.2f}、C≤1/M、连续至少 {config['screening']['minimum_run']} 试次的预定探索性筛选下，找到 {len(low_support)} 个区间。这个筛选检查当前低支持；只有再证实此前曾获得支持，才涉及‘后来不再相信’。没有通过筛选不排除较轻的支持下降，也不能推出人没有记忆或保持困难。不能为了凑齐三种瓶颈修改阈值或把其它波动改名。", "",
              "## 本轮能支持的结论", "",
              "1. 三个过程量可以分开提取；S129 的执行量必须记为不适用。",
              "2. 现有结果能形成随阶段变化的描述性画像，但两人来自不同任务，不能据此做人群划分或能力比较。",
              "3. 最值得继续检查的是 S129 的规则进入过程，以及 S229 t964–968 前后的支持—使用关系。",
              "4. 搜索补偿、旧判断减少重复试错、模型优于替代模型、可靠人群分组，均未由本轮验证。", "",
              "## 下一步优先项", "",
              "- 先复核这些窗口的同选择类别口述；文字变化不自动等于规则变化。",
              "- 比较近优参数候选，检验上述过程分离是否依赖一个 best 参数点。",
              "- 定向验证首次进入、支持下降、支持与执行分离的联合状态恢复；补充逐规则 β 诊断。",
              "- 然后扩展同任务被试并留出后续试次。当前全序列估参不支持前瞻预测结论。", "",
              "## 计算定义与证据边界", "",
              f"- 每人保留全部试次和全部规则，行为曲线为 {config['rolling_window']} 试次尾随均值；开始不足一个窗口时不画均值，但原始正确/错误点仍显示。",
              "- 16 次 PF 等权平均概率；C 用平均 Q/平均 A，不平均各 PF 的比值。四阶段 C 用阶段内 ΣQ/ΣA。阶段均为等试次数分段，不按结果挑选。",
              f"- C 原值保留于表中；图上仅在 A≥{config['conditional_display_min_active']:.2f} 时显示。灰色 N/A 是结构性不适用。",
              "- 图中不提供心理状态置信区间；CSV 中 seed SD 只表示固定参数下的计算重复差异。没有将 PF repeats 计为被试。",
              "- 所有逐试次的‘考虑/支持/执行’均为当前选择前的边际量；估参用了全序列，因此仍是回顾性拟合描述。未使用终端祖先平滑路径。",
              "- 保留原 saved_valid_trial_mask 与 score_trial_mask；首试次等不计入评分的设置不改变状态/原始行为展示。NLL 按原 score mask、先平均概率后取 log。",
              "- 口述 sigma 固定为 0.05。当前报告权重与 latest-by-category 状态分别显示；后者含其它类别的旧报告。无有效编码不等于没有想法或未报告目标规则。",
              "- 原始报告文本保存在逐试次 CSV；同 choice 的文字变化只是复核线索，不是已完成的语义事件标注。",
              "- 候选区间用模型阈值筛选，不能拿同一筛选变量的差异再作显著性验证。候选不等于联合粒子路径中的真实事件；不产生人群标签。",
              "- S229 的 condition 2 精度更新仍有既知解释风险，逐规则精度和近优参数不确定性未覆盖。既有拟合版本由各 manifest 留痕，本轮未重算认知模型。", "",
              "## 图注", "",
              "**完整时间轴。** n=2 名来自不同任务的参与者，分别 256/1088 试次；每人 16 次固定参数 PF 数值重复。A 为目标规则活跃边际概率，C 为在活跃条件下的平均信念支持，Q 为边际信念质量，E 为执行规则概率。报告在选择后、反馈前采集。曲线无统计区间；没有群体推断。", "",
              "**阶段画像。** 每人四个连续等长试次区间，展示 A、合并条件支持 C、Q、适用时的 E、实际正确率和新报告时点的累积口述目标权重。颜色统一 0–1，但不同量不是同一个心理尺度，不直接跨任务比较。", "",
              "**局部片段。** 首个持续高 A/高 C 的片段，以及首个支持—执行分离筛选片段；各向前后扩展 16 试次。高亮仅表示筛选区间，不是被试真正改变想法的精确时间。窗口选择写入 manifest，窗口内不删选观察。", ""]
    (output / "README.md").write_text("\n".join(lines))


def build(config_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    config = json.loads(config_path.read_text())
    data = pd.read_csv(ROOT / config["data"])
    cases = [read_case(spec, config, data) for spec in config["cases"]]
    profiles = pd.concat([stage_profiles(case, config["stage_count"]) for case in cases], ignore_index=True)
    screens = [screen_episodes(case, config) for case in cases]
    episodes = pd.concat([item[0] for item in screens], ignore_index=True)
    sensitivity = pd.concat([item[1] for item in screens], ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    for case in cases:
        save_case_sources(case, output / f"subject_{case['metadata']['subject']}")
    profiles.to_csv(output / "stage_profiles.csv", index=False)
    episodes.to_csv(output / "candidate_episodes.csv", index=False)
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
    style()
    render_timeline(cases, output, config)
    render_profiles(profiles, cases, output, config)
    zooms = render_zoom(cases, episodes, output, config)
    write_report(cases, profiles, episodes, sensitivity, zooms, output, config)
    files = [Path(__file__), Path(__file__).with_name("bottleneck_analysis.py"), config_path]
    for path in files:
        shutil.copy2(path, output / path.name)
    manifest = {
        "analysis": "descriptive_bottleneck_case_pilot", "backend": "python",
        "figure_contract": {
            "claim": "Saved fits distinguish rule availability, conditional support and execution; external correspondence remains to be tested.",
            "archetype": "quantitative grid", "format": "PNG only", "width_mm": 183,
            "n_people": len(cases), "population_inference": False, "new_fitting_or_simulation": False,
        },
        "config": config, "cases": [case["metadata"] for case in cases],
        "zoom_selection": zooms,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "validation": {"all_saved_repeats_used": True, "all_input_trials_retained": True,
                       "all_catalogue_rules_exported": True, "choice_feedback_order_matched": True,
                       "normalized_probabilities": True, "active_probability_sum_equals_capacity": True,
                       "belief_and_execution_mass_not_above_active_probability": True,
                       "oral_sigma": config["oral_sigma"], "same_source_masks_across_repeats": True},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("bottleneck_case_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
