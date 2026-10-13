"""Separate PNG panels for behavioral accuracy and external oral alignment.

Each panel uses the supplied scored observations and retains the complete input
table as its plot source. Behavioral accuracy compares pre-choice probabilities
with actual correctness. Oral probabilities describe the observed report after
the observed choice and before feedback. These panels add no fitted parameters,
uncertainty intervals, or acceptance thresholds.
"""
from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model0923_postfit import BLUE, GRAY, ORANGE, save, style


BEHAVIOR_COLUMNS = (
    "valid_trial", "scored_window", "observed_rolling_exact_accuracy",
    "conditional_rolling_correct_probability", "rolling_accuracy_residual",
)
ORAL_COLUMNS = (
    "trial", "parse_status", "scored", "oral_score_mask",
    "report_probability", "report_top1_hit", "report_top3_hit",
    "report_zero_probability",
)


def _style() -> None:
    style()
    # The standard report uses Chinese explanations alongside scientific terms.
    plt.rcParams.update({
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "Arial",
                            "Helvetica", "DejaVu Sans", "Droid Sans Fallback"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def _destinations(output: Path, stem: str) -> tuple[Path, Path]:
    output = Path(output)
    paths = (output / f"{stem}.png", output / f"{stem}_plot_source.csv")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Alignment output already exists: {path}")
    return paths


def _trial_ids(values: pd.Series) -> np.ndarray:
    trials = values.to_numpy(dtype=float)
    if (not np.isfinite(trials).all() or np.any(trials != np.floor(trials))
            or np.any(trials < 1) or np.any(np.diff(trials) <= 0)):
        raise ValueError("Alignment trial identifiers must be increasing positive integers")
    return trials


def _boolean(values: pd.Series, name: str) -> np.ndarray:
    if values.empty:
        return np.empty(0, dtype=bool)
    if values.dtype.kind != "b":
        raise ValueError(f"{name} must contain Boolean values")
    return values.to_numpy(dtype=bool)


def _probabilities(values: np.ndarray, *, name: str) -> None:
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")


def behavior_alignment_figures(
    subject: int,
    condition: int,
    curves: pd.DataFrame,
    output: Path,
) -> tuple[Path, Path]:
    """Compare actual and predicted rolling accuracy, preserving scored windows.

    The signed residual is observed minus predicted: positive values indicate
    that actual correctness exceeds the model's prediction. Condition 3 adds
    family accuracy and mean reward when those supplied columns are present.
    """
    image_path, source_path = _destinations(output, "behavior_alignment")
    missing = sorted(set(BEHAVIOR_COLUMNS) - set(curves.columns))
    if missing:
        raise ValueError(f"Behavior alignment columns are missing: {missing}")
    trials = _trial_ids(curves["valid_trial"])
    mask = _boolean(curves["scored_window"], "scored_window")
    metrics = [(
        "完全正确率" if condition != 3 else "种正确率（完全正确）",
        "observed_rolling_exact_accuracy", "conditional_rolling_correct_probability",
        "rolling_accuracy_residual",
    )]
    if condition == 3:
        for spec in (
            ("科正确率（至少答对科）", "observed_rolling_family_accuracy",
             "conditional_rolling_family_probability", "rolling_family_residual"),
            ("平均得分（0、0.5、1）", "observed_rolling_mean_reward",
             "conditional_rolling_expected_reward", "rolling_reward_residual"),
        ):
            present = set(spec[1:]) & set(curves.columns)
            if present and present != set(spec[1:]):
                raise ValueError("Condition 3 alignment requires each complete metric column set")
            if present:
                metrics.append(spec)
    for _, observed, predicted, residual in metrics:
        first, second, difference = [curves[name].to_numpy(dtype=float) for name in
                                      (observed, predicted, residual)]
        _probabilities(first[mask], name=observed)
        _probabilities(second[mask], name=predicted)
        if (not np.isfinite(difference[mask]).all()
                or not np.allclose(difference[mask], first[mask] - second[mask],
                                   rtol=0, atol=1e-10)):
            raise ValueError("Accuracy residual must equal observed minus predicted")

    Path(output).mkdir(parents=True, exist_ok=True)
    _style()
    fig, axes = plt.subplots(
        len(metrics), 2, figsize=(7.2047244094, 3.1 if len(metrics) == 1 else 7.0),
        squeeze=False, layout="constrained", sharex=True,
        gridspec_kw={"width_ratios": [1.5, 1],
                     "height_ratios": [1.25, *([1.] * (len(metrics) - 1))]},
    )
    for row, (label, observed, predicted, residual) in enumerate(metrics):
        comparison, errors = axes[row]
        first, second, difference = [np.where(mask, curves[name], np.nan) for name in
                                      (observed, predicted, residual)]
        comparison.plot(trials, first, color=GRAY, label="真实作答", lw=1.)
        comparison.plot(trials, second, color=BLUE, label="模型预测", lw=1., ls="--")
        comparison.set(ylabel=label, ylim=(0, 1.03),
                       title=f"{chr(97 + row * 2)}  真实与模型的滚动比较")
        if row == 0:
            comparison.legend(fontsize=7, ncol=2, loc="upper left")
        errors.axhline(0, color=GRAY, lw=.6)
        errors.plot(trials, difference, color=ORANGE, lw=.9)
        extent = max(.1, float(np.max(np.abs(difference[mask]))) * 1.15) if mask.any() else 1.
        errors.set(ylabel="真实 − 模型", ylim=(-extent, extent),
                   title=f"{chr(98 + row * 2)}  偏差的方向与大小")
        if not mask.any():
            comparison.text(.5, .5, "当前序列没有完整的计分窗口", ha="center", va="center",
                            transform=comparison.transAxes, color=GRAY)
        for ax in (comparison, errors):
            if len(trials):
                ax.set_xlim(trials[0] - .5, trials[-1] + .5)
    for ax in axes[-1]:
        ax.set_xlabel("有效试次（窗口终点）")
    fig.suptitle(
        f"S{subject} · 条件{condition} · behavioral accuracy 对齐\n"
        "模型沿真实选择与反馈历史预测；正偏差表示真实正确率高于预测",
        fontsize=9,
    )
    save(fig, image_path)
    curves.to_csv(source_path, index=False)
    return image_path, source_path


def _wrapped(reason: str) -> str:
    # CJK characters occupy approximately twice a Latin character's width.
    width = 38 if any("\u4e00" <= char <= "\u9fff" for char in reason) else 85
    return "\n".join(textwrap.wrap(reason, width=width))


def oral_alignment_figures(
    subject: int,
    condition: int,
    oral_trials: pd.DataFrame | None,
    summary: dict[str, Any],
    output: Path,
) -> tuple[Path, Path]:
    """Render oral scores or a clear status panel with the same stable filename.

    Zero probability is retained as an observed model failure. Missing and
    unresolved reports stay in the source table and parsing counts; only scored,
    fully encoded reports contribute report probabilities and top-k hits.
    Rolling quantities are supplied by the scoring workflow, which records its
    window and minimum report count. This renderer performs no oral encoding.
    """
    image_path, source_path = _destinations(output, "oral_alignment")
    status = str(summary.get("status", "not_available"))
    completed = status == "completed"
    if completed:
        if oral_trials is None:
            raise ValueError("Completed oral alignment requires its scored trial table")
        missing = sorted(set(ORAL_COLUMNS) - set(oral_trials.columns))
        if missing:
            raise ValueError(f"Oral alignment columns are missing: {missing}")
        trials = _trial_ids(oral_trials["trial"])
        scored = _boolean(oral_trials["scored"], "scored")
        eligible = _boolean(oral_trials["oral_score_mask"], "oral_score_mask")
        zeros = _boolean(oral_trials["report_zero_probability"], "report_zero_probability")
        if not np.array_equal(eligible, scored & oral_trials["parse_status"].eq("coded")):
            raise ValueError("Oral score mask must select scored, completely encoded reports")
        if not oral_trials["parse_status"].isin(["coded", "missing", "needs_review"]).all():
            raise ValueError("Unknown oral parsing status")
        for name in ("report_probability", "report_top1_hit", "report_top3_hit"):
            _probabilities(oral_trials.loc[eligible, name].to_numpy(dtype=float), name=name)
        if np.any(oral_trials.loc[eligible, "report_top1_hit"] >
                  oral_trials.loc[eligible, "report_top3_hit"] + 1e-12):
            raise ValueError("Oral top-1 hit cannot exceed top-3 hit")
        if not np.array_equal(zeros, eligible & oral_trials.report_probability.eq(0)):
            raise ValueError("Zero-support mask must preserve eligible zero-probability reports")
        for name in ("rolling_report_probability", "rolling_top1_hit", "rolling_top3_hit"):
            if name in oral_trials:
                values = oral_trials[name].to_numpy(dtype=float)
                _probabilities(values[~np.isnan(values)], name=name)
    elif oral_trials is not None and len(oral_trials):
        raise ValueError("Unavailable oral alignment cannot display scored trial measurements")

    Path(output).mkdir(parents=True, exist_ok=True)
    _style()
    if not completed:
        fig, ax = plt.subplots(figsize=(7.2047244094, 3.1), layout="constrained")
        ax.set_axis_off()
        reason = str(summary.get("reason") or "尚未提供与当前被试和拟合对应的完整口述评估结果。")
        ax.text(.03, .78, "口述对齐：当前尚无可报告的评估结果", transform=ax.transAxes,
                fontsize=12, color=GRAY, va="top")
        ax.text(.03, .57, _wrapped(reason), transform=ax.transAxes, fontsize=9,
                color=GRAY, va="top", linespacing=1.6)
        ax.text(.03, .12, "本页在每位被试的标准报告中保留；执行状态与原因记录在 oral_alignment.json。",
                transform=ax.transAxes, fontsize=7, color=GRAY)
        source = pd.DataFrame(columns=ORAL_COLUMNS) if oral_trials is None else oral_trials
    else:
        metrics = summary.get("metrics", {})
        fig = plt.figure(figsize=(7.2047244094, 5.5), layout="constrained")
        grid = fig.add_gridspec(3, 2, height_ratios=[1.3, 1., .8])
        probability = fig.add_subplot(grid[0, :])
        hits = fig.add_subplot(grid[1, 0])
        coverage = fig.add_subplot(grid[1, 1])
        notes = fig.add_subplot(grid[2, :])
        probability.scatter(trials[eligible], oral_trials.loc[eligible, "report_probability"],
                            s=5, color=GRAY, alpha=.55, label="真实报告的逐题概率")
        if zeros.any():
            probability.scatter(trials[zeros], np.zeros(zeros.sum()), s=10, marker="x",
                                color=ORANGE, label="模型给出零概率")
        rolling_probability = oral_trials.get("rolling_report_probability")
        if rolling_probability is not None and rolling_probability.notna().any():
            probability.plot(trials, rolling_probability, color=BLUE, lw=1., label="滚动平均概率")
        if not eligible.any():
            probability.text(.5, .5, "当前序列没有可计分的完整口述报告", ha="center", va="center",
                             transform=probability.transAxes, color=GRAY)
        probability.set(ylabel="模型赋予真实口述的概率", ylim=(-.035, 1.035),
                        xlabel="有效试次", title="a  模型是否支持真实报告")
        probability.legend(fontsize=7, ncol=3, loc="upper right")
        if len(trials):
            probability.set_xlim(trials[0] - .5, trials[-1] + .5)

        any_rolling_hits = False
        for name, label, color, line_style in (
            ("rolling_top1_hit", "最可能报告命中率", BLUE, "-"),
            ("rolling_top3_hit", "前三个报告命中率", ORANGE, "--"),
        ):
            if name in oral_trials and oral_trials[name].notna().any():
                hits.plot(trials, oral_trials[name], color=color, lw=.9, ls=line_style, label=label)
                any_rolling_hits = True
        if any_rolling_hits:
            hits.legend(fontsize=7, loc="upper left")
        else:
            hits.text(.5, .5, "有效报告不足或尚无滚动汇总\n当前未绘制滚动命中率", ha="center", va="center",
                      transform=hits.transAxes, fontsize=8, color=GRAY)
        hits.set(ylim=(0, 1.03), ylabel="命中率", xlabel="有效试次",
                 title="b  真实报告在模型排序中的位置")
        if len(trials):
            hits.set_xlim(trials[0] - .5, trials[-1] + .5)

        labels, values = [], []
        for name, label in (
            ("parser_coverage", "计分题：可完整编码"),
            ("vocabulary_coverage", "已编码：进入模型词表"),
            ("positive_support_coverage", "已编码：获得正概率"),
        ):
            value = metrics.get(name)
            if value is not None and np.isfinite(float(value)):
                _probabilities(np.asarray([value], dtype=float), name=name)
                labels.append(label)
                values.append(float(value))
        if labels:
            positions = np.arange(len(labels))
            coverage.barh(positions, values, color=[GRAY, BLUE, ORANGE][:len(labels)], height=.55)
            coverage.set_yticks(positions, labels, fontsize=7)
            coverage.invert_yaxis()
            for position, value in zip(positions, values):
                coverage.text(min(value + .025, 1.01), position, f"{value:.1%}", fontsize=7, va="center")
        else:
            coverage.text(.5, .5, "当前未提供覆盖率汇总", ha="center", va="center",
                          transform=coverage.transAxes, color=GRAY)
        coverage.set(xlim=(0, 1.23), xlabel="比例（分母见行标签）", title="c  可解析性与预测覆盖")

        notes.set_axis_off()
        counts = [f"计分题 {int(scored.sum())}；完整编码 {int(eligible.sum())}；"
                  f"缺失 {int((scored & oral_trials.parse_status.eq('missing')).sum())}；"
                  f"待复核 {int((scored & oral_trials.parse_status.eq('needs_review')).sum())}；"
                  f"零概率 {int(zeros.sum())}。"]
        if "window_size" in summary and "min_coded_reports" in summary:
            counts.append(f"滚动窗口 {summary['window_size']} 个实际试次；"
                          f"至少 {summary['min_coded_reports']} 条可计分报告。")
        if metrics.get("mean_report_nll") is not None:
            value = float(metrics["mean_report_nll"])
            label = "∞（含零概率报告）" if np.isposinf(value) else f"{value:.3f}"
            counts.append(f"精确平均口述 NLL：{label}。")
        counts.append("口述概率取当前作答后、当前反馈前的状态；缺失和待复核记录进入覆盖率审查。")
        notes.text(0, .9, "\n".join(counts), transform=notes.transAxes,
                   va="top", fontsize=7, linespacing=1.5, color=GRAY)
        source = oral_trials
    scope = "（短序列通路检查）" if summary.get("scope") == "smoke_only" else ""
    fig.suptitle(f"S{subject} · 条件{condition} · oral 对齐{scope}", fontsize=10)
    save(fig, image_path)
    source.to_csv(source_path, index=False)
    return image_path, source_path
