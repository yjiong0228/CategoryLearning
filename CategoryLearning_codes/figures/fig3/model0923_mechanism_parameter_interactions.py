"""Plot audited mechanism/parameter interactions for development participant 208.

From the repository root, run this module with --cross, --behavior-audit,
--oral-audit, --precision-audit and --output NEW_DIRECTORY. The first PNG joins
the five completed SEARCH cross cells to both oral readings. The second PNG
shows every cell of the training-only belief/precision probe, including the
off-grid original anchor. All source tables and audit hashes are retained.

Figure contract: mechanism effects depend on the behavior-selected parameter
point; reducing precision learning changes, but does not remove, the observed
training tradeoff. Quantitative grids, Python only, 183 mm wide, PNG at 300 dpi.
This is a development diagnostic for one participant. Error bars describe
Monte Carlo computation, and parameter nominations remain unchanged. Existing
figure conventions supply palette and typography only; statistics are read
from independently audited tables.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


CELLS = ("RETAIN_RETAIN", "ONLINE_RETAIN", "RETAIN_ONLINE", "ONLINE_ONLINE", "ONLINE_NEAR")
ANCHORS = ("4596d22a0bbe600f", "bbdae14d0eaf44cb")
RATES = (0.003, 0.03, 0.3, 1.0)
GAMMAS = (0.0, 0.6, 1.0)
COLORS = {"RETAIN": "#4678A5", "ONLINE": "#CC8152"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_inputs(path: Path, inputs: dict[str, str]) -> dict:
    proof = json.loads(path.read_text())
    if proof["status"] != "complete":
        raise ValueError(f"Incomplete audit: {path}")
    recorded = dict(proof["inputs_sha256"])
    recorded[str(path.resolve())] = digest(path)
    for name, sha in proof.get("artifacts", {}).items():
        recorded[str((path.parent / name).resolve())] = sha
    for name, sha in recorded.items():
        if name in inputs and inputs[name] != sha:
            raise ValueError(f"Audits disagree on input: {name}")
        inputs[name] = sha
    return proof


def verify_hashes(inputs: dict[str, str]) -> None:
    for name, sha in inputs.items():
        if digest(Path(name)) != sha:
            raise ValueError(f"Audited input changed: {name}")


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def cross_figure(scores: pd.DataFrame, oral: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.20472, 6.29921))  # 183 by 160 mm.
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.86, hspace=0.52, wspace=0.36)
    titles = ("Training choice prediction", "Later choice prediction",
              "Whole autonomous distribution", "Later complete-report support")
    for letter, ax, title in zip("abcd", axes.flat, titles):
        ax.set_title(title, loc="left", pad=11)
        ax.text(-0.19, 1.10, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.set(xticks=[0, 1, 2], xticklabels=["Old", "New", "Near"], xlim=(-0.18, 2.18))
        ax.set_xlabel("Behavior-selected parameter point")
    for mechanism, marker in (("RETAIN", "s"), ("ONLINE", "o")):
        labels = [f"{mechanism}_RETAIN", f"{mechanism}_ONLINE"]
        if mechanism == "ONLINE":
            labels.append("ONLINE_NEAR")
        for ax, phase, field in ((axes[0, 0], "training", "choice_nll"),
                                 (axes[0, 1], "test", "choice_nll"),
                                 (axes[1, 0], "whole", "energy")):
            values = scores[scores.phase == phase].set_index("cell").loc[labels, field]
            ax.plot(range(len(labels)), values.to_numpy(), marker=marker,
                    color=COLORS[mechanism], lw=1.2, ms=4)
        for reading, linestyle in (("midpoint", "-"), ("relative", "--")):
            rows = oral[(oral.reading == reading) & (oral.window == "test")
                        & (oral.domain == "functional") & (oral.subset == "complete")
                        & (oral.metric == "conditioned")].set_index("arm").loc[labels]
            axes[1, 1].errorbar(range(len(labels)), rows.support, yerr=rows.support_mcse,
                               marker=marker, ms=4, color=COLORS[mechanism],
                               ls=linestyle, lw=1.2, capsize=2,
                               mfc=COLORS[mechanism] if reading == "midpoint" else "white")
    for ax in axes[0]:
        ax.set_ylabel("Mean choice NLL (lower is better)")
    axes[1, 0].set_ylabel("Energy (lower is better)")
    axes[1, 1].set(ylabel="Functional support mass", ylim=(0, 1))
    handles = [Line2D([], [], color=COLORS[m], marker=k, label=m, lw=1.2)
               for m, k in (("RETAIN", "s"), ("ONLINE", "o"))]
    handles += [Line2D([], [], color=".35", ls=ls, label=label, lw=1.2)
                for ls, label in (("-", "Oral: midpoint"), ("--", "Oral: relative"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.071), ncol=2)
    fig.suptitle("Mechanism effects reverse across parameter points", y=0.98, fontsize=11)
    fig.text(0.5, 0.92, "Participant 208 | SEARCH | 8 filters and 128 autonomous paths per cell",
             ha="center", fontsize=8)
    fig.text(0.5, 0.025, "Oral: 119 later reports; bars = 1 Monte Carlo SE; readings nearly overlap.",
             ha="center", fontsize=7)
    fig.savefig(output / "mechanism_parameter_cross.png", dpi=300)
    plt.close(fig)


def precision_figure(scores: pd.DataFrame, output: Path) -> None:
    if not (scores.learning_rate > 0).all():
        raise ValueError("Log-axis learning rates must be strictly positive")
    fig, axes = plt.subplots(2, 2, figsize=(7.20472, 6.29921), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.87, hspace=0.36, wspace=0.20)
    colors = ("#555555", "#458C8A", "#9270A5")
    markers = ("o", "s", "^")
    for row, anchor in enumerate(ANCHORS):
        for col, mechanism in enumerate(("ONLINE", "SEPARATE")):
            ax = axes[row, col]
            part = scores[(scores.mechanism == mechanism) & (scores.anchor == anchor)]
            for gamma, color, marker in zip(GAMMAS, colors, markers):
                curve = part[(part.gamma == gamma) & (part.learning_rate.isin(RATES))].sort_values("learning_rate")
                ax.errorbar(curve.learning_rate, curve.choice_nll, yerr=curve.choice_nll_mc_se,
                            color=color, marker=marker, ms=4, lw=1.2, capsize=2)
            original = part[~part.learning_rate.isin(RATES)]
            if len(original):
                ax.errorbar(original.learning_rate, original.choice_nll, yerr=original.choice_nll_mc_se,
                            color=colors[2], marker="D", mfc="white", ms=5, ls="none", capsize=2)
            ax.set_xscale("log")
            ax.set(ylim=(0.94, 1.28), xticks=RATES, xticklabels=["0.003", "0.03", "0.3", "1"])
            ax.set_title(f"Anchor {'AB'[row]} / {mechanism}", loc="left", pad=10)
            ax.text(-0.14, 1.07, "abcd"[2 * row + col], transform=ax.transAxes,
                    fontweight="bold", fontsize=10)
            if col == 0:
                ax.set_ylabel("Training choice NLL (lower is better)")
            if row == 1:
                ax.set_xlabel("Precision learning rate (log scale)")
    handles = [Line2D([], [], color=c, marker=m, label=f"γ = {g:g}", lw=1.2)
               for g, c, m in zip(GAMMAS, colors, markers)]
    handles.append(Line2D([], [], color=colors[2], marker="D", mfc="white", ls="none",
                          label="Original B: γ = 1, rate = 0.06"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2)
    fig.suptitle("Slower precision learning changes the retention tradeoff", y=0.98, fontsize=11)
    fig.text(0.5, 0.925, "Participant 208 | All 25 points × 2 mechanisms | Original 384 training choices",
             ha="center", fontsize=8)
    fig.text(0.5, 0.025, "Bars = 1 Monte Carlo SE from 2 filters (128 particles); one participant.",
             ha="center", fontsize=7)
    fig.savefig(output / "belief_precision_interaction.png", dpi=300)
    plt.close(fig)


def draw(cross: Path, behavior_audit: Path, oral_audit: Path, precision_audit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new interaction-figure output directory")
    inputs: dict[str, str] = {}
    audit_inputs(behavior_audit / "validation.json", inputs)
    oral_proof = audit_inputs(oral_audit / "completion.json", inputs)
    precision_proof = audit_inputs(precision_audit / "completion.json", inputs)
    for name in ("scores.csv", "contrasts.csv", "protocol.json", "completion.json"):
        path = cross / name
        if str(path.resolve()) not in inputs:
            raise ValueError(f"Cross file is not bound by independent audit: {path}")
    if oral_proof["cell_scores"] != 600 or precision_proof["cells"] != 50:
        raise ValueError("Expected complete five-cell oral cross and 50 probe cells")
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__))
    verify_hashes(inputs)
    behavior = pd.read_csv(cross / "scores.csv")
    oral = pd.read_csv(oral_audit / "cell_scores.csv")
    precision = pd.read_csv(precision_audit / "cell_scores.csv")
    if len(behavior) != 15 or behavior.duplicated(["cell", "phase"]).any() or set(behavior.cell) != set(CELLS):
        raise ValueError("Incomplete or duplicate behavioral cross")
    if set(behavior.phase) != {"training", "test", "whole"}:
        raise ValueError("Unexpected behavioral windows")
    selected_oral = oral[(oral.subset == "complete") & (oral.metric == "conditioned")]
    if selected_oral.duplicated(["arm", "reading", "window", "domain"]).any():
        raise ValueError("Duplicate oral cell")
    if not np.allclose(selected_oral[["support", "conflict", "undetermined"]].sum(axis=1), 1):
        raise ValueError("Oral masses must sum to one")
    for window, count in (("whole", 391), ("training", 272), ("test", 119)):
        rows = selected_oral[selected_oral.window == window]
        if len(rows) != 20 or not (rows.reports == count).all() or not (rows.filter_seeds == 8).all():
            raise ValueError("Oral cells must use the same reports and eight filters")
    training = precision[precision.window == "training"].copy()
    training["anchor"] = training.anchor_ids.map(lambda value: json.loads(value)[0])
    if len(training) != 50 or training.duplicated(["mechanism", "candidate"]).any():
        raise ValueError("Probe must retain all 50 cells")
    for anchor in ANCHORS:
        for mechanism in ("ONLINE", "SEPARATE"):
            rows = training[(training.anchor == anchor) & (training.mechanism == mechanism)]
            if set(zip(rows.gamma, rows.learning_rate)) != (
                {(g, r) for g in GAMMAS for r in RATES}
                | ({(1.0, 0.06)} if anchor == ANCHORS[1] else set())
            ):
                raise ValueError("Missing or unexpected probe grid cell")
    for frame, columns in ((behavior, ["choice_nll", "energy"]),
                           (selected_oral, ["support", "conflict", "undetermined", "support_mcse"]),
                           (training, ["choice_nll", "choice_nll_mc_se", "correct_loss", "error_loss"])):
        values = frame[columns].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Non-finite or negative plotted metric")
    np.testing.assert_allclose(training.correct_loss + training.error_loss, training.choice_nll, atol=1e-12)
    output.mkdir(parents=True)
    style()
    cross_figure(behavior, oral, output)
    precision_figure(training, output)
    sources = {
        "behavior_scores.csv": cross / "scores.csv",
        "behavior_contrasts.csv": cross / "contrasts.csv",
        "oral_scores.csv": oral_audit / "cell_scores.csv",
        "oral_contrasts.csv": oral_audit / "cross_contrasts.csv",
        "precision_scores.csv": precision_audit / "cell_scores.csv",
        "precision_contrasts.csv": precision_audit / "interactions.csv",
    }
    for name, path in sources.items():
        shutil.copyfile(path, output / name)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>208：机制、参数与口述的交叉检验</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto;line-height:1.7}img{width:100%}</style>
<h1>机制效果随参数而变，精度学习影响保持率的取舍</h1>
<p>被试208，SEARCH提议。参数由训练选择估计，口述用于外部检验。
旧参数来自RETAIN，新参数及接近候选来自ONLINE。原提名保持不变。</p>
<img src="mechanism_parameter_cross.png" alt="五格交叉检验的前后段选择损失、全程自主Energy和后段口述支持">
<p>a、b：前384题与后192题的选择负对数似然，越低越好，每格八次512粒子滤波。
c：全部576题自主曲线的Energy，每格全部128条路径，越低越好。
d：后段119份完整口述的整句功能支持质量，越高表示更多读出权重支持报告中的判断。
中点与相对读法统一应用于全部方案，曲线接近重合；误差棒是一次完整滤波删除所得的模拟标准误。
三个点是离散参数组合，连线帮助对照，不表示连续参数插值。RETAIN未运行接近候选。</p>
<p>旧参数下，ONLINE提高后段口述支持约19.9个百分点；新参数下则降低约24.3个百分点。
ONLINE内部，两点全程选择损失接近，后段口述支持仍相差约19.5个百分点。
当前选择数据对内部过程的约束有限，口述能够检验这些差异。支持、冲突和未确定质量及配对误差均保留在源表。</p>
<img src="belief_precision_interaction.png" alt="两组固定背景参数下，保持率与精度学习率对训练选择损失的作用">
<p>第二张图保留25点、两种机制的全部50项训练评分。每项两个共同种子、128粒子；只使用原384题训练选择。
A、B分别为既有SEPARATE训练候选中γ＝0与γ＝1的领先背景点。
每条线仅改变精度学习率；同一面板的三条线改变γ，其余参数固定。空心菱形保留B的原学习率0.06。
误差棒表示两个完整滤波重复的模拟波动；一名开发被试的局部网格限定本轮结论的适用范围。</p>
<p>A背景中，SEPARATE在学习率0.3时把γ从0改为1，训练损失增加约0.0965；
学习率0.03时，同一改变减少约0.0209。但减慢学习后，γ＝1的损失仍约1.0423，高于原γ＝0点的0.9707。
B的原γ＝1点为1.0318。本轮网格显示速度与保持的交互，同时保留未解决的训练代价。
γ还影响M记忆后验及其参与的搜索、退出；它在当前实现中涉及多个通道。</p>
<p>图中统计量均来自已完成的独立核验。一名被试；无群体假设检验、p值或多重比较推断。
a—c显示点估计，配对差值与模拟标准误见行为源表；口述和网格误差棒仅描述计算波动。
本轮结果用于机制开发，参数联合恢复和跨被试验证继续按研究计划完成。</p>
<p><a href="behavior_scores.csv">行为得分</a> · <a href="behavior_contrasts.csv">行为配对差</a> ·
<a href="oral_scores.csv">完整口述得分</a> · <a href="oral_contrasts.csv">口述配对差</a> ·
<a href="precision_scores.csv">精度网格得分</a> · <a href="precision_contrasts.csv">网格交互</a></p></html>'''
    (output / "index.html").write_text(page)
    verify_hashes(inputs)
    manifest = {
        "status": "rendered_pending_visual_review", "inputs_sha256": inputs,
        "cross_cells": 5, "oral_reports": 391, "probe_cells": 50,
        "source_tables_copied_exactly": len(sources), "nominations_changed": False,
        "figure_contract": "One-participant mechanism/parameter interaction and local training tradeoff; quantitative grids; Python; PNG only",
        "export_qa": "Repository instruction selects PNG only; SVG/PDF/TIFF requirements in generic source preflight do not apply. Draft resolution is 300 dpi, dimensions 183 by 160 mm.",
        "display_selection": "Cross: training/test NLL, whole Energy, later complete functional oral support in both readings; full tables retained. Probe: all 50 whole-training cells; six block windows retained in tables.",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")},
        "artifacts": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({"status": manifest["status"], "inputs": len(inputs), "cross_cells": 5, "probe_cells": 50}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--behavior-audit", type=Path, required=True)
    parser.add_argument("--oral-audit", type=Path, required=True)
    parser.add_argument("--precision-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.cross, args.behavior_audit, args.oral_audit, args.precision_audit, args.output)


if __name__ == "__main__":
    main()
