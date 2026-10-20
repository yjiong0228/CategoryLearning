"""Draw the registered candidate evidence without selecting a winning version.

Run --source COLLECTED_EVIDENCE --output NEW_DIRECTORY from the repository root.
A partial collection requires --allow-partial and labels every image with the
missing participants. All candidates and autonomous paths are retained. Four
183-mm PNGs show paired behavior effects, autonomous distributions and both
oral domains under the two fixed readings. Full source tables are copied.

Behavior bars show ±2 paired simulation MCSE at fitted parameters. Path bands
show 10th–90th percentiles. Oral panels use completely encoded reports and
preserve support, conflict, undetermined and coverage. Other oral strata and
between-participant variation remain in the source tables. No population
confidence interval, parameter uncertainty or final recommendation is implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_finalization_evidence import CANDIDATES, CONTRASTS
from src.Bayesian_state.evaluation.model_0923_h_validation import digest, load

REGISTERED = [108, 205, 306]
COLORS = {"SEARCH_ADD10": "#527697", "SEARCH_LOG25": "#3B8C85", "POOL_LOG25": "#BE8355"}
NAMES = {"SEARCH_ADD10": "SEARCH · ADD10", "SEARCH_LOG25": "SEARCH · LOG25", "POOL_LOG25": "POOL · LOG25"}
SHORT = {"SEARCH_ADD10": "S-A", "SEARCH_LOG25": "S-L", "POOL_LOG25": "P-L"}
PHASES = {"training": "Training", "test": "Later trials", "whole": "Whole sequence"}
LABELS = ("support", "conflict", "undetermined")
MASS_COLORS = ("#538B84", "#C58B60", "#D9DDE2")


def check_scope(manifest: dict, allow_partial: bool) -> list[int]:
    subjects = manifest.get("subjects", [])
    if (manifest.get("status") != "complete" or manifest.get("registered_subjects") != REGISTERED
            or not subjects or subjects != sorted(set(subjects)) or not set(subjects) <= set(REGISTERED)
            or manifest.get("registered_cells") != 9 or manifest.get("collected_cells") != 3 * len(subjects)
            or manifest.get("missing_subjects") != sorted(set(REGISTERED) - set(subjects))
            or manifest.get("complete_primary_matrix") is not (subjects == REGISTERED)):
        raise ValueError("Require the intact registered participant-by-candidate collection")
    if subjects != REGISTERED and not allow_partial:
        raise ValueError("Partial evidence requires --allow-partial and visible missing-participant labels")
    return subjects


def check_oral(frame: pd.DataFrame) -> None:
    """A missing report set remains undefined, never a zero-support result."""
    if frame.duplicated(["subject", "candidate", "reading", "window", "domain", "subset"]).any():
        raise ValueError("Duplicated oral evidence row")
    values = frame[list(LABELS)].to_numpy(float)
    present = frame.reports.to_numpy() > 0
    if (not np.isfinite(values[present]).all() or np.any(values[present] < -1e-12)
            or np.any(values[present] > 1 + 1e-12) or not np.isnan(values[~present]).all()):
        raise ValueError("Invalid oral probability mass or zero-report score")
    np.testing.assert_allclose(values[present].sum(1), 1., atol=1e-12, rtol=0)


def read_evidence(source: Path, allow_partial: bool) -> tuple[dict, dict, dict, pd.DataFrame, int]:
    """Check plotted values and all saved path blocks against frozen sources."""
    manifest = load(source / "manifest.json")
    subjects = check_scope(manifest, allow_partial)
    bindings = {str((source / "manifest.json").resolve()): digest(source / "manifest.json")}

    def bind(path, expected):
        path = Path(path).resolve()
        if digest(path) != expected or (str(path) in bindings and bindings[str(path)] != expected):
            raise ValueError(f"Figure source changed: {path}")
        bindings[str(path)] = expected
        return path

    tables = {}
    for name, sha in manifest["artifacts_sha256"].items():
        file = bind(source / name, sha)
        if file.suffix == ".csv":
            tables[file.stem] = pd.read_csv(file)
    cells, scores, oral, curves = [tables[k] for k in ("cells", "behavior_scores", "oral_scores", "curves")]
    keys = {(s, c) for s in subjects for c in CANDIDATES}
    if (cells.duplicated(["subject", "candidate"]).any()
            or set(cells[["subject", "candidate"]].itertuples(index=False, name=None)) != keys
            or scores.duplicated(["subject", "candidate", "phase"]).any()
            or set(scores[["subject", "candidate", "phase"]].itertuples(index=False, name=None))
            != {(s, c, p) for s, c in keys for p in PHASES}
            or curves.duplicated(["subject", "candidate", "trial"]).any()):
        raise ValueError("Missing or duplicated candidate, phase or curve")
    check_oral(oral)
    expected_oral = {(s, c, r, w, d, u) for s, c in keys for r in ("midpoint", "relative")
                    for w in ("whole", "training", "test", "first64", "last64")
                    for d in ("functional", "structural") for u in ("complete", "catalogue_supported")}
    if set(oral[["subject", "candidate", "reading", "window", "domain", "subset"]].itertuples(index=False, name=None)) != expected_oral:
        raise ValueError("Incomplete fixed-reading oral matrix")
    checks, blocks = 0, []

    def close(actual, expected):
        nonlocal checks
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)
        checks += np.asarray(actual).size

    for row in cells.itertuples(index=False):
        if (row.precision_family, row.arm) != CANDIDATES[row.candidate]:
            raise ValueError("Candidate label differs from its executed proposal/precision family")
        summary = Path(row.evaluation_summary)
        declared = load(bind(summary / "manifest.json", manifest["inputs_sha256"][str((summary / "manifest.json").resolve())]))
        source_blocks = pd.read_csv(bind(summary / "path_blocks.csv", declared["summary_sha256"]["path_blocks.csv"]))
        subset = source_blocks.loc[source_blocks.subject.eq(row.subject) & source_blocks.model.eq(row.arm)].copy()
        own = curves.loc[curves.subject.eq(row.subject) & curves.candidate.eq(row.candidate)].sort_values("trial")
        paths = tables["paths"].loc[tables["paths"].subject.eq(row.subject) & tables["paths"].candidate.eq(row.candidate)]
        pivot = subset.pivot(index="seed", columns="trial", values="accuracy")
        width, cut, length = declared["block_width"], row.training_trials, row.training_trials + row.later_trials
        ends = np.arange(width, length + width, width).clip(max=length)
        if (width != 32 or paths.seed.duplicated().any() or len(paths) < 2 or set(pivot.index) != set(paths.seed)
                or not np.isfinite(pivot.to_numpy()).all() or (pivot.to_numpy() < 0).any()
                or (pivot.to_numpy() > 1).any() or list(pivot.columns) != list(ends)
                or cut not in ends):
            raise ValueError("Path seeds, block sequence or chronological boundary changed")
        close(own.trial, ends); close(own.training_trials, cut)
        close(own.autonomous, pivot.mean(axis=0))
        close(own.lower, np.quantile(pivot, .1, axis=0)); close(own.upper, np.quantile(pivot, .9, axis=0))
        widths = np.diff(np.r_[0, ends])
        for phase in PHASES:
            mask = ends <= cut if phase == "training" else ends > cut if phase == "test" else np.ones(len(ends), bool)
            score = scores.loc[scores.subject.eq(row.subject) & scores.candidate.eq(row.candidate) & scores.phase.eq(phase)].iloc[0]
            close(score.trials, widths[mask].sum())
            close(score.observed_accuracy, np.average(own.observed.to_numpy()[mask], weights=widths[mask]))
            close(score.autonomous_accuracy, np.average(pivot.to_numpy()[:, mask], axis=1, weights=widths[mask]).mean())
            close(score.choice_nll, np.average((own.correct_loss + own.error_loss).to_numpy()[mask], weights=widths[mask]))
        blocks.append(subset.assign(candidate=row.candidate))
    paired = tables["contrasts"]
    for row in paired.loc[paired.evidence.eq("behavior")].itertuples(index=False):
        before, after = CONTRASTS[row.contrast]
        values = scores.loc[scores.subject.eq(row.subject) & scores.phase.eq(row.window)].set_index("candidate")
        close(row.difference, values.loc[after, row.metric] - values.loc[before, row.metric])
        if not np.isfinite(row.mcse) or row.mcse < 0:
            raise ValueError("Invalid paired simulation error")
    for _, group in curves.groupby(["subject", "trial"]):
        close(group.observed, group.observed.iloc[0])
    for _, group in oral.groupby(["subject", "reading", "window", "domain", "subset"]):
        close(group.reports, group.reports.iloc[0]); close(group.window_trials, group.window_trials.iloc[0])
    return manifest, tables, bindings, pd.concat(blocks, ignore_index=True), checks


def banner(manifest: dict) -> str:
    if manifest["complete_primary_matrix"]:
        return "Registered development comparison · 3 participants · 9 candidate cells"
    return "PARTIAL EVIDENCE · awaiting " + ", ".join(f"S{s}" for s in manifest["missing_subjects"])


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=300, facecolor="white")
    plt.close(fig)


def behavior_figure(manifest: dict, tables: dict, output: Path) -> None:
    subjects = manifest["subjects"]
    full = manifest["complete_primary_matrix"]
    labels = [f"S{s}" for s in subjects] + (["Equal mean"] if full else [])
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 120 / 25.4))
    colors = {"precision": COLORS["SEARCH_LOG25"], "proposal": COLORS["POOL_LOG25"]}
    for i, metric in enumerate(("choice_nll", "energy")):
        for j, phase in enumerate(PHASES):
            ax = axes[i, j]
            ax.axvline(0, color="#969CA4", lw=.7)
            for contrast, offset in (("precision", -.13), ("proposal", .13)):
                rows = tables["contrasts"].query("evidence == 'behavior' and contrast == @contrast and window == @phase and metric == @metric").set_index("subject").loc[subjects]
                x, error = rows.difference.to_numpy(), rows.mcse.to_numpy()
                ax.errorbar(x, np.arange(len(subjects)) + offset, xerr=2 * error, fmt="o",
                            color=colors[contrast], ms=3, capsize=2, lw=.8)
                if full:
                    mean, = tables["equal_subject_contrasts"].query("evidence == 'behavior' and contrast == @contrast and window == @phase and metric == @metric").itertuples(index=False)
                    np.testing.assert_allclose(mean.equal_subject_difference, x.mean(), atol=1e-12, rtol=0)
                    np.testing.assert_allclose(mean.equal_subject_mcse, np.linalg.norm(error) / len(subjects), atol=1e-12, rtol=0)
                    ax.errorbar(mean.equal_subject_difference, len(subjects) + offset,
                        xerr=2 * mean.equal_subject_mcse, fmt="D", color=colors[contrast], ms=3, capsize=2, lw=.8)
            ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels)-.4, -.6),
                   title=f"{'abcdef'[3*i+j]}  {PHASES[phase]}", xlabel="Δ NLL / trial" if i == 0 else "Δ Energy score")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.suptitle("Paired behavior effects across learning phases", fontsize=8, y=.98)
    fig.text(.5, .925, banner(manifest), ha="center", fontsize=6)
    fig.legend(handles=[Line2D([], [], color=colors["precision"], marker="o", ls="", label="Precision: SEARCH LOG25 − SEARCH ADD10"),
                        Line2D([], [], color=colors["proposal"], marker="o", ls="", label="Proposal: SEARCH LOG25 − POOL LOG25")],
               loc="upper center", bbox_to_anchor=(.5, .90), ncol=1, fontsize=6)
    fig.text(.5, .018, "Negative favors SEARCH LOG25; bars: ±2 paired simulation MCSE at fitted parameters", ha="center", fontsize=6)
    fig.subplots_adjust(left=.105, right=.96, top=.72, bottom=.12, wspace=.8, hspace=.65)
    save(fig, output, "behavior_contrasts.png")


def trajectory_figure(manifest: dict, tables: dict, blocks: pd.DataFrame, output: Path) -> None:
    subjects = manifest["subjects"]
    fig, axes = plt.subplots(len(subjects), 3, figsize=(183 / 25.4, (52 + 43 * len(subjects)) / 25.4),
                             squeeze=False, sharey=True)
    for i, subject in enumerate(subjects):
        for j, candidate in enumerate(CANDIDATES):
            ax = axes[i, j]
            curve = tables["curves"].query("subject == @subject and candidate == @candidate").sort_values("trial")
            paths = blocks.query("subject == @subject and candidate == @candidate")
            for _, group in paths.groupby("seed"):
                group = group.sort_values("trial")
                ax.plot(group.trial, group.accuracy, color=COLORS[candidate], alpha=.06, lw=.35)
            ax.fill_between(curve.trial, curve.lower, curve.upper, color=COLORS[candidate], alpha=.14)
            ax.plot(curve.trial, curve.autonomous, color=COLORS[candidate], lw=1.35)
            ax.plot(curve.trial, curve.observed, color="#252A31", lw=1.1)
            ax.axvline(curve.training_trials.iloc[0], color="#777777", lw=.7, ls=":")
            ax.set(title=f"{'abcdefghi'[3*i+j]}  S{subject} · {NAMES[candidate]}",
                   xlim=(0, curve.trial.max()), ylim=(-.02, 1.02))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if j == 0: ax.set_ylabel("Block accuracy")
            if i == len(subjects)-1: ax.set_xlabel("Trial")
    fig.suptitle("Autonomous learning: all paths and observed behavior", fontsize=8, y=.98)
    fig.text(.5, .925, banner(manifest), ha="center", fontsize=6)
    fig.legend(handles=[Line2D([], [], color="#252A31", lw=1.1, label="Observed"),
                        Line2D([], [], color="#6A7887", lw=1.35, label="Autonomous mean"),
                        Patch(facecolor="#AEB7C2", alpha=.4, label="10th–90th path percentiles")],
               loc="upper center", ncol=3, bbox_to_anchor=(.5, .91))
    fig.text(.5, .015, "All saved paths retained; 32-trial blocks; dotted line: end of behavior fitting", ha="center", fontsize=6)
    fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.14, wspace=.18, hspace=.46)
    save(fig, output, "autonomous_trajectories.png")


def oral_figure(manifest: dict, tables: dict, domain: str, output: Path) -> None:
    keys = [(s, c) for s in manifest["subjects"] for c in CANDIDATES]
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, (80 + 8 * len(keys)) / 25.4))
    for i, reading in enumerate(("midpoint", "relative")):
        for j, phase in enumerate(("training", "test")):
            ax = axes[i, j]
            data = tables["oral_scores"].query("domain == @domain and reading == @reading and window == @phase and subset == 'complete'").set_index(["subject", "candidate"])
            for y, key in enumerate(keys):
                row = data.loc[key]
                if row.reports:
                    left = 0.
                    for label, color in zip(LABELS, MASS_COLORS):
                        ax.barh(y, row[label], left=left, height=.67, color=color, edgecolor="white", linewidth=.3)
                        left += row[label]
                else:
                    ax.text(.5, y, "No complete reports", ha="center", va="center", fontsize=6)
                ax.text(1.025, y, f"{int(row.reports)}/{int(row.window_trials)}", va="center", fontsize=6)
            ax.set(yticks=range(len(keys)), yticklabels=[f"S{s}  {SHORT[c]}" for s, c in keys],
                   ylim=(len(keys)-.4, -.6), xlim=(0, 1), xticks=[0, .5, 1], xlabel="Probability mass",
                   title=f"{'abcd'[2*i+j]}  {PHASES[phase]} · {reading}")
            for y in range(3, len(keys), 3): ax.axhline(y-.5, color="#EEEEEE", lw=.7)
    fig.suptitle(f"Oral evidence: {domain} support and unresolved content", fontsize=8, y=.98)
    fig.text(.5, .935, banner(manifest), ha="center", fontsize=6)
    fig.legend(handles=[Patch(facecolor=c, label=l.capitalize()) for c, l in zip(MASS_COLORS, LABELS)],
               ncol=3, loc="upper center", bbox_to_anchor=(.5, .92))
    fig.text(.5, .035, "S-A: SEARCH ADD10 · S-L: SEARCH LOG25 · P-L: POOL LOG25; counts: complete reports / trials", ha="center", fontsize=6)
    fig.text(.5, .012, "Two fixed readings; reports are noisy external evidence; unresolved mass remains explicit", ha="center", fontsize=6)
    fig.subplots_adjust(left=.12, right=.89, top=.81, bottom=.14, wspace=.62, hspace=.48)
    save(fig, output, f"oral_{domain}.png")


def draw(source: Path, output: Path, allow_partial: bool = False) -> None:
    if output.exists(): raise FileExistsError("Use a new finalization figure directory")
    manifest, tables, bindings, blocks, checks = read_evidence(source, allow_partial)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "svg.fonttype": "none", "pdf.fonttype": 42})
    output.mkdir(parents=True)
    contract = dict(conclusion="Expose participant-level gains and costs across the three registered candidates.",
        archetype="quantitative grid", backend="python", width_mm=183, dpi=300, formats=["PNG"],
        participants=manifest["subjects"], complete_primary_matrix=manifest["complete_primary_matrix"],
        missing_subjects=manifest["missing_subjects"], exclusions=[],
        display_selection=dict(behavior="NLL and Energy in all three phases; other metrics retained in CSV",
            oral="Complete-report training/later rows in both domains and readings; other strata retained in CSV"),
        uncertainty="±2 paired simulation MCSE; 10th–90th path percentiles; no population confidence intervals",
        caveats="Behavior-prefix fits; later development diagnosis; noisy oral evidence; fixed-parameter simulation uncertainty",
        verification="Source-table bindings, paired differences and all path blocks checked; upstream fitting audits retained by reference",
        image_integrity="Numerical plots only; no image adjustments; original data retained")
    (output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    behavior_figure(manifest, tables, output)
    trajectory_figure(manifest, tables, blocks, output)
    for domain in ("functional", "structural"): oral_figure(manifest, tables, domain, output)
    for name in tables: shutil.copy2(source / (name + ".csv"), output / (name + ".csv"))
    blocks.to_csv(output / "path_blocks.csv", index=False)
    shutil.copy2(__file__, output / "figure_source.py")
    for path, sha in bindings.items():
        if digest(Path(path)) != sha: raise ValueError("Figure input changed during drawing")
    images = ["behavior_contrasts.png", "autonomous_trajectories.png", "oral_functional.png", "oral_structural.png"]
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>模型定稿比较</title>'
        '<style>body{max-width:1100px;margin:2em auto;font:16px/1.6 sans-serif;padding:0 1em}img{width:100%}</style>'
        '<h1>固定候选的行为、学习过程与口述证据</h1><p>' + banner(manifest) + '</p>'
        '<p>参数由原训练段选择数据估计。自主轨迹从第一题生成自己的回答与反馈，所有保存的路径均保留。'
        '路径分位带描述生成分布；评分误差线描述固定参数下的模拟波动。被试间差异另见等权汇总表。</p>'
        '<p>口述作为带噪声的外部证据。两种长短读法按原标准分别呈现，未确定质量与报告覆盖保留。'
        '图中口述集合为完整编码报告；目录可支持子集、其他时间窗口及分数误差均保存在源表。</p>'
        '<p>来源表中标记为verified_zero_rate_reuse的比较格复用已通过零学习率等价核验的评价轨迹，'
        '对应的新拟合计算量另行保留。图中的相同曲线反映这些已核验的相同预测。</p>'
        + ''.join(f'<p><a href="{n}"><img src="{n}" alt="{n}"></a></p>' for n in images)
        + '<p><a href="equal_subject_contrasts.csv">被试等权差值及被试间变异</a> · '
        '<a href="cells.csv">候选与来源</a> · <a href="figure_contract.json">图表定义</a></p>', encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(dict(status="complete", source=str(source.resolve()),
        subjects=manifest["subjects"], complete_primary_matrix=manifest["complete_primary_matrix"],
        missing_subjects=manifest["missing_subjects"], numeric_checks=checks, path_blocks=len(blocks),
        candidate_path_count=len(blocks[["subject", "candidate", "seed"]].drop_duplicates()),
        inputs_sha256=bindings, source_sha256=digest(Path(__file__)), visual_review="pending", images=images,
        artifacts_sha256={p.name: digest(p) for p in output.iterdir()},
        versions=dict(python=sys.version, matplotlib=matplotlib.__version__, numpy=np.__version__, pandas=pd.__version__)), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    draw(args.source, args.output, args.allow_partial)
