"""Render the completed S107 fixed-state choice-reason diagnostic.

Contract: reported features can describe the perceived stimulus accurately while
having little effect on the model's choice. A joint report contrast can inherit
its effect from one feature. Four-panel quantitative grid, Python/matplotlib,
183 x 166 mm, PNG only at 300 dpi for research review. This is a new composition;
existing figure code supplies typography conventions only.

Use the completed oral review and original fixed-decision sweep directories:
python -m CategoryLearning_codes.figures.fig3.model0923_fixed_decision_effects \
    --review RESULTS_ORAL_REVIEW --sweep RESULTS_FIXED_SWEEP --output NEW_DIRECTORY

All 35 complete reports in the original later 64 trials enter the figure. The
first 128 trials are outside this prespecified diagnostic window; all 75 coded
reports remain in source tables. Both readings and both policies are retained.
Each policy has one original, preselected 512-particle filter. Points are trials
or predicates within one participant, not independent experimental replicates.
No significance tests, intervals, oral likelihood or population claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FEATURES = ("neck", "head", "leg", "tail")
LABELS = ("脖子", "头", "腿", "尾巴")
POLICIES = ("MIX", "SELECT")
COLORS = {"MIX": "#3578A7", "SELECT": "#CB7932"}
MARKERS = {"MIX": "o", "SELECT": "s"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(review: Path, sweep: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Validate saved identities and preserve each original report in the plot data."""
    inputs = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"Changed input: {path}")
        inputs[str(path.resolve())] = value
        return path

    done = json.loads(bind(review / "completion.json").read_text())
    if done["status"] != "complete" or done["subject"] != 107:
        raise ValueError("Expected the complete S107 oral diagnostic")
    for name, sha in done["artifacts"].items():
        bind(review / name, sha)
    tables = {name: pd.read_csv(review / (name + ".csv")) for name in
              ("feature_trials", "feature_summary", "predicate_contrasts", "joint_contrasts")}
    features = tables["feature_trials"]
    # Feature identity is invariant to the two retained readings; only predicate
    # semantics change. Display each feature once while keeping both source rows.
    keys = ["policy", "weighting", "state_trial", "feature"]
    cols = keys + ["complete_report", "reported_feature", "mean_particle_range"]
    a = features[features.reading == "midpoint"][cols].sort_values(keys).reset_index(drop=True)
    b = features[features.reading == "relative"][cols].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    joint_rows = []
    for policy in POLICIES:
        directory = sweep / policy.lower()
        record = json.loads(bind(directory / "completion.json").read_text())
        if (record["status"] != "complete" or record["trials"] != 192 or
                record["particles"] != 512 or len(record["seeds"]) != 1 or record["short_diagnostic"]):
            raise ValueError("Expected complete original 192-trial, 512-particle states")
        source = bind(directory / "particle_effects.npz", record["outputs_sha256"]["particle_effects.npz"])
        with np.load(source, allow_pickle=False) as saved:
            q = saved["joint_reflection_probability"]
            weights = saved["conditional_weights"]
            masks = saved["reflection_masks"]
            np.testing.assert_array_equal(masks, ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool))
            np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(saved["grid"], np.linspace(0, 1, 11))
            for reading in ("midpoint", "relative"):
                rows = tables["joint_contrasts"]
                rows = rows[(rows.policy == policy) & (rows.reading == reading) &
                            (rows.weighting == "choice_conditioned")].sort_values("state_trial")
                if len(rows) != 75 or (rows.phase == "test").sum() != 35:
                    raise ValueError("Complete report identities or phase boundary changed")
                for row in rows.itertuples(index=False):
                    t, mask = int(row.state_trial), int(row.reflection_mask)
                    joint = float(weights[t] @ (q[t, :, 0] - q[t, :, mask]))
                    np.testing.assert_allclose(joint, row.joint_probability_effect, atol=1e-12, rtol=0)
                    # Reflect only the reported neck feature; no neck report means
                    # this restricted contrast leaves the original stimulus intact.
                    neck = float(weights[t] @ (q[t, :, 0] - q[t, :, mask & 1]))
                    joint_rows.append(dict(policy=policy, reading=reading, state_trial=t,
                        phase=row.phase, reflection_mask=mask, joint_effect=joint,
                        reported_neck_effect=neck, other_feature_residual=joint-neck))
    tables["joint_feature_source"] = pd.DataFrame(joint_rows)
    return tables, inputs


def draw(tables: dict[str, pd.DataFrame], output: Path) -> dict:
    """Display original units; small effects and exact zeros keep their values."""
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244, 6.5354331))  # 183 x 166 mm
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.88, hspace=.65, wspace=.35)
    titles = ["逐项改变：口述特征是否影响选择？", "联合改变：效应来自哪些特征？",
              "中点读法：描述相符与作答作用", "相对读法：描述相符与作答作用"]
    for letter, ax, title in zip("abcd", axes.flat, titles):
        ax.text(-.12, 1.09, letter, transform=ax.transAxes, weight="bold", fontsize=9)
        ax.set_title(title, loc="left", fontsize=8, pad=10)
        ax.axhline(0, lw=.6, color="#BBBBBB", zorder=0)
    f = tables["feature_trials"]
    f = f[(f.reading == "midpoint") & (f.weighting == "choice_conditioned") &
          (f.phase == "test") & f.complete_report & f.reported_feature]
    counts, means = {}, {}
    for policy, offset in (("MIX", -.14), ("SELECT", .14)):
        for i, feature in enumerate(FEATURES):
            rows = f[(f.policy == policy) & (f.feature == feature)].sort_values("state_trial")
            y = rows.mean_particle_range.to_numpy()
            counts[feature] = len(rows)
            means[policy + "_" + feature] = float(y.mean())
            # Ordered horizontal offsets make coincident observations visible;
            # the x-axis remains categorical and carries no continuous variable.
            x = i + offset + np.linspace(-.065, .065, len(rows))
            axes[0, 0].scatter(x, y, s=8, color=COLORS[policy], alpha=.26,
                               marker=MARKERS[policy], linewidths=0)
            axes[0, 0].scatter(i+offset, y.mean(), s=32, color=COLORS[policy],
                               marker=MARKERS[policy], edgecolors="white", linewidths=.5, zorder=5)
    axes[0, 0].set(xticks=np.arange(4), xticklabels=[f"{x}\n(n={counts[f]})" for x, f in zip(LABELS, FEATURES)],
                    ylim=(-.055, 1.05), ylabel="选择概率变化幅度（0–1全范围网格）")
    handles = [Line2D([], [], marker=MARKERS[p], color=COLORS[p], linestyle="none", label=p, markersize=4)
               for p in POLICIES]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=7)
    effect_text = (f"MIX：头 {means['MIX_head']:.1e}；腿 {means['MIX_leg']:.1e}\n"
                   f"SELECT：头 {means['SELECT_head']:.1g}；腿 {means['SELECT_leg']:.1g}；"
                   f"尾巴 {means['SELECT_tail']:.1g}")
    axes[0, 0].text(.98, .61, effect_text,
                    ha="right", va="top", transform=axes[0, 0].transAxes, fontsize=6.5)
    j = tables["joint_feature_source"]
    j = j[(j.reading == "midpoint") & (j.phase == "test")]
    axes[0, 1].plot([-1, 1], [-1, 1], color="#AAAAAA", lw=.8, ls="--")
    for policy in POLICIES:
        rows = j[j.policy == policy]
        axes[0, 1].scatter(rows.reported_neck_effect, rows.joint_effect, marker=MARKERS[policy],
            s=18, facecolors="none" if policy == "SELECT" else COLORS[policy],
            edgecolors=COLORS[policy], linewidths=.6, alpha=.7)
    axes[0, 1].set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
        xlabel="只镜像改变报告中的脖子：概率下降", ylabel="镜像改变报告的全部部位：概率下降")
    axes[0, 1].text(.03, .96, "每种方式35份报告\n虚线：两种改变的效应相同", fontsize=6.5,
                    ha="left", va="top", transform=axes[0, 1].transAxes)
    categories = (("neck", "脖子单项", "#3578A7"),
                  ("with_neck", "含脖子的联合谓词", "#7868A6"),
                  ("other", "其余谓词", "#CB7932"))
    predicate_counts = {}
    for reading, ax in zip(("midpoint", "relative"), axes[1]):
        rows = tables["predicate_contrasts"]
        rows = rows[(rows.reading == reading) & (rows.weighting == "choice_conditioned") & (rows.phase == "test")].copy()
        bits = rows.reflection_mask.to_numpy(int)
        rows["kind"] = np.where(bits == 1, "neck", np.where(bits & 1, "with_neck", "other"))
        predicate_counts[reading] = int(len(rows) // 2)
        for kind, label, color in categories:
            for policy in POLICIES:
                group = rows[(rows.kind == kind) & (rows.policy == policy)]
                ax.scatter(group.perceived_predicate_truth, group.truth_aligned_probability_effect,
                    marker=MARKERS[policy], s=15, facecolors="none" if policy == "SELECT" else color,
                    edgecolors=color, linewidths=.5, alpha=.6)
        ax.set(xlim=(-.03, 1.045), ylim=(-1.04, 1.04),
            xlabel="内部知觉满足口述谓词的权重", ylabel="按谓词真值定向的选择概率差")
        ax.text(.03, .96, f"每种方式{predicate_counts[reading]}项谓词\n圆点MIX；方框SELECT", fontsize=6.5,
                va="top", transform=ax.transAxes)
    axes[1, 1].legend(handles=[Line2D([], [], color=color, marker="o", linestyle="none", label=label, markersize=4)
                              for _, label, color in categories], loc="lower left", fontsize=6.5)
    fig.suptitle("107：口述中的特征相符，仍需逐项核对其选择作用", x=.10, ha="left", y=.97, fontsize=10)
    fig.text(.10, .925, "冻结H与作答状态；后64题中35份完整报告；每种方式一次预选滤波、512粒子", fontsize=7)
    fig.text(.10, .065, "a  小点为逐题效应，大点为题目均值；每题先计算粒子内幅度，再按原选择条件权重平均。\n"
             "b–d  镜像将相应知觉长度x替换为1−x，规则、精度、贡献和粒子权重保持原值。\n"
             "c–d  谓词来自两种冻结读法；一个联合谓词可涉及多个部位。全部点均保留。\n"
             "这是单个被试的模型内部作用诊断；图中差值具有各自定义，口述概率与群体效应尚待检验。", fontsize=6.3)
    filename = "Figure3_s107_fixed_decision_effects.png"
    fig.savefig(output / filename, dpi=300, facecolor="white")
    plt.close(fig)
    return dict(figure=filename, reported_feature_counts=counts, mean_feature_ranges=means,
                predicates_per_reading_policy=predicate_counts,
                maximum_joint_minus_neck_by_policy=j.groupby("policy").other_feature_residual.apply(lambda x: float(x.abs().max())).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tables, inputs = prepare(args.review, args.sweep)
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__))
    args.output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(args.output / (name + ".csv"), index=False)
    details = draw(tables, args.output)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Input changed during plotting: {path}")
    manifest = dict(status="complete", subject=107, backend="python", figure_contract=__doc__,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        inputs_sha256=inputs, figure_details=details,
        exclusions="Figure uses all35 complete reports in original later64;40 earlier complete reports remain in source tables",
        uncertainty="One original filter/policy; points are within-subject reports or predicates; no CI or significance test",
        export_exceptions="Research-review PNG only per repository policy;300dpi;no SVG/PDF/TIFF or editable-text output",
        artifacts={p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(details, ensure_ascii=False))


if __name__ == "__main__":
    main()
