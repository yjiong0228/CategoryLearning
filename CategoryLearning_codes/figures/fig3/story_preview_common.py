"""Read-only source loading and shared rendering for six story previews.

Contract: compare candidate visual arguments, not establish new model results.
Python; schematic-led composites and quantitative grids; 183 x 205 mm; PNG only.
Observed summaries, saved full-sequence fits, historical autonomous simulations,
and analytical/design schematics have distinct panel badges and provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

TASKS = ((1, "Task 1"), (3, "Task 2"), (2, "Task 3"))
COLORS = {1: "#4E7995", 3: "#AA8158", 2: "#568B80"}
INK, MUTED, PALE = "#253541", "#64727C", "#F1F4F6"
LOCAL, GLOBAL = "#568B80", "#BC8555"
BLUE = LinearSegmentedColormap.from_list("belief_blue", ["#FAFCFD", "#245F82"])
BLUE.set_bad("#E6EAED")
STORIES = [
    ("representation", "理解如何形成与重组", "规则内容 → 分类功能 → 错误与泛化"),
    ("timing", "改变的时机为何重要", "反馈经历 → 规则修订 → 改变的价值"),
    ("individual", "相同成绩背后的不同困难", "完整学习路径 → 个体差异 → 困难环节"),
    ("task", "任务怎样塑造学习路径", "任务信息 → 学习顺序 → 分支差异"),
    ("prediction", "内部状态能否预测未来", "可测量状态 → 前推评价 → 诊断性新刺激"),
    ("generation", "同一机制能否产生多样学习", "真实轨迹差异 → 自主生成 → 机制对照"),
]


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "legend.frameon": False, "lines.linewidth": 1.2,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#A0ABB3",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
        "savefig.dpi": 300,
    })


class PreviewData:
    """Load frozen figure sources; preserve subject and trial identities."""
    def __init__(self) -> None:
        self.inputs: dict[str, str] = {}
        base = Path("CategoryLearning_codes/figures/outputs")
        self.fig1 = base / "fig1/fig1_v10"
        self.fig2 = base / "fig2/fig2_complete_v15"
        self.fig3 = base / "fig3/fig3_v4/source_data"
        self.trials = self.read(self.fig1 / "trial_source.csv")
        self.summary = self.read(self.fig1 / "subject_summary.csv")
        self.phase = self.read(self.fig1 / "oral_phase_summary.csv")
        self.rt = self.read(self.fig1 / "nonoral_trial_source.csv")
        self.feedback_rt = self.read(self.fig1 / "nonoral_feedback_rt.csv")
        self.representatives = self.read(self.fig1 / "representatives.csv")
        assert not self.trials.duplicated(["condition", "iSub", "trial"]).any()
        assert len(self.summary) == self.trials.iSub.nunique() == 96
        assert len(self.trials) == 62720
        np.testing.assert_array_equal(self.trials.correct, self.trials.feedback.eq(1).astype(int))
        assert self.trials.groupby("iSub").trial.apply(lambda x: np.array_equal(x, np.arange(1, len(x) + 1))).all()
        self.cases = {}
        for sid, folder in [(129, "case_sources"), (229, "task3_case_sources")]:
            metrics = self.read(self.fig3 / f"subject_{sid}_trial_metrics.csv")
            mass = self.read(self.fig3 / f"subject_{sid}_rule_mass.csv")
            belief = mass.pivot(index="trial", columns="rule_id", values="online_prior").to_numpy()
            oral = mass.pivot(index="trial", columns="rule_id", values="oral_mass").to_numpy()
            np.testing.assert_allclose(belief.sum(axis=1), 1, atol=1e-9)
            fields = [f"predictive_strategy_{f}" for f in ["exploit", "local_explore", "global_explore"]]
            np.testing.assert_allclose(metrics[fields].sum(axis=1), 1, atol=1e-9)
            assert np.isfinite(belief).all() and (belief >= 0).all()
            case = self.read(self.fig2 / folder / "trial_source.csv")
            target = self.read(self.fig2 / folder / "target_alignment.csv")
            assert len(case) == len(metrics) == len(target) == belief.shape[0]
            self.cases[sid] = dict(metrics=metrics, belief=belief, oral=oral, case=case, target=target,
                excerpt=self.read(self.fig3 / f"subject_{sid}_excerpt.csv"),
                reports=self.read(self.fig3 / f"subject_{sid}_same_choice_reports.csv"))
        auto = Path("results/model_0826/cond1/subject_129/pipeline_20260908_v1/models/PMH/autonomous_trajectories")
        self.auto_summary = self.read(auto / "autonomous_trajectory_summary.csv")
        self.record(auto / "analysis_manifest.json")
        self.auto_manifest = json.loads((auto / "analysis_manifest.json").read_text())
        self.record(auto / "autonomous_trajectory_arrays.npz")
        with np.load(auto / "autonomous_trajectory_arrays.npz") as arrays:
            self.auto = {key: arrays[key].copy() for key in arrays.files}
        assert self.auto["rolling_accuracy"].shape == (500, 240)
        assert np.isfinite(self.auto["rolling_accuracy"]).all()
        self.derived: dict[str, pd.DataFrame] = {}
        self.panels: list[dict] = []

    def record(self, path: Path) -> None:
        self.inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(self, path: Path) -> pd.DataFrame:
        self.record(path)
        return pd.read_csv(path)

    def export(self, output: Path) -> None:
        source = output / "source_data"
        source.mkdir()
        self.summary.to_csv(source / "all_subject_summaries.csv", index=False)
        self.phase.to_csv(source / "oral_phase_summaries.csv", index=False)
        self.auto_summary.to_csv(source / "saved_autonomous_summary.csv", index=False)
        for name, frame in self.derived.items():
            frame.to_csv(source / f"{name}.csv", index=False)
        manifest = {
            "purpose": "six alternative visual arguments; review previews, not completed scientific tests",
            "width_mm": 183, "height_mm": 205, "dpi": 300, "format": "PNG",
            "observed_subjects": 96, "observed_trials": 62720,
            "task_to_condition": {"1": 1, "2": 3, "3": 2},
            "panels": self.panels, "input_sha256": self.inputs,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
            "interpretation": [
                "Named-feature changes are not verified semantic rule-revision events.",
                "Two fitted cases use saved full-sequence parameters and mean pre-choice states over 16 PF runs.",
                "Autonomous figures use all 500 saved S129 PMH rollouts; no new simulation or fitting.",
                "Design and rule-geometry schematics contain no invented empirical effects or p values.",
                "No inferential statistics, confidence intervals, or causal claims are reported.",
            ],
            "exclusions": {
                "behavior": "None; all 62720 trials retained, including ambiguous stimuli.",
                "phase_pairs": "At least 128 trials for disjoint first/last 64; S105 is in behavior views but has no phase pair.",
                "feature_count": "Only reports with recognized named features, as defined by the existing encoder; 59703/62720 trials.",
                "candidate_changes": "Existing same-choice same-session comparison validity and max gap retained; non-comparable reports remain missing.",
                "rolling_curves": "Initial window-incomplete observations are unavailable, not assigned zero.",
                "autonomous_onset": "491 reached, 9 did not; all 500 retained and non-attainment displayed separately.",
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def figure(story: int, number: int, title: str, subtitle: str):
    fig = plt.figure(figsize=(183 / 25.4, 205 / 25.4), facecolor="white")
    fig.text(.075, .963, f"方案 {story + 1}  |  Fig. {number}  {title}", fontsize=10.5, weight="bold")
    fig.text(.075, .936, subtitle, fontsize=6.5, color=MUTED)
    fig.text(.075, .908, STORIES[story][2], fontsize=7, color=COLORS[1])
    fig._story = story
    fig._number = number
    return fig


def panel(fig, data: PreviewData, rect, letter: str, title: str, kind: str, source: str):
    ax = fig.add_axes(rect)
    x, y, w, h = rect
    fig.text(x - .043, y + h + .033, letter, weight="bold", fontsize=10)
    fig.text(x, y + h + .033, title, weight="bold", fontsize=7.8)
    fig.text(x, y + h + .009, kind, fontsize=5.8, color=MUTED)
    data.panels.append(dict(story=fig._story + 1, figure=fig._number, panel=letter,
                            title=title, evidence=kind, source=source))
    return ax


HERO = [.115, .575, .81, .245]
LEFT = [.115, .155, .34, .275]
RIGHT = [.585, .155, .34, .275]


def footer(fig, text: str) -> None:
    lines = textwrap.wrap(text, width=68)
    fig.text(.075, .058, "\n".join(lines), fontsize=6, color=MUTED, linespacing=1.5, va="top")
    fig.text(.075, .016, "研究构思预览  ·  状态标签区分真实数据、已有拟合与设计示意", fontsize=5.5, color="#8D989F")


def save(fig, output: Path, name: str) -> Path:
    path = output / f"{name}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def note(ax, text: str, y=.95, **kwargs) -> None:
    ax.text(.02, y, text, transform=ax.transAxes, fontsize=6, color=MUTED, va="top", **kwargs)


def box(ax, x, y, w, h, text, color=COLORS[1], fontsize=7):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                          edgecolor=color, facecolor=color + "13", linewidth=.8, transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=fontsize, linespacing=1.6, color=INK)


def arrow(ax, start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                color=color, lw=.8, transform=ax.transAxes))


def case_curve(ax, data, sid=129, target=False):
    c = data.cases[sid]
    table = c["target"] if target else c["case"]
    if target:
        ax.plot(table.trial, table.model_target_mass, color=COLORS[1], label="模型目标规则信念")
        valid = c["metrics"].valid_oral_report.astype(bool)
        ax.scatter(table.loc[valid, "trial"], table.loc[valid, "oral_target_mass"], s=4,
                   color=GLOBAL, alpha=.45, label="口述目标质量")
        ax.set_ylabel("目标规则质量")
    else:
        ax.plot(table.trial, table.observed_accuracy_rolling32, color=INK, label="真实正确率")
        ax.plot(table.trial, table.model_correct_probability_rolling32, color=LOCAL, label="模型正确概率")
        ax.set_ylabel("32 试次移动平均")
    ax.set(xlim=(1, len(table)), ylim=(0, 1.03), xlabel="试次")
    ax.legend(loc="upper left", ncol=2, fontsize=5.8)


def paired_phases(ax, data, metric="feature_count", ylabel="报告提及特征数"):
    frame = data.phase[(data.phase.metric == metric) & data.phase.disjoint_periods_eligible].copy()
    rows = []
    for j, (condition, task) in enumerate(TASKS):
        wide = frame[frame.condition == condition].pivot(index="iSub", columns="phase", values="value")
        valid = wide[["first", "last"]].notna().all(axis=1)
        wide = wide.loc[valid]
        for sid, row in wide.iterrows():
            ax.plot([j-.17, j+.17], [row["first"], row["last"]], color=COLORS[condition], alpha=.2, lw=.6)
            rows.append(dict(condition=condition, iSub=sid, first=row["first"], last=row["last"], metric=metric))
        ax.plot([j-.17, j+.17], wide[["first", "last"]].mean(), color=COLORS[condition], lw=2, marker="o", ms=3)
        ax.text(j, -.22, f"{task} · n={len(wide)}", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=6)
    ax.set(xticks=np.ravel([[j-.17, j+.17] for j in range(3)]),
           xticklabels=["首64", "末64"]*3, xlim=(-.5, 2.5), ylabel=ylabel)
    data.derived[f"paired_{metric}"] = pd.DataFrame(rows)


def subject_scatter(ax, data, x, y, xlabel, ylabel, condition=None):
    groups = TASKS if condition is None else [p for p in TASKS if p[0] == condition]
    for cond, task in groups:
        f = data.summary[data.summary.condition == cond]
        valid = f[[x, y]].notna().all(axis=1)
        ax.scatter(f.loc[valid, x], f.loc[valid, y], s=12, c=COLORS[cond], alpha=.65,
                   edgecolors="white", linewidths=.3, label=f"{task} (n={int(valid.sum())})")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best", fontsize=5.5)


def heatmap_subjects(ax, data, condition=1):
    frame = data.trials[data.trials.condition == condition]
    order = data.summary[data.summary.condition == condition].sort_values(["n_trials", "iSub"]).iSub.tolist()
    matrix = frame.pivot(index="iSub", columns="trial", values="rolling_accuracy").reindex(order).to_numpy()
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=BLUE, vmin=0, vmax=1,
              extent=(.5, matrix.shape[1]+.5, len(order)+.5, .5))
    ax.set(xlabel="真实试次（灰色为无完整窗口或记录结束）", ylabel="被试（按记录长度排列）",
           yticks=[1, len(order)], xticks=[32, matrix.shape[1]//2, matrix.shape[1]])
    return matrix


def original_examples(ax, data, condition=1):
    f = data.trials[data.trials.condition == condition]
    for sid, group in f.groupby("iSub"):
        ax.plot(group.trial, group.rolling_accuracy, color="#ABB4BA", alpha=.22, lw=.5)
    reps = data.representatives[data.representatives.condition == condition]
    for color, sid in zip([COLORS[condition], GLOBAL, INK], reps.iSub):
        g=f[f.iSub == sid]
        ax.plot(g.trial, g.rolling_accuracy, color=color, lw=1.3, label=f"S{sid}")
    ax.set(xlabel="真实试次", ylabel="32 试次正确率", ylim=(0, 1.05))
    ax.legend(ncol=3, fontsize=6)


def rule_heat(ax, data, sid=129):
    b=data.cases[sid]["belief"]
    ax.imshow(b.T, aspect="auto", origin="lower", cmap=BLUE, vmin=0, vmax=1,
              extent=(.5, len(b)+.5, -.5, b.shape[1]-.5), interpolation="nearest")
    ax.set(xlabel="试次", ylabel="全部规则编号", yticks=[0, b.shape[1]-1])


def report_cards(ax, data):
    ax.axis("off")
    for j, (sid, oldt, newt, old, new) in enumerate([
        (129, 136, 138, "脖子长", "尾巴长"),
        (229, 988, 992, "腿长，脖子短", "腿长，尾巴短")]):
        y = .64-j*.46
        reports=data.cases[sid]["reports"].set_index("sequence_trial")
        assert old in reports.loc[oldt, "text"] and new in reports.loc[newt, "text"]
        ax.text(.02, y+.2, f"S{sid} · 同一选择类别", fontsize=7, weight="bold", transform=ax.transAxes)
        box(ax, .02, y-.07, .35, .21, f"t={oldt}   {old}", COLORS[1], 7)
        box(ax, .6, y-.07, .37, .21, f"t={newt}   {new}", GLOBAL, 7)
        arrow(ax, (.4,y+.03), (.57,y+.03))
        ax.text(.485, y-.075, "两次报告之间", ha="center", fontsize=5.8, color=MUTED, transform=ax.transAxes)


def excerpts(ax, data):
    ax.axis("off")
    for j, (sid, old, new) in enumerate([(129,136,138),(229,988,992)]):
        inner=ax.inset_axes([.02+j*.53,.2,.43,.67])
        f=data.cases[sid]["excerpt"]
        inner.axvspan(old,new,color="#EFE3D1",alpha=.8)
        inner.plot(f.sequence_trial,f.local_search_probability,color=LOCAL,label="局部搜索")
        inner.plot(f.sequence_trial,f.global_search_probability,color=GLOBAL,label="全局搜索")
        inner.scatter(f.sequence_trial,np.full(len(f),-.035),c=np.where(f.feedback.eq(1),INK,"white"),
                      edgecolor=INK,lw=.5,s=10,clip_on=False)
        inner.set(ylim=(-.09,.55),xlabel="试次", title=f"S{sid} · 阴影为报告间隔")
        if j==0:
            inner.set_ylabel("搜索概率")
            inner.legend(fontsize=5.5,loc="upper left")
        inner.tick_params(labelsize=5.8)
    ax.text(.02,-.045,"底部实心点：正确反馈；空心点：错误反馈。反馈用于下一试次更新。",transform=ax.transAxes,fontsize=5.8,color=MUTED)


def geometry(ax, mode="boundaries"):
    """Analytical rules on a fixed plane; no fitted or simulated observations."""
    ax.axis("off")
    v=np.linspace(0,1,80)
    xx,yy=np.meshgrid(v,v)
    cmap=ListedColormap(["#D9E6EE", "#EBCDAE"])
    if mode == "boundaries":
        for j, (m,title) in enumerate([(xx>.5,"规则 A：看 F1"),(yy>.5,"规则 B：看 F2")]):
            a=ax.inset_axes([.06+j*.51,.25,.39,.59])
            a.imshow(m,origin="lower",extent=(0,1,0,1),cmap=cmap,interpolation="nearest")
            a.set(xlabel="F1",ylabel="F2",xticks=[0,1],yticks=[0,1],title=title)
            a.tick_params(labelsize=5.5)
            a.title.set_fontsize(6.5)
        ax.text(.02,.04,"蓝 / 橙：两类预测；使用同一刺激集合。",fontsize=6,transform=ax.transAxes)
    else:
        a=ax.inset_axes([.12,.26,.73,.65])
        disagree=(xx>.5)!=(yy>.5)
        a.imshow(disagree,origin="lower",extent=(0,1,0,1),cmap=ListedColormap(["#F1F4F6","#D5B58E"]),interpolation="nearest")
        a.axvline(.5,color=MUTED,lw=.8);a.axhline(.5,color=MUTED,lw=.8)
        a.scatter([.2,.8],[.8,.2],marker="*",s=70,color=INK)
        a.set(xlabel="F1",ylabel="F2",xticks=[0,.5,1],yticks=[0,.5,1])
        ax.text(.02,.06,"着色：A/B 预测不同\n星号：候选探针，非真实测量",fontsize=6,transform=ax.transAxes)


def phase_confusions(ax, data):
    ax.axis("off")
    records=[]
    for j,(cond,task) in enumerate(TASKS):
        k=2 if cond==1 else 4
        subjects=data.summary.loc[(data.summary.condition==cond)&(data.summary.n_trials>=128),"iSub"]
        for r,phase in enumerate(["first","last"]):
            matrices=[]
            for sid in subjects:
                g=data.trials[data.trials.iSub==sid]
                g=g.iloc[:64] if phase=="first" else g.iloc[-64:]
                counts=pd.crosstab(g.category,g.choice).reindex(index=range(1,k+1),columns=range(1,k+1),fill_value=0)
                sums=counts.sum(axis=1).replace(0,np.nan)
                matrices.append(counts.div(sums,axis=0).to_numpy())
            mean=np.nanmean(matrices,axis=0)
            inner=ax.inset_axes([j*.34+.02,.57-r*.55,.25,.39])
            inner.imshow(mean,cmap=BLUE,vmin=0,vmax=1)
            inner.set(xticks=range(k),yticks=range(k),xticklabels=range(1,k+1),yticklabels=range(1,k+1))
            inner.tick_params(labelsize=5,length=1)
            inner.set_title(f"{task} · {'首' if r==0 else '末'}64",fontsize=6)
            if j==0:inner.set_ylabel("真实类别",fontsize=5.5)
            if r==1:inner.set_xlabel("选择类别",fontsize=5.5)
            for truth in range(k):
                for choice in range(k):
                    valid_n=int(np.isfinite(np.asarray(matrices)[:,truth,choice]).sum())
                    records.append(dict(condition=cond,phase=phase,category=truth+1,choice=choice+1,mean_probability=mean[truth,choice],n_subjects=valid_n,total_eligible_subjects=len(subjects)))
    data.derived["phase_confusions"]=pd.DataFrame(records)


def candidate_feedback(ax, data):
    keys=["condition","iSub","iSession","iBlock","iTrial","trial"]
    joined=data.trials.merge(data.rt[keys+["previous_feedback"]],on=keys,validate="one_to_one")
    rows=joined[joined.report_set_change.notna() & joined.previous_feedback.notna()]
    grouped=rows.groupby(["condition","iSub","previous_feedback"]).report_set_change.agg(["mean","count"]).reset_index()
    data.derived["candidate_feature_changes_by_previous_feedback"]=grouped
    for j,(cond,task) in enumerate(TASKS):
        f=grouped[grouped.condition==cond]
        summary=f.groupby("previous_feedback")["mean"].mean()
        ax.plot(summary.index,summary.values,"o-",color=COLORS[cond],ms=3,label=task)
    ax.set(xlabel="上一试次反馈",ylabel="特征集合变化比例",xticks=[0,.5,1],ylim=(0,1))
    ax.legend(fontsize=5.5)


def rt_feedback(ax,data):
    for cond,task in TASKS:
        f=data.feedback_rt[data.feedback_rt.condition==cond]
        med=f.groupby("previous_feedback").relative_rt.median()
        ax.plot(med.index,med.values,"o-",color=COLORS[cond],ms=3,label=task)
    ax.axhline(1,ls=":",color=MUTED,lw=.7)
    ax.set(xlabel="上一试次反馈",ylabel="相对反应时（被试中位数）",xticks=[0,.5,1])
    ax.legend(fontsize=5.5)


def learner_paths(ax):
    ax.axis("off")
    colors=["#BFCAD2",GLOBAL,LOCAL]
    rows=[("较晚发现",[(0,.63,0),(.63,.75,1),(.75,1,2)]),
          ("发现后反复",[(0,.16,0),(.16,.32,2),(.32,.56,1),(.56,.7,2),(.7,.82,1),(.82,1,2)]),
          ("逐步接近",[(0,.24,0),(.24,.60,1),(.60,1,2)])]
    for j,(label,segs) in enumerate(rows):
        y=.77-j*.28
        ax.text(.01,y+.03,label,fontsize=6.5,transform=ax.transAxes)
        for start,end,state in segs:
            ax.add_patch(plt.Rectangle((.27+.68*start,y),.68*(end-start)-.008,.12,color=colors[state],transform=ax.transAxes))
    ax.text(.27,.04,"早                         学习过程                         晚",fontsize=5.5,transform=ax.transAxes)
    ax.text(.01,-.09,"灰：未明确 · 橙：部分 / 错误 · 绿：有效",fontsize=5.5,color=MUTED,transform=ax.transAxes)


def design_split(ax):
    ax.axis("off")
    box(ax,.04,.57,.48,.22,"前段选择：拟合参数",COLORS[1],7)
    box(ax,.61,.57,.35,.22,"后段选择：评价",LOCAL,7)
    arrow(ax,(.54,.68),(.59,.68))
    ax.plot([.565,.565],[.4,.91],ls="--",color=MUTED,lw=.8,transform=ax.transAxes)
    ax.text(.565,.93,"时间切分",ha="center",fontsize=6,transform=ax.transAxes)
    box(ax,.04,.13,.40,.20,"行为基线\n近期正确率、反馈",COLORS[1],6)
    box(ax,.56,.13,.40,.20,"相同基线 + 过程状态\n规则信念、搜索",LOCAL,6)
    ax.text(.5,.43,"用同一后段、同一评分比较",ha="center",fontsize=6.5,transform=ax.transAxes)


def mechanism_design(ax):
    ax.axis("off")
    for j,(title,color) in enumerate([("完整模型\n搜索随经历调整",LOCAL),("对照模型\n匹配平均搜索率",GLOBAL)]):
        box(ax,.08,.69-j*.45,.84,.22,title,color,6.5)
    arrow(ax,(.5,.64),(.5,.51))
    ax.text(.5,.535,"同样刺激；比较完整轨迹分布",ha="center",fontsize=5.6,transform=ax.transAxes)
    ax.text(.5,.025,"比较设计；效果尚待检验",ha="center",fontsize=6,color=MUTED,transform=ax.transAxes)


def autonomous_curves(ax,data):
    x=data.auto["rolling_trial"]
    curves=data.auto["rolling_accuracy"]
    lo,q1,med,q3,hi=np.quantile(curves,[.05,.25,.5,.75,.95],axis=0)
    ax.fill_between(x,lo,hi,color=LOCAL,alpha=.13,label="逐试次 5–95% 分位")
    ax.fill_between(x,q1,q3,color=LOCAL,alpha=.23)
    ax.plot(x,med,color=LOCAL,label="500 条模拟的中位数")
    ax.plot(x,data.auto["observed_rolling_accuracy"],color=INK,label="真实 S129")
    ax.set(xlabel="试次",ylabel="16 试次正确率",ylim=(0,1.02),xlim=(17,256))
    ax.legend(fontsize=5.8,ncol=2,loc="lower right")
    data.derived["autonomous_pointwise_quantiles"]=pd.DataFrame(dict(trial=x,q05=lo,q25=q1,median=med,q75=q3,q95=hi,observed=data.auto["observed_rolling_accuracy"]))


def autonomous_onsets(ax,data):
    f=data.auto_summary.sustained_mastery_onset
    valid=f.notna()
    ax.hist(f.loc[valid],bins=np.arange(16,273,16),color=LOCAL,alpha=.8,edgecolor="white",lw=.4)
    observed=data.auto_manifest["mastery"]["observed_onset"]
    ax.axvline(observed,color=INK,ls="--",lw=1.3,label=f"S129: {int(observed)}")
    ax.set(xlabel="首次满足持续标准的试次",ylabel="模拟次数")
    note(ax,f"已达标 {int(valid.sum())}/500\n未达标 {int((~valid).sum())}/500",.95)
    ax.legend(fontsize=5.8,loc="upper right")


def task_feature_heat(ax,data):
    rows=[]
    for cond,task in TASKS:
        summaries=[]
        eligible=data.summary[(data.summary.condition==cond)&(data.summary.n_trials>=128)].iSub
        for sid in eligible:
            f=data.trials[data.trials.iSub==sid]
            for phase,g in [("first",f.iloc[:64]),("last",f.iloc[-64:])]:
                g=g[g.report_present]
                for feature in range(1,5):
                    rows.append(dict(condition=cond,iSub=sid,phase=phase,feature=feature,mention_rate=g[f"feature{feature}_use"].mean(),reports=len(g)))
    source=pd.DataFrame(rows)
    data.derived["task_feature_phase_rates"]=source
    matrix=source.groupby(["condition","phase","feature"]).mention_rate.mean()
    array=np.array([[matrix.loc[(cond,phase,f)] for f in range(1,5)] for cond,_ in TASKS for phase in ["first","last"]])
    im=ax.imshow(array,cmap=BLUE,vmin=0,vmax=1,aspect="auto")
    for i in range(6):
        for j in range(4):
            ax.text(j,i,f"{array[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if array[i,j]>.6 else INK)
    ax.set(xticks=range(4),xticklabels=["F1","F2","F3","F4"],yticks=range(6),
           yticklabels=[f"{task} · {p}64" for _,task in TASKS for p in ["首","末"]],xlabel="按任务对齐的特征")
    return im
