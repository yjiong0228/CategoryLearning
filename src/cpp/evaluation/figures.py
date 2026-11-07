"""Plain PNG-only evaluation plots; matplotlib and evaluation imports are lazy.

State panels aggregate uncommitted, example, generalization and catalogue-rule
contents by representation kind. Functions return the saved Path and never
overwrite an existing file. No global matplotlib style/backend is changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from textwrap import fill
from typing import Any

import numpy as np


def _destination(path: str | Path) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("Figure exports must be PNG")
    if output.exists():
        raise FileExistsError(output)
    return output


def _save(fig: Any, output: Path) -> Path:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150, facecolor="white")
    finally:
        fig.clear()
    return output


def _figure(
    panels: int, *, height_ratios: list[float] | None = None
) -> tuple[Any, np.ndarray]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    height = max(3, 1.8 * sum(height_ratios or [1.0] * panels))
    fig = Figure(figsize=(11, height), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(
        panels, 1, squeeze=False, gridspec_kw={"height_ratios": height_ratios}
    )
    return fig, axes[:, 0]


def _legend(ax: Any) -> None:
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def _band(
    ax: Any, x: np.ndarray, bands: Mapping[str, Any], label: str, color: str = "#267c9b"
) -> None:
    for lower, upper, alpha, text in (
        ("q00", "q100", 0.09, "100% empirical"),
        ("q05", "q95", 0.17, "90%"),
        ("q25", "q75", 0.29, "50%"),
    ):
        ax.fill_between(
            x,
            bands[lower],
            bands[upper],
            color=color,
            alpha=alpha,
            linewidth=0,
            label=f"{label} {text}",
        )
    ax.plot(x, bands["mean"], color=color, linewidth=1.5, label=f"{label} mean")


def plot_behavior(
    data: Any,
    autonomous_bands: Mapping[str, Any],
    output_path: str | Path,
    *,
    window: int = 32,
    conditional_bands: Mapping[str, Any] | None = None,
    feedback_bands: Mapping[str, Any] | None = None,
) -> Path:
    """Observed true-response correctness and received credit in separate panels.

    Autonomous bands are empirical path quantiles. Optional conditional bands
    are independent outcome draws given observed history, not PF-repeat error.
    Omit a binary feedback panel only when its observed curve and every band
    duplicate correctness. The saved numerical metrics remain separate.
    """
    output = _destination(output_path)
    if __package__:
        from .plot_results import behavior_accuracy, rolling_accuracy
    else:
        from plot_results import behavior_accuracy, rolling_accuracy

    metrics = behavior_accuracy(data)
    credit_redundant = (
        data.condition in (1, 2)
        and feedback_bands is not None
        and np.array_equal(metrics["choice_mask"], metrics["feedback_mask"])
        and np.array_equal(metrics["choice_correct"][metrics["choice_mask"]],
                           metrics["feedback_correct"][metrics["feedback_mask"]])
        and all(np.array_equal(autonomous_bands[key], feedback_bands[key], equal_nan=True)
                for key in ("q00", "q05", "q25", "q50", "q75", "q95", "q100", "mean"))
    )
    x = np.arange(1, len(data.choices) + 1)
    panel_specs = [
        (
            "choice_correct",
            "choice_mask",
            "True-response accuracy",
            autonomous_bands,
            "Autonomous",
            "#267c9b",
        )
    ]
    if conditional_bands is not None:
        panel_specs.insert(
            0,
            (
                "choice_correct",
                "choice_mask",
                "True-response accuracy",
                conditional_bands,
                "Conditional outcomes",
                "#bd5c35",
            ),
        )
    if not credit_redundant:
        panel_specs.append(
            (
                "feedback_correct",
                "feedback_mask",
                "Received feedback credit",
                feedback_bands,
                "Autonomous credit",
                "#267c9b",
            )
        )
    fig, axes = _figure(len(panel_specs), height_ratios=[1.5] * len(panel_specs))
    for ax, (field, mask, ylabel, bands, label, color) in zip(axes, panel_specs):
        curve = rolling_accuracy(
            metrics[field], window, valid=metrics[mask], session_ids=data.session_ids
        )
        ax.plot(x, curve, color="0.15", linewidth=1.5, label="Observed")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(ylabel, fontsize=9)
        if bands is not None:
            _band(ax, x, bands, label, color)
        ax.set_title(
            "Feedback can differ from true correctness"
            if field == "feedback_correct"
            else (
                "Conditional outcomes given observed history"
                if label == "Conditional outcomes"
                else "Independent autonomous paths"
            ),
            fontsize=9,
            loc="left",
        )
        ax.set_xlabel("Trial in retained order", fontsize=9)
        ax.set_xlim(0.5, max(1.5, len(x) + 0.5))
        _legend(ax)
        _style(ax)
    equivalence = "\nFeedback credit = correctness; duplicate binary panel omitted" if credit_redundant else ""
    fig.suptitle(
        f"Subject {data.subject_id} | Condition {data.condition} | Rolling window {window}\nWindows reset at session boundaries; 100% bands are finite-ensemble ranges{equivalence}",
        fontsize=10,
    )
    return _save(fig, output)


def plot_states(
    states: Mapping[str, Any],
    output_path: str | Path,
    *,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    title: str = "Conditional PF state diagnostics (not forecasts)",
) -> Path:
    """Representation-kind heatmaps, search/count traces, kind mass and pairing.

    Matrix beta values are plotted directly as means per active item.
    Pairing[T,4,3] is filtered joint kind and
    pairing mass: sum over kinds without prior reweighting. Full arrays remain
    in NPZ. Undefined inactive entries contribute zero, not fabricated beliefs.
    """
    output = _destination(output_path)
    matrices = [
        (name, np.asarray(states[name], float))
        for name in (
            "prior",
            "filtered_prior",
            "posterior",
            "beta",
            "active_probability",
            "choice_weights",
            "dominant_probability",
        )
        if name in states
    ]
    traces = [
        (name, np.asarray(states[name], float))
        for name in (
            "failure_pressure",
            "search_probability",
            "global_probability",
            "active_count",
            "retained_count",
            "newcomer_count",
            "dropped_count",
            "newcomer_mass",
            "fallback",
        )
        if name in states
    ]
    pairing = np.asarray(states["pairing"], float) if "pairing" in states else None
    if not matrices and not traces and pairing is None:
        raise ValueError("No supported trialwise states to plot")
    first = (matrices + traces)[0][1] if matrices or traces else pairing
    n = first.shape[0] if first.ndim else 0
    if n == 0 or any(array.shape != (n, 4) for _, array in matrices):
        raise ValueError("State matrices must have shape [T,4] and nonempty trials")
    if any(array.shape != (n,) for _, array in traces):
        raise ValueError("State traces must have shape [T]")
    if pairing is not None and (
        pairing.shape != (n, 4, 3)
    ):
        raise ValueError("pairing must have shape [T,4,3]")
    prior = np.asarray(states["prior"], float) if "prior" in states else None
    family_names = None
    if metadata is not None and prior is not None:
        if len(metadata) != prior.shape[1]:
            raise ValueError("State metadata must align with representation kinds")
        family_names = [
            str(m["kind"]) for m in metadata
        ]
    probability_traces = [
        (name, array)
        for name, array in traces
        if name in {"search_probability", "global_probability"}
    ]
    count_traces = [(name, array) for name, array in traces if name.endswith("count")]
    pressure_traces = [
        (name, array) for name, array in traces if name == "failure_pressure"
    ]
    groups = [
        ("Search probability", probability_traces),
        ("Active-set counts", count_traces),
        ("Failure pressure", pressure_traces),
        (
            "Recruitment diagnostics",
            [
                (name, array)
                for name, array in traces
                if name in {"newcomer_mass", "fallback"}
            ],
        ),
    ]
    groups = [(title, rows) for title, rows in groups if rows]
    proposal = (
        np.asarray(states["proposal_source"], float)
        if "proposal_source" in states
        else None
    )
    if proposal is not None and proposal.shape != (n, 4):
        raise ValueError("proposal_source must have shape [T,4]")
    panels = (
        len(matrices)
        + len(groups)
        + int(family_names is not None)
        + int(pairing is not None)
        + int(proposal is not None)
    )
    ratios = [1.0] * (len(matrices) + len(groups))
    if family_names is not None:
        ratios.append(max(1.0, 0.23 * len(set(family_names))))
    ratios += [1.0] * (int(pairing is not None) + int(proposal is not None))
    fig, axes = _figure(panels, height_ratios=ratios)
    axes = iter(axes)
    x = np.arange(1, n + 1)
    for name, array in matrices:
        ax = next(axes)
        probability = name not in {"beta", "active_probability"}
        plot = ax.imshow(
            np.ma.masked_invalid(array.T),
            origin="lower",
            aspect="auto",
            extent=(0.5, n + 0.5, -0.5, array.shape[1] - 0.5),
            cmap="viridis",
            vmin=0 if probability else None,
            vmax=1 if probability else None,
            interpolation="nearest",
        )
        ax.set_ylabel("Representation kind", fontsize=8)
        labels = [str(m.get("name", m["kind"])) for m in metadata] if metadata is not None else ["uncommitted", "example", "generalization", "catalogue_rule"]
        ax.set_yticks(range(4), [label.replace("_", " ") for label in labels])
        heading = {"active_probability": "Expected active content count", "beta": "Mean beta per active content"}.get(name, name.replace("_", " ").capitalize())
        ax.set_title(heading, fontsize=9, loc="left")
        fig.colorbar(plot, ax=ax, pad=0.01, fraction=0.025).ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Trial in retained order", fontsize=8)
    for heading, rows in groups:
        ax = next(axes)
        for name, array in rows:
            ax.plot(x, array, linewidth=1.2, label=name.replace("_", " "))
        ax.set_title(heading, fontsize=9, loc="left")
        _legend(ax)
        ax.set_xlabel("Trial in retained order", fontsize=8)
        _style(ax)
    if family_names is not None:
        ax = next(axes)
        names = sorted(set(family_names))
        for name in names:
            selected = np.array([value == name for value in family_names])
            ax.plot(
                x,
                prior[:, selected].sum(axis=1),
                linewidth=1.2,
                label=fill(name.replace("_", " "), width=28),
            )
        ax.set_title("Prior mass by representation kind", fontsize=9, loc="left")
        ax.set_ylim(-0.02, 1.02)
        _legend(ax)
        ax.set_xlabel("Trial in retained order", fontsize=8)
        _style(ax)
    if pairing is not None:
        ax = next(axes)
        finite = np.isfinite(pairing)
        curve = np.where(finite, pairing, 0).sum(axis=1)
        curve[~finite.any(axis=(1, 2))] = np.nan
        heading = "Pairing probability (filtered joint representation-kind marginal)"
        for index in range(3):
            ax.plot(x, curve[:, index], linewidth=1.2, label=f"Pairing {index + 1}")
        ax.set_title(heading, fontsize=9, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Trial in retained order", fontsize=8)
        _legend(ax)
        _style(ax)
    if proposal is not None:
        ax = next(axes)
        for index, name in enumerate(
            ("No proposal", "Encode example", "Drop feature", "Catalogue rule")
        ):
            ax.plot(x, proposal[:, index], linewidth=1.2, label=name)
        ax.set_title("Proposal source probability", fontsize=9, loc="left")
        ax.set_xlabel("Trial in retained order", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        _legend(ax)
        _style(ax)
    fig.suptitle(title, fontsize=11)
    return _save(fig, output)
