"""Fixed PNG report panels for the Model 0923 subject-evaluation protocol.

The existing behavior and representation panels retain their scientific meaning.
The diagnostics panel adds numerical integration checks and pre-choice search
states. Every subject receives the same filenames; a particle-budget comparison
that has not been performed remains explicitly marked in its panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model0923_postfit import (
    BLUE, GRAY, ORANGE, cohort_figures, save, style, subject_figures,
)


TRIAL_COLUMNS = (
    "valid_trial",
    "predictive_ess_fraction",
    "filtered_ess_fraction",
    "resampled_fraction",
    "predictive_search",
    "predictive_global_search",
    "predictive_failure",
    "predictive_search_event",
    "predictive_admission_event",
)
BUDGET_COLUMNS = (
    "valid_trial",
    "mean_abs_probability_difference",
    "max_abs_probability_difference",
)


def _source_frame(
    trials: pd.DataFrame,
    budget_comparison: pd.DataFrame | None,
) -> pd.DataFrame:
    """Preserve trial order and require the declared diagnostic measurements."""
    missing = sorted(set(TRIAL_COLUMNS) - set(trials.columns))
    if missing:
        raise ValueError(f"Diagnostic trial columns are missing: {missing}")
    source = trials.loc[:, TRIAL_COLUMNS].copy()
    values = source.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Diagnostic trial measurements must be finite")
    trial_ids = values[:, 0]
    if not len(trial_ids) or np.any(trial_ids != np.floor(trial_ids)):
        raise ValueError("Diagnostic valid_trial must contain integer trial identifiers")
    if trial_ids[0] < 1 or np.any(np.diff(trial_ids) <= 0):
        raise ValueError("Diagnostic trials must remain in increasing valid-trial order")
    probability_columns = [name for name in TRIAL_COLUMNS[1:] if name != "predictive_failure"]
    probabilities = source[probability_columns].to_numpy(dtype=float)
    if np.any((probabilities < -1e-12) | (probabilities > 1 + 1e-12)):
        raise ValueError("ESS fractions and diagnostic probabilities must lie in [0, 1]")
    if np.any(source["predictive_failure"] < 0):
        raise ValueError("Predictive failure state must be nonnegative")

    if budget_comparison is None:
        for name in BUDGET_COLUMNS[1:]:
            source[name] = np.nan
        source["budget_comparison_status"] = "not_performed"
        return source

    missing = sorted(set(BUDGET_COLUMNS) - set(budget_comparison.columns))
    if missing:
        raise ValueError(f"Particle-budget comparison columns are missing: {missing}")
    comparison = budget_comparison.loc[:, BUDGET_COLUMNS]
    if not np.array_equal(source["valid_trial"].to_numpy(), comparison["valid_trial"].to_numpy()):
        raise ValueError("Particle-budget comparison must use the same trial order")
    differences = comparison.loc[:, BUDGET_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(differences).all() or np.any((differences < 0) | (differences > 1)):
        raise ValueError("Absolute probability differences must be finite and lie in [0, 1]")
    if np.any(differences[:, 0] > differences[:, 1] + 1e-12):
        raise ValueError("Mean absolute difference cannot exceed the maximum difference")
    for name in BUDGET_COLUMNS[1:]:
        source[name] = comparison[name].to_numpy()
    source["budget_comparison_status"] = "completed"
    return source


def diagnostics_figures(
    subject: int,
    condition: int,
    trials: pd.DataFrame,
    output: Path,
    *,
    budget_comparison: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Render fixed numerical and search diagnostics from full trial source data.

    ESS is the effective sample size of the particle weights, divided by the
    particle count. The failure state is recorded before the current choice.
    Budget differences compare category probabilities averaged over matched
    filter seeds at the base and higher particle budgets.
    """
    source = _source_frame(trials, budget_comparison)
    output = Path(output)
    figure_path = output / "diagnostics.png"
    source_path = output / "diagnostics_plot_source.csv"
    for path in (figure_path, source_path):
        if path.exists():
            raise FileExistsError(f"Diagnostic output already exists: {path}")
    output.mkdir(parents=True, exist_ok=True)

    style()
    trial = source["valid_trial"].to_numpy()
    fig, axes = plt.subplots(
        4, 1, figsize=(7.2047244094, 6.6929133858),
        layout="constrained", sharex=True,
    )
    ess, search, events, budget = axes
    for name, label, color in (
        ("predictive_ess_fraction", "Before choice", BLUE),
        ("filtered_ess_fraction", "After choice", ORANGE),
        ("resampled_fraction", "Resampling fraction", GRAY),
    ):
        ess.plot(trial, source[name], label=label, color=color, lw=.8)
    ess.set(ylabel="Fraction", ylim=(0, 1.03), title="a  Particle weights and resampling")
    ess.legend(ncol=3, fontsize=7, loc="upper right")

    search.plot(trial, source["predictive_search"], color=BLUE, lw=.8, label="Search probability")
    search.plot(
        trial, source["predictive_global_search"], color=ORANGE,
        lw=.8, label="Global fraction given search",
    )
    search.set(ylabel="Probability", ylim=(0, 1.03), title="b  Search policy before choice")
    # The accumulated failure state has its own scale; its upper bound depends
    # on the fitted retention parameter and differs from a probability.
    failure = search.twinx()
    failure.spines["right"].set_visible(True)
    failure.plot(trial, source["predictive_failure"], color=GRAY, lw=.7, ls="--", label="Failure state")
    failure.set(ylabel="Failure state", ylim=(0, None))
    lines = [*search.lines, *failure.lines]
    search.legend(lines, [line.get_label() for line in lines], ncol=3, fontsize=7, loc="upper right")

    for name, label, color in (
        ("predictive_search_event", "Search event", BLUE),
        ("predictive_admission_event", "New-rule admission", ORANGE),
    ):
        events.plot(trial, source[name], label=label, color=color, lw=.8)
    events.set(ylabel="Event probability", ylim=(0, 1.03), title="c  Candidate-update events before choice")
    events.legend(ncol=2, fontsize=7, loc="upper right")

    budget.set(ylabel="Absolute probability difference", title="d  Paired particle-budget check")
    if budget_comparison is None:
        budget.set_ylim(0, 1)
        budget.text(
            .5, .5, "Paired particle-budget check: not performed",
            ha="center", va="center", transform=budget.transAxes, color=GRAY,
        )
    else:
        budget.plot(
            trial, source["mean_abs_probability_difference"],
            color=BLUE, lw=.8, label="Mean across categories",
        )
        budget.plot(
            trial, source["max_abs_probability_difference"],
            color=ORANGE, lw=.8, label="Maximum across categories",
        )
        budget.set_ylim(bottom=0)
        budget.legend(ncol=2, fontsize=7, loc="upper right")
    axes[-1].set(xlabel="Valid trial")
    for ax in axes:
        ax.set_xlim(trial[0] - .5, trial[-1] + .5)
    fig.suptitle(
        f"S{subject} · condition {condition} · numerical and search diagnostics\n"
        "Fixed selected parameters; all valid trials",
        fontsize=9,
    )
    save(fig, figure_path)
    source.to_csv(source_path, index=False)
    return figure_path, source_path


def main() -> None:
    """Run the canonical protocol with all mandatory figure renderers."""
    from src.Bayesian_state.utils.parallel import single_threaded_processes
    from src.Bayesian_state.utils.paths import ROOT_DIR
    from src.Bayesian_state.workflows.analysis.evaluate_model_0923 import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/exp123/specific_models/model_0923_evaluation_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output_dir is None and not args.dry_run:
        parser.error("--smoke requires a new --output-dir")

    def resolve(path: Path | None) -> Path | None:
        return path if path is None or path.is_absolute() else ROOT_DIR / path

    with single_threaded_processes():
        result = run(
            resolve(args.config), resolve(args.output_dir), args.smoke,
            figure_output=resolve(args.figure_output_dir), dry_run=args.dry_run,
            plot_subject=subject_figures, plot_cohort=cohort_figures,
            plot_diagnostics=diagnostics_figures,
        )
    summary = {"status": result["status"]}
    if args.dry_run and 'settings' in result:
        settings = result['settings']
        summary.update(protocol=result['protocol'], subjects=settings['subjects'],
                       source_kind=result['fit_source']['source_kind'],
                       fit_acceptance=result['fit_source']['subject_acceptance'],
                       output_dir=result['output_dir'], figure_directory=result['figure_directory'],
                       particle_count=settings['particle_count'], pf_repeats=settings['pf_repeats'],
                       autonomous_rollouts=settings['autonomous_rollouts'], smoke=args.smoke)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
