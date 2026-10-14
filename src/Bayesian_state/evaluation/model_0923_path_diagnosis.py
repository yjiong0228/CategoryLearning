"""Descriptive path metrics with explicit eligibility and conditioning sets."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def residence_retention(episodes: pd.DataFrame, trials: int,
                        horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)) -> pd.DataFrame:
    """At each horizon, include entries with enough remaining recorded trials.

    A residence lasting k trials supplies k feedback observations. The final
    recorded residence is right-censored, so short remaining records cannot
    count as early failures. These horizon-specific cohorts are descriptive,
    not a Kaplan–Meier estimator or independent samples of people.
    """
    rows = []
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("Residence horizons must be positive")
        eligible = episodes[episodes.start_trial <= trials-horizon+1]
        retained = int((eligible.duration >= horizon).sum())
        rows.append(dict(horizon=horizon, eligible=len(eligible), retained=retained,
                         fraction=retained/len(eligible) if len(eligible) else np.nan))
    return pd.DataFrame(rows)


def trace_tables(directory: Path, reference: int, late_window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize every stored path, retaining batch-specific diagnostic rows."""
    curves, batches, feedback_rows, exits = [], [], [], []
    for file in sorted((directory / "traces").glob("*.npz")):
        with np.load(file, allow_pickle=False) as saved:
            a = {key: saved[key] for key in saved.files}
        kind = file.stem[0]
        mask = a["rule"] == reference
        active = mask.any(axis=-1)
        ever = np.maximum.accumulate(active, axis=0)
        n, count = active.shape
        late = np.arange(n) >= max(1, n-late_window)
        values = {key: np.where(mask, a[key], 0).sum(axis=-1) for key in
                  ("prior", "posterior", "forgotten", "contribution", "rule_correct",
                   "beta_pre", "beta_post", "pairing_pre_true", "pairing_post_true")}
        for trial, path in np.argwhere(active[:-1] & ~active[1:]):
            cause = "capacity" if a["capacity_dropped"][trial+1, path] == reference else "autonomous_exit"
            exits.append(dict(condition=kind, batch=file.stem, path=int(path),
                exit_trial=int(trial)+2, previous_feedback=a["feedback"][trial, path],
                cause=cause, **{key: float(value[trial, path]) for key, value in values.items()}))
        for trial in range(n):
            curves.append(dict(condition=kind, batch=file.stem, trial=trial+1, paths=count,
                               ever=float(ever[trial].mean()), active=float(active[trial].mean()),
                               contribution=float(values["contribution"][trial].mean())))
        late_active = active & late[:, None]
        row = dict(condition=kind, batch=file.stem, paths=count, trials=n,
                   initial_paths=int(active[0].sum()), ever_paths=int(ever[-1].sum()),
                   proposed=int((a["proposed"] == reference).sum()),
                   all_proposals=int((a["proposed"] >= 0).sum()),
                   active_trials=int(active.sum()), late_active_trials=int(late_active.sum()),
                   ever_supported_paths=int((values["posterior"] >= .5).any(axis=0).sum()),
                   ever_contributing_paths=int((values["contribution"] >= .5).any(axis=0).sum()))
        for key, value in values.items():
            row["late_"+key+"_sum"] = float(value[late_active].sum())
        for source, code in (("global", 1), ("structural", 2), ("functional", 3)):
            selected = a["source"] == code
            row[source+"_proposals"] = int(selected.sum())
            row[source+"_target_proposals"] = int(((a["proposed"] == reference) & selected).sum())
        batches.append(row)
        for feedback in (0., .5, 1.):
            selected = active & (a["feedback"] == feedback)
            size = int(selected.sum())
            feedback_rows.append(dict(condition=kind, batch=file.stem, feedback=feedback, path_trials=size,
                support_gain_sum=float((values["posterior"]-values["forgotten"])[selected].sum()),
                net_belief_gain_sum=float((values["posterior"]-values["prior"])[selected].sum()),
                beta_gain_sum=float((values["beta_post"]-values["beta_pre"])[selected].sum()),
                positive_support_gain=int(((values["posterior"]-values["forgotten"] > 1e-12) & selected).sum())))
    return pd.DataFrame(curves), pd.DataFrame(batches), pd.DataFrame(feedback_rows), pd.DataFrame(exits)
