#!/usr/bin/env python3
"""Conditional predictive separability for frozen C0 and C1 generators.

For each subject, independently split each model's autonomous suffix
rollouts into a reference distribution and pseudo-observations. A family is
recovered when its own held-out pseudo-observation receives a lower proper
score under the matching reference distribution than under the competing
family. This is conditional suffix separability, not recovery of a unique
latent rho path or a full closed-loop refit from a simulated experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    METRIC_SPECS,
    block_bounded_rolling_accuracy,
    load_subject_cache,
    robust_scales,
    trajectory_metrics,
    write_json,
)
from src.Bayesian_state.optimization.optimizer_common import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-dir", type=Path, required=True)
    parser.add_argument("--c1-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20261031)
    parser.add_argument("--max-pseudo-per-family", type=int, default=192)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cache_paths(directory: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in directory.glob(
        "cache/subject_*/particles_*/rollouts_*.npz"
    ):
        cache = load_subject_cache(path)
        paths[int(cache["subject_id"])] = path
    if not paths:
        raise ValueError(f"No subject caches found under {directory}")
    return paths


def empirical_crps_batch(
    observations: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observations, dtype=float)
    samples = np.asarray(reference, dtype=float)
    n = samples.shape[0]
    ordered = np.sort(samples, axis=0)
    coefficients = (
        2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    )[:, None]
    pair_penalty = np.sum(
        coefficients * ordered,
        axis=0,
    ) / float(n * n)
    return np.mean(
        np.mean(
            np.abs(
                observed[:, None, :] - samples[None, :, :]
            ),
            axis=1,
        )
        - pair_penalty[None, :],
        axis=1,
    )


def energy_score_batch(
    observations: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observations, dtype=float)
    samples = np.asarray(reference, dtype=float)
    first = np.mean(
        np.linalg.norm(
            observed[:, None, :] - samples[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    pairwise = np.linalg.norm(
        samples[:, None, :] - samples[None, :, :],
        axis=2,
    )
    return first - 0.5 * float(np.mean(pairwise))


def trajectory_matrix(
    feedback: np.ndarray,
    choices: np.ndarray,
    block_ids: np.ndarray,
    window: int,
) -> np.ndarray:
    rows = [
        trajectory_metrics(
            feedback[index],
            choices[index],
            window=int(window),
            block_ids=block_ids,
        )
        for index in range(feedback.shape[0])
    ]
    names = [spec.metric for spec in METRIC_SPECS]
    return np.asarray(
        [[row[name] for name in names] for row in rows],
        dtype=float,
    )


def split_indices(
    n: int,
    *,
    seed: int,
    max_pseudo: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n))
    split = max(20, int(n) // 2)
    split = min(split, int(n) - 20)
    reference = order[:split]
    pseudo = order[split : split + int(max_pseudo)]
    return reference, pseudo


def main() -> None:
    args = parse_args()
    if args.window < 4 or args.max_pseudo_per_family < 20:
        raise ValueError("window and max pseudo count are too small.")
    c0_paths = cache_paths(args.c0_dir)
    c1_paths = cache_paths(args.c1_dir)
    subjects = sorted(set(c0_paths) & set(c1_paths))
    if not subjects:
        raise ValueError("C0 and C1 directories have no common subjects.")

    rows: list[dict[str, Any]] = []
    for subject in subjects:
        c0 = load_subject_cache(c0_paths[subject])
        c1 = load_subject_cache(c1_paths[subject])
        for key in ("test_iSession", "test_iBlock", "test_iTrial"):
            if not np.array_equal(c0[key], c1[key]):
                raise ValueError(
                    f"Subject {subject} has misaligned C0/C1 {key}."
                )
        block_ids = np.asarray(
            [
                f"{int(session)}:{int(block)}"
                for session, block in zip(
                    c0["test_iSession"],
                    c0["test_iBlock"],
                )
            ]
        )
        c0_feedback = np.asarray(c0["feedback"], dtype=float)
        c1_feedback = np.asarray(c1["feedback"], dtype=float)
        c0_choices = np.asarray(c0["choices"], dtype=int)
        c1_choices = np.asarray(c1["choices"], dtype=int)
        c0_curve, _ = block_bounded_rolling_accuracy(
            c0_feedback,
            block_ids,
            int(args.window),
        )
        c1_curve, _ = block_bounded_rolling_accuracy(
            c1_feedback,
            block_ids,
            int(args.window),
        )
        c0_metrics = trajectory_matrix(
            c0_feedback,
            c0_choices,
            block_ids,
            int(args.window),
        )
        c1_metrics = trajectory_matrix(
            c1_feedback,
            c1_choices,
            block_ids,
            int(args.window),
        )
        pooled = np.vstack([c0_metrics, c1_metrics])
        for column in range(pooled.shape[1]):
            values = pooled[:, column]
            finite = np.isfinite(values)
            replacement = (
                float(np.nanmedian(values)) if np.any(finite) else 0.0
            )
            c0_metrics[~np.isfinite(c0_metrics[:, column]), column] = (
                replacement
            )
            c1_metrics[~np.isfinite(c1_metrics[:, column]), column] = (
                replacement
            )
        pooled = np.vstack([c0_metrics, c1_metrics])
        resolutions = np.full(pooled.shape[1], 1e-3, dtype=float)
        median, scale = robust_scales(pooled, resolutions)
        c0_metrics = (c0_metrics - median) / scale
        c1_metrics = (c1_metrics - median) / scale

        c0_ref, c0_pseudo = split_indices(
            len(c0_curve),
            seed=stable_seed(
                {
                    "role": "dynamic_rho_separability_c0",
                    "base_seed": int(args.base_seed),
                    "subject": int(subject),
                }
            ),
            max_pseudo=int(args.max_pseudo_per_family),
        )
        c1_ref, c1_pseudo = split_indices(
            len(c1_curve),
            seed=stable_seed(
                {
                    "role": "dynamic_rho_separability_c1",
                    "base_seed": int(args.base_seed),
                    "subject": int(subject),
                }
            ),
            max_pseudo=int(args.max_pseudo_per_family),
        )
        for true_family, pseudo_indices, curve, metrics in (
            ("C0", c0_pseudo, c0_curve, c0_metrics),
            ("C1", c1_pseudo, c1_curve, c1_metrics),
        ):
            curve_c0 = empirical_crps_batch(
                curve[pseudo_indices],
                c0_curve[c0_ref],
            )
            curve_c1 = empirical_crps_batch(
                curve[pseudo_indices],
                c1_curve[c1_ref],
            )
            energy_c0 = energy_score_batch(
                metrics[pseudo_indices],
                c0_metrics[c0_ref],
            )
            energy_c1 = energy_score_batch(
                metrics[pseudo_indices],
                c1_metrics[c1_ref],
            )
            for local_index, pseudo_index in enumerate(pseudo_indices):
                curve_selected = (
                    "C0"
                    if curve_c0[local_index] <= curve_c1[local_index]
                    else "C1"
                )
                summary_selected = (
                    "C0"
                    if energy_c0[local_index] <= energy_c1[local_index]
                    else "C1"
                )
                rows.append(
                    {
                        "iSub": int(subject),
                        "true_family": true_family,
                        "pseudo_index": int(pseudo_index),
                        "curve_score_C0": float(curve_c0[local_index]),
                        "curve_score_C1": float(curve_c1[local_index]),
                        "curve_selected": curve_selected,
                        "curve_recovered": bool(
                            curve_selected == true_family
                        ),
                        "summary_score_C0": float(
                            energy_c0[local_index]
                        ),
                        "summary_score_C1": float(
                            energy_c1[local_index]
                        ),
                        "summary_selected": summary_selected,
                        "summary_recovered": bool(
                            summary_selected == true_family
                        ),
                    }
                )

    results = pd.DataFrame(rows)
    by_subject = (
        results.groupby(["iSub", "true_family"], as_index=False)
        .agg(
            pseudo_n=("pseudo_index", "size"),
            curve_recovery_rate=("curve_recovered", "mean"),
            summary_recovery_rate=("summary_recovered", "mean"),
        )
        .sort_values(["iSub", "true_family"])
    )
    family = (
        results.groupby("true_family", as_index=False)
        .agg(
            pseudo_n=("pseudo_index", "size"),
            curve_recovery_rate=("curve_recovered", "mean"),
            summary_recovery_rate=("summary_recovered", "mean"),
        )
        .sort_values("true_family")
    )
    curve_balanced = float(family["curve_recovery_rate"].mean())
    summary_balanced = float(family["summary_recovery_rate"].mean())
    subject_curve = (
        by_subject.pivot(
            index="iSub",
            columns="true_family",
            values="curve_recovery_rate",
        )
        .reindex(columns=["C0", "C1"])
        .mean(axis=1)
        .to_numpy(dtype=float)
    )
    subject_summary_score = (
        by_subject.pivot(
            index="iSub",
            columns="true_family",
            values="summary_recovery_rate",
        )
        .reindex(columns=["C0", "C1"])
        .mean(axis=1)
        .to_numpy(dtype=float)
    )
    bootstrap_rng = np.random.default_rng(
        stable_seed(
            {
                "role": "dynamic_rho_separability_subject_bootstrap",
                "base_seed": int(args.base_seed),
                "subjects": subjects,
            }
        )
    )
    bootstrap_indices = bootstrap_rng.integers(
        0,
        len(subjects),
        size=(5000, len(subjects)),
    )
    curve_bootstrap = np.mean(
        subject_curve[bootstrap_indices],
        axis=1,
    )
    summary_bootstrap = np.mean(
        subject_summary_score[bootstrap_indices],
        axis=1,
    )
    summary = {
        "analysis": "conditional_predictive_separability_C0_C1",
        "subjects": subjects,
        "subject_n": int(len(subjects)),
        "curve_balanced_accuracy": curve_balanced,
        "curve_subject_bootstrap_ci95": [
            float(value)
            for value in np.quantile(
                curve_bootstrap,
                [0.025, 0.975],
            )
        ],
        "summary_balanced_accuracy": summary_balanced,
        "summary_subject_bootstrap_ci95": [
            float(value)
            for value in np.quantile(
                summary_bootstrap,
                [0.025, 0.975],
            )
        ],
        "family_results": family.to_dict(orient="records"),
        "interpretation": (
            "This quantifies conditional suffix distribution separability "
            "after the actual prefix. It is not a full closed-loop recovery "
            "test and does not recover the latent trialwise rho path."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "pseudo_scores.csv", index=False)
    by_subject.to_csv(
        args.output_dir / "subject_separability.csv",
        index=False,
    )
    family.to_csv(
        args.output_dir / "family_separability.csv",
        index=False,
    )
    write_json(args.output_dir / "separability_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
