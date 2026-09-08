#!/usr/bin/env python3
"""Run the one-factor Model 0826 belief-transport sensitivity.

The primary model uses slot-proportional semantic belief reallocation. The
counterfactual preserves survivor mass and gives newcomers exactly the mass of
the discarded rules. No parameters are refit and the ordered PF seeds are
shared, so the prior-assignment method is the only changed model setting.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.workflows.runs.run_model_0818_boundary_recovery import (  # noqa: E402
    FEATURE_COLUMNS,
    _load_subject_frames,
    _readout_args,
    _subject_engine,
    resolve_filter_seeds,
)
from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs/exp123/specific_models/model_0826_belief_transport_counterfactual.yaml"
)
PRIOR_ASSIGNMENT_PATH = (
    "modules.hypo_transitions_mod.kwargs.prior_assignment.method"
)
VALID_METHODS = {
    "similarity_transport",
    "mass_preserving_similarity_transport",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"cannot traverse {path!r} at {part!r}")
        current = child
    current[parts[-1]] = value


def _flatten(root: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in root.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def build_variant_engine(
    base_engine: Mapping[str, Any],
    prior_assignment_method: str,
) -> dict[str, Any]:
    """Change exactly the declared prior-assignment method."""

    method = str(prior_assignment_method)
    if method not in VALID_METHODS:
        raise ValueError(f"unsupported prior-assignment method: {method!r}")
    engine = deepcopy(dict(base_engine))
    _set_path(engine, PRIOR_ASSIGNMENT_PATH, method)
    base_flat = _flatten(base_engine)
    engine_flat = _flatten(engine)
    changed = {
        path
        for path in set(base_flat) | set(engine_flat)
        if base_flat.get(path) != engine_flat.get(path)
    }
    if changed - {PRIOR_ASSIGNMENT_PATH}:
        raise RuntimeError(
            f"counterfactual changed undeclared paths: {sorted(changed)}"
        )
    return engine


def validate_variants(
    variants: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    ids: set[str] = set()
    methods: set[str] = set()
    for raw in variants:
        variant_id = str(raw["variant_id"])
        method = str(raw["prior_assignment_method"])
        if variant_id in ids:
            raise ValueError(f"duplicate variant_id: {variant_id}")
        if method not in VALID_METHODS:
            raise ValueError(f"unsupported prior-assignment method: {method!r}")
        ids.add(variant_id)
        methods.add(method)
        resolved.append(
            {
                "variant_id": variant_id,
                "prior_assignment_method": method,
                "role": str(raw.get("role", "")),
            }
        )
    if methods != VALID_METHODS or len(resolved) != 2:
        raise ValueError(
            "variant bank must contain exactly the primary and mass-preserving methods"
        )
    return resolved


def _choice_nll(probabilities: np.ndarray, choices: np.ndarray) -> float:
    chosen = probabilities[np.arange(choices.size), choices - 1]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).sum())


def _choice_brier(probabilities: np.ndarray, choices: np.ndarray) -> float:
    target = np.zeros_like(probabilities)
    target[np.arange(choices.size), choices - 1] = 1.0
    return float(np.mean(np.sum(np.square(probabilities - target), axis=1)))


def _mean_js(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("JS inputs must have equal two-dimensional shapes")
    first = first / np.sum(first, axis=1, keepdims=True)
    second = second / np.sum(second, axis=1, keepdims=True)
    midpoint = 0.5 * (first + second)

    def kl(values: np.ndarray) -> np.ndarray:
        contribution = np.zeros_like(values)
        mask = values > 0.0
        contribution[mask] = values[mask] * np.log(
            values[mask] / np.clip(midpoint[mask], 1e-12, None)
        )
        return np.sum(contribution, axis=1)

    return float(np.mean(0.5 * kl(first) + 0.5 * kl(second)))


def run_counterfactual(
    config: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    design = dict(config["design"])
    variants = validate_variants(design["variants"])
    subjects = [int(value) for value in design["subjects"]]
    trials = int(design["trials_per_subject"])
    particle_count = int(design["particle_count"])
    seed_count = int(design["filter_seed_count"])
    if smoke:
        subjects = subjects[:1]
        trials = min(trials, 32)
        particle_count = min(particle_count, 8)
        seed_count = 2
    if seed_count < 2:
        raise ValueError("counterfactual requires at least two shared PF seeds")

    base_path = _repo_path(config["base_simulation_config"])
    base_config = load_yaml(base_path)
    frames, dataset_paths = _load_subject_frames(base_config, base_path, subjects)
    subject_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for subject_id in subjects:
        frame = frames[subject_id].iloc[:trials].copy()
        stimuli = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        choices = frame["choice"].to_numpy(dtype=int)
        feedback = frame["feedback"].to_numpy(dtype=float)
        base_engine = _subject_engine(base_config, base_path, subject_id)
        seed_panel = resolve_filter_seeds(
            dataset_id=f"transport_subject_{subject_id}_{len(frame)}",
            base_seed=int(design["base_seed"]),
            particle_count=particle_count,
            filter_seed_count=seed_count,
            seed_family="model0826_transport_counterfactual_v1",
        )
        by_variant: dict[str, dict[str, np.ndarray | float]] = {}
        for variant in variants:
            engine = build_variant_engine(
                base_engine,
                variant["prior_assignment_method"],
            )
            probabilities: list[np.ndarray] = []
            priors: list[np.ndarray] = []
            seed_nll: list[float] = []
            for filter_seed in seed_panel:
                result = run_state_model_particle_filter(
                    engine_config=engine,
                    subject_id=subject_id,
                    stimulus=stimuli,
                    choices=choices,
                    feedback=feedback,
                    particle_count=particle_count,
                    filter_seed=int(filter_seed),
                    resample_threshold_fraction=float(
                        design["resample_threshold_fraction"]
                    ),
                    processed_data_dir=dataset_paths["processed_dir"],
                    dataset_paths=dataset_paths,
                    **_readout_args(engine),
                )
                probability = np.asarray(result.marginal_probabilities, dtype=float)
                prior = np.asarray(result.marginal_hypothesis_prior, dtype=float)
                probabilities.append(probability)
                priors.append(prior)
                seed_nll.append(_choice_nll(probability, choices))
            mean_probability = np.mean(np.stack(probabilities), axis=0)
            mean_prior = np.mean(np.stack(priors), axis=0)
            row = {
                "subject_id": subject_id,
                "variant_id": variant["variant_id"],
                "prior_assignment_method": variant["prior_assignment_method"],
                "trial_count": int(len(frame)),
                "particle_count": particle_count,
                "filter_seed_count": seed_count,
                "ensemble_choice_nll": _choice_nll(mean_probability, choices),
                "ensemble_choice_brier": _choice_brier(mean_probability, choices),
                "seed_choice_nll_mean": float(np.mean(seed_nll)),
                "seed_choice_nll_sd": float(np.std(seed_nll, ddof=1)),
            }
            subject_rows.append(row)
            by_variant[variant["prior_assignment_method"]] = {
                "probability": mean_probability,
                "prior": mean_prior,
                "nll": float(row["ensemble_choice_nll"]),
                "brier": float(row["ensemble_choice_brier"]),
            }

        primary = by_variant["similarity_transport"]
        counterfactual = by_variant["mass_preserving_similarity_transport"]
        primary_probability = np.asarray(primary["probability"], dtype=float)
        counterfactual_probability = np.asarray(
            counterfactual["probability"], dtype=float
        )
        contrast_rows.append(
            {
                "subject_id": subject_id,
                "trial_count": int(len(frame)),
                "mass_minus_primary_choice_nll": float(counterfactual["nll"])
                - float(primary["nll"]),
                "mass_minus_primary_choice_brier": float(
                    counterfactual["brier"]
                )
                - float(primary["brier"]),
                "choice_probability_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                counterfactual_probability - primary_probability
                            )
                        )
                    )
                ),
                "maximum_choice_probability_difference": float(
                    np.max(
                        np.abs(
                            counterfactual_probability - primary_probability
                        )
                    )
                ),
                "predictive_workspace_belief_js": _mean_js(
                    np.asarray(primary["prior"], dtype=float),
                    np.asarray(counterfactual["prior"], dtype=float),
                ),
            }
        )

    return {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "smoke": bool(smoke),
        "only_changed_path": PRIOR_ASSIGNMENT_PATH,
        "subjects": subjects,
        "trial_count_per_subject": trials,
        "particle_count": particle_count,
        "filter_seed_count": seed_count,
        "variant_summary": subject_rows,
        "contrast_summary": contrast_rows,
        "interpretation_boundary": str(config["interpretation"]["boundary"]),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = run_counterfactual(config, smoke=bool(args.smoke))
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _repo_path(config["output_dir"])
    )
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / (
        "smoke_summary.json" if args.smoke else "counterfactual_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
