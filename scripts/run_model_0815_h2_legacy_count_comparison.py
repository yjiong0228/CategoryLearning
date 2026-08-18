#!/usr/bin/env python3
"""Compare the exact legacy variable-count H policy with adaptive H under PF."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0815_b0_adaptive_controller_pilot import (  # noqa: E402
    ADAPTIVE_CLASS,
    H1_PROFILE,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _choice_brier,
    _choice_nll,
    _filter_seeds,
    _mean_js,
    _relative,
    _repo_path,
    _sha256,
    validate_minimal_adaptive_engine,
)
from src.Bayesian_state.simulation.config import (  # noqa: E402
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.simulation.runner import StateModelSimulationRunner  # noqa: E402
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_h2_legacy_count_comparison.yaml"
)
LEGACY_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition.fixed_strategy."
    "FixedStrategyHypothesisTransitionModule"
)
SEED_ROLE = "model0815_h2_legacy_count_vs_adaptive"
TRACE_KEYS = (
    "marginal_prior",
    "marginal_active_probability",
    "pre_choice_ess",
    "post_choice_ess",
    "resampled",
    "predictive_transition_rate",
    "predictive_search_range",
    "predictive_swap_probability",
    "swap_event_probability",
    "replacement_count",
    "replacement_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _resolved_design(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    design = deepcopy(dict(config["design"]))
    design["subjects"] = [int(value) for value in design["subjects"]]
    if smoke:
        design.update(
            {
                "subjects": [design["subjects"][0]],
                "trials_per_subject": 24,
                "train_trials": 12,
                "particle_count": 8,
                "seed_count": 2,
                "n_jobs": 1,
            }
        )
    if not 1 < int(design["train_trials"]) < int(design["trials_per_subject"]):
        raise ValueError("train_trials must divide the trial sequence")
    if int(design["seed_count"]) < 2:
        raise ValueError("the paired PF screen requires at least two seeds")
    return design


def _without_h(engine: Mapping[str, Any]) -> dict[str, Any]:
    common = deepcopy(dict(engine))
    common.pop("provenance", None)
    common["modules"].pop("hypo_transitions_mod", None)
    return common


def validate_legacy_engine(engine: Mapping[str, Any]) -> None:
    modules = engine["modules"]
    if any("mapping" in str(name).lower() for name in modules):
        raise ValueError("legacy H screen must use fixed task-label mapping")
    transition = modules["hypo_transitions_mod"]
    if str(transition["class"]) != LEGACY_CLASS:
        raise ValueError("legacy branch must use FixedStrategyHypothesisTransitionModule")
    kwargs = transition["kwargs"]
    expected = [
        {
            "label": "retain_random",
            "amount": "random_4",
            "method": "random_posterior",
            "pool": "active",
        },
        {
            "label": "explore_random",
            "amount": "opp_random_4",
            "method": "random",
            "pool": "inactive",
        },
    ]
    if kwargs.get("strategies") != expected:
        raise ValueError("legacy branch must preserve the exact earliest strategy chain")
    if int(kwargs.get("init_num", -1)) != 2:
        raise ValueError("legacy branch requires init_num=2")
    if int(kwargs.get("max_active_hypotheses", -1)) != 3:
        raise ValueError("legacy branch requires max_active_hypotheses=3")
    if str(kwargs.get("post_to_prior", {}).get("method")) != "similarity_novelty":
        raise ValueError("legacy branch requires similarity_novelty prior assignment")


def validate_engine_pair(
    adaptive: Mapping[str, Any], legacy: Mapping[str, Any]
) -> None:
    validate_minimal_adaptive_engine(adaptive)
    validate_legacy_engine(legacy)
    if adaptive["provenance"]["architecture_profile"] != H1_PROFILE:
        raise ValueError("adaptive comparator must use the H1 common architecture")
    if str(adaptive["modules"]["hypo_transitions_mod"]["class"]) != ADAPTIVE_CLASS:
        raise ValueError("adaptive branch uses an unexpected H module")
    if _without_h(adaptive) != _without_h(legacy):
        raise ValueError("adaptive and legacy branches differ outside H/provenance")


def _load_subject_engine(
    simulation_path: Path,
    simulation: Mapping[str, Any],
    subject_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subject_cfg = resolve_subject_config(simulation, subject_id)
    engine = resolve_engine_config(subject_cfg, simulation_path.parent, subject_id=subject_id)
    fixed = {
        **infer_fixed_hyperparams_from_engine_config(engine),
        **dict(subject_cfg.get("fixed_hyperparams") or {}),
    }
    return apply_fixed_hyperparams_to_engine_config(engine, fixed), fixed


def validate_panel(panel: Mapping[str, Any]) -> dict[str, np.ndarray]:
    probability = np.asarray(panel["choice_probability"], dtype=float)
    prior = np.asarray(panel["marginal_prior"], dtype=float)
    active = np.asarray(panel["marginal_active_probability"], dtype=float)
    if probability.ndim != 3 or probability.shape[0] < 2 or probability.shape[2] != 2:
        raise ValueError("choice_probability must have shape (seeds, trials, 2)")
    if prior.ndim != 3 or prior.shape[:2] != probability.shape[:2]:
        raise ValueError("marginal_prior has incompatible dimensions")
    if active.shape != prior.shape:
        raise ValueError("marginal_active_probability must match marginal_prior")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("choice probabilities must be finite and non-negative")
    if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
        raise ValueError("marginal priors must be finite and non-negative")
    if not np.all(np.isfinite(active)) or np.any(
        (active < -1e-10) | (active > 1.0 + 1e-10)
    ):
        raise ValueError("active probabilities must lie in [0,1]")
    active = np.clip(active, 0.0, 1.0)
    probability = probability / np.sum(probability, axis=2, keepdims=True)
    prior = prior / np.sum(prior, axis=2, keepdims=True)
    seed_n, trial_n = probability.shape[:2]
    output = {
        "choice_probability": probability,
        "marginal_prior": prior,
        "marginal_active_probability": active,
        "filter_seed": np.asarray(panel["filter_seed"], dtype=np.uint64).reshape(-1),
        "repeat_index": np.asarray(panel["repeat_index"], dtype=int).reshape(-1),
        "observed_choice_index": np.asarray(
            panel["observed_choice_index"], dtype=int
        ).reshape(-1),
        "valid_trial_mask": np.asarray(panel["valid_trial_mask"], dtype=bool).reshape(-1),
    }
    for name in TRACE_KEYS[2:]:
        values = np.asarray(panel[name])
        if values.shape != (seed_n, trial_n):
            raise ValueError(f"{name} has invalid shape {values.shape}")
        if name != "resampled" and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        output[name] = values.astype(bool) if name == "resampled" else values.astype(float)
    if output["filter_seed"].size != seed_n:
        raise ValueError("filter seed count does not match panel")
    if np.unique(output["filter_seed"]).size != seed_n:
        raise ValueError("filter seeds must be unique")
    if not np.array_equal(output["repeat_index"], np.arange(seed_n)):
        raise ValueError("repeat indices must be contiguous")
    if output["observed_choice_index"].size != trial_n:
        raise ValueError("observed choices do not match trial count")
    if output["valid_trial_mask"].size != trial_n:
        raise ValueError("valid mask does not match trial count")
    return output


def _cache_paths(output: Path, subject_id: int, variant_id: str) -> tuple[Path, Path]:
    stem = f"subject_{int(subject_id)}_{variant_id}"
    return output / "cache" / f"{stem}.npz", output / "cache" / f"{stem}.json"


def _load_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return validate_panel({key: bundle[key] for key in bundle.files})


def _run_panel(
    *,
    simulation_path: Path,
    simulation: Mapping[str, Any],
    output: Path,
    subject_id: int,
    variant_id: str,
    engine: Mapping[str, Any],
    fixed_hyperparams: Mapping[str, Any],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    npz_path, json_path = _cache_paths(output, subject_id, variant_id)
    expected_seeds = _filter_seeds(
        int(design["base_seed"]),
        subject_id,
        int(design["seed_count"]),
        seed_role=SEED_ROLE,
    )
    if npz_path.exists() and json_path.exists():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        panel = _load_panel(npz_path)
        if _sha256(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"cache hash mismatch: {npz_path}")
        if not np.array_equal(panel["filter_seed"], expected_seeds):
            raise ValueError(f"cached seed panel differs from design: {npz_path}")
        return metadata
    if npz_path.exists() != json_path.exists():
        raise FileExistsError(f"incomplete cache pair requires review: {npz_path}")

    resolved_engine = deepcopy(dict(engine))
    resolved_engine.setdefault("inference", {})["particle_count"] = int(
        design["particle_count"]
    )
    engine_path = output / "resolved_engines" / f"subject_{subject_id}_{variant_id}.json"
    if engine_path.exists():
        existing_engine = json.loads(engine_path.read_text(encoding="utf-8"))
        if existing_engine != resolved_engine:
            raise FileExistsError(
                f"existing resolved engine differs from requested engine: {engine_path}"
            )
    else:
        _atomic_json(engine_path, resolved_engine)

    subject_cfg = resolve_subject_config(simulation, subject_id)
    dataset_paths = resolve_dataset_paths(subject_cfg, simulation_path.parent)
    runner = StateModelSimulationRunner(
        engine_config=resolved_engine,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=int(design["n_jobs"]),
    )
    runner.prepare_data(dataset_paths["learning_data"])
    prediction_mode, selection_mode = resolve_prediction_modes(subject_cfg)
    loss_metric = resolve_loss_metric(subject_cfg)
    result = runner.simulate_subject(
        subject_id=int(subject_id),
        simulation_repeats=int(design["seed_count"]),
        fixed_hyperparams=dict(fixed_hyperparams),
        window_size=resolve_window_size(subject_cfg, subject_id, [subject_id]),
        stop_at=float(subject_cfg.get("stop_at", 1.0)),
        max_trials=int(design["trials_per_subject"]),
        keep_logs=True,
        prediction_mode=prediction_mode,
        selection_prediction_mode=selection_mode,
        loss_metric=loss_metric,
        loss_delta=resolve_loss_delta(subject_cfg, loss_metric),
        hyper_candidate_seed=int(design["base_seed"]),
        trajectory_seeds=[int(seed) for seed in expected_seeds],
        compute_statistics=False,
        repeat_aggregation="mean_probability",
        evaluation_protocol=subject_cfg.get("evaluation_protocol"),
    )
    raw_runs = list(result["best"].raw_runs or [])
    if len(raw_runs) != int(design["seed_count"]):
        raise RuntimeError("PF panel did not return every requested seed")

    probabilities: list[np.ndarray] = []
    stacked: dict[str, list[np.ndarray]] = {key: [] for key in TRACE_KEYS}
    observed_choices = None
    valid_mask = None
    observed_seeds: list[int] = []
    for run in raw_runs:
        metrics = run["metrics_by_mode"][selection_mode]
        current_choices = np.asarray(metrics["observed_choice_index"], dtype=int)
        current_valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
        if observed_choices is None:
            observed_choices = current_choices
            valid_mask = current_valid
        elif not np.array_equal(observed_choices, current_choices) or not np.array_equal(
            valid_mask, current_valid
        ):
            raise ValueError("observed data changed across paired seeds")
        probabilities.append(np.asarray(metrics["pred_category_probs"], dtype=float))
        state_log = run.get("state_log") or {}
        for key in TRACE_KEYS:
            stacked[key].append(np.asarray(state_log[key]))
        observed_seeds.append(int(run["trajectory_seed"]))
    if observed_choices is None or valid_mask is None:
        raise RuntimeError("PF panel returned no observed trials")
    panel = validate_panel(
        {
            "choice_probability": np.stack(probabilities),
            **{key: np.stack(values) for key, values in stacked.items()},
            "filter_seed": np.asarray(observed_seeds, dtype=np.uint64),
            "repeat_index": np.arange(int(design["seed_count"]), dtype=int),
            "observed_choice_index": observed_choices,
            "valid_trial_mask": valid_mask,
        }
    )
    if not np.array_equal(panel["filter_seed"], expected_seeds):
        raise ValueError("PF seeds were returned in an unexpected order")
    _atomic_npz(npz_path, panel)
    metadata = {
        "subject_id": int(subject_id),
        "variant_id": str(variant_id),
        "particle_count": int(design["particle_count"]),
        "seed_count": int(design["seed_count"]),
        "trial_count": int(panel["choice_probability"].shape[1]),
        "valid_choice_trial_count": int(np.sum(panel["valid_trial_mask"])),
        "filter_seeds": [int(value) for value in panel["filter_seed"]],
        "resolved_engine": _relative(engine_path),
        "resolved_engine_sha256": _sha256(engine_path),
        "npz_path": _relative(npz_path),
        "npz_sha256": _sha256(npz_path),
    }
    _atomic_json(json_path, metadata)
    return metadata


def _masks(valid: np.ndarray, train_trials: int) -> dict[str, np.ndarray]:
    index = np.arange(valid.size)
    train = valid & (index < int(train_trials))
    heldout = valid & (index >= int(train_trials))
    heldout_rows = np.flatnonzero(heldout)
    early_n = min(16, max(1, heldout_rows.size // 2))
    early = np.zeros(valid.size, dtype=bool)
    early[heldout_rows[:early_n]] = True
    late = heldout & ~early
    if not np.any(train) or not np.any(heldout):
        raise ValueError("train/heldout split contains an empty segment")
    return {"train": train, "heldout": heldout, "early_heldout": early, "late_heldout": late}


def _first_trial(values: np.ndarray, threshold: float) -> int:
    rows = np.flatnonzero(np.asarray(values, dtype=float) >= float(threshold))
    return int(rows[0] + 1) if rows.size else -1


def summarize_variant(
    panel: Mapping[str, Any],
    *,
    subject_id: int,
    variant_id: str,
    particle_count: int,
    train_trials: int,
    target_hypothesis: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values = validate_panel(panel)
    masks = _masks(values["valid_trial_mask"], train_trials)
    probability = values["choice_probability"]
    choices = values["observed_choice_index"]
    mean_probability = np.mean(probability, axis=0)
    mean_prior = np.mean(values["marginal_prior"], axis=0)
    mean_active = np.mean(values["marginal_active_probability"], axis=0)
    active_total = np.sum(values["marginal_active_probability"], axis=2)
    arrays: dict[str, np.ndarray] = {
        "mean_choice_probability": mean_probability,
        "mean_marginal_prior": mean_prior,
        "mean_active_probability": mean_active,
        "observed_choice_index": choices,
        "valid_trial_mask": values["valid_trial_mask"],
        "filter_seed": values["filter_seed"],
    }
    row: dict[str, Any] = {
        "subject_id": int(subject_id),
        "variant_id": str(variant_id),
        "particle_count": int(particle_count),
        "seed_count": int(probability.shape[0]),
        "valid_train_trials": int(np.sum(masks["train"])),
        "valid_heldout_trials": int(np.sum(masks["heldout"])),
        "median_post_choice_ess_fraction": float(
            np.median(values["post_choice_ess"] / float(particle_count))
        ),
        "mean_resampling_fraction": float(np.mean(values["resampled"])),
        "mean_active_total": float(np.mean(active_total)),
        "temporal_sd_mean_active_total": float(np.std(np.mean(active_total, axis=0))),
        "mean_filtered_explore_count": float(np.mean(values["replacement_count"])),
        "first_target_active_majority_trial": _first_trial(
            mean_active[:, target_hypothesis], 0.5
        ),
        "first_target_prior_majority_trial": _first_trial(
            mean_prior[:, target_hypothesis], 0.5
        ),
    }
    for segment, mask in masks.items():
        run_nll = np.asarray(
            [_choice_nll(run, choices, mask) for run in probability], dtype=float
        )
        arrays[f"run_nll_{segment}"] = run_nll
        row[f"ensemble_nll_{segment}"] = _choice_nll(mean_probability, choices, mask)
        row[f"run_nll_mean_{segment}"] = float(np.mean(run_nll))
        row[f"run_nll_sd_{segment}"] = float(np.std(run_nll, ddof=1))
        row[f"ensemble_brier_{segment}"] = _choice_brier(mean_probability, choices, mask)
        row[f"mean_active_total_{segment}"] = float(np.mean(active_total[:, mask]))
        row[f"mean_explore_count_{segment}"] = float(
            np.mean(values["replacement_count"][:, mask])
        )
        row[f"mean_target_prior_{segment}"] = float(
            np.mean(mean_prior[mask, target_hypothesis])
        )
        row[f"mean_target_active_probability_{segment}"] = float(
            np.mean(mean_active[mask, target_hypothesis])
        )
    return row, arrays


def summarize_contrast(
    legacy_row: Mapping[str, Any],
    legacy: Mapping[str, np.ndarray],
    adaptive_row: Mapping[str, Any],
    adaptive: Mapping[str, np.ndarray],
    *,
    train_trials: int,
    practical_fraction: float,
    seed_noise_multiplier: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.array_equal(legacy["filter_seed"], adaptive["filter_seed"]):
        raise ValueError("variants do not share ordered PF seeds")
    if not np.array_equal(
        legacy["observed_choice_index"], adaptive["observed_choice_index"]
    ) or not np.array_equal(legacy["valid_trial_mask"], adaptive["valid_trial_mask"]):
        raise ValueError("variants do not share observed trials")
    masks = _masks(legacy["valid_trial_mask"], train_trials)
    row: dict[str, Any] = {
        "subject_id": int(legacy_row["subject_id"]),
        "legacy_variant": str(legacy_row["variant_id"]),
        "adaptive_variant": str(adaptive_row["variant_id"]),
        "particle_count": int(legacy_row["particle_count"]),
        "seed_count": int(legacy_row["seed_count"]),
    }
    seed_rows: list[dict[str, Any]] = []
    for segment in masks:
        delta = np.asarray(legacy[f"run_nll_{segment}"]) - np.asarray(
            adaptive[f"run_nll_{segment}"]
        )
        row[f"paired_delta_nll_{segment}"] = float(np.mean(delta))
        row[f"paired_delta_nll_sd_{segment}"] = float(np.std(delta, ddof=1))
        row[f"paired_delta_nll_mcse_{segment}"] = float(
            np.std(delta, ddof=1) / np.sqrt(delta.size)
        )
        row[f"ensemble_delta_nll_{segment}"] = float(
            legacy_row[f"ensemble_nll_{segment}"]
            - adaptive_row[f"ensemble_nll_{segment}"]
        )
        row[f"positive_seed_fraction_{segment}"] = float(np.mean(delta > 0.0))
        for index, seed in enumerate(legacy["filter_seed"]):
            seed_rows.append(
                {
                    "subject_id": int(legacy_row["subject_id"]),
                    "segment": segment,
                    "repeat_index": int(index),
                    "filter_seed": int(seed),
                    "legacy_nll": float(legacy[f"run_nll_{segment}"][index]),
                    "adaptive_nll": float(adaptive[f"run_nll_{segment}"][index]),
                    "paired_delta_nll": float(delta[index]),
                }
            )
    heldout = masks["heldout"]
    row["heldout_choice_probability_rmse"] = float(
        np.sqrt(
            np.mean(
                np.square(
                    legacy["mean_choice_probability"][heldout]
                    - adaptive["mean_choice_probability"][heldout]
                )
            )
        )
    )
    row["heldout_predictive_geometry_prior_js"] = _mean_js(
        legacy["mean_marginal_prior"], adaptive["mean_marginal_prior"], heldout
    )
    row["heldout_active_probability_mae"] = float(
        np.mean(
            np.abs(
                legacy["mean_active_probability"][heldout]
                - adaptive["mean_active_probability"][heldout]
            )
        )
    )
    threshold = float(practical_fraction * legacy_row["ensemble_nll_heldout"])
    delta = float(row["paired_delta_nll_heldout"])
    mcse = float(row["paired_delta_nll_mcse_heldout"])
    row["practical_delta_threshold"] = threshold
    row["exceeds_positive_practical_threshold"] = bool(delta > threshold)
    row["exceeds_negative_practical_threshold"] = bool(delta < -threshold)
    row["heldout_effect_exceeds_seed_noise"] = bool(
        abs(delta) > seed_noise_multiplier * mcse
    )
    return row, seed_rows


def _write_readme(
    output: Path,
    variant_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    by_variant = variant_df.set_index(["subject_id", "variant_id"])
    legacy_id = str(summary["legacy_variant_id"])
    adaptive_id = str(summary["adaptive_variant_id"])
    lines = [
        "# Model 0815 H2: legacy variable-count H vs adaptive continuous H",
        "",
        "## Question",
        "",
        (
            "Does the earliest stationary reactive H policy—fixed selection/prior "
            "logic but trial-varying retained/explored counts—remain competitive "
            "when both architectures are inferred with the current particle filter?"
        ),
        "",
        (
            "Both branches use boundary emission, fixed task labels, measured perception "
            "noise, one fading memory channel (gamma=0.80, w0=0), unified dynamic beta, "
            "expectation readout, execution off, strategy confidence off, and zero lapse."
        ),
        "",
        "Positive delta NLL (legacy minus adaptive) favors adaptive H.",
        "",
        "## Held-out results",
        "",
        "| subject | legacy NLL | adaptive NLL | delta | MCSE | positive seeds | geometry JS | active MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in contrast_df.to_dict(orient="records"):
        subject = int(row["subject_id"])
        lines.append(
            "| {subject} | {legacy:.5f} | {adaptive:.5f} | {delta:+.5f} | "
            "{mcse:.5f} | {positive:.2f} | {js:.5f} | {active:.5f} |".format(
                subject=subject,
                legacy=float(by_variant.loc[(subject, legacy_id), "ensemble_nll_heldout"]),
                adaptive=float(by_variant.loc[(subject, adaptive_id), "ensemble_nll_heldout"]),
                delta=float(row["paired_delta_nll_heldout"]),
                mcse=float(row["paired_delta_nll_mcse_heldout"]),
                positive=float(row["positive_seed_fraction_heldout"]),
                js=float(row["heldout_predictive_geometry_prior_js"]),
                active=float(row["heldout_active_probability_mae"]),
            )
        )
    cohort = summary["cohort"]
    lines.extend(
        [
            "",
            "## Cohort screen",
            "",
            f"- Mean held-out paired delta NLL: {cohort['mean_paired_delta_nll_heldout']:+.5f}.",
            f"- Adaptive-favoring subjects: {cohort['positive_subject_count']}/{cohort['subject_count']}.",
            f"- Seed-noise-resolved subjects: {cohort['seed_noise_resolved_subject_count']}/{cohort['subject_count']}.",
            f"- Provisional interpretation: **{summary['provisional_decision']}**.",
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a package-level H architecture comparison, not a single-mechanism "
                "ablation: active-set cardinality, survivor selection, newcomer proposal, "
                "and prior assignment all differ together. A legacy loss does not falsify "
                "all stationary reactive policies; a tie or win would motivate a smaller "
                "follow-up isolating which legacy ingredient is sufficient."
            ),
            "",
            (
                "The legacy PF compatibility fields summarize realized particle transitions. "
                "They are diagnostics only and do not alter selection or choice generation."
            ),
            "",
        ]
    )
    path = output / "README.md"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_yaml(config_path)
    design = _resolved_design(config, smoke=bool(args.smoke))
    if args.n_jobs is not None:
        design["n_jobs"] = int(args.n_jobs)
    output = _repo_path(args.output_dir) if args.output_dir else _repo_path(config["output_dir"])
    if args.smoke:
        output = output / "smoke"
    if (output / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    adaptive_path = _repo_path(config["adaptive_simulation_config"])
    legacy_path = _repo_path(config["legacy_simulation_config"])
    adaptive_simulation = load_yaml(adaptive_path)
    legacy_simulation = load_yaml(legacy_path)
    adaptive_id = str(design["adaptive_variant_id"])
    legacy_id = str(design["legacy_variant_id"])
    metadata_rows: list[dict[str, Any]] = []

    if args.phase in {"run", "all"}:
        for subject_id in design["subjects"]:
            adaptive_engine, adaptive_fixed = _load_subject_engine(
                adaptive_path, adaptive_simulation, subject_id
            )
            legacy_engine, legacy_fixed = _load_subject_engine(
                legacy_path, legacy_simulation, subject_id
            )
            validate_engine_pair(adaptive_engine, legacy_engine)
            for variant_id, simulation_path, simulation, engine, fixed in (
                (
                    adaptive_id,
                    adaptive_path,
                    adaptive_simulation,
                    adaptive_engine,
                    adaptive_fixed,
                ),
                (legacy_id, legacy_path, legacy_simulation, legacy_engine, legacy_fixed),
            ):
                metadata_rows.append(
                    _run_panel(
                        simulation_path=simulation_path,
                        simulation=simulation,
                        output=output,
                        subject_id=subject_id,
                        variant_id=variant_id,
                        engine=engine,
                        fixed_hyperparams=fixed,
                        design=design,
                    )
                )
        _atomic_json(
            output / "run_manifest.json",
            {
                "analysis_id": config["analysis_id"],
                "config": _relative(config_path),
                "config_sha256": _sha256(config_path),
                "adaptive_simulation_config": _relative(adaptive_path),
                "adaptive_simulation_config_sha256": _sha256(adaptive_path),
                "legacy_simulation_config": _relative(legacy_path),
                "legacy_simulation_config_sha256": _sha256(legacy_path),
                "design": design,
                "runs": metadata_rows,
            },
        )

    if args.phase in {"summarize", "all"}:
        manifest_path = output / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = {
            (int(row["subject_id"]), str(row["variant_id"])): row
            for row in manifest["runs"]
        }
        variant_rows: list[dict[str, Any]] = []
        contrast_rows: list[dict[str, Any]] = []
        seed_rows: list[dict[str, Any]] = []
        summarized: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        for subject_id in design["subjects"]:
            current_rows: dict[str, dict[str, Any]] = {}
            for variant_id in (adaptive_id, legacy_id):
                metadata_row = metadata[(int(subject_id), variant_id)]
                panel = _load_panel(_repo_path(metadata_row["npz_path"]))
                variant_row, arrays = summarize_variant(
                    panel,
                    subject_id=subject_id,
                    variant_id=variant_id,
                    particle_count=int(design["particle_count"]),
                    train_trials=int(design["train_trials"]),
                    target_hypothesis=int(design["target_hypothesis"]),
                )
                variant_rows.append(variant_row)
                current_rows[variant_id] = variant_row
                summarized[(int(subject_id), variant_id)] = arrays
            contrast, seeds = summarize_contrast(
                current_rows[legacy_id],
                summarized[(int(subject_id), legacy_id)],
                current_rows[adaptive_id],
                summarized[(int(subject_id), adaptive_id)],
                train_trials=int(design["train_trials"]),
                practical_fraction=float(
                    config["screening_rule"][
                        "practical_fraction_of_legacy_heldout_nll"
                    ]
                ),
                seed_noise_multiplier=float(
                    config["screening_rule"]["seed_noise_multiplier"]
                ),
            )
            contrast_rows.append(contrast)
            seed_rows.extend(seeds)

        variant_df = pd.DataFrame(variant_rows)
        contrast_df = pd.DataFrame(contrast_rows)
        seed_df = pd.DataFrame(seed_rows)
        cohort_delta = float(np.mean(contrast_df["paired_delta_nll_heldout"]))
        legacy_heldout = variant_df.loc[
            variant_df["variant_id"] == legacy_id, "ensemble_nll_heldout"
        ]
        threshold = float(
            config["screening_rule"]["practical_fraction_of_legacy_heldout_nll"]
            * np.mean(legacy_heldout)
        )
        positive_count = int(np.sum(contrast_df["paired_delta_nll_heldout"] > 0.0))
        if cohort_delta > threshold and positive_count >= 3:
            decision = "adaptive H favored in this targeted screen"
        elif cohort_delta < -threshold and positive_count <= 2:
            decision = "legacy variable-count H favored in this targeted screen"
        else:
            decision = "architectures practically comparable or heterogeneous"
        summary = {
            "analysis_id": config["analysis_id"],
            "subjects": [int(value) for value in design["subjects"]],
            "adaptive_variant_id": adaptive_id,
            "legacy_variant_id": legacy_id,
            "train_trials": int(design["train_trials"]),
            "trials_per_subject": int(design["trials_per_subject"]),
            "particle_count": int(design["particle_count"]),
            "seed_count": int(design["seed_count"]),
            "cohort": {
                "subject_count": int(len(contrast_df)),
                "mean_paired_delta_nll_train": float(
                    np.mean(contrast_df["paired_delta_nll_train"])
                ),
                "mean_paired_delta_nll_heldout": cohort_delta,
                "mean_ensemble_delta_nll_heldout": float(
                    np.mean(contrast_df["ensemble_delta_nll_heldout"])
                ),
                "cohort_practical_delta_threshold": threshold,
                "positive_subject_count": positive_count,
                "seed_noise_resolved_subject_count": int(
                    np.sum(contrast_df["heldout_effect_exceeds_seed_noise"])
                ),
            },
            "provisional_decision": decision,
            "decision_is_final": False,
        }
        _atomic_csv(output / "variant_summary.csv", variant_df)
        _atomic_csv(output / "contrast_summary.csv", contrast_df)
        _atomic_csv(output / "paired_seed_effects.csv", seed_df)
        _atomic_json(output / "summary.json", summary)
        _write_readme(output, variant_df, contrast_df, summary)


if __name__ == "__main__":
    main()
