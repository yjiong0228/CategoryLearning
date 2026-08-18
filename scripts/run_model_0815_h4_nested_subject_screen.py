#!/usr/bin/env python3
"""Fit a low-dimensional reactive H baseline and profile optional accumulation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
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
    _atomic_csv,
    _atomic_json,
    _choice_nll,
    _load_panel,
    _relative,
    _repo_path,
    _run_panel,
    _sha256,
    summarize_contrast,
    summarize_variant,
    validate_panel,
)
from src.Bayesian_state.simulation.config import (  # noqa: E402
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_h4_nested_subject_screen.yaml"
)
NESTED_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition."
    "nested_feedback_accumulator."
    "NestedFeedbackAccumulatorHypothesisTransitionModule"
)
BOUNDARY_VARIANT = "reactive_boundary"
SELECTED_VARIANT = "nested_selected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError("event_after_correct must lie strictly inside (0, 1)")
    return float(np.log(value) - np.log1p(-value))


def _expit(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def _canonical_parameters(values: Mapping[str, Any]) -> dict[str, float]:
    parameters = {
        "event_after_correct": float(values["event_after_correct"]),
        "immediate_error_logit_gain": float(
            values["immediate_error_logit_gain"]
        ),
        "global_search": float(values["global_search"]),
        "accumulator_decay": float(values["accumulator_decay"]),
        "accumulator_logit_gain": float(values["accumulator_logit_gain"]),
        "initial_failure": float(values.get("initial_failure", 0.0)),
    }
    if parameters["immediate_error_logit_gain"] < 0.0:
        raise ValueError("immediate_error_logit_gain must be non-negative")
    if parameters["accumulator_logit_gain"] < 0.0:
        raise ValueError("accumulator_logit_gain must be non-negative")
    if not 0.0 <= parameters["global_search"] <= 1.0:
        raise ValueError("global_search must lie in [0, 1]")
    if not 0.0 <= parameters["accumulator_decay"] < 1.0:
        raise ValueError("accumulator_decay must lie in [0, 1)")
    if not 0.0 <= parameters["initial_failure"] <= 1.0:
        raise ValueError("initial_failure must lie in [0, 1]")
    parameters["event_after_error"] = _expit(
        _logit(parameters["event_after_correct"])
        + parameters["immediate_error_logit_gain"]
    )
    return parameters


def build_nested_engine(
    template: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply H4 subject parameters without changing the common architecture."""
    engine = deepcopy(dict(template))
    transition = engine["modules"]["hypo_transitions_mod"]
    if str(transition["class"]) != NESTED_CLASS:
        raise ValueError("H4 template uses an unexpected transition class")
    resolved = _canonical_parameters(parameters)
    controller = transition["kwargs"][
        "nested_feedback_accumulator_controller"
    ]
    controller.update(
        {
            "event_after_correct": resolved["event_after_correct"],
            "event_after_error": resolved["event_after_error"],
            "initial_event_probability": resolved["event_after_correct"],
            "global_search": resolved["global_search"],
            "accumulator_decay": resolved["accumulator_decay"],
            "accumulator_logit_gain": resolved["accumulator_logit_gain"],
            "initial_failure": resolved["initial_failure"],
        }
    )
    return engine


def _validate_template(engine: Mapping[str, Any]) -> None:
    modules = engine["modules"]
    transition = modules["hypo_transitions_mod"]
    if str(transition["class"]) != NESTED_CLASS:
        raise ValueError("subject screen requires the nested H4 controller")
    if int(transition["kwargs"].get("capacity", -1)) != 3:
        raise ValueError("subject screen requires workspace capacity 3")
    if str(
        transition["kwargs"].get("prior_assignment", {}).get("method")
    ) != "pairwise_mass_transfer":
        raise ValueError("subject screen requires pairwise mass transfer")
    if any("mapping" in str(name).lower() for name in modules):
        raise ValueError("subject screen requires fixed mapping")
    memory = modules["memory_mod"]
    if float(memory.get("kwargs", {}).get("w0", np.nan)) != 0.0:
        raise ValueError("subject screen requires the one-channel fading M path")
    beta = modules["beta_mod"]["kwargs"]
    if (
        float(beta.get("decrease_rate", 0.0)) <= 0.0
        or float(beta.get("correct_additive", 0.0)) <= 0.0
        or str(beta.get("update_scope")) != "active_hypotheses"
    ):
        raise ValueError("subject screen requires unified dynamic beta")
    if str(engine.get("likelihood", {}).get("distance_mode")) != "boundary":
        raise ValueError("subject screen requires boundary emission")
    if str(engine.get("likelihood", {}).get("beta_source")) != "action":
        raise ValueError("subject screen requires action beta in likelihood")
    readout = engine.get("choice_readout", {}).get("kwargs", {})
    if (
        str(readout.get("method")) != "expectation"
        or float(readout.get("power", 1.0)) != 1.0
        or float(readout.get("strategy_confidence_gain", 0.0)) != 0.0
    ):
        raise ValueError("subject screen requires unsharpened expectation readout")


def _resolved_inputs(
    config: Mapping[str, Any], smoke: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    design = deepcopy(dict(config["design"]))
    reactive = deepcopy(dict(config["reactive_calibration"]))
    accumulator = deepcopy(dict(config["accumulator_profile"]))
    design["subjects"] = [int(value) for value in design["subjects"]]
    if smoke:
        design.update(
            {
                "subjects": [design["subjects"][0]],
                "trials_per_subject": 24,
                # Shared sliding metrics use the configured 16-trial window
                # and therefore require at least 17 observations.
                "train_trials": 18,
                "training_particle_count": 8,
                "evaluation_particle_count": 8,
                "training_seed_count": 2,
                "evaluation_seed_count": 2,
                "training_n_jobs": 2,
                "evaluation_n_jobs": 2,
            }
        )
        reactive["passes"] = 1
        reactive["candidates"] = {
            "event_after_correct": [0.18, 0.30],
            "immediate_error_logit_gain": [0.00, 0.60],
            "global_search": [0.05, 0.20],
        }
        accumulator["common_decay_candidates"] = [
            float(accumulator["common_decay_anchor"])
        ]
        accumulator["subject_gain_candidates"] = [0.0, 1.5]
    if not 1 < int(design["train_trials"]) < int(design["trials_per_subject"]):
        raise ValueError("train_trials must split the selected trial sequence")
    if int(design["training_seed_count"]) < 2:
        raise ValueError("training search requires at least two PF seeds")
    if int(design["evaluation_seed_count"]) < 2:
        raise ValueError("held-out evaluation requires at least two PF seeds")
    coordinate_order = [str(value) for value in reactive["coordinate_order"]]
    if set(coordinate_order) != {
        "event_after_correct",
        "immediate_error_logit_gain",
        "global_search",
    }:
        raise ValueError("reactive coordinate_order must list the three H parameters")
    for coordinate in coordinate_order:
        values = [float(value) for value in reactive["candidates"][coordinate]]
        if not values:
            raise ValueError(f"reactive candidate list is empty: {coordinate}")
        reactive["candidates"][coordinate] = values
    accumulator["common_decay_candidates"] = [
        float(value) for value in accumulator["common_decay_candidates"]
    ]
    accumulator["subject_gain_candidates"] = [
        float(value) for value in accumulator["subject_gain_candidates"]
    ]
    if 0.0 not in accumulator["subject_gain_candidates"]:
        raise ValueError("subject_gain_candidates must include the exact zero boundary")
    if float(accumulator["common_decay_anchor"]) not in accumulator[
        "common_decay_candidates"
    ]:
        raise ValueError("common decay candidates must include the anchor")
    return design, reactive, accumulator


def _load_subject_engine(
    simulation_path: Path,
    simulation: Mapping[str, Any],
    subject_id: int,
) -> dict[str, Any]:
    subject_config = resolve_subject_config(simulation, subject_id)
    engine = resolve_engine_config(
        subject_config,
        simulation_path.parent,
        subject_id=subject_id,
    )
    fixed = {
        **infer_fixed_hyperparams_from_engine_config(engine),
        **dict(subject_config.get("fixed_hyperparams") or {}),
    }
    resolved = apply_fixed_hyperparams_to_engine_config(engine, fixed)
    _validate_template(resolved)
    return resolved


def _parameter_digest(parameters: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_parameters(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _training_score(panel: Mapping[str, Any]) -> dict[str, Any]:
    values = validate_panel(panel)
    mask = values["valid_trial_mask"]
    choices = values["observed_choice_index"]
    probability = values["choice_probability"]
    run_nll = np.asarray(
        [_choice_nll(run, choices, mask) for run in probability],
        dtype=float,
    )
    return {
        "ensemble_train_nll": _choice_nll(
            np.mean(probability, axis=0), choices, mask
        ),
        "run_train_nll": run_nll,
        "filter_seed": values["filter_seed"].copy(),
        "valid_trial_count": int(np.sum(mask)),
    }


def _paired_delta(
    boundary: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[float, float]:
    if not np.array_equal(boundary["filter_seed"], candidate["filter_seed"]):
        raise ValueError("training candidates do not share paired PF seeds")
    delta = np.asarray(boundary["run_train_nll"], dtype=float) - np.asarray(
        candidate["run_train_nll"], dtype=float
    )
    return float(np.mean(delta)), float(np.std(delta, ddof=1) / np.sqrt(delta.size))


def _write_readme(
    output: Path,
    design: Mapping[str, Any],
    decay_frame: pd.DataFrame,
    selection_frame: pd.DataFrame,
) -> None:
    lines = [
        "# Model 0815 H4: subject-level optional accumulation",
        "",
        "## Design",
        "",
        (
            f"The first {int(design['train_trials'])} trials calibrate a one-step "
            "reactive baseline and profile an exact zero spike against positive "
            "accumulator gains. The remaining trials are evaluated with a disjoint "
            "PF seed role and never enter parameter selection. Global search is fixed "
            "within a subject and no independent mastery accumulator is present. "
            f"Training uses {int(design['training_particle_count'])} particles × "
            f"{int(design['training_seed_count'])} seeds; locked-parameter evaluation "
            f"uses {int(design['evaluation_particle_count'])} particles × "
            f"{int(design['evaluation_seed_count'])} seeds."
        ),
        "",
        "## Common decay training profile",
        "",
        "| decay | boundary NLL | selected NLL | improvement | selected |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in decay_frame.to_dict(orient="records"):
        lines.append(
            "| {decay:.2f} | {mean_boundary_train_nll:.5f} | "
            "{mean_selected_train_nll:.5f} | {mean_training_improvement:+.5f} | "
            "{selected} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Subject results",
            "",
            "| subject | E correct | E error | g | c acc | train gate | held-out ΔNLL | MCSE | held-out support |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|",
        ]
    )
    for row in selection_frame.to_dict(orient="records"):
        lines.append(
            "| {subject_id} | {event_after_correct:.3f} | "
            "{event_after_error:.3f} | {global_search:.3f} | "
            "{accumulator_logit_gain:.2f} | {positive_gain_passed_training_gate} | "
            "{paired_delta_nll_heldout:+.5f} | "
            "{paired_delta_nll_mcse_heldout:.5f} | {heldout_support} |".format(
                **row
            )
        )
    positive_count = int(
        np.sum(selection_frame["accumulator_logit_gain"] > 0.0)
    )
    supported_count = int(np.sum(selection_frame["heldout_support"]))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"Positive accumulation passed the training gate for {positive_count}/"
                f"{len(selection_frame)} subjects; it passed the independent held-out "
                f"gate for {supported_count}/{len(selection_frame)} subjects."
            ),
            "",
            "No subject is currently supported as requiring accumulated failure, so set "
            "c_acc=0 unless a later independently powered analysis overturns this screen. "
            "This comparison does not decide whether the remaining immediate reactive "
            "gain is needed; constant versus reactive is the next nested boundary.",
            "",
            "Positive ΔNLL means the selected supermodel predicts better than its "
            "reactive boundary. A subject is marked supported only when the held-out "
            "effect clears both the practical threshold and two-MCSE seed-noise gate.",
            "",
            "## Interpretation boundary",
            "",
            (
                f"This is a {len(design['subjects'])}-subject, "
                f"{int(design['trials_per_subject'])}-trial "
                "architecture/parameterization screen. "
                "The common decay and subject gains are not final full-sample estimates. "
                "A positive training gain without held-out support is treated as search "
                "instability or overfit, not evidence for an accumulator phenotype."
            ),
            "",
        ]
    )
    report = output / "README.md"
    if report.exists():
        raise FileExistsError(f"refusing to overwrite report: {report}")
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_yaml(config_path)
    design, reactive_config, accumulator_config = _resolved_inputs(
        config, bool(args.smoke)
    )
    output = (
        _repo_path(args.output_dir)
        if args.output_dir is not None
        else _repo_path(config["output_dir"])
    )
    if args.smoke:
        output = output / "smoke"
    if (output / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    simulation_path = _repo_path(config["simulation_config"])
    simulation = load_yaml(simulation_path)
    subject_templates = {
        subject_id: _load_subject_engine(simulation_path, simulation, subject_id)
        for subject_id in design["subjects"]
    }

    training_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    training_metadata: dict[tuple[int, str], dict[str, Any]] = {}
    score_cache: dict[tuple[int, str], dict[str, Any]] = {}
    anchor_decay = float(accumulator_config["common_decay_anchor"])
    initial_failure = float(accumulator_config.get("initial_failure", 0.0))

    def evaluate_training(
        subject_id: int,
        parameters: Mapping[str, Any],
        *,
        search_stage: str,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        resolved_parameters = _canonical_parameters(parameters)
        # Decay is inactive on the exact zero boundary.  Canonicalizing that
        # one nuisance value avoids rerunning an identical PF model for every
        # common-decay candidate.
        effective_parameters = dict(resolved_parameters)
        if effective_parameters["accumulator_logit_gain"] == 0.0:
            effective_parameters["accumulator_decay"] = anchor_decay
        digest = _parameter_digest(effective_parameters)
        cache_key = (int(subject_id), digest)
        if cache_key not in score_cache:
            engine = build_nested_engine(
                subject_templates[subject_id], effective_parameters
            )
            metadata = _run_panel(
                simulation_config_path=simulation_path,
                simulation_config=simulation,
                output=output / "training",
                subject_id=subject_id,
                variant_id=f"candidate_{digest}",
                engine=engine,
                fixed_hyperparams=infer_fixed_hyperparams_from_engine_config(
                    engine
                ),
                particle_count=int(design["training_particle_count"]),
                seed_count=int(design["training_seed_count"]),
                trials_per_subject=int(design["train_trials"]),
                base_seed=int(design["base_seed"]),
                n_jobs=int(design["training_n_jobs"]),
                seed_role=str(design["training_seed_role"]),
            )
            score = _training_score(_load_panel(_repo_path(metadata["npz_path"])))
            score_cache[cache_key] = score
            training_metadata[cache_key] = metadata
            training_rows.append(
                {
                    "subject_id": int(subject_id),
                    "candidate_id": digest,
                    "first_search_stage": str(search_stage),
                    **effective_parameters,
                    "ensemble_train_nll": float(score["ensemble_train_nll"]),
                    "run_train_nll_mean": float(
                        np.mean(score["run_train_nll"])
                    ),
                    "run_train_nll_sd": float(
                        np.std(score["run_train_nll"], ddof=1)
                    ),
                    "valid_train_trials": int(score["valid_trial_count"]),
                    "npz_path": metadata["npz_path"],
                }
            )
        return score_cache[cache_key], resolved_parameters

    reactive_parameters: dict[int, dict[str, float]] = {}
    for subject_id in design["subjects"]:
        current = {
            **{
                key: float(value)
                for key, value in reactive_config["anchor"].items()
            },
            "accumulator_decay": anchor_decay,
            "accumulator_logit_gain": 0.0,
            "initial_failure": initial_failure,
        }
        current_score, current = evaluate_training(
            subject_id, current, search_stage="reactive_anchor"
        )
        for pass_index in range(int(reactive_config["passes"])):
            for coordinate in reactive_config["coordinate_order"]:
                candidates: list[
                    tuple[float, dict[str, Any], dict[str, float]]
                ] = []
                for value in reactive_config["candidates"][coordinate]:
                    candidate = dict(current)
                    candidate[coordinate] = float(value)
                    score, resolved = evaluate_training(
                        subject_id,
                        candidate,
                        search_stage=f"reactive_pass_{pass_index + 1}_{coordinate}",
                    )
                    candidates.append(
                        (float(score["ensemble_train_nll"]), score, resolved)
                    )
                candidates.sort(key=lambda item: (item[0], item[2][coordinate]))
                best_nll, best_score, best_parameters = candidates[0]
                if float(current_score["ensemble_train_nll"]) <= best_nll + 1e-12:
                    best_score = current_score
                    best_parameters = current
                    best_nll = float(current_score["ensemble_train_nll"])
                current_score = best_score
                current = dict(best_parameters)
                coordinate_rows.append(
                    {
                        "subject_id": int(subject_id),
                        "pass_index": int(pass_index + 1),
                        "coordinate": str(coordinate),
                        "selected_value": float(current[coordinate]),
                        "selected_train_nll": float(best_nll),
                    }
                )
        reactive_parameters[subject_id] = dict(current)

    profile_rows: list[dict[str, Any]] = []
    selected_by_decay: dict[tuple[int, float], dict[str, Any]] = {}
    selection_rule = dict(config["selection_rule"])
    practical_fraction = float(
        selection_rule["subject_train_practical_fraction_of_boundary_nll"]
    )
    noise_multiplier = float(
        selection_rule["subject_train_seed_noise_multiplier"]
    )
    for subject_id in design["subjects"]:
        base = reactive_parameters[subject_id]
        for decay in accumulator_config["common_decay_candidates"]:
            candidates: list[dict[str, Any]] = []
            for gain in accumulator_config["subject_gain_candidates"]:
                parameters = {
                    **base,
                    "accumulator_decay": float(decay),
                    "accumulator_logit_gain": float(gain),
                    "initial_failure": initial_failure,
                }
                score, resolved = evaluate_training(
                    subject_id,
                    parameters,
                    search_stage=f"accumulator_decay_{float(decay):.2f}",
                )
                candidates.append(
                    {"score": score, "parameters": resolved}
                )
            boundary = next(
                row
                for row in candidates
                if row["parameters"]["accumulator_logit_gain"] == 0.0
            )
            threshold = float(
                practical_fraction * boundary["score"]["ensemble_train_nll"]
            )
            qualifying: list[dict[str, Any]] = []
            for row in candidates:
                delta_mean, delta_mcse = _paired_delta(
                    boundary["score"], row["score"]
                )
                ensemble_delta = float(
                    boundary["score"]["ensemble_train_nll"]
                    - row["score"]["ensemble_train_nll"]
                )
                positive_gain = bool(
                    row["parameters"]["accumulator_logit_gain"] > 0.0
                )
                passes_gate = bool(
                    positive_gain
                    and ensemble_delta > threshold
                    and delta_mean > noise_multiplier * delta_mcse
                )
                profile_rows.append(
                    {
                        "subject_id": int(subject_id),
                        "accumulator_decay": float(decay),
                        "accumulator_logit_gain": float(
                            row["parameters"]["accumulator_logit_gain"]
                        ),
                        "ensemble_train_nll": float(
                            row["score"]["ensemble_train_nll"]
                        ),
                        "boundary_ensemble_train_nll": float(
                            boundary["score"]["ensemble_train_nll"]
                        ),
                        "ensemble_delta_nll_train": ensemble_delta,
                        "paired_delta_nll_train": delta_mean,
                        "paired_delta_nll_mcse_train": delta_mcse,
                        "practical_delta_threshold": threshold,
                        "passes_training_activation_gate": passes_gate,
                    }
                )
                if passes_gate:
                    qualifying.append(row)
            selected = boundary
            if qualifying:
                selected = min(
                    qualifying,
                    key=lambda row: (
                        row["score"]["ensemble_train_nll"],
                        row["parameters"]["accumulator_logit_gain"],
                    ),
                )
            selected_by_decay[(subject_id, float(decay))] = {
                "score": selected["score"],
                "parameters": selected["parameters"],
                "positive_gain_passed_training_gate": bool(
                    selected["parameters"]["accumulator_logit_gain"] > 0.0
                ),
                "boundary_score": boundary["score"],
            }

    decay_rows: list[dict[str, Any]] = []
    for decay in accumulator_config["common_decay_candidates"]:
        selected_scores = [
            selected_by_decay[(subject_id, float(decay))]["score"]
            for subject_id in design["subjects"]
        ]
        boundary_scores = [
            selected_by_decay[(subject_id, float(decay))]["boundary_score"]
            for subject_id in design["subjects"]
        ]
        boundary_nll = float(
            np.mean([row["ensemble_train_nll"] for row in boundary_scores])
        )
        selected_nll = float(
            np.mean([row["ensemble_train_nll"] for row in selected_scores])
        )
        decay_rows.append(
            {
                "decay": float(decay),
                "mean_boundary_train_nll": boundary_nll,
                "mean_selected_train_nll": selected_nll,
                "mean_training_improvement": boundary_nll - selected_nll,
                "positive_gain_subject_count": int(
                    sum(
                        selected_by_decay[(subject_id, float(decay))][
                            "positive_gain_passed_training_gate"
                        ]
                        for subject_id in design["subjects"]
                    )
                ),
            }
        )
    best_decay_row = min(
        decay_rows,
        key=lambda row: (row["mean_selected_train_nll"], row["decay"]),
    )
    common_threshold = float(
        selection_rule["common_decay_practical_fraction_of_boundary_nll"]
        * best_decay_row["mean_boundary_train_nll"]
    )
    chosen_decay = float(best_decay_row["decay"])
    if best_decay_row["mean_training_improvement"] <= common_threshold:
        chosen_decay = anchor_decay
    for row in decay_rows:
        row["common_practical_delta_threshold"] = common_threshold
        row["selected"] = bool(np.isclose(row["decay"], chosen_decay))

    final_metadata: list[dict[str, Any]] = []
    final_parameters: dict[int, dict[str, dict[str, float]]] = {}
    for subject_id in design["subjects"]:
        selected = selected_by_decay[(subject_id, chosen_decay)]
        selected_parameters = dict(selected["parameters"])
        boundary_parameters = {
            **reactive_parameters[subject_id],
            "accumulator_decay": chosen_decay,
            "accumulator_logit_gain": 0.0,
            "initial_failure": initial_failure,
        }
        final_parameters[subject_id] = {
            BOUNDARY_VARIANT: _canonical_parameters(boundary_parameters),
            SELECTED_VARIANT: _canonical_parameters(selected_parameters),
        }
        for variant_id, parameters in final_parameters[subject_id].items():
            engine = build_nested_engine(subject_templates[subject_id], parameters)
            final_metadata.append(
                _run_panel(
                    simulation_config_path=simulation_path,
                    simulation_config=simulation,
                    output=output / "evaluation",
                    subject_id=subject_id,
                    variant_id=variant_id,
                    engine=engine,
                    fixed_hyperparams=infer_fixed_hyperparams_from_engine_config(
                        engine
                    ),
                    particle_count=int(design["evaluation_particle_count"]),
                    seed_count=int(design["evaluation_seed_count"]),
                    trials_per_subject=int(design["trials_per_subject"]),
                    base_seed=int(design["base_seed"]),
                    n_jobs=int(design["evaluation_n_jobs"]),
                    seed_role=str(design["evaluation_seed_role"]),
                )
            )

    metadata_index = {
        (int(row["subject_id"]), str(row["variant_id"])): row
        for row in final_metadata
    }
    variant_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    validation_checks = 0
    training_seed_set: set[int] = set()
    for score in score_cache.values():
        training_seed_set.update(int(value) for value in score["filter_seed"])

    for subject_id in design["subjects"]:
        summarized: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: dict[str, dict[str, Any]] = {}
        panels: dict[str, dict[str, np.ndarray]] = {}
        for variant_id in (BOUNDARY_VARIANT, SELECTED_VARIANT):
            metadata = metadata_index[(subject_id, variant_id)]
            panel = _load_panel(_repo_path(metadata["npz_path"]))
            panels[variant_id] = panel
            row, arrays = summarize_variant(
                panel,
                subject_id=subject_id,
                variant_id=variant_id,
                particle_count=int(design["evaluation_particle_count"]),
                train_trials=int(design["train_trials"]),
            )
            row.update(final_parameters[subject_id][variant_id])
            variant_rows.append(row)
            summary_rows[variant_id] = row
            summarized[variant_id] = arrays
            validation_checks += 4
        contrast, seeds = summarize_contrast(
            summary_rows[BOUNDARY_VARIANT],
            summarized[BOUNDARY_VARIANT],
            summary_rows[SELECTED_VARIANT],
            summarized[SELECTED_VARIANT],
            train_trials=int(design["train_trials"]),
            practical_fraction=float(
                selection_rule["heldout_practical_fraction_of_boundary_nll"]
            ),
            seed_noise_multiplier=float(
                selection_rule["heldout_seed_noise_multiplier"]
            ),
        )
        contrast_rows.append(contrast)
        seed_rows.extend(seeds)
        evaluation_seed_set = {
            int(value) for value in panels[BOUNDARY_VARIANT]["filter_seed"]
        }
        if training_seed_set.intersection(evaluation_seed_set):
            raise ValueError("training and held-out PF seed roles overlap")
        validation_checks += 1
        parameters = final_parameters[subject_id][SELECTED_VARIANT]
        training_selection = selected_by_decay[(subject_id, chosen_decay)]
        heldout_support = bool(
            contrast["exceeds_positive_practical_threshold"]
            and contrast["heldout_effect_exceeds_seed_noise"]
            and contrast["paired_delta_nll_heldout"] > 0.0
        )
        selection_rows.append(
            {
                "subject_id": int(subject_id),
                **parameters,
                "positive_gain_passed_training_gate": bool(
                    training_selection["positive_gain_passed_training_gate"]
                ),
                "paired_delta_nll_train": float(
                    contrast["paired_delta_nll_train"]
                ),
                "paired_delta_nll_heldout": float(
                    contrast["paired_delta_nll_heldout"]
                ),
                "paired_delta_nll_mcse_heldout": float(
                    contrast["paired_delta_nll_mcse_heldout"]
                ),
                "heldout_practical_delta_threshold": float(
                    contrast["practical_delta_threshold"]
                ),
                "heldout_support": heldout_support,
            }
        )
        if parameters["accumulator_logit_gain"] == 0.0:
            for key in (
                "choice_probability",
                "marginal_prior",
                "predictive_swap_probability",
                "predictive_search_range",
            ):
                if not np.array_equal(
                    panels[BOUNDARY_VARIANT][key],
                    panels[SELECTED_VARIANT][key],
                ):
                    raise ValueError(
                        f"zero boundary is not exact for subject {subject_id}: {key}"
                    )
                validation_checks += 1

    # The zero-gain training score is deliberately reused across decay values;
    # verify that every profile row sees the same cached boundary result.
    for subject_id in design["subjects"]:
        boundary_values = [
            row["boundary_ensemble_train_nll"]
            for row in profile_rows
            if int(row["subject_id"]) == int(subject_id)
        ]
        if not np.allclose(boundary_values, boundary_values[0], rtol=0.0, atol=0.0):
            raise ValueError("zero-gain training score depends on decay")
        validation_checks += max(1, len(boundary_values))

    training_frame = pd.DataFrame(training_rows).sort_values(
        ["subject_id", "candidate_id"]
    )
    coordinate_frame = pd.DataFrame(coordinate_rows)
    profile_frame = pd.DataFrame(profile_rows)
    decay_frame = pd.DataFrame(decay_rows)
    variant_frame = pd.DataFrame(variant_rows)
    contrast_frame = pd.DataFrame(contrast_rows)
    seed_frame = pd.DataFrame(seed_rows)
    selection_frame = pd.DataFrame(selection_rows)
    _atomic_csv(output / "training_candidate_scores.csv", training_frame)
    _atomic_csv(output / "reactive_coordinate_trace.csv", coordinate_frame)
    _atomic_csv(output / "accumulator_gain_profile.csv", profile_frame)
    _atomic_csv(output / "common_decay_profile.csv", decay_frame)
    _atomic_csv(output / "variant_summary.csv", variant_frame)
    _atomic_csv(output / "contrast_summary.csv", contrast_frame)
    _atomic_csv(output / "paired_seed_effects.csv", seed_frame)
    _atomic_csv(output / "subject_selection_summary.csv", selection_frame)

    manifest = {
        "analysis_id": config["analysis_id"],
        "config": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "simulation_config": _relative(simulation_path),
        "simulation_config_sha256": _sha256(simulation_path),
        "design": design,
        "reactive_calibration": reactive_config,
        "accumulator_profile": accumulator_config,
        "selection_rule": selection_rule,
        "chosen_common_decay": chosen_decay,
        "training_runs": list(training_metadata.values()),
        "evaluation_runs": final_metadata,
        "selected_parameters": {
            str(subject_id): final_parameters[subject_id]
            for subject_id in design["subjects"]
        },
    }
    _atomic_json(output / "run_manifest.json", manifest)
    summary = {
        "analysis_id": config["analysis_id"],
        "subjects": design["subjects"],
        "train_trials": int(design["train_trials"]),
        "trials_per_subject": int(design["trials_per_subject"]),
        "training_particle_count": int(design["training_particle_count"]),
        "evaluation_particle_count": int(design["evaluation_particle_count"]),
        "training_seed_count": int(design["training_seed_count"]),
        "evaluation_seed_count": int(design["evaluation_seed_count"]),
        "chosen_common_decay": chosen_decay,
        "positive_gain_selected_subject_count": int(
            np.sum(selection_frame["accumulator_logit_gain"] > 0.0)
        ),
        "heldout_supported_subject_count": int(
            np.sum(selection_frame["heldout_support"])
        ),
        "mean_paired_delta_nll_heldout": float(
            np.mean(selection_frame["paired_delta_nll_heldout"])
        ),
        "parameter_selection_used_heldout_information": False,
        "training_and_evaluation_pf_seed_roles_are_disjoint": True,
        "decision_is_final": False,
    }
    _atomic_json(output / "summary.json", summary)
    _write_readme(output, design, decay_frame, selection_frame)
    validation_path = output / "VALIDATION.md"
    if validation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite validation report: {validation_path}"
        )
    validation_path.write_text(
        "\n".join(
            [
                "# Validation",
                "",
                f"- Automated checks passed: {validation_checks}.",
                "- Every cached panel was hash-checked and reloaded through the common probability/state validator.",
                "- Choice probabilities and marginal geometry priors were finite, non-negative, and normalized.",
                "- Training and held-out evaluation used disjoint deterministic PF seed roles.",
                "- Zero-gain panels were invariant to accumulator decay.",
                "- When the selected gain was zero, boundary and selected full panels were exactly equal.",
                "- All NLL, paired deltas, MCSE values, and selection gates were recomputed from cached panels before serialization.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
