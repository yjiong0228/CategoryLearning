#!/usr/bin/env python3
"""Project and test a one-step feedback-reactive H controller."""

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
    _atomic_csv,
    _atomic_json,
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
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_h3_feedback_reactive_comparison.yaml"
)
REACTIVE_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition.feedback_reactive."
    "FeedbackReactiveHypothesisTransitionModule"
)
SOURCE_SEED_ROLE = "model0815_b0_adaptive_controller_pilot"


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
        raise ValueError("train_trials must divide the selected trial sequence")
    if int(design["seed_count"]) < 2:
        raise ValueError("nested H screen requires at least two paired PF seeds")
    return design


def _slice_panel(
    panel: Mapping[str, Any], *, seed_count: int, trial_count: int
) -> dict[str, np.ndarray]:
    values = validate_panel(panel)
    if values["choice_probability"].shape[0] < int(seed_count):
        raise ValueError("source panel has fewer PF seeds than requested")
    if values["choice_probability"].shape[1] < int(trial_count):
        raise ValueError("source panel has fewer trials than requested")
    sliced: dict[str, np.ndarray] = {}
    for key, value in values.items():
        array = np.asarray(value)
        if key in {"observed_choice_index", "valid_trial_mask"}:
            sliced[key] = array[:trial_count]
        elif key in {"filter_seed", "repeat_index"}:
            sliced[key] = array[:seed_count]
        else:
            sliced[key] = array[:seed_count, :trial_count]
    sliced["repeat_index"] = np.arange(int(seed_count), dtype=int)
    return validate_panel(sliced)


def project_feedback_reactive_controls(
    adaptive_panel: Mapping[str, Any],
    feedback: np.ndarray,
    *,
    train_trials: int,
    enforce_error_not_less: bool,
) -> dict[str, Any]:
    """Project adaptive train traces without optimizing behavioral NLL."""
    values = validate_panel(adaptive_panel)
    trial_n = values["choice_probability"].shape[1]
    feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if feedback.size != trial_n:
        raise ValueError("feedback length does not match the adaptive panel")
    if not np.all(np.isfinite(feedback)) or np.any((feedback < 0.0) | (feedback > 1.0)):
        raise ValueError("feedback must be finite and lie in [0,1]")
    if not 1 < int(train_trials) < trial_n:
        raise ValueError("train_trials must split the adaptive panel")

    trial_index = np.arange(trial_n)
    calibration = (
        values["valid_trial_mask"]
        & (trial_index > 0)
        & (trial_index < int(train_trials))
    )
    previous_feedback = np.full(trial_n, np.nan, dtype=float)
    previous_feedback[1:] = feedback[:-1]
    after_correct = calibration & np.isclose(previous_feedback, 1.0)
    after_error = calibration & np.isclose(previous_feedback, 0.0)
    if not np.any(after_correct) or not np.any(after_error):
        raise ValueError("projection requires both previous-correct and previous-error trials")

    event = values["predictive_swap_probability"]
    global_search = values["predictive_search_range"]
    event_correct = float(np.mean(event[:, after_correct]))
    event_error = float(np.mean(event[:, after_error]))
    global_mean = float(np.mean(global_search[:, calibration]))
    if enforce_error_not_less and event_error + 1e-12 < event_correct:
        raise ValueError(
            "adaptive projection violates event_after_error >= event_after_correct"
        )
    return {
        "event_after_correct": event_correct,
        "event_after_error": event_error,
        "initial_event_probability": event_correct,
        "global_search": global_mean,
        "calibration_trial_count": int(np.sum(calibration)),
        "previous_correct_trial_count": int(np.sum(after_correct)),
        "previous_error_trial_count": int(np.sum(after_error)),
        "adaptive_seed_count": int(event.shape[0]),
        "uses_choice_nll": False,
        "uses_heldout_trials": False,
    }


def build_projected_reactive_engine(
    template: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    engine = deepcopy(dict(template))
    transition = engine["modules"]["hypo_transitions_mod"]
    if str(transition["class"]) != REACTIVE_CLASS:
        raise ValueError("reactive template uses an unexpected H module")
    controller = transition["kwargs"]["feedback_reactive_controller"]
    for key in (
        "event_after_correct",
        "event_after_error",
        "initial_event_probability",
        "global_search",
    ):
        controller[key] = float(projection[key])
    return engine


def _without_h_and_provenance(engine: Mapping[str, Any]) -> dict[str, Any]:
    common = deepcopy(dict(engine))
    common.pop("provenance", None)
    common["modules"].pop("hypo_transitions_mod", None)
    return common


def validate_reactive_pair(
    adaptive_engine: Mapping[str, Any], reactive_engine: Mapping[str, Any]
) -> None:
    if _without_h_and_provenance(adaptive_engine) != _without_h_and_provenance(
        reactive_engine
    ):
        raise ValueError("reactive and adaptive branches differ outside H/provenance")
    transition = reactive_engine["modules"]["hypo_transitions_mod"]
    if str(transition["class"]) != REACTIVE_CLASS:
        raise ValueError("reactive engine uses an unexpected H class")
    kwargs = transition["kwargs"]
    if int(kwargs.get("capacity", -1)) != 3:
        raise ValueError("reactive screen requires capacity=3")
    if str(kwargs.get("prior_assignment", {}).get("method")) != "pairwise_mass_transfer":
        raise ValueError("reactive screen requires pairwise mass transfer")
    controller = kwargs["feedback_reactive_controller"]
    if float(controller["event_after_error"]) < float(controller["event_after_correct"]):
        raise ValueError("reactive screen requires E_error >= E_correct")


def _load_reactive_subject_engine(
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


def _subject_feedback(
    simulation_path: Path,
    simulation: Mapping[str, Any],
    subject_id: int,
    trial_count: int,
) -> np.ndarray:
    subject_cfg = resolve_subject_config(simulation, subject_id)
    paths = resolve_dataset_paths(subject_cfg, simulation_path.parent)
    frame = pd.read_csv(paths["learning_data"], encoding="utf-8-sig")
    subject = frame.loc[frame["iSub"] == int(subject_id), "feedback"].to_numpy(
        dtype=float
    )
    if subject.size < int(trial_count):
        raise ValueError(f"subject {subject_id} has fewer than {trial_count} trials")
    return subject[:trial_count]


def _source_index(manifest: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["subject_id"]), str(row["variant_id"])): dict(row)
        for row in manifest["runs"]
    }


def _pair_row(
    simpler_row: Mapping[str, Any],
    simpler: Mapping[str, np.ndarray],
    complex_row: Mapping[str, Any],
    complex_arrays: Mapping[str, np.ndarray],
    *,
    pair_id: str,
    train_trials: int,
    practical_fraction: float,
    seed_noise_multiplier: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, seeds = summarize_contrast(
        simpler_row,
        simpler,
        complex_row,
        complex_arrays,
        train_trials=train_trials,
        practical_fraction=practical_fraction,
        seed_noise_multiplier=seed_noise_multiplier,
    )
    row["pair_id"] = pair_id
    row["simpler_variant"] = str(simpler_row["variant_id"])
    row["complex_variant"] = str(complex_row["variant_id"])
    row.pop("fixed_variant", None)
    row.pop("adaptive_variant", None)
    for seed in seeds:
        seed["pair_id"] = pair_id
        seed["simpler_nll"] = seed.pop("fixed_nll")
        seed["complex_nll"] = seed.pop("adaptive_nll")
    return row, seeds


def _cohort_pair(
    pair_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    *,
    pair_id: str,
    simpler_variant: str,
    practical_fraction: float,
    majority_subjects: int,
) -> dict[str, Any]:
    rows = pair_frame[pair_frame["pair_id"] == pair_id]
    simpler_nll = variant_frame.loc[
        variant_frame["variant_id"] == simpler_variant, "ensemble_nll_heldout"
    ]
    threshold = float(practical_fraction * np.mean(simpler_nll))
    mean_delta = float(np.mean(rows["paired_delta_nll_heldout"]))
    positive = int(np.sum(rows["paired_delta_nll_heldout"] > 0.0))
    return {
        "pair_id": pair_id,
        "mean_paired_delta_nll_train": float(
            np.mean(rows["paired_delta_nll_train"])
        ),
        "mean_paired_delta_nll_heldout": mean_delta,
        "cohort_practical_delta_threshold": threshold,
        "positive_subject_count": positive,
        "subject_count": int(len(rows)),
        "seed_noise_resolved_subject_count": int(
            np.sum(rows["heldout_effect_exceeds_seed_noise"])
        ),
        "complex_favored": bool(
            mean_delta > threshold and positive >= int(majority_subjects)
        ),
    }


def _write_readme(
    output: Path,
    projection_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    ids = summary["variant_ids"]
    by_variant = variant_df.set_index(["subject_id", "variant_id"])
    lines = [
        "# Model 0815 H3: constant → feedback-reactive → accumulator H",
        "",
        "## Design",
        "",
        (
            "The feedback-reactive controller uses the previous completed outcome only. "
            "Its E_correct, E_error, and fixed global-search mixture are projections of "
            f"the adaptive branch's first {summary['train_trials']} trials; choice NLL "
            "and later trials are not "
            "used in the projection. All three variants share bounded-workspace selection, "
            "local/global proposal, pairwise prior transfer, and the same non-H architecture."
        ),
        "",
        "Positive paired delta NLL (simpler minus complex) favors the more complex controller.",
        "",
        "## Projected reactive controls",
        "",
        "| subject | E correct | E error | fixed g | correct trials | error trials |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in projection_df.to_dict(orient="records"):
        lines.append(
            "| {subject_id} | {event_after_correct:.5f} | {event_after_error:.5f} | "
            "{global_search:.5f} | {previous_correct_trial_count} | "
            "{previous_error_trial_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Held-out nested contrasts",
            "",
            "| pair | subject | simpler NLL | complex NLL | delta | MCSE | positive seeds | geometry JS |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrast_df.to_dict(orient="records"):
        subject = int(row["subject_id"])
        simpler = str(row["simpler_variant"])
        complex_variant = str(row["complex_variant"])
        lines.append(
            "| {pair} | {subject} | {simpler_nll:.5f} | {complex_nll:.5f} | "
            "{delta:+.5f} | {mcse:.5f} | {positive:.2f} | {js:.5f} |".format(
                pair=row["pair_id"],
                subject=subject,
                simpler_nll=float(by_variant.loc[(subject, simpler), "ensemble_nll_heldout"]),
                complex_nll=float(
                    by_variant.loc[(subject, complex_variant), "ensemble_nll_heldout"]
                ),
                delta=float(row["paired_delta_nll_heldout"]),
                mcse=float(row["paired_delta_nll_mcse_heldout"]),
                positive=float(row["positive_seed_fraction_heldout"]),
                js=float(row["heldout_predictive_geometry_prior_js"]),
            )
        )
    lines.extend(["", "## Cohort decision", ""])
    for pair in summary["cohort_pairs"]:
        lines.append(
            "- {pair_id}: mean ΔNLL={mean_paired_delta_nll_heldout:+.5f}, "
            "positive subjects={positive_subject_count}/{subject_count}, "
            "complex favored={complex_favored}.".format(**pair)
        )
    lines.extend(
        [
            f"- Provisional H controller: **{summary['provisional_controller']}**.",
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a low-cost nested controller screen with a provisional PF budget. "
                "Projection makes the simpler controllers faithful approximations of the "
                "accumulator's training-period control intensity; it is not a final fit of "
                "subject-level H parameters. Model recovery and final PF calibration remain "
                "downstream of architecture selection."
            ),
            "",
            (
                f"Variant IDs: const={ids['const']}, reactive={ids['reactive']}, "
                f"accumulator={ids['accumulator']}."
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
    design = _resolved_design(config, bool(args.smoke))
    if args.n_jobs is not None:
        design["n_jobs"] = int(args.n_jobs)
    output = _repo_path(args.output_dir) if args.output_dir else _repo_path(config["output_dir"])
    if args.smoke:
        output = output / "smoke"
    if (output / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    simulation_path = _repo_path(config["reactive_simulation_config"])
    simulation = load_yaml(simulation_path)
    source = _repo_path(config["source_analysis_dir"])
    source_manifest_path = source / "run_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"missing source controller panel: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_rows = _source_index(source_manifest)
    const_id = str(design["const_variant_id"])
    reactive_id = str(design["reactive_variant_id"])
    accumulator_id = str(design["accumulator_variant_id"])
    reactive_metadata: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []

    if args.phase in {"run", "all"}:
        for subject_id in design["subjects"]:
            accumulator_meta = source_rows[(subject_id, accumulator_id)]
            accumulator_panel = _slice_panel(
                _load_panel(_repo_path(accumulator_meta["npz_path"])),
                seed_count=int(design["seed_count"]),
                trial_count=int(design["trials_per_subject"]),
            )
            feedback = _subject_feedback(
                simulation_path,
                simulation,
                subject_id,
                int(design["trials_per_subject"]),
            )
            projection = project_feedback_reactive_controls(
                accumulator_panel,
                feedback,
                train_trials=int(design["train_trials"]),
                enforce_error_not_less=bool(
                    config["projection"][
                        "enforce_event_after_error_not_less_than_correct"
                    ]
                ),
            )
            projection_rows.append({"subject_id": subject_id, **projection})

            reactive_template, fixed = _load_reactive_subject_engine(
                simulation_path, simulation, subject_id
            )
            reactive_engine = build_projected_reactive_engine(
                reactive_template, projection
            )
            adaptive_engine = json.loads(
                _repo_path(accumulator_meta["resolved_engine"]).read_text(
                    encoding="utf-8"
                )
            )
            validate_reactive_pair(adaptive_engine, reactive_engine)
            reactive_metadata.append(
                _run_panel(
                    simulation_config_path=simulation_path,
                    simulation_config=simulation,
                    output=output,
                    subject_id=subject_id,
                    variant_id=reactive_id,
                    engine=reactive_engine,
                    fixed_hyperparams=fixed,
                    particle_count=int(design["particle_count"]),
                    seed_count=int(design["seed_count"]),
                    trials_per_subject=int(design["trials_per_subject"]),
                    base_seed=int(design["base_seed"]),
                    n_jobs=int(design["n_jobs"]),
                    seed_role=SOURCE_SEED_ROLE,
                )
            )
        _atomic_csv(output / "projection_summary.csv", pd.DataFrame(projection_rows))
        _atomic_json(
            output / "run_manifest.json",
            {
                "analysis_id": config["analysis_id"],
                "config": _relative(config_path),
                "config_sha256": _sha256(config_path),
                "reactive_simulation_config": _relative(simulation_path),
                "reactive_simulation_config_sha256": _sha256(simulation_path),
                "source_manifest": _relative(source_manifest_path),
                "source_manifest_sha256": _sha256(source_manifest_path),
                "design": design,
                "reactive_runs": reactive_metadata,
                "projections": projection_rows,
            },
        )

    if args.phase in {"summarize", "all"}:
        manifest_path = output / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reactive_index = {
            int(row["subject_id"]): row for row in manifest["reactive_runs"]
        }
        variant_rows: list[dict[str, Any]] = []
        contrast_rows: list[dict[str, Any]] = []
        seed_rows: list[dict[str, Any]] = []
        summarized: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        summary_rows: dict[tuple[int, str], dict[str, Any]] = {}

        for subject_id in design["subjects"]:
            metadata = {
                const_id: source_rows[(subject_id, const_id)],
                accumulator_id: source_rows[(subject_id, accumulator_id)],
                reactive_id: reactive_index[subject_id],
            }
            for variant_id in (const_id, reactive_id, accumulator_id):
                panel = _slice_panel(
                    _load_panel(_repo_path(metadata[variant_id]["npz_path"])),
                    seed_count=int(design["seed_count"]),
                    trial_count=int(design["trials_per_subject"]),
                )
                row, arrays = summarize_variant(
                    panel,
                    subject_id=subject_id,
                    variant_id=variant_id,
                    particle_count=int(design["particle_count"]),
                    train_trials=int(design["train_trials"]),
                )
                variant_rows.append(row)
                summary_rows[(subject_id, variant_id)] = row
                summarized[(subject_id, variant_id)] = arrays

            for pair_id, simpler_id, complex_id in (
                ("const_vs_reactive", const_id, reactive_id),
                ("reactive_vs_accumulator", reactive_id, accumulator_id),
                ("const_vs_accumulator_reference", const_id, accumulator_id),
            ):
                contrast, seeds = _pair_row(
                    summary_rows[(subject_id, simpler_id)],
                    summarized[(subject_id, simpler_id)],
                    summary_rows[(subject_id, complex_id)],
                    summarized[(subject_id, complex_id)],
                    pair_id=pair_id,
                    train_trials=int(design["train_trials"]),
                    practical_fraction=float(
                        config["screening_rule"][
                            "practical_fraction_of_simpler_heldout_nll"
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
        practical_fraction = float(
            config["screening_rule"]["practical_fraction_of_simpler_heldout_nll"]
        )
        majority = int(config["screening_rule"]["majority_subjects"])
        cohort_pairs = [
            _cohort_pair(
                contrast_df,
                variant_df,
                pair_id="const_vs_reactive",
                simpler_variant=const_id,
                practical_fraction=practical_fraction,
                majority_subjects=majority,
            ),
            _cohort_pair(
                contrast_df,
                variant_df,
                pair_id="reactive_vs_accumulator",
                simpler_variant=reactive_id,
                practical_fraction=practical_fraction,
                majority_subjects=majority,
            ),
        ]
        const_to_reactive, reactive_to_accumulator = cohort_pairs
        if reactive_to_accumulator["complex_favored"]:
            controller = "H_accum"
        elif const_to_reactive["complex_favored"]:
            controller = "H_reactive"
        else:
            controller = "H_const"
        summary = {
            "analysis_id": config["analysis_id"],
            "subjects": [int(value) for value in design["subjects"]],
            "variant_ids": {
                "const": const_id,
                "reactive": reactive_id,
                "accumulator": accumulator_id,
            },
            "train_trials": int(design["train_trials"]),
            "trials_per_subject": int(design["trials_per_subject"]),
            "particle_count": int(design["particle_count"]),
            "seed_count": int(design["seed_count"]),
            "cohort_pairs": cohort_pairs,
            "provisional_controller": controller,
            "decision_is_final": False,
        }
        _atomic_csv(output / "variant_summary.csv", variant_df)
        _atomic_csv(output / "contrast_summary.csv", contrast_df)
        _atomic_csv(output / "paired_seed_effects.csv", seed_df)
        _atomic_json(output / "summary.json", summary)
        _write_readme(
            output,
            pd.DataFrame(manifest["projections"]),
            variant_df,
            contrast_df,
            summary,
        )


if __name__ == "__main__":
    main()
