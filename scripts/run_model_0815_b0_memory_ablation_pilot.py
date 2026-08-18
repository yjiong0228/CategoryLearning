#!/usr/bin/env python3
"""Run the nested M-off, leaky-memory, and dual-memory architecture pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0815_b0_adaptive_controller_pilot import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _load_panel,
    _load_subject_base_engine,
    _mean_js,
    _relative,
    _repo_path,
    _run_panel,
    _sha256,
    summarize_variant,
    validate_minimal_adaptive_engine,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_b0_memory_ablation_pilot.yaml"
)
BAYES_ONLY_CLASS = "src.Bayesian_state.model.modules.memory.BayesianMemoryModule"
DUAL_MEMORY_CLASS = "src.Bayesian_state.model.modules.memory.DualMemoryModule"
SEED_ROLE = "model0815_b0_memory_ablation_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _flatten(root: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in root.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(_flatten(value, path))
        else:
            output[path] = value
    return output


def _candidate_id(family: str, gamma: float | None, w0: float | None) -> str:
    if family == "m_off":
        return "m_off"
    if gamma is None or w0 is None:
        raise ValueError(f"{family} requires gamma and w0")
    gamma_code = int(round(100.0 * float(gamma)))
    if family == "m_leaky":
        return f"m_leaky_g{gamma_code:03d}"
    w0_code = int(round(100.0 * float(w0)))
    return f"m_dual_g{gamma_code:03d}_w{w0_code:03d}"


def build_candidate_bank(
    config: Mapping[str, Any], *, smoke: bool = False
) -> list[dict[str, Any]]:
    families = config["memory_families"]
    feedback_gain = float(families.get("feedback_gain", 1.0))
    leaky_gammas = [float(value) for value in families["m_leaky"]["gamma_values"]]
    dual_gammas = [float(value) for value in families["m_dual"]["gamma_values"]]
    dual_w0 = [float(value) for value in families["m_dual"]["w0_values"]]
    if smoke:
        leaky_gammas = [min(leaky_gammas, key=lambda value: abs(value - 0.8))]
        dual_gammas = [min(dual_gammas, key=lambda value: abs(value - 0.8))]
        dual_w0 = [min(dual_w0, key=lambda value: abs(value - 0.15))]

    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "m_off",
            "family": "m_off",
            "gamma": None,
            "w0": None,
            "feedback_gain": None,
        }
    ]
    for gamma in leaky_gammas:
        candidates.append(
            {
                "candidate_id": _candidate_id("m_leaky", gamma, 0.0),
                "family": "m_leaky",
                "gamma": gamma,
                "w0": 0.0,
                "feedback_gain": feedback_gain,
            }
        )
    for gamma in dual_gammas:
        for w0 in dual_w0:
            candidates.append(
                {
                    "candidate_id": _candidate_id("m_dual", gamma, w0),
                    "family": "m_dual",
                    "gamma": gamma,
                    "w0": w0,
                    "feedback_gain": feedback_gain,
                }
            )
    identifiers = [str(value["candidate_id"]) for value in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("memory candidate identifiers must be unique")
    if any(
        value["family"] == "m_dual" and not 0.0 < float(value["w0"]) < 1.0
        for value in candidates
    ):
        raise ValueError("dual-memory candidates require an interior w0")
    if any(
        value["family"] != "m_off" and not 0.0 < float(value["gamma"]) < 1.0
        for value in candidates
    ):
        raise ValueError("screened forgetting candidates require gamma in (0, 1)")
    return candidates


def build_memory_engine(
    base_engine: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Change only the optional M mechanism in the common B0 engine."""
    validate_minimal_adaptive_engine(base_engine)
    engine = deepcopy(dict(base_engine))
    family = str(candidate["family"])
    if family == "m_off":
        engine["modules"]["memory_mod"] = {
            "class": BAYES_ONLY_CLASS,
            "kwargs": {},
        }
    elif family in {"m_leaky", "m_dual"}:
        w0 = float(candidate["w0"])
        if family == "m_leaky" and w0 != 0.0:
            raise ValueError("single-channel leaky memory requires w0=0")
        if family == "m_dual" and not 0.0 < w0 < 1.0:
            raise ValueError("dual memory requires an interior w0")
        engine["modules"]["memory_mod"] = {
            "class": DUAL_MEMORY_CLASS,
            "kwargs": {
                "gamma": float(candidate["gamma"]),
                "w0": w0,
                "feedback_gain": float(candidate["feedback_gain"]),
            },
        }
    else:
        raise ValueError(f"unknown memory family: {family!r}")

    base_flat = _flatten(base_engine)
    candidate_flat = _flatten(engine)
    changed = {
        path
        for path in set(base_flat) | set(candidate_flat)
        if base_flat.get(path) != candidate_flat.get(path)
    }
    illegal = [path for path in changed if not path.startswith("modules.memory_mod.")]
    if illegal:
        raise RuntimeError(f"memory ablation changed non-M paths: {illegal}")
    return engine


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
        raise ValueError("train_trials must split the selected trial sequence")
    if int(design["seed_count"]) < 2:
        raise ValueError("memory screen requires at least two paired PF seeds")
    return design


def _segment_masks(valid: np.ndarray, train_trials: int) -> dict[str, np.ndarray]:
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    index = np.arange(valid.size)
    train = valid & (index < int(train_trials))
    heldout = valid & (index >= int(train_trials))
    heldout_indices = np.flatnonzero(heldout)
    early_n = min(16, max(1, heldout_indices.size // 2))
    early = np.zeros(valid.size, dtype=bool)
    early[heldout_indices[:early_n]] = True
    late = heldout & ~early
    if not np.any(train) or not np.any(heldout):
        raise ValueError("memory train/heldout split contains an empty segment")
    return {
        "train": train,
        "heldout": heldout,
        "early_heldout": early,
        "late_heldout": late,
    }


def select_family_candidate(
    candidate_rows: pd.DataFrame, family: str
) -> Mapping[str, Any]:
    eligible = candidate_rows.loc[candidate_rows["family"] == family].copy()
    if eligible.empty:
        raise ValueError(f"candidate table has no family {family!r}")
    eligible = eligible.sort_values(
        ["ensemble_nll_train", "candidate_id"], kind="mergesort"
    )
    return eligible.iloc[0].to_dict()


def summarize_memory_contrast(
    comparator_row: Mapping[str, Any],
    comparator: Mapping[str, np.ndarray],
    mechanism_row: Mapping[str, Any],
    mechanism: Mapping[str, np.ndarray],
    *,
    contrast: Mapping[str, Any],
    train_trials: int,
    practical_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.array_equal(comparator["filter_seed"], mechanism["filter_seed"]):
        raise ValueError("memory variants do not share ordered PF seeds")
    if not np.array_equal(
        comparator["observed_choice_index"], mechanism["observed_choice_index"]
    ) or not np.array_equal(
        comparator["valid_trial_mask"], mechanism["valid_trial_mask"]
    ):
        raise ValueError("memory variants do not share observed trials")

    masks = _segment_masks(comparator["valid_trial_mask"], train_trials)
    output: dict[str, Any] = {
        "subject_id": int(comparator_row["subject_id"]),
        "contrast_id": str(contrast["contrast_id"]),
        "scientific_question": str(contrast["scientific_question"]),
        "comparator_family": str(contrast["comparator_family"]),
        "mechanism_family": str(contrast["mechanism_family"]),
        "comparator_candidate_id": str(comparator_row["candidate_id"]),
        "mechanism_candidate_id": str(mechanism_row["candidate_id"]),
        "particle_count": int(comparator_row["particle_count"]),
        "seed_count": int(comparator_row["seed_count"]),
    }
    seed_rows: list[dict[str, Any]] = []
    for segment in ("train", "heldout", "early_heldout", "late_heldout"):
        delta = np.asarray(comparator[f"run_nll_{segment}"], dtype=float) - np.asarray(
            mechanism[f"run_nll_{segment}"], dtype=float
        )
        sd = float(np.std(delta, ddof=1))
        output[f"paired_delta_nll_{segment}"] = float(np.mean(delta))
        output[f"paired_delta_nll_sd_{segment}"] = sd
        output[f"paired_delta_nll_mcse_{segment}"] = float(sd / np.sqrt(delta.size))
        output[f"ensemble_delta_nll_{segment}"] = float(
            comparator_row[f"ensemble_nll_{segment}"]
        ) - float(mechanism_row[f"ensemble_nll_{segment}"])
        output[f"positive_seed_fraction_{segment}"] = float(np.mean(delta > 0.0))
        for index, seed in enumerate(comparator["filter_seed"]):
            seed_rows.append(
                {
                    "subject_id": int(comparator_row["subject_id"]),
                    "contrast_id": str(contrast["contrast_id"]),
                    "segment": segment,
                    "repeat_index": int(index),
                    "filter_seed": int(seed),
                    "comparator_nll": float(comparator[f"run_nll_{segment}"][index]),
                    "mechanism_nll": float(mechanism[f"run_nll_{segment}"][index]),
                    "paired_delta_nll": float(delta[index]),
                }
            )

    heldout = masks["heldout"]
    output["heldout_choice_probability_rmse"] = float(
        np.sqrt(
            np.mean(
                np.square(
                    comparator["mean_choice_probability"][heldout]
                    - mechanism["mean_choice_probability"][heldout]
                )
            )
        )
    )
    output["heldout_predictive_geometry_prior_js"] = _mean_js(
        comparator["mean_marginal_prior"],
        mechanism["mean_marginal_prior"],
        heldout,
    )
    threshold = float(
        practical_fraction * float(comparator_row["ensemble_nll_heldout"])
    )
    output["practical_delta_threshold"] = threshold
    output["exceeds_positive_practical_threshold"] = bool(
        float(output["paired_delta_nll_heldout"]) > threshold
    )
    return output, seed_rows


def _write_readme(
    output: Path,
    selections: pd.DataFrame,
    overall_selections: pd.DataFrame,
    contrasts: pd.DataFrame,
    overall_contrasts: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# Model 0815 B0 M-module ablation pilot",
        "",
        "## Model semantics",
        "",
        "- `m_off`: mandatory Bayes-only posterior update; the optional M mechanism is absent.",
        "- `m_leaky`: one fading channel, `w0=0`, with forgetting controlled by `gamma`.",
        "- `m_dual`: fading and static channels mixed by an interior `w0`.",
        "",
        (
            "Every family uses the same provisional adaptive H controller and the same "
            "P, likelihood, beta, execution, readout, and lapse settings. Candidate "
            "selection uses only the training segment; all reported primary effects use "
            "held-out trials."
        ),
        "",
        "## Training-selected candidates",
        "",
        "| subject | family | candidate | gamma | w0 | train NLL | held-out NLL |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in selections.to_dict(orient="records"):
        gamma = "—" if pd.isna(row["gamma"]) else f"{float(row['gamma']):.2f}"
        w0 = "—" if pd.isna(row["w0"]) else f"{float(row['w0']):.2f}"
        lines.append(
            f"| {int(row['subject_id'])} | {row['family']} | {row['candidate_id']} | "
            f"{gamma} | {w0} | {float(row['ensemble_nll_train']):.5f} | "
            f"{float(row['ensemble_nll_heldout']):.5f} |"
        )
    lines.extend(
        [
            "",
            "## Overall train-selected architecture",
            "",
            "| subject | selected family | candidate | train NLL | held-out NLL | held-out delta vs M-off |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    overall_by_subject = overall_contrasts.set_index("subject_id")
    for row in overall_selections.to_dict(orient="records"):
        effect = overall_by_subject.loc[int(row["subject_id"])]
        lines.append(
            f"| {int(row['subject_id'])} | {row['family']} | {row['candidate_id']} | "
            f"{float(row['ensemble_nll_train']):.5f} | "
            f"{float(row['ensemble_nll_heldout']):.5f} | "
            f"{float(effect['paired_delta_nll_heldout']):+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Held-out paired contrasts",
            "",
            "| subject | contrast | delta NLL | MCSE | practical | geometry JS |",
            "|---:|---|---:|---:|:---:|---:|",
        ]
    )
    for row in contrasts.to_dict(orient="records"):
        lines.append(
            "| {subject_id} | {contrast_id} | {delta:+.5f} | {mcse:.5f} | "
            "{practical} | {js:.5f} |".format(
                subject_id=int(row["subject_id"]),
                contrast_id=row["contrast_id"],
                delta=float(row["paired_delta_nll_heldout"]),
                mcse=float(row["paired_delta_nll_mcse_heldout"]),
                practical=(
                    "yes" if row["exceeds_positive_practical_threshold"] else "no"
                ),
                js=float(row["heldout_predictive_geometry_prior_js"]),
            )
        )
    lines.extend(["", "## Cohort screen", ""])
    for contrast_id, values in summary["cohort_by_contrast"].items():
        lines.append(
            f"- `{contrast_id}`: mean delta NLL {values['mean_paired_delta_nll_heldout']:+.5f}; "
            f"positive {values['positive_subject_count']}/{values['subject_count']}; "
            f"practically positive {values['practically_positive_subject_count']}/{values['subject_count']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                f"This is a low-cost architecture screen with {summary['particle_count']} "
                f"particles and {summary['seed_count']} paired PF seeds. It is designed "
                "to identify clearly redundant M levels before "
                "spending on numerical precision; it is not a final memory-model or "
                "parameter decision."
            ),
            "",
            "Raw panels, candidate rankings, selected-family contrasts, resolved engines, and hashes are retained.",
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
    candidates = build_candidate_bank(config, smoke=bool(args.smoke))
    output = _repo_path(args.output_dir) if args.output_dir else _repo_path(config["output_dir"])
    if args.smoke:
        output = output / "smoke"
    if (output / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    simulation_config_path = _repo_path(config["base_simulation_config"])
    simulation_config = load_yaml(simulation_config_path)
    run_rows: list[dict[str, Any]] = []

    if args.phase in {"run", "all"}:
        for subject_id in design["subjects"]:
            base_engine, fixed_hyperparams = _load_subject_base_engine(
                simulation_config_path, simulation_config, subject_id
            )
            for candidate in candidates:
                engine = build_memory_engine(base_engine, candidate)
                metadata = _run_panel(
                    simulation_config_path=simulation_config_path,
                    simulation_config=simulation_config,
                    output=output,
                    subject_id=subject_id,
                    variant_id=str(candidate["candidate_id"]),
                    engine=engine,
                    fixed_hyperparams=fixed_hyperparams,
                    particle_count=int(design["particle_count"]),
                    seed_count=int(design["seed_count"]),
                    trials_per_subject=int(design["trials_per_subject"]),
                    base_seed=int(design["base_seed"]),
                    n_jobs=int(design["n_jobs"]),
                    seed_role=SEED_ROLE,
                )
                run_rows.append({**metadata, **candidate})
        _atomic_json(
            output / "run_manifest.json",
            {
                "analysis_id": config["analysis_id"],
                "config": _relative(config_path),
                "config_sha256": _sha256(config_path),
                "base_simulation_config": _relative(simulation_config_path),
                "base_simulation_config_sha256": _sha256(simulation_config_path),
                "design": design,
                "candidates": candidates,
                "runs": run_rows,
            },
        )

    if args.phase in {"summarize", "all"}:
        manifest_path = output / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = {
            (int(row["subject_id"]), str(row["candidate_id"])): row
            for row in manifest["runs"]
        }
        candidate_rows: list[dict[str, Any]] = []
        arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        for subject_id in design["subjects"]:
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                meta = metadata[(int(subject_id), candidate_id)]
                panel = _load_panel(_repo_path(meta["npz_path"]))
                row, current_arrays = summarize_variant(
                    panel,
                    subject_id=subject_id,
                    variant_id=candidate_id,
                    particle_count=int(design["particle_count"]),
                    train_trials=int(design["train_trials"]),
                )
                candidate_rows.append(
                    {
                        **row,
                        "candidate_id": candidate_id,
                        "family": str(candidate["family"]),
                        "gamma": candidate["gamma"],
                        "w0": candidate["w0"],
                        "feedback_gain": candidate["feedback_gain"],
                    }
                )
                arrays[(int(subject_id), candidate_id)] = current_arrays
        candidate_df = pd.DataFrame(candidate_rows)

        selection_rows: list[dict[str, Any]] = []
        overall_selection_rows: list[dict[str, Any]] = []
        selected: dict[tuple[int, str], Mapping[str, Any]] = {}
        for subject_id in design["subjects"]:
            subject_rows = candidate_df.loc[candidate_df["subject_id"] == subject_id]
            for family in ("m_off", "m_leaky", "m_dual"):
                chosen = select_family_candidate(subject_rows, family)
                selection_rows.append(dict(chosen))
                selected[(int(subject_id), family)] = chosen
            overall = (
                subject_rows.sort_values(
                    ["ensemble_nll_train", "candidate_id"], kind="mergesort"
                )
                .iloc[0]
                .to_dict()
            )
            overall_selection_rows.append(dict(overall))
        selection_df = pd.DataFrame(selection_rows)
        overall_selection_df = pd.DataFrame(overall_selection_rows)

        contrast_rows: list[dict[str, Any]] = []
        seed_rows: list[dict[str, Any]] = []
        for subject_id in design["subjects"]:
            for contrast in config["contrasts"]:
                comparator_row = selected[
                    (int(subject_id), str(contrast["comparator_family"]))
                ]
                mechanism_row = selected[
                    (int(subject_id), str(contrast["mechanism_family"]))
                ]
                comparator_id = str(comparator_row["candidate_id"])
                mechanism_id = str(mechanism_row["candidate_id"])
                row, current_seed_rows = summarize_memory_contrast(
                    comparator_row,
                    arrays[(int(subject_id), comparator_id)],
                    mechanism_row,
                    arrays[(int(subject_id), mechanism_id)],
                    contrast=contrast,
                    train_trials=int(design["train_trials"]),
                    practical_fraction=float(
                        config["screening_rule"][
                            "practical_fraction_of_comparator_heldout_nll"
                        ]
                    ),
                )
                contrast_rows.append(row)
                seed_rows.extend(current_seed_rows)
        contrast_df = pd.DataFrame(contrast_rows)
        seed_df = pd.DataFrame(seed_rows)

        overall_contrast_rows: list[dict[str, Any]] = []
        overall_seed_rows: list[dict[str, Any]] = []
        for overall in overall_selection_rows:
            subject_id = int(overall["subject_id"])
            comparator = selected[(subject_id, "m_off")]
            overall_contrast = {
                "contrast_id": "train_selected_architecture_vs_M_off",
                "comparator_family": "m_off",
                "mechanism_family": str(overall["family"]),
                "scientific_question": "does_train_selected_optional_M_generalize",
            }
            row, current_seed_rows = summarize_memory_contrast(
                comparator,
                arrays[(subject_id, str(comparator["candidate_id"]))],
                overall,
                arrays[(subject_id, str(overall["candidate_id"]))],
                contrast=overall_contrast,
                train_trials=int(design["train_trials"]),
                practical_fraction=float(
                    config["screening_rule"][
                        "practical_fraction_of_comparator_heldout_nll"
                    ]
                ),
            )
            overall_contrast_rows.append(row)
            overall_seed_rows.extend(current_seed_rows)
        overall_contrast_df = pd.DataFrame(overall_contrast_rows)
        overall_seed_df = pd.DataFrame(overall_seed_rows)

        cohort_by_contrast: dict[str, Any] = {}
        for contrast_id, frame in contrast_df.groupby("contrast_id", sort=False):
            cohort_by_contrast[str(contrast_id)] = {
                "subject_count": int(len(frame)),
                "mean_paired_delta_nll_train": float(
                    np.mean(frame["paired_delta_nll_train"])
                ),
                "mean_paired_delta_nll_heldout": float(
                    np.mean(frame["paired_delta_nll_heldout"])
                ),
                "mean_ensemble_delta_nll_heldout": float(
                    np.mean(frame["ensemble_delta_nll_heldout"])
                ),
                "positive_subject_count": int(
                    np.sum(frame["paired_delta_nll_heldout"] > 0.0)
                ),
                "practically_positive_subject_count": int(
                    np.sum(frame["exceeds_positive_practical_threshold"])
                ),
                "mean_heldout_geometry_prior_js": float(
                    np.mean(frame["heldout_predictive_geometry_prior_js"])
                ),
            }
        summary = {
            "analysis_id": config["analysis_id"],
            "subjects": [int(value) for value in design["subjects"]],
            "train_trials": int(design["train_trials"]),
            "trials_per_subject": int(design["trials_per_subject"]),
            "particle_count": int(design["particle_count"]),
            "seed_count": int(design["seed_count"]),
            "candidate_count": int(len(candidates)),
            "cohort_by_contrast": cohort_by_contrast,
            "overall_train_selected_architecture": {
                "family_counts": {
                    str(key): int(value)
                    for key, value in overall_selection_df["family"]
                    .value_counts()
                    .items()
                },
                "mean_paired_delta_nll_heldout_vs_M_off": float(
                    np.mean(overall_contrast_df["paired_delta_nll_heldout"])
                ),
                "mean_ensemble_delta_nll_heldout_vs_M_off": float(
                    np.mean(overall_contrast_df["ensemble_delta_nll_heldout"])
                ),
            },
            "decision_is_final": False,
        }
        _atomic_csv(output / "candidate_summary.csv", candidate_df)
        _atomic_csv(output / "training_selected_candidates.csv", selection_df)
        _atomic_csv(
            output / "overall_training_selected_candidates.csv",
            overall_selection_df,
        )
        _atomic_csv(output / "contrast_summary.csv", contrast_df)
        _atomic_csv(output / "paired_seed_effects.csv", seed_df)
        _atomic_csv(
            output / "overall_selection_vs_M_off.csv", overall_contrast_df
        )
        _atomic_csv(
            output / "overall_selection_paired_seed_effects.csv", overall_seed_df
        )
        _atomic_json(output / "summary.json", summary)
        _write_readme(
            output,
            selection_df,
            overall_selection_df,
            contrast_df,
            overall_contrast_df,
            summary,
        )


if __name__ == "__main__":
    main()
