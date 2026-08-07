#!/usr/bin/env python3
"""Paired mechanism ablations for the frozen Cond1 V14 subject configurations.

Every variant for a subject uses the same trajectory seeds.  The script changes
one mechanism at a time after applying the frozen subject-specific
hyperparameters, so deltas are interpretable as paired performance effects
rather than Monte-Carlo variation.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cond1_v14_fine import (  # noqa: E402
    GENERATED_CONFIG,
    MEMORY_PATH,
    TRANSITION_PATH,
)
from scripts.run_cond1_v14_frozen_confirmation import (  # noqa: E402
    SUBJECTS,
    state_mode,
    toggle_state,
)
from scripts.run_cond1_v14_pilot import result_row, write_json  # noqa: E402
from src.Bayesian_state.run_simulation import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.optimization.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.optimization.optimizer_simulation import (  # noqa: E402
    StateModelSimulationRunner,
)


SIM_CONFIG = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
OUTPUT_DIR = ROOT / "results/model_architecture_effect_audit/paired_ablations"
CHOICE_READOUT_PATH = "engine.choice_readout.kwargs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_frozen_config() -> dict[str, Any]:
    payload = yaml.safe_load(GENERATED_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {GENERATED_CONFIG}")
    return payload


def frozen_params(config: Mapping[str, Any], subject_id: int) -> dict[str, Any]:
    return deepcopy(
        config["subject_overrides"][str(subject_id)]["fixed_hyperparams"]
    )


def has_choice_informed_policy(params: Mapping[str, Any]) -> bool:
    transition = params[TRANSITION_PATH]
    profiles = (transition.get("strategy_controller") or {}).get("profiles", [])
    return any(
        profile.get("survivor_score") == "posterior_choice"
        or profile.get("newcomer_score") in {"recent_choice", "recent_error_choice"}
        for profile in profiles
    )


def remove_choice_evidence(params: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(params))
    profiles = (
        out[TRANSITION_PATH].get("strategy_controller") or {}
    ).get("profiles", [])
    for profile in profiles:
        if profile.get("survivor_score") == "posterior_choice":
            profile["survivor_score"] = "posterior"
        if profile.get("newcomer_score") in {"recent_choice", "recent_error_choice"}:
            profile["newcomer_score"] = "random"
    return out


def remove_controller_profile(
    params: Mapping[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    out = deepcopy(dict(params))
    controller = out[TRANSITION_PATH].get("strategy_controller") or {}
    profiles = list(controller.get("profiles", []))
    kept = [profile for profile in profiles if profile.get("id") != profile_id]
    if len(kept) != len(profiles) - 1:
        raise ValueError(
            f"Expected exactly one controller profile {profile_id!r}, "
            f"found {len(profiles) - len(kept)}"
        )
    controller["profiles"] = kept
    return out


def build_variants(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    baseline = deepcopy(dict(params))
    state_counterfactual, state_counterfactual_id = toggle_state(baseline)

    readout_expectation = deepcopy(baseline)
    readout_expectation[CHOICE_READOUT_PATH] = {"method": "expectation"}

    fade_only = deepcopy(baseline)
    fade_only[MEMORY_PATH]["w0"] = 0.0

    static_only = deepcopy(baseline)
    static_only[MEMORY_PATH]["w0"] = 1.0

    variants = [
        {
            "variant_id": "baseline_corrected",
            "model_family": "corrected_v14",
            "controller_id": "subject_frozen",
            "hyperparams": baseline,
            "engine_ablation": None,
        },
        {
            "variant_id": state_counterfactual_id,
            "model_family": "latent_state_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": state_counterfactual,
            "engine_ablation": None,
        },
        {
            "variant_id": "readout_expectation",
            "model_family": "readout_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": readout_expectation,
            "engine_ablation": None,
        },
        {
            "variant_id": "beta_static_5",
            "model_family": "beta_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": baseline,
            "engine_ablation": "beta_static_5",
        },
        {
            "variant_id": "memory_fade_only",
            "model_family": "memory_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": fade_only,
            "engine_ablation": None,
        },
        {
            "variant_id": "memory_static_only",
            "model_family": "memory_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": static_only,
            "engine_ablation": None,
        },
        {
            "variant_id": "perception_noise_off",
            "model_family": "perception_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": baseline,
            "engine_ablation": "perception_noise_off",
        },
        {
            "variant_id": "label_reversals_off",
            "model_family": "hypothesis_space_ablation",
            "controller_id": "subject_frozen",
            "hyperparams": baseline,
            "engine_ablation": "label_reversals_off",
        },
        {
            "variant_id": "dynamic_hypothesis_selection_off",
            "model_family": "active_set_ablation",
            "controller_id": "full_hypothesis_set",
            "hyperparams": baseline,
            "engine_ablation": "dynamic_hypothesis_selection_off",
        },
    ]

    if has_choice_informed_policy(baseline):
        variants.append(
            {
                "variant_id": "choice_evidence_off",
                "model_family": "choice_evidence_ablation",
                "controller_id": "subject_frozen",
                "hyperparams": remove_choice_evidence(baseline),
                "engine_ablation": None,
            }
        )
    for profile_id in ("conservative", "stable", "aggressive", "stubborn"):
        variants.append(
            {
                "variant_id": f"profile_{profile_id}_off",
                "model_family": "controller_profile_ablation",
                "controller_id": "subject_frozen",
                "hyperparams": remove_controller_profile(baseline, profile_id),
                "engine_ablation": None,
            }
        )
    return variants


def apply_engine_ablation(
    engine: dict[str, Any],
    ablation: str | None,
) -> dict[str, Any]:
    out = deepcopy(engine)
    if ablation is None:
        return out
    if ablation == "beta_static_5":
        # Keep the beta array interface because choice-informed transition
        # profiles query it.  Zero update rates and disable prior scaling so
        # every active hypothesis remains at beta=5; this isolates beta
        # evolution without accidentally ablating controller functionality.
        beta_kwargs = out["modules"]["beta_mod"].setdefault("kwargs", {})
        beta_kwargs.update(
            {
                "beta_init": 5.0,
                "decrease_rate": 0.0,
                "correct_additive": 0.0,
                "use_prior_scaling": False,
                "prior_beta_scale": 0.0,
            }
        )
        return out
    if ablation == "perception_noise_off":
        kwargs = out["modules"]["perception_mod"].setdefault("kwargs", {})
        kwargs.update(
            {
                "noise_mode": "normal",
                "mean": [0.0, 0.0, 0.0, 0.0],
                "std": [0.0, 0.0, 0.0, 0.0],
            }
        )
        return out
    if ablation == "label_reversals_off":
        out["partition"]["kwargs"]["include_label_reversals"] = False
        transition = out["modules"]["hypo_transitions_mod"].setdefault("kwargs", {})
        if transition.get("init_pool") == "label_permuted":
            transition["init_pool"] = "all"
        return out
    if ablation == "dynamic_hypothesis_selection_off":
        out["modules"].pop("hypo_transitions_mod", None)
        out["agenda"] = [
            name for name in out["agenda"] if name != "hypo_transitions_mod"
        ]
        return out
    raise ValueError(f"Unknown engine ablation: {ablation}")


def load_completed_rows(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "subjects").glob("subject_*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = dict(payload["row"])
        row["engine_ablation"] = payload.get("engine_ablation")
        row["frozen_state_mode"] = payload.get("frozen_state_mode")
        row["choice_informed_baseline"] = payload.get("choice_informed_baseline")
        rows.append(row)
    return pd.DataFrame(rows)


def write_rows(output_dir: Path) -> None:
    rows = load_completed_rows(output_dir)
    if not rows.empty:
        rows.sort_values(["subject_id", "variant_id"]).to_csv(
            output_dir / "paired_rows.csv", index=False
        )


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("--repeats and --n-jobs must be positive")

    frozen_config = load_frozen_config()
    sim_config = load_yaml(SIM_CONFIG)
    base_engine = resolve_engine_config(sim_config, SIM_CONFIG.parent)
    dataset_paths = resolve_dataset_paths(
        sim_config, SIM_CONFIG.parent, DEFAULT_DATA_PATH
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output_dir / "manifest.json",
        {
            "subjects": args.subjects,
            "simulation_repeats": args.repeats,
            "hyper_candidate_seed": 260726,
            "common_random_numbers_within_subject": True,
            "baseline": "frozen V14 configuration after causal-beta and memory-prior fixes",
            "one_mechanism_changed_per_variant": True,
            "frozen_source_config": str(GENERATED_CONFIG.relative_to(ROOT)),
            "state_counterfactual_note": (
                "Uses the pre-registered V14 matched toggle; state-on and state-off "
                "subjects therefore receive opposite counterfactuals."
            ),
        },
    )

    for subject_id in args.subjects:
        params = frozen_params(frozen_config, subject_id)
        variants = build_variants(params)
        choice_informed = has_choice_informed_policy(params)
        for variant in variants:
            output_path = (
                args.output_dir
                / "subjects"
                / f"subject_{subject_id}"
                / f"{variant['variant_id']}.json"
            )
            if output_path.exists() and not args.force:
                print(
                    f"SKIP subject={subject_id} variant={variant['variant_id']}",
                    flush=True,
                )
                continue

            engine = apply_fixed_hyperparams_to_engine_config(
                base_engine, variant["hyperparams"]
            )
            engine = apply_engine_ablation(engine, variant["engine_ablation"])
            runner = StateModelSimulationRunner(
                engine_config=engine,
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
                n_jobs=args.n_jobs,
            )
            runner.prepare_data(dataset_paths["learning_data"])
            print(
                f"RUN subject={subject_id} variant={variant['variant_id']}",
                flush=True,
            )
            result = runner.simulate_subject(
                subject_id=subject_id,
                simulation_repeats=args.repeats,
                fixed_hyperparams=variant["hyperparams"],
                window_size=16,
                keep_logs=False,
                prediction_mode="prior_t",
                selection_prediction_mode="prior_t",
                loss_metric="choice_brier",
                hyper_candidate_seed=260726,
                seed_hyperparams={
                    "paired_seed_group": f"architecture_audit_subject_{subject_id}"
                },
                statistics_config=sim_config.get("statistics_config"),
            )
            row = result_row(
                subject_id=subject_id,
                variant=variant,
                result=result,
            )
            write_json(
                output_path,
                {
                    "row": row,
                    "engine_ablation": variant["engine_ablation"],
                    "frozen_state_mode": state_mode(params),
                    "choice_informed_baseline": choice_informed,
                    "statistics": result["best"].statistics_summary,
                    "sample_errors": result["best"].sample_errors,
                    "hyperparams": variant["hyperparams"],
                },
            )
            write_rows(args.output_dir)
            print(
                f"DONE subject={subject_id} variant={variant['variant_id']} "
                f"brier={row['marginal_choice_brier']:.6f} "
                f"crps={row['trajectory_crps']:.6f}",
                flush=True,
            )

    write_rows(args.output_dir)
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
