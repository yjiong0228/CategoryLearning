#!/usr/bin/env python3
"""Paired V13/V14 pilot for the eight representative condition-1 subjects.

The pilot deliberately freezes each subject's V13 memory parameters.  It uses
common trajectory seeds across variants, so the comparison isolates the V14
controller-state and readout changes instead of Monte-Carlo seed noise.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.run_simulation import apply_fixed_hyperparams_to_engine_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.optimization.optimization_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
    recursive_to_builtin,
    resolve_engine_config,
)
from src.Bayesian_state.optimization.optimizer_simulation import StateModelSimulationRunner


DEFAULT_SUBJECTS = (103, 105, 111, 112, 117, 118, 127, 131)
CORE6 = {
    "c1_v13_stable_dominant",
    "c1_v13_choice_volatile_refresh",
    "c1_v13_conservative_heavy",
    "c1_v13_early_explore_late_stable",
    "c1_v13_error_choice_newcomer",
    "c1_v13_error_aggressive",
}
READOUTS = {
    "expectation": {"method": "expectation"},
    "sharp2": {"method": "sharpened_expectation", "power": 2.0},
    "sharp4": {"method": "sharpened_expectation", "power": 4.0},
    "map": {"method": "map_hypothesis"},
}
METRIC_FIELDS = (
    "mean_run_choice_brier",
    "marginal_choice_brier",
    "marginal_choice_nll",
    "trajectory_crps",
    "trajectory_mean_mae",
    "trajectory_median_mae",
    "trajectory_coverage_90",
    "trajectory_median_vol_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--models", choices=("core", "unified", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--error-gain", type=float, default=0.35)
    parser.add_argument("--decay", type=float, default=0.80)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_v14/pilot_state_readout",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], Path, dict[str, Any]]:
    best_path = ROOT / "results/zhuran/cond1_v13/cd/cond1_v13/best_hyperparams.json"
    candidate_path = (
        ROOT
        / "src/Bayesian_state/problems/modules/hypo_transition/candidates"
        / "hypo_transition_profile_v13_candidates.json"
    )
    sim_path = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v13.yaml"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))["cond1_v13"]
    sim_cfg = load_yaml(sim_path)
    return best, candidates, sim_path, sim_cfg


def selected_hyperparams(best: Mapping[str, Any], subject_id: int) -> dict[str, Any]:
    payload = best["per_subject_best"][str(subject_id)]
    return deepcopy(payload["selected"]["best_hyperparams"])


def candidate_id_for(
    hypo_kwargs: Mapping[str, Any], candidates: list[dict[str, Any]]
) -> str:
    matches = [
        str(candidate["id"])
        for candidate in candidates
        if candidate["hypo_transitions_kwargs"] == hypo_kwargs
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one V13 controller match, found {matches!r}")
    return matches[0]


def candidate_kwargs(candidates: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for candidate in candidates:
        if candidate["id"] == candidate_id:
            return deepcopy(candidate["hypo_transitions_kwargs"])
    raise KeyError(candidate_id)


def enable_v14_state(
    hypo_kwargs: Mapping[str, Any], *, error_gain: float, decay: float, threshold: float
) -> dict[str, Any]:
    """Turn controller error evidence into a persistent belief-instability state."""
    out = deepcopy(dict(hypo_kwargs))
    out.update(
        {
            "latent_volatility_base": 0.0,
            "latent_volatility_error_gain": float(error_gain),
            "latent_volatility_low_accuracy_gain": 0.0,
            "latent_volatility_threshold": float(threshold),
            "latent_volatility_window": 8,
            "latent_volatility_decay": float(decay),
            "latent_volatility_max": 1.0,
            "latent_volatility_feedback_mode": "exact",
            "latent_volatility_signal": "confidence_weighted_error",
            "latent_volatility_pressure_slope": 8.0,
        }
    )
    controller = out.get("state_controller") or {}
    aggressive_count = 0
    for profile in controller.get("states", []):
        if profile.get("policy_method") != "aggressive":
            continue
        aggressive_count += 1
        activation = profile.setdefault("activation", {})
        activation.pop("recent_error", None)
        activation["latent_volatility_pressure"] = 1.5
    if aggressive_count != 1:
        raise ValueError(f"Expected exactly one aggressive profile, found {aggressive_count}")
    return out


def variants_for_subject(
    hyperparams: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    *,
    models: str,
    error_gain: float,
    decay: float,
    threshold: float,
) -> list[dict[str, Any]]:
    original = deepcopy(dict(hyperparams))
    selected_kwargs = original["engine.modules.hypo_transitions_mod.kwargs"]
    selected_id = candidate_id_for(selected_kwargs, candidates)
    if selected_id not in CORE6:
        raise ValueError(f"Representative subject selected non-core controller {selected_id}")

    variants = [
        {
            "variant_id": "m0_v13_saved",
            "model_family": "m0_v13",
            "controller_id": selected_id,
            "hyperparams": original,
        }
    ]
    families: list[tuple[str, str, dict[str, Any]]] = []
    if models in {"core", "all"}:
        families.append(("m2_core6", selected_id, deepcopy(selected_kwargs)))
    if models in {"unified", "all"}:
        families.append(
            (
                "m3_unified",
                "c1_v13_stable_dominant",
                candidate_kwargs(candidates, "c1_v13_stable_dominant"),
            )
        )

    for family, controller_id, kwargs in families:
        v14_kwargs = enable_v14_state(
            kwargs,
            error_gain=error_gain,
            decay=decay,
            threshold=threshold,
        )
        for readout_id, readout in READOUTS.items():
            params = deepcopy(original)
            params["engine.modules.hypo_transitions_mod.kwargs"] = deepcopy(v14_kwargs)
            params["engine.choice_readout.kwargs"] = deepcopy(readout)
            variants.append(
                {
                    "variant_id": f"{family}_{readout_id}",
                    "model_family": family,
                    "controller_id": controller_id,
                    "hyperparams": params,
                }
            )
    return variants


def nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def result_row(
    *,
    subject_id: int,
    variant: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    best = result["best"]
    stats = best.statistics_summary or {}
    marginal = stats.get("marginal_prediction") or {}
    readout_cfg = variant["hyperparams"].get("engine.choice_readout.kwargs") or {}
    return {
        "subject_id": int(subject_id),
        "variant_id": variant["variant_id"],
        "model_family": variant["model_family"],
        "controller_id": variant["controller_id"],
        "readout": readout_cfg.get("method", "unknown"),
        "readout_power": readout_cfg.get("power", 1.0),
        "mean_run_choice_brier": float(best.mean_error),
        "marginal_choice_brier": float(marginal.get("choice_brier", float("nan"))),
        "marginal_choice_nll": float(marginal.get("choice_nll", float("nan"))),
        "trajectory_crps": float(marginal.get("trajectory_crps", float("nan"))),
        "trajectory_mean_mae": float(
            marginal.get("trajectory_mean_mae", float("nan"))
        ),
        "trajectory_median_mae": float(
            marginal.get("trajectory_median_mae", float("nan"))
        ),
        "trajectory_coverage_90": float(
            marginal.get("trajectory_coverage_90", float("nan"))
        ),
        "trajectory_median_vol_ratio": float(
            marginal.get("trajectory_median_vol_ratio", float("nan"))
        ),
        "simulation_repeats": int(best.simulation_repeats),
        "simulation_point_seed": int(best.simulation_point_seed),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(recursive_to_builtin(payload), ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def load_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_dir / "subjects").glob("subject_*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("row"), dict):
            row = dict(payload["row"])
            readout_cfg = (payload.get("hyperparams") or {}).get(
                "engine.choice_readout.kwargs"
            ) or {}
            row["readout"] = readout_cfg.get("method", row.get("readout", "unknown"))
            row["readout_power"] = readout_cfg.get("power", row.get("readout_power", 1.0))
            rows.append(row)
    return rows


def summarize(output_dir: Path) -> None:
    rows = load_rows(output_dir)
    if not rows:
        return
    csv_path = output_dir / "pilot_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = {
        int(row["subject_id"]): row
        for row in rows
        if row["variant_id"] == "m0_v13_saved"
    }
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_id"]), []).append(row)
    groups = []
    for variant_id, group_rows in sorted(by_variant.items()):
        item: dict[str, Any] = {
            "variant_id": variant_id,
            "subject_count": len(group_rows),
            "model_family": group_rows[0]["model_family"],
        }
        for field in METRIC_FIELDS:
            values = [float(row[field]) for row in group_rows]
            finite = [value for value in values if value == value]
            item[f"mean_{field}"] = mean(finite) if finite else float("nan")
            paired = [
                float(row[field]) - float(baseline[int(row["subject_id"])][field])
                for row in group_rows
                if int(row["subject_id"]) in baseline
                and float(row[field]) == float(row[field])
                and float(baseline[int(row["subject_id"])][field])
                == float(baseline[int(row["subject_id"])][field])
            ]
            item[f"paired_delta_{field}"] = mean(paired) if paired else float("nan")
        groups.append(item)
    groups.sort(key=lambda item: item["mean_marginal_choice_brier"])
    write_json(
        output_dir / "aggregate_summary.json",
        {
            "ranking_rule": "mean marginal choice Brier; trajectory CRPS is a guardrail",
            "rows_completed": len(rows),
            "groups": groups,
        },
    )


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("--repeats and --n-jobs must be positive")
    best, candidates, sim_path, sim_cfg = load_inputs()
    base_engine = resolve_engine_config(sim_cfg, sim_path.parent)
    dataset_paths = resolve_dataset_paths(sim_cfg, sim_path.parent, DEFAULT_DATA_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "subjects": args.subjects,
            "models": args.models,
            "simulation_repeats": args.repeats,
            "common_random_numbers": True,
            "state": {
                "signal": "confidence_weighted_error",
                "error_gain": args.error_gain,
                "decay": args.decay,
                "threshold": args.threshold,
                "max": 1.0,
                "pressure_slope": 8.0,
                "aggressive_pressure_weight": 1.5,
            },
            "readouts": READOUTS,
            "core6": sorted(CORE6),
        },
    )

    for subject_id in args.subjects:
        subject_hyperparams = selected_hyperparams(best, subject_id)
        variants = variants_for_subject(
            subject_hyperparams,
            candidates,
            models=args.models,
            error_gain=args.error_gain,
            decay=args.decay,
            threshold=args.threshold,
        )
        for variant in variants:
            variant_path = (
                args.output_dir
                / "subjects"
                / f"subject_{subject_id}"
                / f"{variant['variant_id']}.json"
            )
            if variant_path.exists() and not args.force:
                print(f"SKIP subject={subject_id} variant={variant['variant_id']}", flush=True)
                continue
            print(f"RUN subject={subject_id} variant={variant['variant_id']}", flush=True)
            engine = apply_fixed_hyperparams_to_engine_config(
                base_engine, variant["hyperparams"]
            )
            runner = StateModelSimulationRunner(
                engine_config=engine,
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
                n_jobs=args.n_jobs,
            )
            runner.prepare_data(dataset_paths["learning_data"])
            result = runner.simulate_subject(
                subject_id=subject_id,
                simulation_repeats=args.repeats,
                fixed_hyperparams=variant["hyperparams"],
                window_size=16,
                keep_logs=False,
                prediction_mode="prior_t",
                selection_prediction_mode="prior_t",
                loss_metric="choice_brier",
                hyper_candidate_seed=140014,
                seed_hyperparams={"paired_seed_group": "cond1_v14_pilot"},
                statistics_config=sim_cfg.get("statistics_config"),
            )
            row = result_row(subject_id=subject_id, variant=variant, result=result)
            write_json(
                variant_path,
                {
                    "row": row,
                    "statistics": result["best"].statistics_summary,
                    "sample_errors": result["best"].sample_errors,
                    "hyperparams": variant["hyperparams"],
                },
            )
            summarize(args.output_dir)
            print(
                "DONE "
                f"subject={subject_id} variant={variant['variant_id']} "
                f"marginal_brier={row['marginal_choice_brier']:.6f} "
                f"trajectory_crps={row['trajectory_crps']:.6f}",
                flush=True,
            )
    summarize(args.output_dir)
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
