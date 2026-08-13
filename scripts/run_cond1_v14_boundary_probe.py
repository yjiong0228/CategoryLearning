#!/usr/bin/env python3
"""Paired memory-boundary probe for the Cond1 V14 fine winners."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cond1_v14_fine import (  # noqa: E402
    BRIER_PATH,
    CANDIDATES,
    CRPS_PATH,
    GENERATED_CONFIG,
    MEMORY_PATH,
    identify_candidate,
)
from scripts.run_cond1_v14_pilot import result_row, summarize, write_json  # noqa: E402
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.optimization.objectives import (  # noqa: E402
    resolve_objective_order,
    select_best_by_objectives,
)
from src.Bayesian_state.simulation.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.runner import (  # noqa: E402
    StateModelSimulationRunner,
)


SIM_CONFIG = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
HYPER_CONFIG = ROOT / "configs/hyper_cd_cfg/pmh_cond1_hyper_cd_v14.yaml"
OUTPUT_DIR = ROOT / "results/cond1_v14/boundary_probe"
DOC_PATH = ROOT / "docs/model_v14_boundary_checkpoint.md"
BOUNDARY_GRIDS = {
    103: {"gamma": [0.10, 0.25], "w0": [0.005, 0.010]},
    112: {"gamma": [0.10, 0.25], "w0": [0.500, 0.750]},
    117: {"gamma": [0.70], "w0": [0.005, 0.010]},
    127: {"gamma": [0.10, 0.25], "w0": [0.050]},
    131: {"gamma": [0.50], "w0": [0.500, 0.750]},
}


def variant_id(gamma: float, w0: float) -> str:
    gamma_id = f"{gamma:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    w0_id = f"{w0:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"gamma_{gamma_id}_w0_{w0_id}"


def load_fine_config() -> dict[str, Any]:
    loaded = yaml.safe_load(GENERATED_CONFIG.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {GENERATED_CONFIG}")
    return loaded


def fixed_params(config: dict[str, Any], subject_id: int) -> dict[str, Any]:
    return deepcopy(
        config["subject_overrides"][str(subject_id)]["fixed_hyperparams"]
    )


def build_variants(
    params: dict[str, Any],
    *,
    subject_id: int,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = identify_candidate(params, candidates)
    variants = []
    grid = BOUNDARY_GRIDS[subject_id]
    baseline_memory = params[MEMORY_PATH]
    for gamma, w0 in product(grid["gamma"], grid["w0"]):
        candidate_params = deepcopy(params)
        candidate_params[MEMORY_PATH] = {"gamma": float(gamma), "w0": float(w0)}
        is_baseline = (
            float(gamma) == float(baseline_memory["gamma"])
            and float(w0) == float(baseline_memory["w0"])
        )
        variants.append(
            {
                "variant_id": (
                    "fine_boundary_baseline"
                    if is_baseline
                    else variant_id(float(gamma), float(w0))
                ),
                "model_family": "v14_memory_boundary",
                "controller_id": candidate["id"],
                "hyperparams": candidate_params,
            }
        )
    if sum(v["variant_id"] == "fine_boundary_baseline" for v in variants) != 1:
        raise ValueError(f"Subject {subject_id} boundary grid omitted fine baseline")
    return variants


def objective_values(row: dict[str, Any]) -> dict[str, float]:
    return {
        BRIER_PATH: float(row["marginal_choice_brier"]),
        CRPS_PATH: float(row["trajectory_crps"]),
    }


def choose_boundaries(
    *,
    fine_config: dict[str, Any],
    objective_specs: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = pd.read_csv(OUTPUT_DIR / "pilot_rows.csv")
    selected_rows: list[dict[str, Any]] = []
    contexts: dict[str, Any] = {}
    for subject_id in BOUNDARY_GRIDS:
        subject_rows = rows.loc[rows["subject_id"].eq(subject_id)].to_dict(
            orient="records"
        )
        selected, context = select_best_by_objectives(
            subject_rows,
            objective_values,
            objective_specs,
            tie_breaker=lambda row: str(row["variant_id"]),
        )
        baseline = next(
            row
            for row in subject_rows
            if row["variant_id"] == "fine_boundary_baseline"
        )
        selected["delta_brier_vs_boundary_baseline"] = (
            float(selected["marginal_choice_brier"])
            - float(baseline["marginal_choice_brier"])
        )
        selected["delta_crps_vs_boundary_baseline"] = (
            float(selected["trajectory_crps"]) - float(baseline["trajectory_crps"])
        )
        selected_rows.append(selected)
        contexts[str(subject_id)] = context

        params = json.loads(
            (
                OUTPUT_DIR
                / "subjects"
                / f"subject_{subject_id}"
                / f"{selected['variant_id']}.json"
            ).read_text()
        )["hyperparams"]
        memory = deepcopy(params[MEMORY_PATH])
        override = fine_config["subject_overrides"][str(subject_id)]
        override["fixed_hyperparams"][MEMORY_PATH] = memory
        override["engine_config"]["modules"]["memory_mod"]["kwargs"] = deepcopy(
            memory
        )

    selection = {
        "subjects": selected_rows,
        "selection_context": contexts,
        "objective_order": [
            {
                "path": spec.path,
                "rel_tolerance": spec.rel_tolerance,
                "abs_tolerance": spec.abs_tolerance,
                "scale_floor": spec.scale_floor,
                "anchor_guard": spec.anchor_guard,
            }
            for spec in objective_specs
        ],
    }
    return selected_rows, selection


def write_checkpoint_doc(selected: list[dict[str, Any]]) -> None:
    changed = sum(row["variant_id"] != "fine_boundary_baseline" for row in selected)
    lines = [
        "# Cond1 V14 memory-boundary checkpoint",
        "",
        "- Design: paired common-random-number probe around fine memory boundaries",
        "- Repeats: 512 per configuration",
        "- Candidate seed: 140016",
        f"- Fine memory settings changed: {changed}/{len(selected)}",
        "",
        "| Subject | Selected variant | Brier | Δ Brier vs fine boundary baseline | CRPS | Δ CRPS |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {int(row['subject_id'])} | {row['variant_id']} | "
            f"{float(row['marginal_choice_brier']):.6f} | "
            f"{float(row['delta_brier_vs_boundary_baseline']):+.6f} | "
            f"{float(row['trajectory_crps']):.6f} | "
            f"{float(row['delta_crps_vs_boundary_baseline']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "The generated subjectwise configuration now contains the selected "
            "memory settings. These choices are still part of model selection; "
            "performance must be measured in the subsequent frozen independent "
            "common-seed confirmation.",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("--repeats and --n-jobs must be positive")

    fine_config = load_fine_config()
    candidates = json.loads(CANDIDATES.read_text())["cond1_v14"]
    sim_config = load_yaml(SIM_CONFIG)
    base_engine = resolve_engine_config(sim_config, SIM_CONFIG.parent)
    dataset_paths = resolve_dataset_paths(
        sim_config, SIM_CONFIG.parent, DEFAULT_DATA_PATH
    )
    hyper_config = load_yaml(HYPER_CONFIG)
    objective_specs = resolve_objective_order(hyper_config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "subjects": list(BOUNDARY_GRIDS),
            "grids": BOUNDARY_GRIDS,
            "simulation_repeats": args.repeats,
            "common_random_numbers": True,
            "hyper_candidate_seed": 140016,
            "source_config": str(GENERATED_CONFIG.relative_to(ROOT)),
        },
    )

    for subject_id in BOUNDARY_GRIDS:
        params = fixed_params(fine_config, subject_id)
        for variant in build_variants(
            params, subject_id=subject_id, candidates=candidates
        ):
            path = (
                OUTPUT_DIR
                / "subjects"
                / f"subject_{subject_id}"
                / f"{variant['variant_id']}.json"
            )
            if path.exists() and not args.force:
                print(
                    f"SKIP subject={subject_id} variant={variant['variant_id']}",
                    flush=True,
                )
                continue
            print(
                f"RUN subject={subject_id} variant={variant['variant_id']}",
                flush=True,
            )
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
                hyper_candidate_seed=140016,
                seed_hyperparams={
                    "paired_seed_group": f"cond1_v14_boundary_subject_{subject_id}"
                },
                statistics_config=sim_config.get("statistics_config"),
            )
            row = result_row(subject_id=subject_id, variant=variant, result=result)
            write_json(
                path,
                {
                    "row": row,
                    "statistics": result["best"].statistics_summary,
                    "sample_errors": result["best"].sample_errors,
                    "hyperparams": variant["hyperparams"],
                },
            )
            summarize(OUTPUT_DIR)
            print(
                f"DONE subject={subject_id} variant={variant['variant_id']} "
                f"brier={row['marginal_choice_brier']:.6f} "
                f"crps={row['trajectory_crps']:.6f}",
                flush=True,
            )

    summarize(OUTPUT_DIR)
    selected, selection = choose_boundaries(
        fine_config=fine_config,
        objective_specs=objective_specs,
    )
    write_json(OUTPUT_DIR / "boundary_selection.json", selection)
    GENERATED_CONFIG.write_text(
        yaml.safe_dump(fine_config, sort_keys=False, allow_unicode=True)
    )
    write_checkpoint_doc(selected)
    print(f"COMPLETE output={OUTPUT_DIR}")
    print(f"UPDATED config={GENERATED_CONFIG}")
    print(f"WROTE checkpoint={DOC_PATH}")


if __name__ == "__main__":
    main()
