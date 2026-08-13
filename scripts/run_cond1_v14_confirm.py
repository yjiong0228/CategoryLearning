#!/usr/bin/env python3
"""Independent-seed confirmation of V14 pilot choices and state gain."""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_v14_pilot import (  # noqa: E402
    DEFAULT_SUBJECTS,
    READOUTS,
    candidate_kwargs,
    enable_v14_state,
    load_inputs,
    result_row,
    selected_hyperparams,
    summarize,
    write_json,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.simulation.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.runner import (  # noqa: E402
    StateModelSimulationRunner,
)


def readout_id(variant_id: str) -> str:
    for name in sorted(READOUTS, key=len, reverse=True):
        if variant_id.endswith(f"_{name}"):
            return name
    raise ValueError(f"Cannot infer readout from {variant_id}")


def pilot_selection(subject_id: int, pilot_rows: pd.DataFrame) -> dict[str, str]:
    rows = pilot_rows[
        (pilot_rows.subject_id == subject_id)
        & (pilot_rows.model_family.isin(["m2_core6", "m3_unified"]))
    ].sort_values(
        ["marginal_choice_brier", "trajectory_crps", "model_family"],
        ascending=[True, True, True],
    )
    if rows.empty:
        raise ValueError(f"No completed V14 pilot rows for subject {subject_id}")
    row = rows.iloc[0]
    return {
        "controller_id": str(row.controller_id),
        "readout_id": readout_id(str(row.variant_id)),
        "pilot_variant_id": str(row.variant_id),
    }


def build_variants(
    original: dict,
    candidates: list[dict],
    selection: dict[str, str],
    gains: list[float],
) -> list[dict]:
    selected_controller = candidate_kwargs(candidates, selection["controller_id"])
    readout = deepcopy(READOUTS[selection["readout_id"]])
    variants = [
        {
            "variant_id": "m0_v13_saved",
            "model_family": "m0_v13",
            "controller_id": "v13_subject_selected",
            "hyperparams": deepcopy(original),
        }
    ]
    off = deepcopy(original)
    off["engine.modules.hypo_transitions_mod.kwargs"] = deepcopy(selected_controller)
    off["engine.choice_readout.kwargs"] = deepcopy(readout)
    variants.append(
        {
            "variant_id": "m1_selected_state_off",
            "model_family": "m1_state_off",
            "controller_id": selection["controller_id"],
            "hyperparams": off,
        }
    )
    for gain in gains:
        params = deepcopy(off)
        params["engine.modules.hypo_transitions_mod.kwargs"] = enable_v14_state(
            selected_controller,
            error_gain=gain,
            decay=0.80,
            threshold=0.55,
        )
        gain_id = f"{gain:.2f}".replace(".", "p")
        variants.append(
            {
                "variant_id": f"m2_v14_gain_{gain_id}",
                "model_family": "m2_v14_gain",
                "controller_id": selection["controller_id"],
                "hyperparams": params,
            }
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--repeats", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--gains", type=float, nargs="+", default=[0.20, 0.35, 0.50])
    parser.add_argument(
        "--pilot-rows",
        type=Path,
        default=ROOT / "results/cond1_v14/pilot_state_readout/pilot_rows.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/cond1_v14/confirm_gain_readout",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    best, candidates, sim_path, sim_cfg = load_inputs()
    base_engine = resolve_engine_config(sim_cfg, sim_path.parent)
    dataset_paths = resolve_dataset_paths(sim_cfg, sim_path.parent, DEFAULT_DATA_PATH)
    pilot_rows = pd.read_csv(args.pilot_rows)
    write_json(
        args.output_dir / "manifest.json",
        {
            "subjects": args.subjects,
            "simulation_repeats": args.repeats,
            "hyper_candidate_seed": 140015,
            "independent_of_pilot_seed": True,
            "gains": args.gains,
            "selection_rule": [
                "lowest pilot marginal choice Brier",
                "then lowest trajectory CRPS",
                "then core6 before unified on exact ties",
            ],
        },
    )

    for subject_id in args.subjects:
        original = selected_hyperparams(best, subject_id)
        selection = pilot_selection(subject_id, pilot_rows)
        for variant in build_variants(original, candidates, selection, args.gains):
            path = (
                args.output_dir
                / "subjects"
                / f"subject_{subject_id}"
                / f"{variant['variant_id']}.json"
            )
            if path.exists() and not args.force:
                print(f"SKIP subject={subject_id} variant={variant['variant_id']}", flush=True)
                continue
            print(f"RUN subject={subject_id} variant={variant['variant_id']}", flush=True)
            engine = apply_fixed_hyperparams_to_engine_config(
                base_engine, variant["hyperparams"]
            )
            runner = StateModelSimulationRunner(
                engine,
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
                n_jobs=args.n_jobs,
            )
            runner.prepare_data(dataset_paths["learning_data"])
            result = runner.simulate_subject(
                subject_id,
                simulation_repeats=args.repeats,
                fixed_hyperparams=variant["hyperparams"],
                window_size=16,
                keep_logs=False,
                prediction_mode="prior_t",
                selection_prediction_mode="prior_t",
                loss_metric="choice_brier",
                hyper_candidate_seed=140015,
                seed_hyperparams={"paired_seed_group": "cond1_v14_confirm"},
            )
            row = result_row(
                subject_id=subject_id,
                variant=variant,
                result=result,
            )
            write_json(
                path,
                {
                    "row": row,
                    "pilot_selection": selection,
                    "statistics": result["best"].statistics_summary,
                    "sample_errors": result["best"].sample_errors,
                    "hyperparams": variant["hyperparams"],
                },
            )
            summarize(args.output_dir)
            print(
                f"DONE subject={subject_id} variant={variant['variant_id']} "
                f"marginal_brier={row['marginal_choice_brier']:.6f} "
                f"trajectory_crps={row['trajectory_crps']:.6f}",
                flush=True,
            )
    summarize(args.output_dir)
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
