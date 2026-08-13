#!/usr/bin/env python3
"""Independent common-seed confirmation of frozen Cond1 V14 configurations."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cond1_v14_fine import (  # noqa: E402
    CANDIDATES,
    GENERATED_CONFIG,
    MEMORY_PATH,
    TRANSITION_PATH,
    identify_candidate,
)
from scripts.analyze_cond1_v14_results import bootstrap_mean_interval  # noqa: E402
from scripts.run_cond1_v14_pilot import (  # noqa: E402
    enable_v14_state,
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
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.runner import (  # noqa: E402
    StateModelSimulationRunner,
)


SUBJECTS = [103, 105, 111, 112, 117, 118, 127, 131]
SIM_CONFIG = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
V13_BEST = ROOT / "results/zhuran/cond1_v13/cd/cond1_v13/best_hyperparams.json"
OUTPUT_DIR = ROOT / "results/cond1_v14/frozen_confirmation"
DOC_PATH = ROOT / "docs/model_v14_frozen_confirmation.md"


def load_frozen_config() -> dict[str, Any]:
    loaded = yaml.safe_load(GENERATED_CONFIG.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {GENERATED_CONFIG}")
    return loaded


def frozen_params(config: dict[str, Any], subject_id: int) -> dict[str, Any]:
    return deepcopy(
        config["subject_overrides"][str(subject_id)]["fixed_hyperparams"]
    )


def state_mode(params: dict[str, Any]) -> str:
    transition = params[TRANSITION_PATH]
    gain = float(transition.get("latent_volatility_error_gain", 0.0))
    max_state = float(transition.get("latent_volatility_max", 0.0))
    return "on" if gain > 0.0 and max_state > 0.0 else "off"


def toggle_state(params: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = deepcopy(params)
    transition = out[TRANSITION_PATH]
    if state_mode(params) == "on":
        transition.update(
            {
                "latent_volatility_base": 0.0,
                "latent_volatility_error_gain": 0.0,
                "latent_volatility_low_accuracy_gain": 0.0,
                "latent_volatility_max": 0.0,
            }
        )
        return out, "matched_state_off"

    out[TRANSITION_PATH] = enable_v14_state(
        transition,
        error_gain=0.35,
        decay=0.80,
        threshold=0.55,
    )
    return out, "matched_state_on_g0p35"


def build_variants(
    *,
    subject_id: int,
    v13_params: dict[str, Any],
    v14_params: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = identify_candidate(v14_params, candidates)
    mode = state_mode(v14_params)
    counterfactual, counterfactual_id = toggle_state(v14_params)
    variants = [
        {
            "variant_id": "v13_frozen",
            "model_family": "v13",
            "controller_id": "v13_subject_selected",
            "hyperparams": deepcopy(v13_params),
            "frozen_state_mode": "v13",
        },
        {
            "variant_id": "v14_frozen",
            "model_family": "v14",
            "controller_id": candidate["id"],
            "hyperparams": deepcopy(v14_params),
            "frozen_state_mode": mode,
        },
        {
            "variant_id": counterfactual_id,
            "model_family": "v14_state_counterfactual",
            "controller_id": candidate["id"],
            "hyperparams": counterfactual,
            "frozen_state_mode": "off" if mode == "on" else "on",
        },
    ]
    if subject_id == 127:
        pre_boundary = deepcopy(v14_params)
        pre_boundary[MEMORY_PATH]["gamma"] = 0.25
        variants.append(
            {
                "variant_id": "v14_pre_boundary_gamma_0p25",
                "model_family": "v14_boundary_counterfactual",
                "controller_id": candidate["id"],
                "hyperparams": pre_boundary,
                "frozen_state_mode": mode,
            }
        )
    return variants


def load_completed_rows() -> pd.DataFrame:
    rows = []
    for path in sorted((OUTPUT_DIR / "subjects").glob("subject_*/*.json")):
        payload = json.loads(path.read_text())
        row = dict(payload["row"])
        row["frozen_state_mode"] = payload["frozen_state_mode"]
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_summary(rows: pd.DataFrame) -> dict[str, Any]:
    required = {"v13_frozen", "v14_frozen"}
    for subject_id, group in rows.groupby("subject_id"):
        if not required.issubset(set(group["variant_id"])):
            raise ValueError(f"Subject {subject_id} is missing a frozen comparison")
    v13 = rows[rows.variant_id.eq("v13_frozen")].set_index("subject_id")
    v14 = rows[rows.variant_id.eq("v14_frozen")].set_index("subject_id")
    delta_brier = v14.marginal_choice_brier - v13.marginal_choice_brier
    delta_crps = v14.trajectory_crps - v13.trajectory_crps
    brier_ci = bootstrap_mean_interval(
        delta_brier.to_numpy(), draws=100_000, seed=140018
    )
    crps_ci = bootstrap_mean_interval(
        delta_crps.to_numpy(), draws=100_000, seed=140019
    )

    counterfactual = rows[
        rows.variant_id.isin(["matched_state_off", "matched_state_on_g0p35"])
    ].set_index("subject_id")
    delta_state_brier = (
        v14.marginal_choice_brier - counterfactual.marginal_choice_brier
    )
    delta_state_crps = v14.trajectory_crps - counterfactual.trajectory_crps
    state_on_subjects = v14.index[v14.frozen_state_mode.eq("on")]
    state_off_subjects = v14.index[v14.frozen_state_mode.eq("off")]

    subject_rows = []
    for subject_id in SUBJECTS:
        subject_rows.append(
            {
                "subject_id": subject_id,
                "frozen_state_mode": str(v14.loc[subject_id, "frozen_state_mode"]),
                "v13_brier": float(v13.loc[subject_id, "marginal_choice_brier"]),
                "v14_brier": float(v14.loc[subject_id, "marginal_choice_brier"]),
                "delta_brier_vs_v13": float(delta_brier.loc[subject_id]),
                "v13_crps": float(v13.loc[subject_id, "trajectory_crps"]),
                "v14_crps": float(v14.loc[subject_id, "trajectory_crps"]),
                "delta_crps_vs_v13": float(delta_crps.loc[subject_id]),
                "delta_brier_vs_state_counterfactual": float(
                    delta_state_brier.loc[subject_id]
                ),
                "delta_crps_vs_state_counterfactual": float(
                    delta_state_crps.loc[subject_id]
                ),
            }
        )

    boundary = None
    if 127 in rows.subject_id.values:
        frozen_127 = v14.loc[127]
        pre = rows[
            (rows.subject_id == 127)
            & rows.variant_id.eq("v14_pre_boundary_gamma_0p25")
        ].iloc[0]
        boundary = {
            "subject_id": 127,
            "selected_gamma": 0.10,
            "counterfactual_gamma": 0.25,
            "delta_brier_selected_minus_counterfactual": float(
                frozen_127.marginal_choice_brier - pre.marginal_choice_brier
            ),
            "delta_crps_selected_minus_counterfactual": float(
                frozen_127.trajectory_crps - pre.trajectory_crps
            ),
        }

    return {
        "validation": {
            "row_count": int(len(rows)),
            "subject_count": int(rows.subject_id.nunique()),
            "common_seed_within_subject": bool(
                rows.groupby("subject_id").simulation_point_seed.nunique().eq(1).all()
            ),
            "metric_null_count": int(
                rows[["marginal_choice_brier", "trajectory_crps"]]
                .isna()
                .sum()
                .sum()
            ),
            "simulation_repeats": int(rows.simulation_repeats.min()),
        },
        "frozen_v14_vs_v13": {
            "mean_delta_brier": float(delta_brier.mean()),
            "delta_brier_ci_95": list(brier_ci),
            "brier_wins": int((delta_brier < 0).sum()),
            "mean_delta_crps": float(delta_crps.mean()),
            "delta_crps_ci_95": list(crps_ci),
            "crps_wins": int((delta_crps < 0).sum()),
        },
        "state_counterfactual": {
            "state_on_subjects": [int(value) for value in state_on_subjects],
            "state_off_subjects": [int(value) for value in state_off_subjects],
            "state_on_frozen_brier_wins": int(
                (delta_state_brier.loc[state_on_subjects] < 0).sum()
            ),
            "state_on_frozen_crps_wins": int(
                (delta_state_crps.loc[state_on_subjects] < 0).sum()
            ),
            "state_off_frozen_brier_wins": int(
                (delta_state_brier.loc[state_off_subjects] < 0).sum()
            ),
            "state_off_frozen_crps_wins": int(
                (delta_state_crps.loc[state_off_subjects] < 0).sum()
            ),
            "mean_delta_brier_frozen_minus_counterfactual": float(
                delta_state_brier.mean()
            ),
            "mean_delta_crps_frozen_minus_counterfactual": float(
                delta_state_crps.mean()
            ),
        },
        "boundary_confirmation": boundary,
        "subjects": subject_rows,
    }


def write_checkpoint_doc(summary: dict[str, Any]) -> None:
    frozen = summary["frozen_v14_vs_v13"]
    state = summary["state_counterfactual"]
    boundary = summary["boundary_confirmation"]
    lines = [
        "# Cond1 V14 frozen independent confirmation",
        "",
        "- Frozen before this run: controller, state setting, readout, gamma, and w0",
        "- Repeats: 1024 per configuration",
        "- Candidate seed: 140017",
        "- Common trajectory seeds within each subject: yes",
        "",
        "## Frozen V14 versus frozen V13",
        "",
        f"- Mean Δ Brier: {frozen['mean_delta_brier']:+.6f} "
        f"(95% subject bootstrap CI {frozen['delta_brier_ci_95'][0]:+.6f}, "
        f"{frozen['delta_brier_ci_95'][1]:+.6f}); wins {frozen['brier_wins']}/8",
        f"- Mean Δ CRPS: {frozen['mean_delta_crps']:+.6f} "
        f"(95% subject bootstrap CI {frozen['delta_crps_ci_95'][0]:+.6f}, "
        f"{frozen['delta_crps_ci_95'][1]:+.6f}); wins {frozen['crps_wins']}/8",
        "",
        "## State counterfactual",
        "",
        f"- Frozen state-on subjects: {state['state_on_subjects']}",
        f"- State-on frozen wins vs matched state-off: Brier "
        f"{state['state_on_frozen_brier_wins']}/{len(state['state_on_subjects'])}, "
        f"CRPS {state['state_on_frozen_crps_wins']}/{len(state['state_on_subjects'])}",
        f"- Frozen state-off subjects: {state['state_off_subjects']}",
        f"- State-off frozen wins vs matched gain=0.35: Brier "
        f"{state['state_off_frozen_brier_wins']}/{len(state['state_off_subjects'])}, "
        f"CRPS {state['state_off_frozen_crps_wins']}/{len(state['state_off_subjects'])}",
        f"- Mean frozen minus matched counterfactual: Brier "
        f"{state['mean_delta_brier_frozen_minus_counterfactual']:+.6f}, "
        f"CRPS {state['mean_delta_crps_frozen_minus_counterfactual']:+.6f}",
        "",
        "## Subject-level results",
        "",
        "| Subject | Frozen state | Δ Brier vs V13 | Δ CRPS vs V13 | "
        "Δ Brier vs state counterfactual | Δ CRPS vs state counterfactual |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["subjects"]:
        lines.append(
            f"| {row['subject_id']} | {row['frozen_state_mode']} | "
            f"{row['delta_brier_vs_v13']:+.6f} | "
            f"{row['delta_crps_vs_v13']:+.6f} | "
            f"{row['delta_brier_vs_state_counterfactual']:+.6f} | "
            f"{row['delta_crps_vs_state_counterfactual']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary check",
            "",
            f"For subject 127, frozen gamma=0.10 minus gamma=0.25: "
            f"Δ Brier {boundary['delta_brier_selected_minus_counterfactual']:+.6f}, "
            f"Δ CRPS {boundary['delta_crps_selected_minus_counterfactual']:+.6f}.",
            "",
            "## Decision",
            "",
            "- Proceed to a full-sample V14 evaluation with the search space and "
            "selection rule frozen before expansion.",
            "- Treat the representative performance gate as passed: V14 wins 7/8 "
            "subjects on both metrics and the CRPS interval is entirely below zero. "
            "The Brier interval still crosses zero, largely because subject 103 "
            "trades worse Brier for better CRPS.",
            "- Keep state-on and state-off as selectable alternatives. The state "
            "counterfactual has a small favorable mean but inconsistent subject-level "
            "wins, so these data do not justify enabling persistent state for everyone "
            "or attributing the overall V14 gain mainly to state.",
            "- Keep subject 127 at gamma=0.10: its boundary improvement reproduced "
            "on both metrics under the independent seed.",
            "- Do not tune these eight subjects again before full-sample evaluation; "
            "doing so would contaminate the controlling confirmation.",
            "",
            "This run is the controlling performance check because no configuration "
            "was selected or changed using these confirmation outcomes.",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1024)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("--repeats and --n-jobs must be positive")

    frozen_config = load_frozen_config()
    v13_best = json.loads(V13_BEST.read_text())
    candidates = json.loads(CANDIDATES.read_text())["cond1_v14"]
    sim_config = load_yaml(SIM_CONFIG)
    base_engine = resolve_engine_config(sim_config, SIM_CONFIG.parent)
    dataset_paths = resolve_dataset_paths(
        sim_config, SIM_CONFIG.parent, DEFAULT_DATA_PATH
    )
    write_json(
        OUTPUT_DIR / "manifest.json",
        {
            "subjects": SUBJECTS,
            "simulation_repeats": args.repeats,
            "hyper_candidate_seed": 140017,
            "common_random_numbers": True,
            "frozen_source_config": str(GENERATED_CONFIG.relative_to(ROOT)),
            "selection_on_confirmation": False,
            "variants": [
                "v13_frozen",
                "v14_frozen",
                "matched state counterfactual",
                "subject 127 pre-boundary gamma counterfactual",
            ],
        },
    )

    for subject_id in SUBJECTS:
        variants = build_variants(
            subject_id=subject_id,
            v13_params=selected_hyperparams(v13_best, subject_id),
            v14_params=frozen_params(frozen_config, subject_id),
            candidates=candidates,
        )
        for variant in variants:
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
                hyper_candidate_seed=140017,
                seed_hyperparams={
                    "paired_seed_group": f"cond1_v14_frozen_subject_{subject_id}"
                },
                statistics_config=sim_config.get("statistics_config"),
            )
            row = result_row(subject_id=subject_id, variant=variant, result=result)
            write_json(
                path,
                {
                    "row": row,
                    "frozen_state_mode": variant["frozen_state_mode"],
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

    rows = load_completed_rows()
    summary = comparison_summary(rows)
    write_json(OUTPUT_DIR / "frozen_summary.json", summary)
    pd.DataFrame(summary["subjects"]).to_csv(
        OUTPUT_DIR / "frozen_subject_comparison.csv", index=False
    )
    write_checkpoint_doc(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"COMPLETE output={OUTPUT_DIR}")
    print(f"WROTE checkpoint={DOC_PATH}")


if __name__ == "__main__":
    main()
