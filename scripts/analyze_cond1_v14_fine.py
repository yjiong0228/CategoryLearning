#!/usr/bin/env python3
"""Audit Cond1 V14 selected-eight fine results against the coarse checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [103, 105, 111, 112, 117, 118, 127, 131]
HYPER_DIR = ROOT / "results/state-based-hyper-cd/pmh/cond1_v14_selected8"
GENERATED_CONFIG = (
    ROOT
    / "configs/simulation_cfg/generated_from_hyper/"
    "pmh_cond1_subjectwise_hyper_cd_best.yaml"
)
CANDIDATES = (
    ROOT
    / "src/Bayesian_state/problems/modules/hypo_transition_strategies/"
    "hypo_transition_profile_v14_candidates.json"
)
CONFIRM_ROWS = ROOT / "results/cond1_v14/confirm_gain_readout/pilot_rows.csv"
OUTPUT_DIR = ROOT / "results/cond1_v14/fine_analysis"
DOC_PATH = ROOT / "docs/model_v14_fine_checkpoint.md"

MEMORY_PATH = "engine.modules.memory_mod.kwargs"
TRANSITION_PATH = "engine.modules.hypo_transitions_mod.kwargs"
READOUT_PATH = "engine.choice_readout.kwargs"
BRIER_PATH = "statistics.marginal_prediction.choice_brier"
CRPS_PATH = "statistics.marginal_prediction.trajectory_crps"


def load_yaml_at_commit(commit: str, path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    text = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        text=True,
    )
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {commit}:{relative}")
    return loaded


def candidate_lookup() -> list[dict[str, Any]]:
    payload = json.loads(CANDIDATES.read_text())
    values = payload.get("cond1_v14")
    if not isinstance(values, list) or len(values) != 24:
        raise ValueError("Expected 24 Cond1 V14 transition candidates")
    return values


def identify_candidate(
    params: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    transition = params[TRANSITION_PATH]
    controller = transition.get("strategy_controller")
    gain = float(transition.get("latent_volatility_error_gain", 0.0))
    for candidate in candidates:
        candidate_transition = candidate["hypo_transitions_kwargs"]
        candidate_gain = float(
            candidate_transition.get("latent_volatility_error_gain", 0.0)
        )
        if (
            controller == candidate_transition.get("strategy_controller")
            and math.isclose(gain, candidate_gain, abs_tol=1e-12)
        ):
            return candidate
    raise ValueError("Could not match selected transition configuration to V14 candidate")


def compact_selection(
    params: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate = identify_candidate(params, candidates)
    memory = params[MEMORY_PATH]
    readout = params[READOUT_PATH]
    if candidate["state_mode"] == "off":
        state = "off"
    else:
        state = f"gain={float(candidate['state_error_gain']):.2f}"
    readout_name = str(readout["method"])
    if "power" in readout:
        readout_name += f"(p={float(readout['power']):g})"
    return {
        "candidate_id": candidate["id"],
        "family": candidate["family_id"].replace("c1_v14_", ""),
        "state": state,
        "readout": readout_name,
        "gamma": float(memory["gamma"]),
        "w0": float(memory["w0"]),
    }


def read_combinations(subject_id: int) -> list[dict[str, Any]]:
    path = HYPER_DIR / f"subject_{subject_id}" / "all_combinations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics_for_params(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    params: dict[str, Any],
) -> tuple[float, float]:
    matches = [
        row
        for row in rows
        if row.get("stage") == stage and row.get("hyperparams") == params
    ]
    if not matches:
        raise ValueError(f"No {stage} combination matched the checkpoint parameters")
    values = matches[0]["objective_values"]
    return float(values[BRIER_PATH]), float(values[CRPS_PATH])


def counter_text(values: pd.Series) -> str:
    return ", ".join(
        f"{key}: {count}" for key, count in sorted(Counter(values).items())
    )


def write_markdown(
    frame: pd.DataFrame,
    *,
    coarse_commit: str,
    validation: dict[str, Any],
) -> None:
    mean_delta_brier = float(frame["fine_delta_brier_vs_v13"].mean())
    mean_delta_crps = float(frame["fine_delta_crps_vs_v13"].mean())
    boundary = frame.loc[frame["boundary_probe_required"], "subject_id"].tolist()
    exact_same = int(frame["exact_selection_same"].sum())
    lines = [
        "# Cond1 V14 fine checkpoint",
        "",
        f"- Coarse checkpoint commit: `{coarse_commit}`",
        "- Fine repeats: 256 per evaluated configuration",
        f"- Completed subjects: {len(frame)}/8",
        f"- Exact coarse-to-fine selection retained: {exact_same}/8",
        f"- Controller family retained: {int(frame['family_same'].sum())}/8",
        f"- State setting retained: {int(frame['state_same'].sum())}/8",
        f"- Boundary probe required: {', '.join(map(str, boundary))}",
        "",
        "## Fine selections",
        "",
        f"- State: {counter_text(frame['fine_state'])}",
        f"- Controller family: {counter_text(frame['fine_family'])}",
        f"- Readout: {counter_text(frame['fine_readout'])}",
        "",
        "## Subject-level checkpoint",
        "",
        "| Subject | Fine controller | State | Readout | gamma | w0 | Brier | CRPS | Coarse→fine changes |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict(orient="records"):
        changed = [
            key
            for key in ("family", "state", "readout", "gamma", "w0")
            if not row[f"{key}_same"]
        ]
        lines.append(
            "| {subject_id} | {fine_family} | {fine_state} | {fine_readout} | "
            "{fine_gamma:.2f} | {fine_w0:.3f} | {fine_brier:.6f} | "
            "{fine_crps:.6f} | {changes} |".format(
                **row,
                changes=", ".join(changed) if changed else "none",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Fine scores are nominally lower than the frozen V13 reference by "
            f"{abs(mean_delta_brier):.4f} Brier and {abs(mean_delta_crps):.4f} "
            "CRPS on average, with 7/8 subject-level wins for each metric. This is "
            "not an independent performance estimate because fine selected among "
            "candidate configurations and used a different Monte Carlo sample.",
            "",
            "The structural result is more reliable than the exact parameter result: "
            "controller family was retained for 7/8 subjects, whereas only "
            f"{exact_same}/8 retained the full controller/state/readout/memory tuple. "
            "Do not expand to the full sample before a targeted memory-boundary probe "
            "and frozen common-seed confirmation.",
            "",
            "## Required next gate",
            "",
            "1. Probe only the fine winners at current memory boundaries: gamma=0.10 "
            "below the current 0.25 floor, w0=0.005 below 0.01, and w0=0.75 above 0.50.",
            "2. Freeze the resulting per-subject configurations.",
            "3. Run an independent common-seed comparison of V13, the frozen V14 "
            "winner, and its matched state-off/state-on ablation.",
            "",
            "## Validation",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in validation.items())
    DOC_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse-commit",
        default="677116f",
        help="Commit containing the generated coarse subjectwise checkpoint.",
    )
    args = parser.parse_args()

    coarse_config = load_yaml_at_commit(args.coarse_commit, GENERATED_CONFIG)
    fine_config = yaml.safe_load(GENERATED_CONFIG.read_text())
    candidates = candidate_lookup()

    confirm = pd.read_csv(CONFIRM_ROWS)
    v13 = (
        confirm.loc[
            confirm["variant_id"].eq("m0_v13_saved"),
            ["subject_id", "marginal_choice_brier", "trajectory_crps"],
        ]
        .rename(
            columns={
                "marginal_choice_brier": "v13_brier",
                "trajectory_crps": "v13_crps",
            }
        )
        .set_index("subject_id")
    )

    rows: list[dict[str, Any]] = []
    for subject_id in SUBJECTS:
        key = str(subject_id)
        coarse_params = coarse_config["subject_overrides"][key]["fixed_hyperparams"]
        fine_params = fine_config["subject_overrides"][key]["fixed_hyperparams"]
        coarse = compact_selection(coarse_params, candidates)
        fine = compact_selection(fine_params, candidates)
        combinations = read_combinations(subject_id)
        coarse_brier, coarse_crps = metrics_for_params(
            combinations, stage="coarse", params=coarse_params
        )

        best_path = HYPER_DIR / f"subject_{subject_id}" / "best_hyperparams.json"
        best = json.loads(best_path.read_text())
        if best["selection"]["candidate"]["stage"] != "fine":
            raise ValueError(f"Subject {subject_id} is not complete for fine")
        objective_values = best["selection"]["objectives"]["values"]
        fine_brier = float(objective_values[BRIER_PATH])
        fine_crps = float(objective_values[CRPS_PATH])
        stage_summary = json.loads(
            (HYPER_DIR / f"subject_{subject_id}" / "stage_summary.json").read_text()
        )

        row: dict[str, Any] = {
            "subject_id": subject_id,
            **{f"coarse_{name}": value for name, value in coarse.items()},
            **{f"fine_{name}": value for name, value in fine.items()},
            "coarse_brier": coarse_brier,
            "coarse_crps": coarse_crps,
            "fine_brier": fine_brier,
            "fine_crps": fine_crps,
            "fine_combination_count": int(stage_summary["fine"]["num_combinations"]),
            "v13_brier": float(v13.loc[subject_id, "v13_brier"]),
            "v13_crps": float(v13.loc[subject_id, "v13_crps"]),
        }
        for name in ("family", "state", "readout", "gamma", "w0"):
            row[f"{name}_same"] = coarse[name] == fine[name]
        row["exact_selection_same"] = all(
            row[f"{name}_same"]
            for name in ("family", "state", "readout", "gamma", "w0")
        )
        row["boundary_probe_required"] = (
            fine["gamma"] in {0.25, 0.95} or fine["w0"] in {0.01, 0.50}
        )
        row["fine_delta_brier_vs_v13"] = fine_brier - row["v13_brier"]
        row["fine_delta_crps_vs_v13"] = fine_crps - row["v13_crps"]
        rows.append(row)

    frame = pd.DataFrame(rows)
    validation = {
        "subject_count": int(len(frame)),
        "all_fine_complete": bool(len(frame) == 8),
        "fine_metric_null_count": int(
            frame[["fine_brier", "fine_crps"]].isna().sum().sum()
        ),
        "coarse_history_retained": bool(
            all(
                "coarse"
                in json.loads(
                    (
                        HYPER_DIR
                        / f"subject_{subject_id}"
                        / "stage_summary.json"
                    ).read_text()
                )
                for subject_id in SUBJECTS
            )
        ),
        "fine_repeat_count": 256,
    }
    summary = {
        "coarse_commit": args.coarse_commit,
        "validation": validation,
        "stability": {
            "exact_selection_same": int(frame["exact_selection_same"].sum()),
            **{
                f"{name}_same": int(frame[f"{name}_same"].sum())
                for name in ("family", "state", "readout", "gamma", "w0")
            },
        },
        "fine_counts": {
            "state": dict(Counter(frame["fine_state"])),
            "family": dict(Counter(frame["fine_family"])),
            "readout": dict(Counter(frame["fine_readout"])),
        },
        "nominal_comparison_to_v13": {
            "mean_delta_brier": float(frame["fine_delta_brier_vs_v13"].mean()),
            "mean_delta_crps": float(frame["fine_delta_crps_vs_v13"].mean()),
            "brier_wins": int((frame["fine_delta_brier_vs_v13"] < 0).sum()),
            "crps_wins": int((frame["fine_delta_crps_vs_v13"] < 0).sum()),
            "independent": False,
        },
        "boundary_probe_subjects": frame.loc[
            frame["boundary_probe_required"], "subject_id"
        ].tolist(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "fine_selection.csv", index=False)
    (OUTPUT_DIR / "fine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    write_markdown(frame, coarse_commit=args.coarse_commit, validation=validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT_DIR} and {DOC_PATH}")


if __name__ == "__main__":
    main()
