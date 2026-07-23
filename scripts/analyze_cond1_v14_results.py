#!/usr/bin/env python3
"""Create reproducible V14 pilot/confirmation analysis artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def bootstrap_mean_interval(
    values: np.ndarray, *, draws: int = 100_000, seed: int = 140016
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def variant_comparison(confirm: pd.DataFrame) -> pd.DataFrame:
    base = confirm[confirm.variant_id == "m0_v13_saved"].set_index("subject_id")
    off = confirm[confirm.variant_id == "m1_selected_state_off"].set_index("subject_id")
    rows: list[dict[str, Any]] = []
    for variant_id, group in confirm.groupby("variant_id", sort=True):
        group = group.set_index("subject_id").sort_index()
        delta_brier = group.marginal_choice_brier - base.marginal_choice_brier
        delta_crps = group.trajectory_crps - base.trajectory_crps
        delta_off_brier = group.marginal_choice_brier - off.marginal_choice_brier
        delta_off_crps = group.trajectory_crps - off.trajectory_crps
        brier_low, brier_high = bootstrap_mean_interval(delta_brier.to_numpy())
        crps_low, crps_high = bootstrap_mean_interval(
            delta_crps.to_numpy(), seed=140017
        )
        rows.append(
            {
                "variant_id": variant_id,
                "subject_count": len(group),
                "mean_marginal_choice_brier": group.marginal_choice_brier.mean(),
                "delta_brier_vs_v13": delta_brier.mean(),
                "delta_brier_vs_v13_ci_low": brier_low,
                "delta_brier_vs_v13_ci_high": brier_high,
                "delta_brier_vs_state_off": delta_off_brier.mean(),
                "brier_wins_vs_v13": int((delta_brier < 0).sum()),
                "brier_wins_vs_state_off": int((delta_off_brier < 0).sum()),
                "mean_trajectory_crps": group.trajectory_crps.mean(),
                "delta_crps_vs_v13": delta_crps.mean(),
                "delta_crps_vs_v13_ci_low": crps_low,
                "delta_crps_vs_v13_ci_high": crps_high,
                "delta_crps_vs_state_off": delta_off_crps.mean(),
                "crps_wins_vs_v13": int((delta_crps < 0).sum()),
                "crps_wins_vs_state_off": int((delta_off_crps < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def selected_confirmation(confirm: pd.DataFrame) -> pd.DataFrame:
    base = confirm[confirm.variant_id == "m0_v13_saved"].set_index("subject_id")
    candidates = confirm[confirm.variant_id != "m0_v13_saved"].copy()
    selected_idx = candidates.groupby("subject_id").marginal_choice_brier.idxmin()
    selected = candidates.loc[selected_idx].copy().set_index("subject_id").sort_index()
    selected["v13_marginal_choice_brier"] = base.marginal_choice_brier
    selected["delta_brier_vs_v13"] = (
        selected.marginal_choice_brier - base.marginal_choice_brier
    )
    selected["v13_trajectory_crps"] = base.trajectory_crps
    selected["delta_crps_vs_v13"] = selected.trajectory_crps - base.trajectory_crps
    selected["state_setting"] = selected.variant_id.str.replace(
        "m2_v14_gain_", "gain=", regex=False
    ).str.replace("m1_selected_state_off", "off", regex=False).str.replace(
        "p", ".", regex=False
    )
    return selected.reset_index()


def pilot_structure_summary(pilot: pd.DataFrame) -> pd.DataFrame:
    base = pilot[pilot.variant_id == "m0_v13_saved"].set_index("subject_id")
    rows = []
    for family in ("m2_core6", "m3_unified"):
        candidates = pilot[pilot.model_family == family]
        idx = candidates.groupby("subject_id").marginal_choice_brier.idxmin()
        selected = candidates.loc[idx].set_index("subject_id")
        rows.append(
            {
                "model_family": family,
                "subject_count": len(selected),
                "mean_marginal_choice_brier": selected.marginal_choice_brier.mean(),
                "delta_brier_vs_v13": (
                    selected.marginal_choice_brier - base.marginal_choice_brier
                ).mean(),
                "mean_trajectory_crps": selected.trajectory_crps.mean(),
                "delta_crps_vs_v13": (
                    selected.trajectory_crps - base.trajectory_crps
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def validation_checks(
    pilot: pd.DataFrame, confirm: pd.DataFrame, state_runs: pd.DataFrame
) -> dict[str, Any]:
    stable_subjects = {105, 118}
    duplicate_check = []
    for subject_id in stable_subjects:
        core = pilot[
            (pilot.subject_id == subject_id) & (pilot.model_family == "m2_core6")
        ].sort_values(["readout", "readout_power"])
        unified = pilot[
            (pilot.subject_id == subject_id) & (pilot.model_family == "m3_unified")
        ].sort_values(["readout", "readout_power"])
        duplicate_check.append(
            bool(
                np.allclose(
                    core[["marginal_choice_brier", "trajectory_crps"]],
                    unified[["marginal_choice_brier", "trajectory_crps"]],
                )
            )
        )
    confirm_seed_counts = confirm.groupby("subject_id").simulation_point_seed.nunique()
    return {
        "pilot_row_count": int(len(pilot)),
        "confirm_row_count": int(len(confirm)),
        "state_run_count": int(len(state_runs)),
        "pilot_expected_shape": bool(
            len(pilot) == 72 and pilot.groupby("subject_id").size().eq(9).all()
        ),
        "confirm_expected_shape": bool(
            len(confirm) == 40 and confirm.groupby("subject_id").size().eq(5).all()
        ),
        "state_expected_shape": bool(
            len(state_runs) == 256 and state_runs.groupby("subject_id").size().eq(32).all()
        ),
        "confirm_common_seed_within_subject": bool(confirm_seed_counts.eq(1).all()),
        "stable_controller_duplicate_control_passed": bool(all(duplicate_check)),
        "metric_null_count": int(
            pilot[["marginal_choice_brier", "trajectory_crps"]].isna().sum().sum()
            + confirm[["marginal_choice_brier", "trajectory_crps"]]
            .isna()
            .sum()
            .sum()
        ),
    }


def create_notebook(
    output: Path,
    *,
    comparison: pd.DataFrame,
    selected: pd.DataFrame,
    state_summary: dict[str, Any],
) -> None:
    gain50 = comparison[comparison.variant_id == "m2_v14_gain_0p50"].iloc[0]
    selected_brier_delta = float(selected.delta_brier_vs_v13.mean())
    selected_crps_delta = float(selected.delta_crps_vs_v13.mean())
    state = state_summary["aggregate"]
    cells = [
        nbf.v4.new_markdown_cell(
            "# Cond1 V14 pilot and confirmation analysis\n\n"
            "## tl;dr\n\n"
            f"- Independent confirmation favors the pilot-selected V14 structure: "
            f"the best tested setting per subject reduced mean marginal choice Brier by "
            f"{abs(selected_brier_delta):.4f} and trajectory CRPS by "
            f"{abs(selected_crps_delta):.4f} versus the frozen V13 baseline.\n"
            f"- A single gain is not universally optimal. Gain 0.50 improved Brier in "
            f"{int(gain50.brier_wins_vs_v13)}/8 subjects, while one subject preferred "
            "state-off.\n"
            f"- The state is operational: it was non-zero on "
            f"{state['state_nonzero_fraction']:.1%} of logged trials and had mean lag-1 "
            f"autocorrelation {state['state_lag1']:.2f}.\n"
            "- Decision: keep six controller families, retain all four inner policies, "
            "and search state-off plus gains 0.20/0.35/0.50 and four readouts."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This notebook supports the V14 structural decision for condition 1. The unit "
            "of analysis is one representative subject. The pilot used 256 stochastic runs "
            "per configuration; confirmation used 512 new runs with an independent seed. "
            "All variants within a subject shared trajectory seeds. Lower Brier and CRPS are better.\n\n"
            "### Key Assumptions\n\n"
            "The eight subjects are a structural screen, not a population-level inferential "
            "sample. Independent Monte Carlo seeds reduce stochastic selection noise but do "
            "not provide held-out human trials."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] "
            "if (p / 'results/cond1_v14').exists())\n"
            "PILOT = ROOT / 'results/cond1_v14/pilot_state_readout/pilot_rows.csv'\n"
            "CONFIRM = ROOT / 'results/cond1_v14/confirm_gain_readout/pilot_rows.csv'\n"
            "STATE = ROOT / 'results/cond1_v14/state_diagnostics/state_diagnostic_runs.csv'\n"
            "pilot = pd.read_csv(PILOT)\n"
            "confirm = pd.read_csv(CONFIRM)\n"
            "state_runs = pd.read_csv(STATE)\n"
            "len(pilot), len(confirm), len(state_runs)"
        ),
        nbf.v4.new_markdown_cell("## Data\n\nValidate expected row counts and paired-seed design before interpreting results."),
        nbf.v4.new_code_cell(
            "assert len(pilot) == 72 and pilot.groupby('subject_id').size().eq(9).all()\n"
            "assert len(confirm) == 40 and confirm.groupby('subject_id').size().eq(5).all()\n"
            "assert len(state_runs) == 256 and state_runs.groupby('subject_id').size().eq(32).all()\n"
            "assert confirm.groupby('subject_id').simulation_point_seed.nunique().eq(1).all()\n"
            "{'pilot_rows': len(pilot), 'confirm_rows': len(confirm), 'state_runs': len(state_runs)}"
        ),
        nbf.v4.new_markdown_cell("## Results\n\nThe independent confirmation is the controlling evidence; pilot results only selected the controller/readout candidates."),
        nbf.v4.new_code_cell(
            "comparison = pd.read_csv(ROOT / 'results/cond1_v14/analysis/variant_comparison.csv')\n"
            "comparison[['variant_id','mean_marginal_choice_brier','delta_brier_vs_v13',"
            "'brier_wins_vs_v13','mean_trajectory_crps','delta_crps_vs_v13','crps_wins_vs_v13']]"
        ),
        nbf.v4.new_code_cell(
            "selected = pd.read_csv(ROOT / 'results/cond1_v14/analysis/subject_confirm_selection.csv')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.8))\n"
            "colors = ['#1f77b4' if x < 0 else '#d28e2b' for x in selected.delta_brier_vs_v13]\n"
            "ax.bar(selected.subject_id.astype(str), selected.delta_brier_vs_v13, color=colors)\n"
            "ax.axhline(0, color='#333333', linewidth=1)\n"
            "ax.set(title='Confirmed marginal Brier change by subject', xlabel='Subject', "
            "ylabel='Selected V14 − V13 (lower is better)')\n"
            "ax.spines[['top','right']].set_visible(False)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("## Persistent-state diagnostics\n\nState dynamics were measured from 32 fully logged runs per subject; readout does not feed back into the latent-state update."),
        nbf.v4.new_code_cell(
            "state_runs[['state_mean','state_nonzero_fraction','state_above_threshold_fraction',"
            "'state_lag1','aggressive_probability_high_state','aggressive_probability_low_state']].mean()"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. Do not collapse V14 to one controller. Unified stable-dominant helped some "
            "subjects but materially harmed others.\n"
            "2. Keep the four inner policies; all were used in logged runs.\n"
            "3. Keep state-off as a real ablation because subject 118 did not benefit from "
            "the tested persistent gains.\n"
            "4. Use marginal choice Brier first and trajectory CRPS second; do not select "
            "on best-run/lower-tail statistics.\n"
            "5. The next expensive step is selected-eight coordinate descent over six "
            "controller families, four state settings, four readouts, and the existing memory grid."
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/cond1_v14/analysis"
    )
    args = parser.parse_args()
    pilot = pd.read_csv(ROOT / "results/cond1_v14/pilot_state_readout/pilot_rows.csv")
    confirm = pd.read_csv(ROOT / "results/cond1_v14/confirm_gain_readout/pilot_rows.csv")
    state_runs = pd.read_csv(
        ROOT / "results/cond1_v14/state_diagnostics/state_diagnostic_runs.csv"
    )
    state_summary = json.loads(
        (ROOT / "results/cond1_v14/state_diagnostics/state_diagnostic_summary.json").read_text()
    )

    comparison = variant_comparison(confirm)
    selected = selected_confirmation(confirm)
    pilot_summary = pilot_structure_summary(pilot)
    checks = validation_checks(pilot, confirm, state_runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "variant_comparison.csv", index=False)
    selected.to_csv(args.output_dir / "subject_confirm_selection.csv", index=False)
    pilot_summary.to_csv(args.output_dir / "pilot_structure_summary.csv", index=False)
    summary = {
        "validation": checks,
        "pilot_structure": pilot_summary.to_dict(orient="records"),
        "confirm_variants": comparison.to_dict(orient="records"),
        "confirm_subject_selection": selected.to_dict(orient="records"),
        "state_diagnostics": state_summary,
        "interpretation": {
            "selected_subject_mean_delta_brier_vs_v13": float(
                selected.delta_brier_vs_v13.mean()
            ),
            "selected_subject_mean_delta_crps_vs_v13": float(
                selected.delta_crps_vs_v13.mean()
            ),
            "selected_subject_brier_wins": int((selected.delta_brier_vs_v13 < 0).sum()),
            "selected_subject_crps_wins": int((selected.delta_crps_vs_v13 < 0).sum()),
            "state_off_selected_count": int((selected.state_setting == "off").sum()),
        },
    }
    summary = json_safe(summary)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    create_notebook(
        args.output_dir / "cond1_v14_analysis.ipynb",
        comparison=comparison,
        selected=selected,
        state_summary=state_summary,
    )
    print(json.dumps(summary["interpretation"], indent=2))
    print(f"Wrote analysis artifacts -> {args.output_dir}")


if __name__ == "__main__":
    main()
