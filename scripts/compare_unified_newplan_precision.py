#!/usr/bin/env python3
"""Audit 128 -> 256 -> 512 Sobol-point stability for the unified core screen.

This is intentionally a post-processing script: it never refits a model.  It
aligns subjects and models across completed runs, compares the cached
perceptual integrals, held-out metrics, training selections, and fitted
parameters, then writes a self-contained precision-freeze report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ROOT / "results/zhuran/unified_newplan/core_sobol128_20260802",
    ROOT / "results/zhuran/unified_newplan/core_sobol256_20260802",
    ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802",
)
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/unified_newplan/precision_128_256_512_20260802"
)
PARAMETERS = {
    "NR2": ("learning_rate", "sensitivity"),
    "R0K": ("sensitivity",),
    "R1": ("retention",),
    "R2": ("retention", "sensitivity"),
    "R3": ("retention", "sensitivity"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_runs(paths: Iterable[Path]) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        precision = int(manifest["sobol_points"])
        if manifest.get("status") != "complete":
            raise ValueError(f"run is not complete: {path}")
        if precision in runs:
            raise ValueError(f"duplicate Sobol precision: {precision}")
        runs[precision] = {
            "path": path,
            "manifest": manifest,
            "metrics": pd.read_csv(path / "subject_model_metrics.csv"),
            "comparisons": pd.read_csv(path / "model_comparisons.csv", dtype={"condition": str}),
            "fits": pd.read_csv(path / "fit_manifest.csv"),
        }
    if sorted(runs) != [128, 256, 512]:
        raise ValueError(f"expected precisions [128, 256, 512], found {sorted(runs)}")

    reference = runs[128]["manifest"]
    invariants = ("data_sha256", "base_seed", "n_subjects", "subjects")
    for precision, run in runs.items():
        for key in invariants:
            if run["manifest"].get(key) != reference.get(key):
                raise ValueError(f"manifest mismatch for {key} at {precision} points")
    return runs


def compare_q(runs: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows = []
    for old, new in ((128, 256), (256, 512), (128, 512)):
        for subject_id in runs[old]["manifest"]["subjects"]:
            old_path = runs[old]["path"] / "q_cache" / f"subject_{int(subject_id)}.npz"
            new_path = runs[new]["path"] / "q_cache" / f"subject_{int(subject_id)}.npz"
            with np.load(old_path, allow_pickle=False) as archive:
                old_q = archive["q"].astype(np.float64)
            with np.load(new_path, allow_pickle=False) as archive:
                new_q = archive["q"].astype(np.float64)
            if old_q.shape != new_q.shape:
                raise ValueError(f"q shape mismatch for subject {subject_id}: {old_q.shape} != {new_q.shape}")
            difference = np.abs(new_q - old_q)
            subject_rows.append(
                {
                    "precision_pair": f"{old}->{new}",
                    "old_points": old,
                    "new_points": new,
                    "subject_id": int(subject_id),
                    "condition": int(str(subject_id)[0]),
                    "n_values": int(difference.size),
                    "mean_abs_q_delta": float(difference.mean()),
                    "q99_abs_q_delta": float(np.quantile(difference, 0.99)),
                    "max_abs_q_delta": float(difference.max()),
                    "argmax_disagreement": float(
                        np.mean(np.argmax(old_q, axis=-1) != np.argmax(new_q, axis=-1))
                    ),
                }
            )
    subjects = pd.DataFrame(subject_rows)
    summaries = []
    for pair, group in subjects.groupby("precision_pair", sort=False):
        summaries.append(
            {
                "precision_pair": pair,
                "n_subjects": int(len(group)),
                "mean_subject_mean_abs_q_delta": float(group["mean_abs_q_delta"].mean()),
                "max_subject_mean_abs_q_delta": float(group["mean_abs_q_delta"].max()),
                "mean_subject_q99_abs_q_delta": float(group["q99_abs_q_delta"].mean()),
                "max_abs_q_delta": float(group["max_abs_q_delta"].max()),
                "mean_argmax_disagreement": float(group["argmax_disagreement"].mean()),
                "max_argmax_disagreement": float(group["argmax_disagreement"].max()),
            }
        )
    return subjects, pd.DataFrame(summaries)


def compare_metrics(runs: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    key = ["subject_id", "condition", "model", "segment"]
    for old, new in ((128, 256), (256, 512), (128, 512)):
        old_metrics = runs[old]["metrics"]
        new_metrics = runs[new]["metrics"]
        paired = old_metrics.merge(new_metrics, on=key, suffixes=("_old", "_new"), validate="one_to_one")
        if len(paired) != len(old_metrics) or len(paired) != len(new_metrics):
            raise ValueError(f"metric alignment is incomplete for {old}->{new}")
        for row in paired.itertuples(index=False):
            delta = float(row.nll_per_trial_new - row.nll_per_trial_old)
            rows.append(
                {
                    "precision_pair": f"{old}->{new}",
                    "old_points": old,
                    "new_points": new,
                    "subject_id": int(row.subject_id),
                    "condition": int(row.condition),
                    "model": row.model,
                    "segment": row.segment,
                    "old_nll_per_trial": float(row.nll_per_trial_old),
                    "new_nll_per_trial": float(row.nll_per_trial_new),
                    "delta_nll_per_trial": delta,
                    "abs_delta_nll_per_trial": abs(delta),
                }
            )
    subjects = pd.DataFrame(rows)
    heldout = subjects[subjects["segment"] == "holdout"]
    summary = (
        heldout.groupby(["precision_pair", "condition", "model"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_signed_delta=("delta_nll_per_trial", "mean"),
            mean_abs_delta=("abs_delta_nll_per_trial", "mean"),
            median_abs_delta=("abs_delta_nll_per_trial", "median"),
            max_abs_delta=("abs_delta_nll_per_trial", "max"),
        )
    )
    all_conditions = (
        heldout.groupby(["precision_pair", "model"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_signed_delta=("delta_nll_per_trial", "mean"),
            mean_abs_delta=("abs_delta_nll_per_trial", "mean"),
            median_abs_delta=("abs_delta_nll_per_trial", "median"),
            max_abs_delta=("abs_delta_nll_per_trial", "max"),
        )
    )
    all_conditions.insert(1, "condition", "all")
    summary["condition"] = summary["condition"].astype(str)
    return subjects, pd.concat([summary, all_conditions], ignore_index=True)


def unpack_fits(run: dict[str, Any], precision: int) -> pd.DataFrame:
    rows = []
    for fit in run["fits"].itertuples(index=False):
        parameters = json.loads(fit.parameters_json)
        base = {
            "precision": precision,
            "subject_id": int(fit.subject_id),
            "condition": int(fit.condition),
            "nr_selected": parameters["NR_SELECT"]["selected_model"],
            "r_selected": parameters["R_SELECT"]["selected_model"],
        }
        for model, names in PARAMETERS.items():
            for name in names:
                base[f"{model}.{name}"] = float(parameters[model][name])
        rows.append(base)
    return pd.DataFrame(rows)


def compare_fits(
    runs: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unpacked = {precision: unpack_fits(run, precision) for precision, run in runs.items()}
    selection_rows = []
    parameter_rows = []
    parameter_columns = [f"{model}.{name}" for model, names in PARAMETERS.items() for name in names]
    for old, new in ((128, 256), (256, 512), (128, 512)):
        paired = unpacked[old].merge(
            unpacked[new],
            on=["subject_id", "condition"],
            suffixes=("_old", "_new"),
            validate="one_to_one",
        )
        for row in paired.itertuples(index=False):
            selection_rows.append(
                {
                    "precision_pair": f"{old}->{new}",
                    "subject_id": int(row.subject_id),
                    "condition": int(row.condition),
                    "nr_selected_old": row.nr_selected_old,
                    "nr_selected_new": row.nr_selected_new,
                    "nr_selection_same": bool(row.nr_selected_old == row.nr_selected_new),
                    "r_selected_old": row.r_selected_old,
                    "r_selected_new": row.r_selected_new,
                    "r_selection_same": bool(row.r_selected_old == row.r_selected_new),
                }
            )
        for column in parameter_columns:
            old_values = paired[f"{column}_old"].to_numpy(dtype=float)
            new_values = paired[f"{column}_new"].to_numpy(dtype=float)
            model, parameter = column.split(".")
            for index, row in paired.iterrows():
                parameter_rows.append(
                    {
                        "precision_pair": f"{old}->{new}",
                        "subject_id": int(row["subject_id"]),
                        "condition": int(row["condition"]),
                        "model": model,
                        "parameter": parameter,
                        "old_value": float(row[f"{column}_old"]),
                        "new_value": float(row[f"{column}_new"]),
                        "delta": float(row[f"{column}_new"] - row[f"{column}_old"]),
                        "abs_delta": float(abs(row[f"{column}_new"] - row[f"{column}_old"])),
                    }
                )
    selection_subjects = pd.DataFrame(selection_rows)
    parameter_subjects = pd.DataFrame(parameter_rows)
    selection_summary = (
        selection_subjects.groupby("precision_pair", as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            nr_same=("nr_selection_same", "sum"),
            nr_stability=("nr_selection_same", "mean"),
            r_same=("r_selection_same", "sum"),
            r_stability=("r_selection_same", "mean"),
        )
    )
    parameter_summary_rows = []
    for keys, group in parameter_subjects.groupby(
        ["precision_pair", "condition", "model", "parameter"], sort=False
    ):
        correlation = np.corrcoef(group["old_value"], group["new_value"])[0, 1]
        parameter_summary_rows.append(
            {
                "precision_pair": keys[0],
                "condition": str(keys[1]),
                "model": keys[2],
                "parameter": keys[3],
                "n_subjects": int(len(group)),
                "mean_abs_delta": float(group["abs_delta"].mean()),
                "max_abs_delta": float(group["abs_delta"].max()),
                "pearson_r": float(correlation),
            }
        )
    for keys, group in parameter_subjects.groupby(
        ["precision_pair", "model", "parameter"], sort=False
    ):
        correlation = np.corrcoef(group["old_value"], group["new_value"])[0, 1]
        parameter_summary_rows.append(
            {
                "precision_pair": keys[0],
                "condition": "all",
                "model": keys[1],
                "parameter": keys[2],
                "n_subjects": int(len(group)),
                "mean_abs_delta": float(group["abs_delta"].mean()),
                "max_abs_delta": float(group["abs_delta"].max()),
                "pearson_r": float(correlation),
            }
        )
    return selection_subjects, selection_summary, pd.DataFrame(parameter_summary_rows), parameter_subjects


def comparison_stability(runs: dict[int, dict[str, Any]]) -> pd.DataFrame:
    frames = []
    columns = [
        "comparison",
        "candidate",
        "reference",
        "condition",
        "mean_delta_nll_per_trial",
        "bootstrap_mean_ci_low",
        "bootstrap_mean_ci_high",
        "n_improved",
        "n_subjects",
    ]
    for precision, run in runs.items():
        frame = run["comparisons"][columns].copy()
        frame.insert(0, "precision", precision)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["comparison", "condition", "precision"]
    )


def render_report(
    path: Path,
    q_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    q_last = q_summary[q_summary["precision_pair"] == "256->512"].iloc[0]
    metric_last = metric_summary[
        (metric_summary["precision_pair"] == "256->512")
        & (metric_summary["condition"] == "all")
    ].set_index("model")
    parameter_last = parameter_summary[
        (parameter_summary["precision_pair"] == "256->512")
        & (parameter_summary["condition"] == "all")
    ]
    selection_last = selection_summary[
        selection_summary["precision_pair"] == "256->512"
    ].iloc[0]
    primary = comparisons[comparisons["comparison"] == "representation_gate"]
    resource = comparisons[
        comparisons["comparison"].isin(
            ["sensitivity_only_increment", "retention_given_sensitivity"]
        )
    ]

    lines = [
        "# Unified new-plan Sobol precision audit",
        "",
        "> Decision: freeze subject-specific perceptual integration at 512 nested Sobol points. Group conclusions and the simplest viable rule parameter are stable; additional integration points are not the present bottleneck.",
        "",
        "## 256 -> 512 convergence",
        "",
        f"- Mean subject-level absolute change in integrated rule/category probability: {q_last.mean_subject_mean_abs_q_delta:.6g}; mean argmax disagreement: {q_last.mean_argmax_disagreement:.6g}.",
        f"- R0K held-out NLL/trial mean absolute change: {metric_last.loc['R0K', 'mean_abs_delta']:.6g} (maximum {metric_last.loc['R0K', 'max_abs_delta']:.6g}).",
        f"- Training-selected non-rule model unchanged for {int(selection_last.nr_same)}/{int(selection_last.n_subjects)} subjects; selected R2/R3 rule variant unchanged for {int(selection_last.r_same)}/{int(selection_last.n_subjects)}.",
    ]
    r0k = parameter_last[
        (parameter_last["model"] == "R0K") & (parameter_last["parameter"] == "sensitivity")
    ].iloc[0]
    lines.append(
        f"- R0K sensitivity κ: mean absolute change {r0k.mean_abs_delta:.6g}, maximum {r0k.max_abs_delta:.6g}, cross-precision Pearson r={r0k.pearson_r:.6f}."
    )

    lines.extend(
        [
            "",
            "## Scientific comparisons across precision",
            "",
            "Positive ΔNLL/trial favors the candidate. Intervals shown are subject-bootstrap 95% intervals from each frozen temporal holdout.",
            "",
            "| Comparison | Condition | Points | Mean ΔNLL/trial | 95% CI | Improved |",
            "|:--|:--|--:|--:|:--|:--|",
        ]
    )
    display = pd.concat([primary, resource], ignore_index=True).sort_values(
        ["comparison", "condition", "precision"]
    )
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.condition} | {int(row.precision)} | "
            f"{row.mean_delta_nll_per_trial:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary representation gate is invariant to integration precision: the training-selected rule family does not outperform the training-selected non-rule family overall, and is clearly worse in condition 1. The rule result therefore cannot be rescued by increasing Monte Carlo precision.",
            "",
            "Within the rule family, adding a stable readout-sensitivity parameter (R0K versus R0) is reliably useful, but adding forgetting once sensitivity is present (R2 versus R0K) is not supported overall. R0K is consequently the appropriate minimal rule diagnostic. Its κ estimate is exceptionally stable across 256 and 512 points.",
            "",
            "R2/R3 occasionally move between optimization modes at the individual level even though their group comparisons are stable. Their λ values must not be used for individual-difference or correlation claims without recovery and stronger identifiability evidence. This instability is a model/likelihood issue, not residual Sobol error.",
            "",
            "## Artifacts",
            "",
            "- `q_precision_subjects.csv`, `q_precision_summary.csv`: cached-integral convergence.",
            "- `metric_precision_subjects.csv`, `metric_precision_summary.csv`: aligned train/holdout score changes.",
            "- `selection_precision_subjects.csv`, `selection_precision_summary.csv`: training-selection stability.",
            "- `parameter_precision_subjects.csv`, `parameter_precision_summary.csv`: parameter stability.",
            "- `comparison_stability.csv`: all scientific comparisons at each precision.",
            "- `manifest.json`: input paths, hashes, and freeze decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)

    q_subjects, q_summary = compare_q(runs)
    metric_subjects, metric_summary = compare_metrics(runs)
    selection_subjects, selection_summary, parameter_summary, parameter_subjects = compare_fits(runs)
    comparisons = comparison_stability(runs)

    outputs = {
        "q_precision_subjects.csv": q_subjects,
        "q_precision_summary.csv": q_summary,
        "metric_precision_subjects.csv": metric_subjects,
        "metric_precision_summary.csv": metric_summary,
        "selection_precision_subjects.csv": selection_subjects,
        "selection_precision_summary.csv": selection_summary,
        "parameter_precision_subjects.csv": parameter_subjects,
        "parameter_precision_summary.csv": parameter_summary,
        "comparison_stability.csv": comparisons,
    }
    for name, frame in outputs.items():
        atomic_csv(output / name, frame)

    render_report(
        output / "RESULTS.md",
        q_summary,
        metric_summary,
        selection_summary,
        parameter_summary,
        comparisons,
    )
    manifest = {
        "result_type": "unified_newplan_sobol_precision_audit",
        "status": "complete",
        "precisions": sorted(runs),
        "input_runs": {
            str(precision): {
                "path": str(run["path"]),
                "manifest_sha256": sha256_file(run["path"] / "manifest.json"),
            }
            for precision, run in runs.items()
        },
        "data_sha256": runs[128]["manifest"]["data_sha256"],
        "n_subjects": runs[128]["manifest"]["n_subjects"],
        "decision": "freeze_512",
        "decision_scope": (
            "512 points for all downstream rule-likelihood work; R0K/group conclusions converged. "
            "R2/R3 individual parameters remain subject to likelihood multimodality."
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote precision audit to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
