#!/usr/bin/env python3
"""Run the condition 1--3 core screen specified in ``model_newplan.tex``.

The script has two CPU-parallel phases:

1. integrate subject-specific perceptual noise with nested Sobol points;
2. fit NR0--NR3 and R0--R3 on each subject's training suffix split and score
   the frozen one-step predictor on the final block.

Outputs are deliberately self-describing.  This is a screening/MAP pipeline,
not the final hierarchical posterior analysis.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.utils.unified_newplan import (  # noqa: E402
    CORE_MODEL_NAMES,
    FEATURE_COLUMNS,
    ORDER_COLUMNS,
    PerceptionSpec,
    audit_dataset,
    build_partition,
    encode_partition_regions,
    fit_core_models,
    integrated_rule_probabilities,
    load_perception_specs,
    metric_rows,
    n_categories,
    score_probabilities,
    sobol_noise,
    subject_seed,
    unique_stimuli,
)


DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/core_sobol128_20260802"
ALL_MODEL_NAMES = (*CORE_MODEL_NAMES, "NR_SELECT", "R_SELECT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sobol-points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--jobs", type=int, default=max(1, min(112, os.cpu_count() or 1)))
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs for a smoke/subset run.",
    )
    parser.add_argument("--force-q", action="store_true")
    parser.add_argument("--skip-q", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["iSub", "condition", *ORDER_COLUMNS, *FEATURE_COLUMNS, "choice", "feedback"]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def q_cache_path(output: Path, subject_id: int) -> Path:
    return output / "q_cache" / f"subject_{int(subject_id)}.npz"


def prediction_path(output: Path, subject_id: int) -> Path:
    return output / "subject_predictions" / f"subject_{int(subject_id)}.npz"


def _cache_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            raw = data["metadata_json"].item()
            metadata = json.loads(str(raw))
        return all(metadata.get(key) == value for key, value in expected.items())
    except Exception:
        return False


def _q_worker(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    stimuli = np.asarray(task["stimuli"], dtype=np.float64)
    spec = PerceptionSpec(
        mode=str(task["perception_mode"]),
        location=np.asarray(task["perception_location"], dtype=float),
        scale=np.asarray(task["perception_scale"], dtype=float),
    )
    noise = sobol_noise(spec, int(task["sobol_points"]), int(task["sobol_seed"]))
    unique, inverse = unique_stimuli(stimuli)
    q_unique = integrated_rule_probabilities(
        unique,
        noise,
        (
            np.asarray(task["region_A"]),
            np.asarray(task["region_b"]),
            np.asarray(task["region_counts"]),
        ),
    )
    q_values = q_unique[inverse]
    metadata = dict(task["expected_metadata"])
    metadata.update(
        {
            "n_trials": int(len(stimuli)),
            "n_unique_stimuli": int(len(unique)),
            "n_hypotheses": int(q_values.shape[1]),
            "n_categories": int(q_values.shape[2]),
            "perception_mode": spec.mode,
            "perception_location": spec.location.tolist(),
            "perception_scale": spec.scale.tolist(),
            "probability_floor": 1e-7,
            "runtime_seconds": float(time.time() - started),
        }
    )
    atomic_savez(
        Path(task["cache_path"]),
        q=q_values.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def _calibration_rows(
    subject_id: int,
    condition: int,
    model: str,
    probabilities: np.ndarray,
    choices: np.ndarray,
    holdout_mask: np.ndarray,
) -> list[dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=float)[holdout_mask]
    observed = np.eye(probabilities.shape[1], dtype=float)[choices[holdout_mask]]
    flat_p = probabilities.ravel()
    flat_y = observed.ravel()
    bins = np.minimum((flat_p * 10.0).astype(int), 9)
    rows = []
    for bin_index in range(10):
        selected = bins == bin_index
        if not np.any(selected):
            continue
        rows.append(
            {
                "subject_id": int(subject_id),
                "condition": int(condition),
                "model": model,
                "bin": int(bin_index),
                "n": int(selected.sum()),
                "sum_probability": float(flat_p[selected].sum()),
                "sum_observed": float(flat_y[selected].sum()),
            }
        )
    return rows


def _block_rows(
    frame: pd.DataFrame,
    model: str,
    probabilities: np.ndarray,
    choices: np.ndarray,
    holdout_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for (session, block), indices in frame.groupby(["iSession", "iBlock"], sort=False).groups.items():
        index = np.asarray(list(indices), dtype=int)
        for segment, segment_mask in (("train", ~holdout_mask), ("holdout", holdout_mask)):
            mask = np.zeros(len(frame), dtype=bool)
            mask[index] = True
            mask &= segment_mask
            if not mask.any():
                continue
            rows.append(
                {
                    "subject_id": int(frame["iSub"].iloc[0]),
                    "condition": int(frame["condition"].iloc[0]),
                    "iSession": int(session),
                    "iBlock": int(block),
                    "model": model,
                    "segment": segment,
                    **score_probabilities(probabilities, choices, mask),
                }
            )
    return rows


def _fit_worker(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    frame = pd.DataFrame(task["frame"]).sort_values(list(ORDER_COLUMNS), kind="stable").reset_index(drop=True)
    subject_id = int(frame["iSub"].iloc[0])
    condition = int(frame["condition"].iloc[0])
    with np.load(Path(task["q_path"]), allow_pickle=False) as q_data:
        q_values = q_data["q"].astype(np.float64)
        q_metadata = json.loads(str(q_data["metadata_json"].item()))
    if q_metadata["frame_fingerprint"] != task["frame_fingerprint"]:
        raise ValueError(f"stale q cache for subject {subject_id}")

    partition = build_partition(condition)
    predictions, parameters, holdout_mask, split_metadata = fit_core_models(
        frame, q_values, partition
    )
    choices = frame["choice"].to_numpy(dtype=np.int64) - 1
    rows = metric_rows(
        subject_id,
        condition,
        predictions,
        parameters,
        choices,
        holdout_mask,
        split_metadata,
    )
    calibration = []
    blocks = []
    output_arrays: dict[str, Any] = {
        "subject_id": np.asarray(subject_id),
        "condition": np.asarray(condition),
        "choice": choices.astype(np.int8),
        "feedback": frame["feedback"].to_numpy(dtype=np.float32),
        "category": (frame["category"].to_numpy(dtype=np.int64) - 1).astype(np.int8),
        "holdout_mask": holdout_mask,
        "iSession": frame["iSession"].to_numpy(dtype=np.int32),
        "iBlock": frame["iBlock"].to_numpy(dtype=np.int32),
        "iTrial": frame["iTrial"].to_numpy(dtype=np.int32),
        "parameters_json": np.asarray(json.dumps(parameters, ensure_ascii=False, sort_keys=True)),
        "split_metadata_json": np.asarray(json.dumps(split_metadata, ensure_ascii=False, sort_keys=True)),
        "q_metadata_json": np.asarray(json.dumps(q_metadata, ensure_ascii=False, sort_keys=True)),
    }
    for model, result in predictions.items():
        output_arrays[f"p_{model}"] = result.probabilities.astype(np.float32)
        output_arrays[f"choice_entropy_{model}"] = result.choice_entropy.astype(np.float32)
        if result.belief_entropy is not None:
            output_arrays[f"belief_entropy_{model}"] = result.belief_entropy.astype(np.float32)
        if result.max_belief is not None:
            output_arrays[f"max_belief_{model}"] = result.max_belief.astype(np.float32)
        calibration.extend(
            _calibration_rows(
                subject_id, condition, model, result.probabilities, choices, holdout_mask
            )
        )
        blocks.extend(
            _block_rows(frame, model, result.probabilities, choices, holdout_mask)
        )
    atomic_savez(Path(task["prediction_path"]), **output_arrays)
    return {
        "subject_id": subject_id,
        "condition": condition,
        "metrics": rows,
        "calibration": calibration,
        "blocks": blocks,
        "parameters": parameters,
        "split": split_metadata,
        "runtime_seconds": float(time.time() - started),
    }


def _bootstrap_interval(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_models(metrics: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = metrics[metrics["segment"] == "holdout"].copy()
    summary_rows = []
    for condition_label, group in [(str(c), holdout[holdout["condition"] == c]) for c in (1, 2, 3)]:
        for model, model_group in group.groupby("model"):
            summary_rows.append(
                {
                    "condition": condition_label,
                    "model": model,
                    "n_subjects": int(model_group["subject_id"].nunique()),
                    "mean_nll_per_trial": float(model_group["nll_per_trial"].mean()),
                    "median_nll_per_trial": float(model_group["nll_per_trial"].median()),
                    "mean_brier": float(model_group["brier"].mean()),
                    "median_brier": float(model_group["brier"].median()),
                    "mean_accuracy": float(model_group["accuracy"].mean()),
                }
            )
    for model, model_group in holdout.groupby("model"):
        summary_rows.append(
            {
                "condition": "all",
                "model": model,
                "n_subjects": int(model_group["subject_id"].nunique()),
                "mean_nll_per_trial": float(model_group["nll_per_trial"].mean()),
                "median_nll_per_trial": float(model_group["nll_per_trial"].median()),
                "mean_brier": float(model_group["brier"].mean()),
                "median_brier": float(model_group["brier"].median()),
                "mean_accuracy": float(model_group["accuracy"].mean()),
            }
        )

    comparisons = [
        ("R_SELECT", "NR_SELECT", "representation_gate"),
        ("R_SELECT", "NR0", "rule_vs_NR0"),
        ("R_SELECT", "NR1", "rule_vs_NR1"),
        ("R_SELECT", "NR2", "rule_vs_NR2"),
        ("R_SELECT", "NR3", "rule_vs_NR3"),
        ("R1", "R0", "retention_increment"),
        ("R0K", "R0", "sensitivity_only_increment"),
        ("R2", "R1", "sensitivity_increment"),
        ("R2", "R0K", "retention_given_sensitivity"),
        ("R2", "R0", "joint_resource_increment"),
        ("R3", "R2", "family_prior_increment"),
    ]
    comparison_rows = []
    for candidate, reference, label in comparisons:
        for condition_label, condition_group in [
            (str(c), holdout[holdout["condition"] == c]) for c in (1, 2, 3)
        ] + [("all", holdout)]:
            candidate_rows = condition_group[condition_group["model"] == candidate]
            reference_rows = condition_group[condition_group["model"] == reference]
            paired = candidate_rows.merge(
                reference_rows,
                on=["subject_id", "condition", "segment"],
                suffixes=("_candidate", "_reference"),
            )
            if paired.empty:
                continue
            delta_total = paired["nll_reference"].to_numpy() - paired["nll_candidate"].to_numpy()
            delta_trial = (
                paired["nll_per_trial_reference"].to_numpy()
                - paired["nll_per_trial_candidate"].to_numpy()
            )
            lower, upper = _bootstrap_interval(
                delta_trial,
                seed + sum(ord(value) for value in f"{candidate}{reference}{condition_label}"),
            )
            nonzero = delta_trial[~np.isclose(delta_trial, 0.0)]
            if len(nonzero):
                try:
                    p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = np.nan
            else:
                p_value = 1.0
            comparison_rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "condition": condition_label,
                    "n_subjects": int(len(paired)),
                    "mean_delta_nll": float(np.mean(delta_total)),
                    "median_delta_nll": float(np.median(delta_total)),
                    "mean_delta_nll_per_trial": float(np.mean(delta_trial)),
                    "median_delta_nll_per_trial": float(np.median(delta_trial)),
                    "bootstrap_mean_ci_low": lower,
                    "bootstrap_mean_ci_high": upper,
                    "n_improved": int(np.sum(delta_trial > 0)),
                    "proportion_improved": float(np.mean(delta_trial > 0)),
                    "wilcoxon_p_uncorrected": p_value,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def aggregate_calibration(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    grouped = (
        rows.groupby(["condition", "model", "bin"], as_index=False)
        .agg(n=("n", "sum"), sum_probability=("sum_probability", "sum"), sum_observed=("sum_observed", "sum"))
    )
    grouped["mean_probability"] = grouped["sum_probability"] / grouped["n"]
    grouped["observed_frequency"] = grouped["sum_observed"] / grouped["n"]
    return grouped


def render_results(
    output: Path,
    audit: dict[str, Any],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    fit_manifest: pd.DataFrame,
    sobol_points: int,
) -> None:
    lines = [
        "# Unified new-plan core screening results",
        "",
        "> Status: first-stage subject-wise optimization/MAP screen. This is not the final hierarchical posterior analysis.",
        "",
        f"Perceptual integration used {int(sobol_points)} fixed nested Sobol points per subject. "
        "All reported model comparisons use a frozen temporal suffix; current-trial feedback is only applied after scoring the current choice.",
        "",
        "## Data and implementation audit",
        "",
        f"- Rows: {audit['n_rows']:,}; subjects: {audit['n_subjects']}.",
    ]
    for condition in (1, 2, 3):
        info = audit["conditions"][str(condition)]
        lines.append(
            f"- Condition {condition}: {info['n_subjects']} subjects, {info['n_rows']:,} trials, "
            f"{info['n_hypotheses']} hypotheses; target-rule/category match "
            f"{info['target_category_match_rate']:.4f}."
        )
    if audit["issues"]:
        lines.extend(
            [
                "- Material audit note: condition 3 subject 319 contains category/feedback inconsistencies in session 5. "
                "The one-step core fit correctly uses the feedback actually recorded as delivered; category-driven autonomous generation for these rows remains blocked pending data provenance resolution.",
            ]
        )

    lines.extend(["", "## Primary representation gate", ""])
    primary = comparisons[comparisons["comparison"] == "representation_gate"]
    if primary.empty:
        lines.append("No completed primary comparisons.")
    else:
        lines.append(
            "Positive ΔNLL means the training-selected rule family predicts held-out choices better than the training-selected non-rule family."
        )
        lines.append("")
        lines.append("| Condition | N | Mean ΔNLL/trial | 95% bootstrap CI | Improved subjects |")
        lines.append("|:--|--:|--:|:--|:--|")
        for row in primary.itertuples(index=False):
            lines.append(
                f"| {row.condition} | {int(row.n_subjects)} | {row.mean_delta_nll_per_trial:.6f} | "
                f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
                f"{int(row.n_improved)}/{int(row.n_subjects)} |"
            )

    lines.extend(["", "## Held-out model summary", ""])
    lines.append("| Condition | Model | Mean NLL/trial | Mean Brier | Mean accuracy |")
    lines.append("|:--|:--|--:|--:|--:|")
    for row in summary.sort_values(["condition", "mean_nll_per_trial", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.model} | {row.mean_nll_per_trial:.6f} | "
            f"{row.mean_brier:.6f} | {row.mean_accuracy:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These results decide whether the rule-representation route is worth advancing. "
            "They do not yet establish stable individual differences, a specific resource mechanism, RT/oral external validity, or autonomous-generative adequacy. "
            "Those claims require recovery, the 128→256 integration stability check, and the frozen downstream validation channels.",
            "",
            "## Reproducible artifacts",
            "",
            "- `subject_model_metrics.csv`: subject-level train/holdout likelihood and Brier metrics.",
            "- `model_summary.csv` and `model_comparisons.csv`: group summaries and paired representation/resource gates.",
            "- `subject_predictions/`: trialwise probabilities and latent uncertainty summaries.",
            "- `q_cache/`: fixed perceptual rule-probability integrals.",
            "- `calibration.csv`, `block_metrics.csv`, `fit_manifest.csv`, `implementation_audit.json`, and `manifest.json`.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    data_path = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "q_cache").mkdir(exist_ok=True)
    (output / "subject_predictions").mkdir(exist_ok=True)

    data = pd.read_csv(data_path, low_memory=False)
    data = data.sort_values(["condition", "iSub", *ORDER_COLUMNS], kind="stable").reset_index(drop=True)
    if args.subjects:
        requested = {int(value.strip()) for value in args.subjects.split(",") if value.strip()}
        data = data[data["iSub"].isin(requested)].copy()
        missing = sorted(requested - set(data["iSub"].astype(int).unique()))
        if missing:
            raise ValueError(f"requested subjects are absent: {missing}")
    subjects = sorted(int(value) for value in data["iSub"].unique())
    if not subjects:
        raise ValueError("no subjects selected")

    full_data = pd.read_csv(data_path, low_memory=False)
    audit = audit_dataset(full_data)
    atomic_json(output / "implementation_audit.json", audit)
    specs = load_perception_specs(data_path.parent, feature_order_path=data_path)
    missing_specs = sorted(set(subjects) - set(specs))
    if missing_specs:
        raise ValueError(f"missing Task-1b perception distributions: {missing_specs}")

    partitions = {condition: build_partition(condition) for condition in (1, 2, 3)}
    region_arrays = {
        condition: encode_partition_regions(partition)
        for condition, partition in partitions.items()
    }

    # Warm numba before forking so workers inherit compiled machine code.
    for condition in sorted(set(int(value) for value in data["condition"].unique())):
        integrated_rule_probabilities(
            np.full((1, 4), 0.5),
            np.zeros((2, 4)),
            region_arrays[condition],
        )

    q_tasks = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        condition = int(frame["condition"].iloc[0])
        frame = frame.sort_values(list(ORDER_COLUMNS), kind="stable").reset_index(drop=True)
        fingerprint = frame_fingerprint(frame)
        expected = {
            "subject_id": subject_id,
            "condition": condition,
            "sobol_points": int(args.sobol_points),
            "sobol_seed": int(subject_seed(args.seed, subject_id)),
            "base_seed": int(args.seed),
            "frame_fingerprint": fingerprint,
        }
        path = q_cache_path(output, subject_id)
        if args.force_q or not _cache_matches(path, expected):
            spec = specs[subject_id]
            A, b, counts = region_arrays[condition]
            q_tasks.append(
                {
                    "stimuli": frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
                    "perception_mode": spec.mode,
                    "perception_location": spec.location,
                    "perception_scale": spec.scale,
                    "sobol_points": int(args.sobol_points),
                    "sobol_seed": expected["sobol_seed"],
                    "region_A": A,
                    "region_b": b,
                    "region_counts": counts,
                    "cache_path": str(path),
                    "expected_metadata": expected,
                }
            )

    if args.skip_q and q_tasks:
        raise ValueError(f"--skip-q requested but {len(q_tasks)} q caches are missing/stale")
    if q_tasks:
        print(f"[q] computing {len(q_tasks)} subject caches with {args.jobs} workers", flush=True)
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(q_tasks)), mp_context=context) as pool:
            futures = {pool.submit(_q_worker, task): task for task in q_tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                metadata = future.result()
                print(
                    f"[q] {completed}/{len(q_tasks)} s{metadata['subject_id']} "
                    f"({metadata['runtime_seconds']:.1f}s)",
                    flush=True,
                )
    else:
        print("[q] all caches are current", flush=True)

    if args.skip_fit:
        print(f"[done] q-only run in {time.time() - started:.1f}s", flush=True)
        return 0

    fit_tasks = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        frame = frame.sort_values(list(ORDER_COLUMNS), kind="stable").reset_index(drop=True)
        fit_tasks.append(
            {
                "frame": frame.to_dict(orient="list"),
                "frame_fingerprint": frame_fingerprint(frame),
                "q_path": str(q_cache_path(output, subject_id)),
                "prediction_path": str(prediction_path(output, subject_id)),
            }
        )

    metric_payload: list[dict[str, Any]] = []
    calibration_payload: list[dict[str, Any]] = []
    block_payload: list[dict[str, Any]] = []
    fit_payload: list[dict[str, Any]] = []
    errors = []
    print(f"[fit] fitting {len(fit_tasks)} subjects with {args.jobs} workers", flush=True)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(fit_tasks)), mp_context=context) as pool:
        futures = {pool.submit(_fit_worker, task): task for task in fit_tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                payload = future.result()
            except Exception as exc:
                errors.append({"error": repr(exc), "traceback": traceback.format_exc()})
                print(f"[fit] ERROR {exc!r}", flush=True)
                continue
            metric_payload.extend(payload["metrics"])
            calibration_payload.extend(payload["calibration"])
            block_payload.extend(payload["blocks"])
            fit_payload.append(
                {
                    "subject_id": payload["subject_id"],
                    "condition": payload["condition"],
                    "runtime_seconds": payload["runtime_seconds"],
                    "split_json": json.dumps(payload["split"], ensure_ascii=False, sort_keys=True),
                    "parameters_json": json.dumps(payload["parameters"], ensure_ascii=False, sort_keys=True),
                }
            )
            print(
                f"[fit] {completed}/{len(fit_tasks)} s{payload['subject_id']} "
                f"({payload['runtime_seconds']:.1f}s)",
                flush=True,
            )
    if errors:
        atomic_json(output / "fit_errors.json", errors)
        raise RuntimeError(f"{len(errors)} subject fits failed; see fit_errors.json")

    metrics = pd.DataFrame(metric_payload).sort_values(
        ["condition", "subject_id", "segment", "model"]
    )
    calibration_subject = pd.DataFrame(calibration_payload)
    calibration = aggregate_calibration(calibration_subject)
    block_metrics = pd.DataFrame(block_payload).sort_values(
        ["condition", "subject_id", "iSession", "iBlock", "segment", "model"]
    )
    fit_manifest = pd.DataFrame(fit_payload).sort_values(["condition", "subject_id"])
    summary, comparisons = summarize_models(metrics, args.seed)

    metrics.to_csv(output / "subject_model_metrics.csv", index=False)
    summary.to_csv(output / "model_summary.csv", index=False)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    calibration_subject.to_csv(output / "calibration_subject_sufficient_stats.csv", index=False)
    block_metrics.to_csv(output / "block_metrics.csv", index=False)
    fit_manifest.to_csv(output / "fit_manifest.csv", index=False)

    manifest = {
        "result_type": "unified_newplan_core_map_screen",
        "status": "complete",
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "sobol_points": int(args.sobol_points),
        "base_seed": int(args.seed),
        "jobs": int(args.jobs),
        "models": list(ALL_MODEL_NAMES),
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "module_sha256": sha256_file(
            ROOT / "src/Bayesian_state/utils/unified_newplan.py"
        ),
        "evidence_scope": (
            "subject-wise optimization/MAP screening with temporal holdout; "
            "not final hierarchical posterior inference"
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    render_results(output, audit, summary, comparisons, fit_manifest, args.sobol_points)
    print(f"[done] wrote {output} in {manifest['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
