#!/usr/bin/env python3
"""Audit FA2R particle-likelihood stability on a frozen synthetic subset."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0803_cond1 import (  # noqa: E402
    build_frozen_geometry,
    load_config,
)
from scripts.run_model_0804_regeneration_recovery import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _ensemble_nll,
    _score_candidate,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_particle_stability_audit.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results/zhuran/model_0804_cond1/particle_stability_audit_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--stop-after",
        choices=("n2048", "n8192", "n32768"),
        default=None,
        help="testing/resume aid; omitted for the frozen complete audit",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("stability config must contain a mapping")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _original_seed_winners(row: Mapping[str, Any]) -> list[str]:
    ranking = list(row["confirmed_ranking"])
    seed_count = len(ranking[0]["seed_nll"])
    return [
        str(min(ranking, key=lambda item: item["seed_nll"][index])["id"])
        for index in range(seed_count)
    ]


def _select_source_rows(
    report: Mapping[str, Any], slots: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["dataset_id"]): row for row in report["datasets"]}
    if len(by_id) != len(report["datasets"]):
        raise ValueError("source report has duplicate dataset ids")
    selected: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for slot in slots:
        dataset_id = str(slot["dataset_id"])
        if dataset_id in observed_ids:
            raise ValueError(f"duplicate selected dataset: {dataset_id}")
        if dataset_id not in by_id:
            raise ValueError(f"selected dataset is absent from source: {dataset_id}")
        row = dict(by_id[dataset_id])
        if int(row["subject_id"]) != int(slot["subject_id"]):
            raise ValueError(f"subject mismatch for {dataset_id}")
        if str(row["scenario_id"]) != str(slot["scenario_id"]):
            raise ValueError(f"scenario mismatch for {dataset_id}")
        winners = _original_seed_winners(row)
        agreement = len(set(winners)) == 1
        if agreement != bool(slot["expected_original_seed_agreement"]):
            raise ValueError(f"original seed stratum mismatch for {dataset_id}")
        row["original_seed_winners"] = winners
        row["original_seed_winner_agreement"] = agreement
        selected.append(row)
        observed_ids.add(dataset_id)
    return selected


def _candidate_lookup(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = {
        str(row["id"]): dict(row) for row in report["candidate_grid"]
    }
    if len(candidates) != len(report["candidate_grid"]):
        raise ValueError("source candidate grid contains duplicate ids")
    return candidates


def _confirmed_candidates(
    source_row: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    confirmed_ids = {str(row["id"]) for row in source_row["confirmed_ranking"]}
    missing = confirmed_ids - set(lookup)
    if missing:
        raise ValueError(f"confirmed candidates absent from grid: {sorted(missing)}")
    candidates = [
        dict(candidate)
        for candidate in lookup.values()
        if str(candidate["id"]) in confirmed_ids
    ]
    if len(candidates) != int(source_row["confirmed_candidate_count"]):
        raise ValueError("confirmed candidate count mismatch")
    return candidates


def _checkpoint_metadata(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "analysis_id": str(payload["analysis_id"]),
            "config_sha256": str(payload["config_sha256"]),
            "dataset_id": str(payload["dataset_id"]),
            "particle_count": int(payload["particle_count"]),
            "seeds": [int(value) for value in payload["seeds"]],
            "candidate_ids": [str(row["id"]) for row in payload["candidates"]],
            "model_sha256": str(payload["model_sha256"]),
            "recovery_sha256": str(payload["recovery_sha256"]),
        },
        sort_keys=True,
    )


def _run_level_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Process boundary: score every confirmed candidate for one N and dataset."""

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(int(payload["worker_blas_threads"]))
    checkpoint = Path(payload["checkpoint_path"])
    metadata = _checkpoint_metadata(payload)
    candidates = list(payload["candidates"])
    seeds = [int(value) for value in payload["seeds"]]
    shape = (len(candidates), len(seeds))
    nll = np.full(shape, np.nan)
    runtime = np.full(shape, np.nan)
    minimum_ess = np.full(shape, np.nan)
    resampling_count = np.full(shape, np.nan)
    if checkpoint.exists() and not bool(payload["force"]):
        with np.load(checkpoint, allow_pickle=False) as stored:
            if str(stored["metadata_json"].item()) == metadata:
                nll = stored["nll"].astype(float)
                runtime = stored["runtime"].astype(float)
                minimum_ess = stored["minimum_ess"].astype(float)
                resampling_count = stored["resampling_count"].astype(float)

    reused = payload.get("reused_nll", {})
    for candidate_index, candidate in enumerate(candidates):
        candidate_values = reused.get(str(candidate["id"]), {})
        for seed_index, seed in enumerate(seeds):
            if str(seed) in candidate_values and not np.isfinite(nll[candidate_index, seed_index]):
                nll[candidate_index, seed_index] = float(candidate_values[str(seed)])

    missing = np.argwhere(~np.isfinite(nll))
    print(
        f"[stability] dataset={payload['dataset_id']} N={payload['particle_count']} "
        f"missing={len(missing)}/{nll.size}",
        flush=True,
    )
    started = time.time()
    for completed, (candidate_index, seed_index) in enumerate(missing, start=1):
        value, diagnostics = _score_candidate(
            np.asarray(payload["q_values"]),
            np.asarray(payload["choices"]),
            np.asarray(payload["feedback"]),
            np.asarray(payload["prior"]),
            payload["kernels"],
            model_id=str(payload["model_id"]),
            architecture=payload["architecture"],
            candidate=candidates[int(candidate_index)],
            particle_count=int(payload["particle_count"]),
            filter_seed=seeds[int(seed_index)],
            resample_threshold_fraction=float(payload["resample_threshold_fraction"]),
        )
        nll[candidate_index, seed_index] = value
        runtime[candidate_index, seed_index] = diagnostics["runtime_seconds"]
        minimum_ess[candidate_index, seed_index] = diagnostics["minimum_pre_choice_ess"]
        resampling_count[candidate_index, seed_index] = diagnostics["resampling_count"]
        _atomic_npz(
            checkpoint,
            metadata_json=np.asarray(metadata),
            nll=nll,
            runtime=runtime,
            minimum_ess=minimum_ess,
            resampling_count=resampling_count,
        )
        if completed % 8 == 0 or completed == len(missing):
            print(
                f"[stability] dataset={payload['dataset_id']} N={payload['particle_count']} "
                f"done={completed}/{len(missing)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    return {
        "dataset_id": str(payload["dataset_id"]),
        "tier_id": str(payload["tier_id"]),
        "particle_count": int(payload["particle_count"]),
        "candidate_ids": [str(row["id"]) for row in candidates],
        "seeds": seeds,
        "nll": nll.tolist(),
        "runtime": runtime.tolist(),
        "minimum_ess": minimum_ess.tolist(),
        "resampling_count": resampling_count.tolist(),
        "new_compute_runtime_seconds": float(np.nansum(runtime)),
    }


def _standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _modal_fraction(values: Sequence[Any]) -> tuple[Any, float]:
    counts = Counter(values)
    mode, count = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0]
    return mode, float(count / len(values))


def _kendall_tau(order_left: Sequence[str], order_right: Sequence[str]) -> float:
    if set(order_left) != set(order_right) or len(order_left) != len(order_right):
        raise ValueError("Kendall orders must contain the same unique ids")
    positions = {value: index for index, value in enumerate(order_right)}
    concordant = 0
    discordant = 0
    for left in range(len(order_left)):
        for right in range(left + 1, len(order_left)):
            if positions[order_left[left]] < positions[order_left[right]]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return 1.0 if total == 0 else float((concordant - discordant) / total)


def _analyse_level(
    raw: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    true_candidate_id: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    nll = np.asarray(raw["nll"], dtype=float)
    if nll.shape != (len(candidates), len(raw["seeds"])) or np.any(~np.isfinite(nll)):
        raise ValueError("complete finite NLL matrix required")
    combined = np.asarray([_ensemble_nll(row) for row in nll])
    order = np.argsort(combined, kind="stable")
    ids = [str(row["id"]) for row in candidates]
    best_index = int(order[0])
    seed_winner_indices = np.argmin(nll, axis=0)
    seed_winner_ids = [ids[int(index)] for index in seed_winner_indices]
    modal_winner, modal_winner_fraction = _modal_fraction(seed_winner_ids)
    seed_rho_classes = [
        "positive" if float(candidates[int(index)]["rho"]) > 0 else "zero"
        for index in seed_winner_indices
    ]
    modal_rho_class, modal_rho_fraction = _modal_fraction(seed_rho_classes)

    delta = combined - combined[best_index]
    plausible = np.flatnonzero(
        delta <= float(thresholds["plausible_candidate_delta_nll"])
    )
    plausible_difference_se = [
        _standard_error(nll[index] - nll[best_index])
        for index in plausible
        if int(index) != best_index
    ]
    maximum_difference_se = float(max(plausible_difference_se, default=0.0))
    true_positions = [index for index, value in enumerate(ids) if value == true_candidate_id]
    if len(true_positions) != 1:
        raise ValueError("true candidate must occur once in confirmed set")
    true_index = int(true_positions[0])
    true_nll_se = _standard_error(nll[true_index])
    numerical_resolved = bool(
        maximum_difference_se
        <= float(thresholds["maximum_plausible_candidate_paired_difference_se_nll"])
        and true_nll_se
        <= float(thresholds["maximum_true_candidate_absolute_nll_se"])
    )

    g_profiles: dict[str, Any] = {}
    for g_value in sorted({float(candidate["g"]) for candidate in candidates}):
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if np.isclose(float(candidate["g"]), g_value, atol=1e-12)
        ]
        winner = min(eligible, key=lambda index: combined[index])
        g_profiles[f"{g_value:.3f}"] = {
            "candidate_id": ids[winner],
            "combined_nll": float(combined[winner]),
            "delta_nll": float(delta[winner]),
            "paired_difference_se_vs_best": _standard_error(
                nll[winner] - nll[best_index]
            ),
        }

    rows = []
    for index in order:
        candidate = candidates[int(index)]
        rows.append(
            {
                **dict(candidate),
                "combined_nll": float(combined[index]),
                "delta_nll": float(delta[index]),
                "seed_nll": nll[index].tolist(),
                "seed_nll_se": _standard_error(nll[index]),
            }
        )
    return {
        "tier_id": str(raw["tier_id"]),
        "particle_count": int(raw["particle_count"]),
        "seeds": [int(value) for value in raw["seeds"]],
        "ensemble_winner": dict(candidates[best_index]),
        "ensemble_order": [ids[int(index)] for index in order],
        "seed_winner_ids": seed_winner_ids,
        "seed_modal_exact_winner": modal_winner,
        "seed_modal_exact_winner_fraction": modal_winner_fraction,
        "seed_rho_classes": seed_rho_classes,
        "seed_modal_rho_class": modal_rho_class,
        "seed_modal_rho_class_fraction": modal_rho_fraction,
        "plausible_candidate_count": int(len(plausible)),
        "maximum_plausible_candidate_paired_difference_se_nll": maximum_difference_se,
        "true_candidate_absolute_nll_se": true_nll_se,
        "true_candidate_delta_nll": float(delta[true_index]),
        "true_candidate_within_2_nll": bool(
            delta[true_index] <= float(thresholds["plausible_candidate_delta_nll"])
        ),
        "numerically_resolved": numerical_resolved,
        "g_profiles": g_profiles,
        "candidate_ranking": rows,
        "new_compute_runtime_seconds": float(raw["new_compute_runtime_seconds"]),
    }


def _requires_escalation(
    baseline: Mapping[str, Any],
    high: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    rules = config["escalation"]
    thresholds = config["diagnostic_thresholds"]
    reasons: list[str] = []
    if bool(rules["escalate_if_n8192_numerically_unresolved"]) and not bool(
        high["numerically_resolved"]
    ):
        reasons.append("n8192_numerically_unresolved")
    if bool(rules["escalate_if_exact_ensemble_winner_changed_from_n2048"]) and str(
        baseline["ensemble_winner"]["id"]
    ) != str(high["ensemble_winner"]["id"]):
        reasons.append("ensemble_winner_changed_from_n2048")
    if bool(rules["escalate_if_seed_modal_exact_winner_below_threshold"]) and float(
        high["seed_modal_exact_winner_fraction"]
    ) < float(thresholds["minimum_seed_modal_exact_winner_fraction"]):
        reasons.append("seed_modal_exact_winner_below_threshold")
    if bool(rules["escalate_if_seed_rho_class_below_threshold"]) and float(
        high["seed_modal_rho_class_fraction"]
    ) < float(thresholds["minimum_seed_modal_rho_class_fraction"]):
        reasons.append("seed_rho_class_below_threshold")
    return bool(reasons), reasons


def _rate(flags: Sequence[bool]) -> dict[str, Any]:
    successes = int(sum(bool(value) for value in flags))
    total = int(len(flags))
    return {"successes": successes, "total": total, "value": float(successes / total)}


def _aggregate(
    datasets: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    thresholds = config["diagnostic_thresholds"]
    final_levels = [row["final_level"] for row in datasets]
    numerical = _rate([bool(row["numerically_resolved"]) for row in final_levels])
    difference_precision = _rate(
        [
            float(row["maximum_plausible_candidate_paired_difference_se_nll"])
            <= float(thresholds["maximum_plausible_candidate_paired_difference_se_nll"])
            for row in final_levels
        ]
    )
    absolute_precision = _rate(
        [
            float(row["true_candidate_absolute_nll_se"])
            <= float(thresholds["maximum_true_candidate_absolute_nll_se"])
            for row in final_levels
        ]
    )
    winner_stable = _rate(
        [
            float(row["seed_modal_exact_winner_fraction"])
            >= float(config["diagnostic_thresholds"]["minimum_seed_modal_exact_winner_fraction"])
            for row in final_levels
        ]
    )
    rho_stable = _rate(
        [
            float(row["seed_modal_rho_class_fraction"])
            >= float(config["diagnostic_thresholds"]["minimum_seed_modal_rho_class_fraction"])
            for row in final_levels
        ]
    )
    g_extremes = [
        row for row in datasets if str(row["scenario_id"]) in {"g_low", "g_high"}
    ]
    g_exact = _rate(
        [
            np.isclose(
                float(row["final_level"]["ensemble_winner"]["g"]),
                float(row["true_candidate"]["g"]),
                atol=1e-12,
            )
            for row in g_extremes
        ]
    )
    g_direction = _rate(
        [
            float(row["final_level"]["ensemble_winner"]["g"]) < 0.35
            if str(row["scenario_id"]) == "g_low"
            else float(row["final_level"]["ensemble_winner"]["g"]) > 0.35
            for row in g_extremes
        ]
    )
    true_within = _rate(
        [bool(row["true_candidate_within_2_nll"]) for row in final_levels]
    )
    rho_class_recovery = _rate(
        [
            (float(dataset["true_candidate"]["rho"]) > 0.0)
            == (float(dataset["final_level"]["ensemble_winner"]["rho"]) > 0.0)
            for dataset in datasets
        ]
    )
    tier_precision = {}
    for tier_id in ("n2048", "n8192"):
        levels = [row["levels"][tier_id] for row in datasets]
        tier_precision[tier_id] = {
            "numerically_resolved_dataset_count": int(
                sum(bool(level["numerically_resolved"]) for level in levels)
            ),
            "median_maximum_plausible_candidate_paired_difference_se_nll": float(
                np.median(
                    [
                        float(level["maximum_plausible_candidate_paired_difference_se_nll"])
                        for level in levels
                    ]
                )
            ),
            "maximum_plausible_candidate_paired_difference_se_nll": float(
                np.max(
                    [
                        float(level["maximum_plausible_candidate_paired_difference_se_nll"])
                        for level in levels
                    ]
                )
            ),
            "median_true_candidate_absolute_nll_se": float(
                np.median(
                    [float(level["true_candidate_absolute_nll_se"]) for level in levels]
                )
            ),
            "maximum_true_candidate_absolute_nll_se": float(
                np.max(
                    [float(level["true_candidate_absolute_nll_se"]) for level in levels]
                )
            ),
        }
    tier_precision["final_adaptive_tier"] = {
        "numerically_resolved_dataset_count": int(numerical["successes"]),
        "median_maximum_plausible_candidate_paired_difference_se_nll": float(
            np.median(
                [
                    float(level["maximum_plausible_candidate_paired_difference_se_nll"])
                    for level in final_levels
                ]
            )
        ),
        "maximum_plausible_candidate_paired_difference_se_nll": float(
            np.max(
                [
                    float(level["maximum_plausible_candidate_paired_difference_se_nll"])
                    for level in final_levels
                ]
            )
        ),
        "median_true_candidate_absolute_nll_se": float(
            np.median(
                [float(level["true_candidate_absolute_nll_se"]) for level in final_levels]
            )
        ),
        "maximum_true_candidate_absolute_nll_se": float(
            np.max(
                [float(level["true_candidate_absolute_nll_se"]) for level in final_levels]
            )
        ),
    }
    threshold = float(
        config["diagnostic_thresholds"]["minimum_numerically_resolved_dataset_fraction"]
    )
    numerically_resolved = bool(numerical["value"] >= threshold)
    route = str(
        config["routing"][
            "route_if_numerically_resolved"
            if numerically_resolved
            else "route_if_numerically_unresolved"
        ]
    )
    return {
        "dataset_count": len(datasets),
        "numerically_resolved": {**numerical, "threshold": threshold, "passed": numerically_resolved},
        "plausible_candidate_difference_precision": difference_precision,
        "true_candidate_absolute_nll_precision": absolute_precision,
        "seed_exact_winner_stable": winner_stable,
        "seed_rho_class_stable": rho_stable,
        "true_candidate_within_2_nll": true_within,
        "rho_zero_vs_positive_ensemble_recovery": rho_class_recovery,
        "g_extreme_ensemble_exact_recovery": g_exact,
        "g_extreme_ensemble_direction_recovery": g_direction,
        "escalated_dataset_count": int(sum(bool(row["escalated"]) for row in datasets)),
        "ensemble_winner_changed_n8192_to_final_count": int(
            sum(
                str(row["levels"]["n8192"]["ensemble_winner"]["id"])
                != str(row["final_level"]["ensemble_winner"]["id"])
                for row in datasets
            )
        ),
        "median_rank_kendall_n8192_vs_n32768_among_escalated": float(
            np.median(
                [
                    float(row["rank_kendall_n8192_vs_n32768"])
                    for row in datasets
                    if bool(row["escalated"])
                ]
            )
        ),
        "particle_tier_precision": tier_precision,
        "interpretation_boundary": (
            "diagnoses_particle_likelihood_precision_on_a_stratified_subset; "
            "does_not_replace_the_180_dataset_recovery_gate"
        ),
    }, route


def _run_payloads(payloads: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_run_level_payload(payload) for payload in payloads]
    results: list[dict[str, Any] | None] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_level_payload, payload): index
            for index, payload in enumerate(payloads)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            print(f"[stability] completed={completed}/{len(payloads)}", flush=True)
    if any(row is None for row in results):
        raise AssertionError("incomplete stability worker results")
    return [row for row in results if row is not None]


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    previous_report_path = output / "particle_stability_report.json"
    previous_compute_wall_runtime = None
    if previous_report_path.exists() and not args.force:
        try:
            previous_report = _load_json(previous_report_path)
            previous_compute_wall_runtime = float(
                previous_report.get(
                    "compute_wall_runtime_seconds",
                    previous_report["runtime_seconds"],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            previous_compute_wall_runtime = None
    config = _load_yaml(config_path)
    source_path = (ROOT / str(config["source"]["report"])).resolve()
    observed_source_hash = _sha256(source_path)
    if observed_source_hash != str(config["source"]["expected_report_sha256"]):
        raise ValueError("source recovery report hash does not match frozen config")
    report = _load_json(source_path)
    if int(report["confirmation_particle_count"]) != int(
        config["source"]["original_particle_count"]
    ):
        raise ValueError("source particle count mismatch")
    original_seeds = [int(value) for value in config["source"]["original_seeds"]]
    slots = list(config["selection"]["datasets"])
    source_rows = _select_source_rows(report, slots)
    agreement_count = int(sum(row["original_seed_winner_agreement"] for row in source_rows))
    if agreement_count * 2 != len(source_rows):
        raise ValueError("selection must balance original agreement/disagreement")
    if sorted({int(row["subject_id"]) for row in source_rows}) != [101, 102, 103, 104]:
        raise ValueError("selection must cover all four frozen q-sequences")

    base_path = Path(report["base_config_path"]).resolve()
    if _sha256(base_path) != str(report["base_config_sha256"]):
        raise ValueError("base config changed since source recovery")
    base = load_config(base_path)
    priors, kernels_by_prior, _ = build_frozen_geometry(base)
    prior_id = str(base["rule_space"]["primary_prior"])
    prior = priors[prior_id]
    kernels = kernels_by_prior[prior_id]
    candidate_lookup = _candidate_lookup(report)
    config_sha256 = _sha256(config_path)
    model_path = ROOT / "src/Bayesian_state/manuscript_models/model_0804.py"
    recovery_path = ROOT / "src/Bayesian_state/manuscript_models/model_0804_recovery.py"
    model_sha256 = _sha256(model_path)
    recovery_sha256 = _sha256(recovery_path)
    if model_sha256 != str(report["implementation_sha256"]["model_0804.py"]):
        raise ValueError("model implementation changed since source recovery")
    if recovery_sha256 != str(report["implementation_sha256"]["model_0804_recovery.py"]):
        raise ValueError("recovery implementation changed since source recovery")

    workers = int(args.workers or config["execution"]["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    worker_threads = int(config["execution"]["worker_blas_threads"])
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(worker_threads)
    seeds = [int(value) for value in config["filter"]["seeds"]]
    if seeds[: len(original_seeds)] != original_seeds:
        raise ValueError("audit seed prefix must equal original confirmation seeds")

    tiers = list(config["filter"]["tiers"])
    tier_by_id = {str(row["id"]): row for row in tiers}
    required_tiers = {"n2048", "n8192", "n32768"}
    if set(tier_by_id) != required_tiers:
        raise ValueError("audit requires exactly n2048, n8192, and n32768 tiers")
    datasets: dict[str, dict[str, Any]] = {}
    common_payloads: dict[str, dict[str, Any]] = {}
    for source_row in source_rows:
        dataset_id = str(source_row["dataset_id"])
        candidates = _confirmed_candidates(source_row, candidate_lookup)
        dataset_path = source_path.parent / "datasets" / f"{dataset_id}.npz"
        with np.load(dataset_path, allow_pickle=False) as stored:
            arrays = {
                name: stored[name].copy()
                for name in ("q_values", "choices", "feedback")
            }
        common_payloads[dataset_id] = {
            "analysis_id": config["analysis_id"],
            "config_sha256": config_sha256,
            "dataset_id": dataset_id,
            "q_values": arrays["q_values"],
            "choices": arrays["choices"],
            "feedback": arrays["feedback"],
            "prior": prior,
            "kernels": kernels,
            "model_id": report["model_id"],
            "architecture": report["architecture"],
            "candidates": candidates,
            "seeds": seeds,
            "resample_threshold_fraction": config["filter"]["resample_threshold_fraction"],
            "worker_blas_threads": worker_threads,
            "model_sha256": model_sha256,
            "recovery_sha256": recovery_sha256,
            "force": bool(args.force),
        }
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "subject_id": int(source_row["subject_id"]),
            "scenario_id": str(source_row["scenario_id"]),
            "replicate": int(source_row["replicate"]),
            "n_trials": int(source_row["n_trials"]),
            "true_candidate": dict(source_row["true_candidate"]),
            "candidate_count": len(candidates),
            "original_seed_winners": source_row["original_seed_winners"],
            "original_seed_winner_agreement": bool(source_row["original_seed_winner_agreement"]),
            "levels": {},
        }

    def payload_for(dataset_id: str, tier_id: str) -> dict[str, Any]:
        tier = tier_by_id[tier_id]
        reused_nll: dict[str, dict[str, float]] = {}
        if bool(tier.get("reuse_original_seed_values", False)):
            source_row = next(row for row in source_rows if row["dataset_id"] == dataset_id)
            for candidate in source_row["confirmed_ranking"]:
                reused_nll[str(candidate["id"])] = {
                    str(seed): float(value)
                    for seed, value in zip(original_seeds, candidate["seed_nll"])
                }
        return {
            **common_payloads[dataset_id],
            "tier_id": tier_id,
            "particle_count": int(tier["particle_count"]),
            "checkpoint_path": output / "checkpoints" / f"{dataset_id}_{tier_id}.npz",
            "reused_nll": reused_nll,
        }

    for tier_id in ("n2048", "n8192"):
        raw_results = _run_payloads(
            [payload_for(dataset_id, tier_id) for dataset_id in datasets], workers
        )
        for raw in raw_results:
            dataset = datasets[str(raw["dataset_id"])]
            level = _analyse_level(
                raw,
                common_payloads[str(raw["dataset_id"])]["candidates"],
                str(dataset["true_candidate"]["id"]),
                config["diagnostic_thresholds"],
            )
            dataset["levels"][tier_id] = level
        if args.stop_after == tier_id:
            break

    complete_through_high = all("n8192" in row["levels"] for row in datasets.values())
    if complete_through_high and args.stop_after not in {"n2048", "n8192"}:
        escalated_ids = []
        for dataset_id, dataset in datasets.items():
            escalated, reasons = _requires_escalation(
                dataset["levels"]["n2048"], dataset["levels"]["n8192"], config
            )
            dataset["escalated"] = escalated
            dataset["escalation_reasons"] = reasons
            if escalated:
                escalated_ids.append(dataset_id)
        raw_results = _run_payloads(
            [payload_for(dataset_id, "n32768") for dataset_id in escalated_ids], workers
        )
        for raw in raw_results:
            dataset = datasets[str(raw["dataset_id"])]
            dataset["levels"]["n32768"] = _analyse_level(
                raw,
                common_payloads[str(raw["dataset_id"])]["candidates"],
                str(dataset["true_candidate"]["id"]),
                config["diagnostic_thresholds"],
            )

    complete = complete_through_high and all(
        (not row.get("escalated", False)) or "n32768" in row["levels"]
        for row in datasets.values()
    )
    rows = list(datasets.values())
    for row in rows:
        if "n8192" in row["levels"]:
            row["rank_kendall_n2048_vs_n8192"] = _kendall_tau(
                row["levels"]["n2048"]["ensemble_order"],
                row["levels"]["n8192"]["ensemble_order"],
            )
        final_tier = "n32768" if "n32768" in row["levels"] else (
            "n8192" if "n8192" in row["levels"] else "n2048"
        )
        row["final_tier_id"] = final_tier
        row["final_level"] = row["levels"][final_tier]
        if "n32768" in row["levels"]:
            row["rank_kendall_n8192_vs_n32768"] = _kendall_tau(
                row["levels"]["n8192"]["ensemble_order"],
                row["levels"]["n32768"]["ensemble_order"],
            )

    if complete:
        aggregate, route = _aggregate(rows, config)
        status = "particle_stability_audit_complete"
    else:
        aggregate = None
        route = "partial_run_no_route_decision"
        status = "particle_stability_audit_partial"
    refresh_runtime = float(time.time() - started)
    compute_wall_runtime = (
        refresh_runtime
        if previous_compute_wall_runtime is None
        else previous_compute_wall_runtime
    )
    report_payload = {
        "analysis_id": config["analysis_id"],
        "status": status,
        "scope": config["scope"],
        "route_decision": route,
        "runtime_seconds": compute_wall_runtime,
        "compute_wall_runtime_seconds": compute_wall_runtime,
        "report_refresh_runtime_seconds": refresh_runtime,
        "source_report_path": str(source_path),
        "source_report_sha256": observed_source_hash,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "implementation_sha256": {
            "model_0804.py": model_sha256,
            "model_0804_recovery.py": recovery_sha256,
            "particle_stability_runner": _sha256(Path(__file__).resolve()),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "selection_policy": config["selection"]["policy"],
        "selected_dataset_count": len(rows),
        "original_seed_agreement_count": agreement_count,
        "original_seed_disagreement_count": len(rows) - agreement_count,
        "diagnostic_thresholds": dict(config["diagnostic_thresholds"]),
        "guardrails": list(config["guardrails"]),
        "aggregate": aggregate,
        "datasets": rows,
    }
    output_path = output / "particle_stability_report.json"
    _atomic_json(output_path, report_payload)
    print(
        f"STABILITY status={status} route={route} output={output_path} "
        f"runtime={report_payload['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
