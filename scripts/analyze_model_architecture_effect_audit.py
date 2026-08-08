#!/usr/bin/env python3
"""Summarize architecture evidence with paired subject-level effect estimates."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_v14_pilot import candidate_id_for, write_json  # noqa: E402


AUDIT_DIR = ROOT / "results/model_architecture_effect_audit"
PAIRED_ROWS = AUDIT_DIR / "paired_ablations/paired_rows.csv"
PILOT_ROWS = ROOT / "results/cond1_v14/pilot_state_readout/pilot_rows.csv"
CONFIRM_ROWS = ROOT / "results/cond1_v14/confirm_gain_readout/pilot_rows.csv"
FROZEN_ROWS = ROOT / "results/cond1_v14/frozen_confirmation/pilot_rows.csv"
V13_BEST = ROOT / "results/zhuran/cond1_v13/cd/cond1_v13/best_hyperparams.json"
V13_CANDIDATES = (
    ROOT
    / "src/Bayesian_state/problems/modules/hypo_transition/candidates"
    / "hypo_transition_profile_v13_candidates.json"
)


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    draws: int = 100_000,
    seed: int = 260727,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def effect_record(
    *,
    mechanism: str,
    evidence_source: str,
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    notes: str,
    seed: int,
) -> dict[str, Any]:
    metrics = ["marginal_choice_brier", "trajectory_crps"]
    left = baseline.set_index("subject_id")[metrics]
    right = ablation.set_index("subject_id")[metrics]
    common = left.index.intersection(right.index)
    left = left.loc[common]
    right = right.loc[common]
    # Positive means the ablation is worse and the mechanism therefore helps.
    delta_brier = right.marginal_choice_brier - left.marginal_choice_brier
    delta_crps = right.trajectory_crps - left.trajectory_crps
    brier_ci = bootstrap_mean_interval(delta_brier, seed=seed)
    crps_ci = bootstrap_mean_interval(delta_crps, seed=seed + 1)
    return {
        "mechanism": mechanism,
        "evidence_source": evidence_source,
        "subject_count": int(len(common)),
        "mean_brier_benefit": float(delta_brier.mean()),
        "brier_ci_low": brier_ci[0],
        "brier_ci_high": brier_ci[1],
        "brier_helped_subjects": int((delta_brier > 0).sum()),
        "mean_crps_benefit": float(delta_crps.mean()),
        "crps_ci_low": crps_ci[0],
        "crps_ci_high": crps_ci[1],
        "crps_helped_subjects": int((delta_crps > 0).sum()),
        "notes": notes,
    }


def paired_ablation_effects(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = rows[rows.variant_id.eq("baseline_corrected")].copy()
    records: list[dict[str, Any]] = []
    subject_records: list[dict[str, Any]] = []
    specs = [
        (
            "latent volatility state (frozen selected mode)",
            rows.variant_id.str.startswith("matched_state_"),
            "The counterfactual direction depends on the frozen subject setting; "
            "the frozen setting was selected on these subjects.",
        ),
        (
            "flexible readout versus expectation",
            rows.variant_id.eq("readout_expectation"),
            "Expectation is the ablation; zero deltas are expected when the frozen "
            "readout was already expectation.",
        ),
        (
            "dynamic hypothesis-specific beta versus static beta=5",
            rows.variant_id.eq("beta_static_5"),
            "Static beta removes feedback-driven per-hypothesis beta evolution.",
        ),
        (
            "dual memory versus fade-only",
            rows.variant_id.eq("memory_fade_only"),
            "Ablation fixes w0=0 while retaining the frozen gamma.",
        ),
        (
            "dual memory versus static-only",
            rows.variant_id.eq("memory_static_only"),
            "Ablation fixes w0=1 while retaining the transition controller.",
        ),
        (
            "calibrated perception noise",
            rows.variant_id.eq("perception_noise_off"),
            "Ablation uses the physical stimulus without sampled perception error.",
        ),
        (
            "label-reversed hypothesis copies",
            rows.variant_id.eq("label_reversals_off"),
            "Subject 103 is additionally forced from label_permuted to all initial "
            "pool because the ablated hypothesis space has no reversed copies.",
        ),
        (
            "choice-informed transition scoring",
            rows.variant_id.eq("choice_evidence_off"),
            "Only frozen controllers that explicitly use posterior_choice or "
            "recent_error_choice contribute.",
        ),
        (
            "dynamic active-set selection versus full hypothesis set",
            rows.variant_id.eq("dynamic_hypothesis_selection_off"),
            "Ablation removes the transition/controller module and updates the "
            "complete configured hypothesis space on every trial.",
        ),
        (
            "controller conservative profile",
            rows.variant_id.eq("profile_conservative_off"),
            "One-profile deletion from each frozen subject-specific controller.",
        ),
        (
            "controller stable profile",
            rows.variant_id.eq("profile_stable_off"),
            "One-profile deletion from each frozen subject-specific controller.",
        ),
        (
            "controller aggressive profile",
            rows.variant_id.eq("profile_aggressive_off"),
            "One-profile deletion from each frozen subject-specific controller.",
        ),
        (
            "controller stubborn profile",
            rows.variant_id.eq("profile_stubborn_off"),
            "One-profile deletion from each frozen subject-specific controller.",
        ),
    ]
    for index, (mechanism, mask, notes) in enumerate(specs):
        ablation = rows[mask].copy()
        if ablation.empty:
            continue
        records.append(
            effect_record(
                mechanism=mechanism,
                evidence_source="corrected paired ablation, 256 repeats/config",
                baseline=baseline,
                ablation=ablation,
                notes=notes,
                seed=260800 + 10 * index,
            )
        )
        base_idx = baseline.set_index("subject_id")
        abl_idx = ablation.set_index("subject_id")
        for subject_id in base_idx.index.intersection(abl_idx.index):
            subject_records.append(
                {
                    "mechanism": mechanism,
                    "subject_id": int(subject_id),
                    "brier_benefit": float(
                        abl_idx.loc[subject_id, "marginal_choice_brier"]
                        - base_idx.loc[subject_id, "marginal_choice_brier"]
                    ),
                    "crps_benefit": float(
                        abl_idx.loc[subject_id, "trajectory_crps"]
                        - base_idx.loc[subject_id, "trajectory_crps"]
                    ),
                }
            )

    label_ablation = rows[
        rows.variant_id.eq("label_reversals_off") & rows.subject_id.ne(103)
    ]
    records.append(
        effect_record(
            mechanism="label-reversed hypothesis copies (unconfounded 7)",
            evidence_source="corrected paired ablation, 256 repeats/config",
            baseline=baseline[baseline.subject_id.ne(103)],
            ablation=label_ablation,
            notes="Excludes subject 103, whose initial pool also changes in this ablation.",
            seed=260890,
        )
    )
    return pd.DataFrame(records), pd.DataFrame(subject_records)


def existing_structural_effects() -> pd.DataFrame:
    rows = pd.read_csv(PILOT_ROWS)
    records: list[dict[str, Any]] = []
    readout_ids = ["sharp2", "sharp4", "map"]
    for index, readout_id in enumerate(readout_ids):
        records.append(
            effect_record(
                mechanism=f"readout {readout_id} versus expectation (pre-fix)",
                evidence_source="V14 structural pilot, 256 repeats/config",
                baseline=rows[rows.variant_id.eq(f"m2_core6_{readout_id}")],
                ablation=rows[rows.variant_id.eq("m2_core6_expectation")],
                notes="Same controller, state gain and random seeds; positive means "
                "the non-expectation readout helps.",
                seed=261000 + 10 * index,
            )
        )
        records.append(
            effect_record(
                mechanism=f"subject controller versus unified, readout={readout_id}",
                evidence_source="V14 structural pilot, 256 repeats/config",
                baseline=rows[rows.variant_id.eq(f"m2_core6_{readout_id}")],
                ablation=rows[rows.variant_id.eq(f"m3_unified_{readout_id}")],
                notes="Unified ablation forces stable_dominant for all subjects.",
                seed=261100 + 10 * index,
            )
        )
    records.append(
        effect_record(
            mechanism="subject controller versus unified, readout=expectation",
            evidence_source="V14 structural pilot, 256 repeats/config",
            baseline=rows[rows.variant_id.eq("m2_core6_expectation")],
            ablation=rows[rows.variant_id.eq("m3_unified_expectation")],
            notes="Unified ablation forces stable_dominant for all subjects.",
            seed=261200,
        )
    )

    confirm = pd.read_csv(CONFIRM_ROWS)
    records.append(
        effect_record(
            mechanism="selected readout flexibility versus frozen V13",
            evidence_source="independent Monte Carlo seed, 512 repeats/config",
            baseline=confirm[confirm.variant_id.eq("m1_selected_state_off")],
            ablation=confirm[confirm.variant_id.eq("m0_v13_saved")],
            notes="Controllers are unchanged; five subjects change readout and three "
            "are exact configuration duplicates.",
            seed=261300,
        )
    )

    frozen = pd.read_csv(FROZEN_ROWS)
    records.append(
        effect_record(
            mechanism="frozen selected latent-state mode versus matched counterfactual (pre-fix)",
            evidence_source="frozen confirmation, 1024 repeats/config",
            baseline=frozen[frozen.variant_id.eq("v14_frozen")],
            ablation=frozen[
                frozen.variant_id.isin(
                    ["matched_state_off", "matched_state_on_g0p35"]
                )
            ],
            notes="Frozen mode was selected before this seed but on the same eight "
            "behavioral trajectories.",
            seed=261400,
        )
    )
    return pd.DataFrame(records)


def full_sample_selection_summary() -> dict[str, Any]:
    best = json.loads(V13_BEST.read_text(encoding="utf-8"))
    candidate_payload = json.loads(V13_CANDIDATES.read_text(encoding="utf-8"))
    candidates = candidate_payload["cond1_v13"]
    controller_counts: Counter[str] = Counter()
    readout_counts: Counter[str] = Counter()
    init_pool_counts: Counter[str] = Counter()
    gamma_counts: Counter[str] = Counter()
    w0_counts: Counter[str] = Counter()
    for subject_id, payload in best["per_subject_best"].items():
        params = payload["selected"]["best_hyperparams"]
        transition = params["engine.modules.hypo_transitions_mod.kwargs"]
        controller_counts[candidate_id_for(transition, candidates)] += 1
        readout = params.get("engine.choice_readout.kwargs", {})
        readout_counts[str(readout.get("method", "expectation"))] += 1
        init_pool_counts[str(transition.get("init_pool", "all"))] += 1
        memory = params["engine.modules.memory_mod.kwargs"]
        gamma_counts[str(memory["gamma"])] += 1
        w0_counts[str(memory["w0"])] += 1
    return {
        "subject_count": len(best["per_subject_best"]),
        "controller_counts": dict(controller_counts.most_common()),
        "controller_family_count_selected": len(controller_counts),
        "readout_counts": dict(readout_counts.most_common()),
        "init_pool_counts": dict(init_pool_counts.most_common()),
        "gamma_counts": dict(gamma_counts.most_common()),
        "w0_counts": dict(w0_counts.most_common()),
        "interpretation": (
            "Selection frequency is in-sample capability evidence, not an "
            "out-of-sample causal ablation."
        ),
    }


def validation_summary(rows: pd.DataFrame) -> dict[str, Any]:
    variant_counts = rows.groupby("subject_id").variant_id.nunique()
    seed_counts = rows.groupby("subject_id").simulation_point_seed.nunique()
    return {
        "row_count": int(len(rows)),
        "subject_count": int(rows.subject_id.nunique()),
        "minimum_variants_per_subject": int(variant_counts.min()),
        "maximum_variants_per_subject": int(variant_counts.max()),
        "common_random_numbers_within_subject": bool(seed_counts.eq(1).all()),
        "simulation_repeats": sorted(
            int(value) for value in rows.simulation_repeats.unique()
        ),
        "metric_null_count": int(
            rows[["marginal_choice_brier", "trajectory_crps"]]
            .isna()
            .sum()
            .sum()
        ),
    }


def main() -> None:
    rows = pd.read_csv(PAIRED_ROWS)
    paired, subject_effects = paired_ablation_effects(rows)
    existing = existing_structural_effects()
    all_effects = pd.concat([paired, existing], ignore_index=True)
    all_effects.to_csv(AUDIT_DIR / "mechanism_effects.csv", index=False)
    subject_effects.to_csv(AUDIT_DIR / "subject_mechanism_effects.csv", index=False)
    payload = {
        "validation": validation_summary(rows),
        "effect_sign_convention": (
            "benefit = loss(ablated model) - loss(baseline); positive values "
            "mean that retaining the mechanism improves fit"
        ),
        "paired_corrected_effects": paired.to_dict(orient="records"),
        "existing_structural_effects": existing.to_dict(orient="records"),
        "full_sample_v13_selection": full_sample_selection_summary(),
        "limitations": [
            "The corrected paired ablations freeze hyperparameters selected under "
            "the pre-fix implementation; they are local effects, not re-optimized "
            "model-family comparisons.",
            "The eight subjects are a representative development set, not a held-out "
            "sample; bootstrap intervals quantify cross-subject dispersion only.",
            "The independent confirmation changes Monte Carlo seeds but reuses the "
            "same observed behavioral trajectories.",
        ],
    }
    write_json(AUDIT_DIR / "effect_audit_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
