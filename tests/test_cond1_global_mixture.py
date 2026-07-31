from __future__ import annotations

import pandas as pd

from scripts.summarize_cond1_global_mechanism_mixture import (
    _candidate_bank,
    _discover_cache_lookup,
)


def test_global_bank_keeps_one_baseline_and_only_nonreference_extensions() -> None:
    rows = []
    candidates = {
        "F": [("F_kappa_1", True), ("F_kappa_2", False)],
        "M": [("M_gamma_0p55", True), ("M_gamma_0p9", False)],
        "H": [("H_capacity_38", True), ("H_capacity_3", False)],
        "P": [("P_zeta_1", True), ("P_zeta_0", False)],
    }
    for family, family_candidates in candidates.items():
        for candidate_id, is_reference in family_candidates:
            rows.append(
                {
                    "readout": "static",
                    "family": family,
                    "candidate_id": candidate_id,
                    "is_reference": is_reference,
                }
            )

    bank = _candidate_bank(pd.DataFrame(rows), "static")

    assert bank[0] == ("BASE", "F", "F_kappa_1")
    assert len(bank) == 5
    assert {entry[2] for entry in bank[1:]} == {
        "F_kappa_2",
        "M_gamma_0p9",
        "H_capacity_3",
        "P_zeta_0",
    }


def test_cache_discovery_works_before_parallel_index_is_written(tmp_path) -> None:
    cache = (
        tmp_path
        / "candidate_runs/static/F/F_kappa_1/cache/subject_101/particles_64/rollouts_256.npz"
    )
    cache.parent.mkdir(parents=True)
    cache.touch()

    lookup = _discover_cache_lookup(tmp_path)

    assert lookup[(101, "static", "F", "F_kappa_1")] == cache
