from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_model_0826_belief_transport_counterfactual import (
    PRIOR_ASSIGNMENT_PATH,
    _mean_js,
    build_variant_engine,
    validate_variants,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "configs/model_struct/pmh_model_cond1_0826.yaml"
AUDIT_CONFIG = (
    ROOT
    / "configs/specific_models/model_0826_belief_transport_counterfactual.yaml"
)


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_counterfactual_bank_changes_only_prior_assignment_method() -> None:
    base = _yaml(MODEL_CONFIG)
    audit = _yaml(AUDIT_CONFIG)
    variants = validate_variants(audit["design"]["variants"])

    primary = build_variant_engine(
        base,
        variants[0]["prior_assignment_method"],
    )
    mass = build_variant_engine(
        base,
        variants[1]["prior_assignment_method"],
    )

    assert primary["modules"]["hypo_transitions_mod"]["kwargs"][
        "prior_assignment"
    ]["method"] == "similarity_transport"
    assert mass["modules"]["hypo_transitions_mod"]["kwargs"][
        "prior_assignment"
    ]["method"] == "mass_preserving_similarity_transport"
    assert audit["interpretation"]["only_changed_path"] == PRIOR_ASSIGNMENT_PATH


def test_counterfactual_bank_rejects_missing_or_duplicate_methods() -> None:
    with pytest.raises(ValueError, match="exactly the primary"):
        validate_variants(
            [
                {
                    "variant_id": "one",
                    "prior_assignment_method": "similarity_transport",
                },
                {
                    "variant_id": "two",
                    "prior_assignment_method": "similarity_transport",
                },
            ]
        )


def test_workspace_belief_js_is_zero_only_for_matching_rows() -> None:
    first = np.asarray([[0.8, 0.2], [0.5, 0.5]], dtype=float)
    second = np.asarray([[0.7, 0.3], [0.5, 0.5]], dtype=float)

    assert _mean_js(first, first) == pytest.approx(0.0)
    assert _mean_js(first, second) > 0.0
