"""Protect extension membership, chronological boundaries and neutral starts."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_extension_setup as setup


def registration():
    selected = [dict(subject=s, condition=c, trials=n, training_trials=cut, later_trials=n-cut)
                for s, c, n, cut in ((110, 1, 448, 288), (209, 2, 512, 352), (309, 3, 768, 512))]
    return (dict(status="complete", selected=selected, model_scores_used=False,
                 behavioral_accuracy_used=False, previous_development_influence="unknown",
                 recommendation_freeze_required=True),
            dict(status="complete", smoke=False, subjects=[110, 209, 309]))


def test_registered_sample_and_original_boundaries_are_preserved():
    eligible, prepared = registration()
    assert [r["subject"] for r in setup.validate_registration(eligible, prepared)] == [110, 209, 309]


@pytest.mark.parametrize("fault", ["score_selection", "accuracy_selection", "new_subject", "boundary",
                                  "smoke", "prior_influence", "remove_freeze_gate"])
def test_changed_sample_or_data_use_cannot_enter_extension(fault):
    eligible, prepared = registration()
    if fault == "score_selection":
        eligible["model_scores_used"] = True
    elif fault == "accuracy_selection":
        eligible["behavioral_accuracy_used"] = True
    elif fault == "new_subject":
        eligible["selected"][0]["subject"] = 111
        prepared["subjects"][0] = 111
    elif fault == "boundary":
        eligible["selected"][0]["training_trials"] += 32
        eligible["selected"][0]["later_trials"] -= 32
    elif fault == "smoke":
        prepared["smoke"] = True
    elif fault == "prior_influence":
        eligible["previous_development_influence"] = "independent"
    else:
        eligible["recommendation_freeze_required"] = False
    with pytest.raises(ValueError):
        setup.validate_registration(eligible, prepared)


def test_neutral_starts_use_supports_and_seeds_without_subject_outcomes():
    space = {setup.recovery.CAPACITY: [1, 2, 4, 8, 16], setup.recovery.TRANSFER: [.01],
             **{f"parameter_{i}": [0., .5, 1.] for i in range(12)}}
    settings = dict(seed=2026101807, initial_candidates=40)
    banks = setup.neutral_banks(settings, space, [110, 209, 309])
    other = deepcopy(settings)
    other.update(previous_nominees="unavailable", oral_scores="unavailable", choices="unavailable")
    assert banks == setup.neutral_banks(other, space, [110, 209, 309])
    assert len({b["seed"] for b in banks}) == 3
    assert banks[0]["rows"] != banks[1]["rows"]
    for bank in banks:
        assert len(bank["rows"]) == 40
        assert bank["required_reference_id"] == bank["rows"][0]["id"]
        assert set(p["hyperparams"][setup.recovery.CAPACITY] for p in bank["rows"]) == {1, 2, 4, 8, 16}
        assert all(p["hyperparams"][setup.recovery.TRANSFER] == .01 for p in bank["rows"])
