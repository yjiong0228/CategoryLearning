"""Seeds archived before adding the optional primary-generation role."""
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_finalist_generation import trajectory_seeds


def test_default_seeds_match_saved_prior_finalist_bank():
    settings = dict(protocol='model0923_h_finalist_generation_v1', seed=2026101505, paths_per_candidate=5)
    assert trajectory_seeds(settings, 101) == [4076189067, 3162194653, 262971346, 2266911519, 1302221021]


def test_autonomous_role_matches_saved_primary_generator_seeds():
    settings = dict(protocol='software_fixture_candidate_validation_v1', seed=2026101503,
                    seed_role='autonomous', paths_per_candidate=2)
    assert trajectory_seeds(settings, 101) == [386800429, 2356494713]


@pytest.mark.parametrize('role', ['', True, 2])
def test_invalid_role_is_rejected(role):
    with pytest.raises(ValueError):
        trajectory_seeds(dict(protocol='x', seed=1, seed_role=role, paths_per_candidate=2), 101)
