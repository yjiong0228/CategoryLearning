"""Prefix construction observation is removed even after filter failure."""
import pytest

from src.Bayesian_state.simulation.filtered_continuation import capture_prefix


def test_failed_filter_restores_original_model_constructor(monkeypatch):
    import src.Bayesian_state.model as models
    from src.Bayesian_state.inference.backends import particle_filter
    original = models.StateModel

    def fail(**kwargs):
        assert models.StateModel is not original
        raise RuntimeError('deliberate filter failure')

    monkeypatch.setattr(particle_filter, 'run_state_model_particle_filter', fail)
    with pytest.raises(RuntimeError, match='deliberate filter failure'):
        capture_prefix(particle_count=2, choices=[1, 2])
    assert models.StateModel is original


@pytest.mark.parametrize('kwargs', [{'particle_blocks': 2}, {'condition_on_observed_choice': False}])
def test_unsupported_filter_controls_leave_constructor_unchanged(kwargs):
    import src.Bayesian_state.model as models
    original = models.StateModel
    with pytest.raises(ValueError):
        capture_prefix(**kwargs)
    assert models.StateModel is original
