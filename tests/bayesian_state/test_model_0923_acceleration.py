"""Exact resource reuse, assignment ties and resampled-particle ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space import ContinuousPartition
from src.Bayesian_state.hypothesis_space import rule_programs
from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION
from src.Bayesian_state.inference.backends.particle_filter import _restore, _snapshot, run_state_model_particle_filter
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.optimization.search.common import deep_update
from src.Bayesian_state.simulation.config import resolve_engine_config


COSTS = {"feature": 1., "relation": 1.5, "reference": 1., "branch": 2.}


@pytest.fixture(autouse=True)
def isolate_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, "RUNTIME_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ContinuousSimilarity, "_confusion_cache", {})
    monkeypatch.setattr(rule_programs, "_EDIT_CACHE", rule_programs.OrderedDict())


def _partition(n_categories=2, **kwargs):
    if n_categories == 4:
        kwargs["structural_extension"] = RELATIONAL_EXTENSION
    return ContinuousPartition(4, n_categories, **kwargs)


@pytest.mark.parametrize("n_categories", [2, 4])
def test_edit_cache_reuses_identical_catalogues_and_preserves_costs(monkeypatch, n_categories):
    first, second = [_partition(n_categories) for _ in range(2)]
    programs, costs = rule_programs.build_rule_edit_costs(first.hypothesis_space, COSTS)
    with monkeypatch.context() as patcher:
        def forbidden(*args):
            raise AssertionError("identical catalogue should reuse compiled rules")
        patcher.setattr(rule_programs, "compile_rule_program", forbidden)
        reused_programs, reused_costs = rule_programs.build_rule_edit_costs(second.hypothesis_space, dict(reversed(list(COSTS.items()))))
    assert reused_programs is programs
    np.testing.assert_array_equal(reused_costs, costs)
    with pytest.raises(ValueError):
        reused_costs.setflags(write=True)
    reused_costs.shape = (reused_costs.size,)
    assert rule_programs.build_rule_edit_costs(first.hypothesis_space, COSTS)[1].shape == costs.shape
    _, changed = rule_programs.build_rule_edit_costs(first.hypothesis_space, {k: 2*v for k, v in COSTS.items()})
    np.testing.assert_array_equal(changed, 2*costs)
    with pytest.raises(ValueError):
        rule_programs.build_rule_edit_costs(first.hypothesis_space, COSTS | {"feature": 0.})


def test_content_key_distinguishes_injected_catalogues_with_identical_metadata():
    first = build_continuous_hypothesis_space(4, 2, center_band_tolerance=.1)
    changed = build_continuous_hypothesis_space(4, 2, center_band_tolerance=.2)
    second = replace(first, hypotheses=changed.hypotheses)
    assert first.signature == second.signature
    assert first.content_signature != second.content_signature
    first_programs, _ = rule_programs.build_rule_edit_costs(first, COSTS)
    second_programs, _ = rule_programs.build_rule_edit_costs(second, COSTS)
    assert first_programs != second_programs


def test_edit_cache_is_bounded():
    space = _partition().hypothesis_space
    for cost in range(1, 12):
        rule_programs.build_rule_edit_costs(space, COSTS | {"feature": float(cost)})
    assert len(rule_programs._EDIT_CACHE) <= 8


@pytest.mark.parametrize("n_categories", [2, 4])
def test_confusion_disk_cache_reuses_exact_counts_across_instances(monkeypatch, tmp_path, n_categories):
    first = _partition(n_categories).similarity
    expected = first.get_confusion_tensor(distance_mode="boundary", n_samples=32, random_state=9)
    paths = list(tmp_path.glob("confusion_*.npz"))
    assert len(paths) == 1
    original_bytes = paths[0].read_bytes()
    ContinuousSimilarity._confusion_cache.clear()
    second = _partition(n_categories).similarity
    monkeypatch.setattr(second.boundary, "category_assignments", lambda *a: pytest.fail("cache miss"))
    actual = second.get_confusion_tensor(distance_mode="boundary", n_samples=32, random_state=9)
    np.testing.assert_array_equal(actual, expected)
    assert paths[0].read_bytes() == original_bytes
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    actual.shape = (actual.size,)
    assert second.get_confusion_tensor(distance_mode="boundary", n_samples=32, random_state=9).shape == expected.shape


@pytest.mark.parametrize("damage", ["truncated", "checksum", "key"])
def test_invalid_confusion_cache_is_recomputed_and_preserved(tmp_path, damage):
    similarity = _partition().similarity
    expected = similarity.get_confusion_tensor(distance_mode="boundary", n_samples=32)
    path = next(tmp_path.glob("confusion_*.npz"))
    if damage == "truncated":
        path.write_bytes(b"PK\x03\x04truncated")
    else:
        with np.load(path) as archive:
            saved = dict(archive)
        saved["tensor_sha256" if damage == "checksum" else "key_sha256"] = "invalid"
        np.savez(path, **saved)
    damaged_bytes = path.read_bytes()
    ContinuousSimilarity._confusion_cache.clear()
    np.testing.assert_array_equal(similarity.get_confusion_tensor(distance_mode="boundary", n_samples=32), expected)
    assert path.read_bytes() == damaged_bytes


def test_confusion_cache_separates_reference_sample_settings(tmp_path):
    similarity = _partition().similarity
    for count, seed in ((32, 9), (32, 10), (33, 9)):
        similarity.get_confusion_tensor(distance_mode="boundary", n_samples=count, random_state=seed)
    assert len(list(tmp_path.glob("confusion_*.npz"))) == 3


def test_concurrent_confusion_publication_is_atomic_and_cleans_temporary_files(tmp_path):
    similarity = _partition().similarity
    tensor = similarity.get_confusion_tensor(distance_mode="boundary", n_samples=32)
    path = tmp_path / "concurrent.npz"
    barrier = Barrier(2)

    def publish(_):
        barrier.wait(timeout=5)
        similarity._save_confusion_tensor(path, "test-key", tensor)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(publish, range(2)))
    np.testing.assert_array_equal(similarity._load_confusion_tensor(path, "test-key"), tensor)
    assert not list(tmp_path.glob(".confusion_*.npz"))


def test_explicit_reference_stimuli_bypass_confusion_disk_cache(tmp_path):
    similarity = _partition().similarity
    points = np.full((3, 4), .5)
    tensor = similarity.get_confusion_tensor(distance_mode="boundary", stimuli=points)
    np.testing.assert_allclose(tensor.sum(axis=(2, 3)), 1.)
    assert not list(tmp_path.glob("confusion_*.npz"))
    assert not ContinuousSimilarity._confusion_cache


@pytest.mark.parametrize("n_categories", [2, 4])
@pytest.mark.parametrize("mode", ["boundary", "prototype"])
def test_similarity_rows_equal_frozen_full_boolean_reduction(n_categories, mode):
    similarity = _partition(n_categories).similarity
    points = np.random.default_rng(18).random((129, 4))
    geometry = similarity.boundary if mode == "boundary" else similarity.prototype
    labels = np.asarray([geometry.category_assignments(h.index, points) for h in similarity.space])
    expected = np.mean(labels[:, None, :] == labels[None, :, :], axis=2)
    np.testing.assert_array_equal(similarity.compute(distance_mode=mode, stimuli=points), expected)


@pytest.mark.parametrize("n_categories", [2, 4])
def test_batched_assignments_preserve_region_ties_and_distance_fallback(n_categories):
    geometry = _partition(n_categories).boundary_geometry
    points = np.r_[np.random.default_rng(9).random((8, 4)),
                   np.array([[.5]*4, [0.]*4, [1.]*4, [-.2, .4, .8, 1.2]])]
    # Frozen pre-optimization assignment procedure, including its tie order.
    for spec in geometry.space:
        distances = geometry.category_distances(spec.index, points)
        expected = distances.argmin(axis=1)
        memberships = np.zeros((len(points), n_categories), dtype=bool)
        scores = np.full_like(distances, np.inf)
        for index, category in enumerate(spec.categories):
            for component in category.components:
                constraints, bounds = geometry._bounded_constraints(component)
                violations = points @ constraints.T - bounds[None, :]
                memberships[:, index] |= np.all(violations <= geometry.tolerance, axis=1)
                scores[:, index] = np.minimum(scores[:, index], np.clip(violations, 0., None).sum(axis=1))
        for row in np.flatnonzero(memberships.any(axis=1)):
            candidates = np.flatnonzero(memberships[row])
            expected[row] = candidates[np.argmin(scores[row, candidates])]
        np.testing.assert_array_equal(geometry.category_assignments(spec.index, points), expected)


@pytest.mark.parametrize("n_categories", [2, 4])
def test_covered_reference_assignments_skip_unused_dykstra_projections(monkeypatch, n_categories):
    geometry = _partition(n_categories).boundary_geometry
    points = np.r_[np.random.default_rng(19).random((64, 4)), np.full((1, 4), .5)]
    monkeypatch.setattr(geometry, "category_distances", lambda *a: pytest.fail("covered labels need no projections"))
    for spec in geometry.space:
        labels = geometry.category_assignments(spec.index, points)
        assert labels.shape == (len(points),)
        assert np.all((labels >= 0) & (labels < n_categories))


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_shared_engine_snapshot_and_siblings_own_independent_buffers(condition):
    path = Path(f"configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml")
    config = yaml.safe_load(path.read_text())
    config["modules"]["perception_mod"]["kwargs"] = {"features": 4, "mean": [0.]*4, "std": [0.]*4}
    config["modules"]["hypo_transitions_mod"]["kwargs"].update(relation_samples=32, module_seed=9)
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 32
    source = StateModel(config, context=ModelContext(condition=condition))
    source.begin_trial(np.array([.3, .5, .7, .8]))
    source.complete_trial(1, .5 if condition == 3 else 0.)
    snapshot = _snapshot(source)
    expected = snapshot.payload["core"]["posterior"].copy()
    source.engine.posterior[:] = 0.
    np.testing.assert_array_equal(snapshot.payload["core"]["posterior"], expected)
    children = [StateModel(config, context=ModelContext(condition=condition),
                           partition=source.partition_model) for _ in range(2)]
    transitions = [child.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for child in children]
    for name in ("edit_distances", "functional_similarity", "_distance_matrix"):
        assert np.shares_memory(getattr(transitions[0], name), getattr(transitions[1], name))
        with pytest.raises(ValueError):
            getattr(transitions[0], name).setflags(write=True)
    for index, child in enumerate(children):
        _restore(child, snapshot, filter_seed=923, trial_index=0, particle_index=index)
        np.testing.assert_array_equal(child.engine.posterior, expected)

    def arrays(payload):
        if isinstance(payload, dict):
            return [a for value in payload.values() for a in arrays(value)]
        if isinstance(payload, (tuple, list)):
            return [a for value in payload for a in arrays(value)]
        return [payload] if isinstance(payload, np.ndarray) else []

    # Inspect live module buffers; state_dict() would itself copy these arrays.
    first_engine, second_engine = [child.engine for child in children]
    for role in (ModuleRole.MEMORY, ModuleRole.BETA, ModuleRole.HYPOTHESIS_TRANSITION):
        first_module, second_module = [engine.get_module(role) for engine in (first_engine, second_engine)]
        for name, value in vars(first_module).items():
            if name in {"engine", "base_prior", "edit_distances", "functional_similarity", "_distance_matrix", "confusion_tensor"}:
                continue
            if isinstance(value, np.ndarray):
                assert not np.shares_memory(value, getattr(second_module, name)), name
                assert all(not np.shares_memory(value, original) for original in arrays(snapshot.payload))
    children[0].engine.posterior[:] = 0.
    np.testing.assert_array_equal(children[1].engine.posterior, expected)
    np.testing.assert_array_equal(snapshot.payload["core"]["posterior"], expected)


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_choice_only_fitting_preserves_predictions_states_and_resampling(condition):
    config = yaml.safe_load(Path(f"configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml").read_text())
    config["modules"]["perception_mod"]["kwargs"] = {"features": 4, "mean": [0.]*4, "std": [0.]*4}
    config["modules"]["hypo_transitions_mod"]["kwargs"].update(relation_samples=32, module_seed=9)
    config["provenance"]["hypothesis_similarity"]["sample_count"] = 32
    stimuli = np.random.default_rng(923).random((8, 4))
    kwargs = dict(subject_id=1, condition=condition, stimulus=stimuli, choices=np.tile([1, 2], 4),
                  feedback=np.tile([0., .5 if condition == 3 else 1.], 4), particle_count=4,
                  choice_readout_power=1., filter_seed=923, resample_threshold_fraction=.99)
    enabled = run_state_model_particle_filter(engine_config=config, **kwargs)
    disabled = run_state_model_particle_filter(engine_config=config | {"oral_readout": None}, **kwargs)
    for name in ("state_probabilities", "latent_summaries"):
        first, second = [getattr(result, name) for result in (enabled, disabled)]
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_equal(second[key], first[key], err_msg=f"{name}/{key}")
    np.testing.assert_array_equal(disabled.observation_probabilities["prior_t"], enabled.observation_probabilities["prior_t"])
    np.testing.assert_array_equal(disabled.resampled, enabled.resampled)
    assert "oral_readout" in enabled.metadata and "oral_readout" not in disabled.metadata


def test_meg_search_disables_oral_predictions_only_in_search_stages():
    path = Path("configs/meg/hyper_cd_cfg/model0923_task3b.yaml").resolve()
    search = yaml.safe_load(path.read_text())
    simulation_path = (path.parent / search["base_sim_config_path"]).resolve()
    simulation = yaml.safe_load(simulation_path.read_text())
    assert resolve_engine_config(simulation, simulation_path.parent)["oral_readout"] is not None
    for stage in [*search["stages"].values(), search["final_rescore"]]:
        cfg = deep_update(simulation, stage["simulation_overrides"])
        assert resolve_engine_config(cfg, simulation_path.parent)["oral_readout"] is None
