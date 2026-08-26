from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize
import yaml

from src.Bayesian_state.hypothesis_space.geometry import BoundaryGeometry
from src.Bayesian_state.hypothesis_space.observation_model import ContinuousPartition
from src.Bayesian_state.hypothesis_space.observation_model import ObservationLikelihood
from src.Bayesian_state.model.assembly import build_observation_likelihood
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.readout import read_choice_probabilities_from_model
from src.Bayesian_state.model.modules.hypothesis_transition.selection import (
    HypothesisSelectionPolicy,
)
from src.Bayesian_state.hypothesis_space.spaces import (
    CategoryRegion,
    ContinuousHypothesisSpace,
    ContinuousHypothesisSpec,
    Polytope,
    build_continuous_hypothesis_space,
)


def _geometry(method: str, *, n_dims: int = 2) -> BoundaryGeometry:
    space = build_continuous_hypothesis_space(n_dims, 2)
    return BoundaryGeometry(space, method=method)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([0.75, 0.25], 0.25),
        ([0.75, 0.75], np.sqrt(0.125)),
        ([1.2, 1.2], np.sqrt(0.98)),
    ],
)
def test_kkt_projects_to_face_edge_and_box_vertex(point, expected) -> None:
    polytope = Polytope(np.eye(2), np.asarray([0.5, 0.5]))
    observed = _geometry("kkt_active_set_projection").distances_to_polytope(
        np.asarray([point]), polytope
    )
    assert np.allclose(observed, [expected], atol=1e-10, rtol=1e-10)


def test_two_solvers_match_for_slanted_plane_and_multiple_components() -> None:
    first = Polytope(np.asarray([[1.0, 1.0]]), np.asarray([1.0]))
    second = Polytope(np.asarray([[-1.0, -1.0]]), np.asarray([-1.4]))
    category = CategoryRegion((first, second))
    points = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.8, 0.4], [1.1, -0.1]])
    dykstra = _geometry("dykstra_iterative_projection").distances_to_category(
        points, category
    )
    kkt = _geometry("kkt_active_set_projection").distances_to_category(
        points, category
    )
    assert np.allclose(kkt, dykstra, atol=1e-7, rtol=1e-6)


def test_assignment_membership_includes_unit_box(monkeypatch) -> None:
    left = CategoryRegion(
        (Polytope(np.asarray([[1.0]]), np.asarray([0.5])),)
    )
    right = CategoryRegion(
        (Polytope(np.asarray([[-1.0]]), np.asarray([-0.5])),)
    )
    specification = ContinuousHypothesisSpec(
        index=0,
        family="test",
        hyperplanes=(),
        categories=(left, right),
        parameters={},
        feedback_neighbors=((), ()),
    )
    space = ContinuousHypothesisSpace(
        n_dims=1,
        n_cats=2,
        hypotheses=(specification,),
        version="test",
        parameters={},
    )
    geometry = BoundaryGeometry(space, method="kkt_active_set_projection")
    monkeypatch.setattr(
        geometry,
        "category_distances",
        lambda hypo, stimuli: np.asarray([[1.0, 0.1]]),
    )

    # The point satisfies the left rule halfspace, but it is outside the unit
    # cube and therefore must not be treated as a member of that category.
    assignment = geometry.category_assignments(0, np.asarray([[-1.0]]))
    assert assignment.tolist() == [1]


def test_kkt_matches_slsqp_oracle() -> None:
    polytope = Polytope(
        np.asarray([[1.0, 1.0], [-1.0, 0.0]]),
        np.asarray([0.8, -0.1]),
    )
    points = np.asarray([[0.8, 0.8], [0.0, 0.6], [1.2, -0.2]])
    geometry = _geometry("kkt_active_set_projection")
    observed = geometry.distances_to_polytope(points, polytope)
    constraint = LinearConstraint(polytope.A, -np.inf, polytope.b)
    expected = []
    for point in points:
        result = minimize(
            lambda value: 0.5 * float(np.sum((value - point) ** 2)),
            x0=np.clip(point, 0.0, 1.0),
            method="SLSQP",
            bounds=Bounds(0.0, 1.0),
            constraints=(constraint,),
            options={"ftol": 1e-12, "maxiter": 500},
        )
        assert result.success, result.message
        expected.append(np.linalg.norm(point - result.x))
    assert np.allclose(observed, expected, atol=1e-7, rtol=1e-6)


def test_geometry_compilation_cache_reuses_equal_polytopes() -> None:
    BoundaryGeometry.clear_compilation_cache()
    first = Polytope(np.asarray([[1.0, -1.0]]), np.asarray([0.0]))
    second = Polytope(first.A.copy(), first.b.copy())
    geometry = _geometry("kkt_active_set_projection")
    geometry.distances_to_polytope(np.asarray([[0.9, 0.1]]), first)
    geometry.distances_to_polytope(np.asarray([[0.9, 0.1]]), second)
    assert BoundaryGeometry.compilation_cache_info() == {"size": 1}


def test_solver_names_are_strict() -> None:
    with pytest.raises(ValueError, match="Unsupported boundary distance method"):
        _geometry("active_set")


def test_continuous_likelihood_requires_explicit_distance_mode() -> None:
    partition = ContinuousPartition(4, 2, similarity_n_samples=8)
    with pytest.raises(ValueError, match="explicitly configured"):
        ObservationLikelihood(partition)


def test_prototype_config_rejects_explicit_boundary_only_parameters() -> None:
    config = {
        "partition": {
            "kwargs": {
                "boundary_distance_method": "dykstra_iterative_projection"
            }
        },
        "likelihood": {"distance_mode": "prototype"},
    }
    partition = ContinuousPartition(4, 2, similarity_n_samples=8)
    with pytest.raises(ValueError, match="boundary-only"):
        build_observation_likelihood(config, partition)


def test_ksimilar_centers_rejects_boundary_encoding() -> None:
    class ConcretePolicy(HypothesisSelectionPolicy):
        def process(self, **kwargs):
            del kwargs

    policy = object.__new__(ConcretePolicy)
    policy.engine = SimpleNamespace(
        distance_mode="boundary",
        partition=ContinuousPartition(4, 2, similarity_n_samples=8),
    )
    with pytest.raises(ValueError, match="prototype-only"):
        policy._validate_ksimilar_partition()


def test_probability_dispatch_does_not_mix_geometries(monkeypatch) -> None:
    partition = ContinuousPartition(4, 2, similarity_n_samples=8)
    data = ([np.asarray([0.1, 0.2, 0.3, 0.4])], [1], [1.0])

    def unexpected(*args, **kwargs):
        raise AssertionError("wrong geometry was called")

    monkeypatch.setattr(
        partition.boundary_geometry, "category_probabilities", unexpected
    )
    partition.get_category_probabilities(
        0, data, beta=5.0, distance_mode="prototype"
    )

    partition = ContinuousPartition(4, 2, similarity_n_samples=8)
    monkeypatch.setattr(
        partition.prototype_geometry, "category_probabilities", unexpected
    )
    partition.get_category_probabilities(
        0, data, beta=5.0, distance_mode="boundary"
    )


@pytest.mark.parametrize("n_cats", [2, 4])
def test_all_builtin_regions_match_between_solvers(n_cats: int) -> None:
    rng = np.random.default_rng(17 + n_cats)
    points = rng.random((4, 4))
    dykstra = ContinuousPartition(
        4,
        n_cats,
        boundary_distance_method="dykstra_iterative_projection",
        similarity_n_samples=8,
    )
    kkt = ContinuousPartition(
        4,
        n_cats,
        boundary_distance_method="kkt_active_set_projection",
        similarity_n_samples=8,
    )
    for hypothesis in range(dykstra.length):
        first = dykstra.boundary_geometry.category_distances(hypothesis, points)
        second = kkt.boundary_geometry.category_distances(hypothesis, points)
        assert np.allclose(first, second, atol=1e-7, rtol=1e-6)
        assert np.array_equal(
            dykstra.boundary_geometry.category_assignments(hypothesis, points),
            kkt.boundary_geometry.category_assignments(hypothesis, points),
        )
        for beta in (0.1, 1.0, 5.0, 15.0, 25.0):
            first_prob = dykstra.boundary_geometry.category_probabilities(
                hypothesis, points, beta
            )
            second_prob = kkt.boundary_geometry.category_probabilities(
                hypothesis, points, beta
            )
            assert np.allclose(first_prob, second_prob, atol=1e-6, rtol=0.0)


def test_binary_label_reversal_preserves_base_order_and_reverses_labels() -> None:
    base = ContinuousPartition(4, 2, similarity_n_samples=8)
    expanded = ContinuousPartition(
        4,
        2,
        label_permutation_policy="binary_identity_and_reverse",
        similarity_n_samples=8,
    )
    assert expanded.length == 2 * base.length
    for original, copied in zip(
        base.hypothesis_space,
        expanded.hypothesis_space.hypotheses[: base.length],
    ):
        assert copied.index == original.index
        assert copied.family == original.family
        assert copied.hyperplanes == original.hyperplanes
    reverse = expanded.hypothesis_space[base.length]
    assert reverse.base_hypothesis_index == 0
    assert reverse.label_permutation == (1, 0)
    assert reverse.is_label_permuted
    points = np.random.default_rng(31).random((12, 4))
    for mode in ("prototype", "boundary"):
        original = expanded.get_category_assignments(0, points, distance_mode=mode)
        reversed_labels = expanded.get_category_assignments(
            base.length, points, distance_mode=mode
        )
        assert np.array_equal(reversed_labels, 1 - original)


def test_four_category_space_rejects_binary_reversal_policy() -> None:
    with pytest.raises(ValueError, match="requires exactly two categories"):
        ContinuousPartition(
            4,
            4,
            label_permutation_policy="binary_identity_and_reverse",
        )


def test_similarity_is_mode_aware_and_reversal_has_zero_agreement() -> None:
    partition = ContinuousPartition(
        4,
        2,
        label_permutation_policy="binary_identity_and_reverse",
        similarity_n_samples=64,
    )
    stimuli = np.random.default_rng(9).random((64, 4))
    for mode in ("prototype", "boundary"):
        matrix = partition.get_similarity_matrix(
            distance_mode=mode,
            stimuli=stimuli,
        )
        assert matrix.shape == (partition.length, partition.length)
        assert np.allclose(np.diag(matrix), 1.0)
        assert matrix[0, partition.length // 2] == 0.0


def test_legacy_similarity_resource_requires_exact_boundary_backend() -> None:
    compatible = ContinuousPartition(
        4,
        2,
        boundary_distance_method="dykstra_iterative_projection",
        similarity_n_samples=8,
    )
    incompatible = ContinuousPartition(
        4,
        2,
        boundary_distance_method="kkt_active_set_projection",
        similarity_n_samples=8,
    )

    compatible_path = compatible.similarity._compatible_resource_path(
        "boundary",
        compatible.similarity.DEFAULT_N_SAMPLES,
        compatible.similarity.DEFAULT_RANDOM_STATE,
    )
    incompatible_path = incompatible.similarity._compatible_resource_path(
        "boundary",
        incompatible.similarity.DEFAULT_N_SAMPLES,
        incompatible.similarity.DEFAULT_RANDOM_STATE,
    )

    assert compatible_path.name.endswith("pairtol0p1_centertol0p1.npy")
    assert incompatible_path.name == "__no_compatible_similarity_resource__"


def test_label_reversed_space_runs_complete_model_trial() -> None:
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (
            root
            / "configs/model_struct/pmh_model_cond1_0815_h5_similarity_transport.yaml"
        ).read_text(encoding="utf-8")
    )
    config["inference"] = {"backend": "trajectory"}
    config["partition"]["kwargs"].update(
        {
            "label_permutation_policy": "binary_identity_and_reverse",
            "boundary_distance_method": "kkt_active_set_projection",
            "boundary_distance_tolerance": 1e-9,
            "boundary_projection_iterations": 100,
            "similarity_n_samples": 64,
        }
    )
    model = StateModel(config, context=ModelContext(condition=1, subject_id=118))
    assert model.partition_model.length == 58
    assert model.engine.prior.shape == (58,)
    assert model.engine.beta.shape == (58,)

    prepared = model.begin_trial(np.asarray([0.1, 0.2, 0.3, 0.4]))
    probabilities = read_choice_probabilities_from_model(
        model, prepared.perceived_stimulus
    )
    assert probabilities.shape == (2,)
    assert np.isclose(probabilities.sum(), 1.0)
    posterior, _, _ = model.complete_trial(choice=1, feedback=1.0)
    assert posterior.shape == (58,)
    assert np.isclose(posterior.sum(), 1.0)
