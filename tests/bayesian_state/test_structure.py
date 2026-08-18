from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]


def test_bayesian_state_core_package_layout_is_explicit() -> None:
    package_root = ROOT_DIR / "src/Bayesian_state"
    core_packages = (
        "model",
        "hypothesis_space",
        "inference",
        "simulation",
        "optimization",
        "evaluation",
        "metrics",
        "reference_models",
    )
    removed_package_names = (
        "problems",
        "inference_engine",
        "model_evaluation",
        "manuscript_models",
        "clustering",
    )

    for package_name in core_packages:
        package_dir = package_root / package_name
        assert package_dir.is_dir()
        assert (package_dir / "__init__.py").is_file()
        assert (package_dir / "README.md").is_file()
    for package_name in removed_package_names:
        assert not (package_root / package_name).exists()


def test_nested_package_names_and_readmes_are_explicit() -> None:
    package_root = ROOT_DIR / "src/Bayesian_state"
    expected_nested_packages = (
        "evaluation/oral",
        "evaluation/particle_filter",
        "hypothesis_space/analysis",
        "hypothesis_space/geometry",
        "hypothesis_space/observation_model",
        "hypothesis_space/spaces",
        "inference/backends",
        "model/modules",
        "model/modules/hypothesis_transition",
        "optimization/diagnostics",
        "optimization/search",
        "reference_models/model_0804",
    )
    for relative_path in expected_nested_packages:
        package_dir = package_root / relative_path
        assert package_dir.is_dir(), relative_path
        assert (package_dir / "__init__.py").is_file(), relative_path
        assert (package_dir / "README.md").is_file(), relative_path


def test_model_owns_engine_and_inference_depends_one_way() -> None:
    package_root = ROOT_DIR / "src/Bayesian_state"
    assert (package_root / "model/engine.py").is_file()
    assert not (package_root / "inference/engine.py").exists()

    for path in (package_root / "model").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                assert imported != "inference", path
                assert "Bayesian_state.inference" not in imported, path


def test_oral_evaluation_separates_mapping_scoring_and_reporting() -> None:
    oral_dir = ROOT_DIR / "src/Bayesian_state/evaluation/oral"
    assert {
        path.name for path in oral_dir.glob("*.py")
    } == {
        "__init__.py",
        "alignment.py",
        "mapping.py",
        "scoring.py",
        "reporting.py",
    }


def test_transition_modes_and_shared_mechanisms_are_separate() -> None:
    transition_dir = (
        ROOT_DIR
        / "src/Bayesian_state/model/modules/hypothesis_transition"
    )
    public_modes = {
        "fixed_strategy.py",
        "dynamic_discrete_strategy.py",
        "dynamic_adaptive_control.py",
        "feedback_reactive.py",
        "nested_feedback_accumulator.py",
    }
    shared_mechanisms = {
        "contracts.py",
        "selection.py",
        "prior_assignment.py",
        "workspace.py",
        "execution.py",
    }
    python_files = {
        path.name for path in transition_dir.glob("*.py") if path.name != "__init__.py"
    }
    assert python_files == public_modes | shared_mechanisms
    assert not (transition_dir / "static.py").exists()
    assert not (transition_dir / "dynamic_discrete.py").exists()
    assert not (transition_dir / "dynamic_continuous.py").exists()


def test_observation_model_is_separate_from_hypothesis_inventory() -> None:
    hypothesis_dir = ROOT_DIR / "src/Bayesian_state/hypothesis_space"
    observation_dir = hypothesis_dir / "observation_model"
    assert {
        path.name for path in observation_dir.glob("*.py")
    } == {
        "__init__.py",
        "base_partition.py",
        "continuous_partition.py",
        "discrete_rule_partition.py",
        "likelihood.py",
    }
    assert (hypothesis_dir / "similarity.py").is_file()
    assert not (hypothesis_dir / "runtime").exists()
    assert not (hypothesis_dir / "partitions").exists()


def test_partition_geometries_share_one_hypothesis_space() -> None:
    from src.Bayesian_state.hypothesis_space import (
        CategoryRegion,
        ContinuousHypothesisSpace,
        ContinuousHypothesisSpec,
    )
    from src.Bayesian_state.hypothesis_space.geometry import (
        BoundaryGeometry,
        PrototypeGeometry,
    )
    from src.Bayesian_state.hypothesis_space.observation_model.base_partition import BasePartition
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, 2)

    assert isinstance(partition.boundary_geometry, BoundaryGeometry)
    assert isinstance(partition.prototype_geometry, PrototypeGeometry)
    assert partition.boundary_geometry.space is partition.hypothesis_space
    assert partition.prototype_geometry.space is partition.hypothesis_space
    assert isinstance(partition, ContinuousPartition)
    assert isinstance(partition, BasePartition)
    assert isinstance(partition.hypothesis_space, ContinuousHypothesisSpace)
    assert isinstance(partition.hypothesis_space[0], ContinuousHypothesisSpec)
    assert isinstance(partition.hypothesis_space[0].categories[0], CategoryRegion)
    assert partition.prototype_method == "component_volume_centroid"
    for removed_view in ("splits", "regions", "prototypes", "prototype_mask"):
        assert not hasattr(partition, removed_view)


def test_discrete_partition_does_not_depend_on_continuous_facade() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.discrete_rule_partition import DiscreteRulePartition
    from src.Bayesian_state.hypothesis_space.observation_model.base_partition import BasePartition

    assert issubclass(DiscreteRulePartition, BasePartition)
    assert DiscreteRulePartition.__mro__[1] is BasePartition


def test_model_package_contains_no_hypothesis_compatibility_paths() -> None:
    from src.Bayesian_state import model
    from src.Bayesian_state import hypothesis_space

    removed_paths = (
        "base_problem.py",
        "partitions.py",
        "discrete_partitions.py",
        "partition_base.py",
        "hypothesis_space_analysis.py",
        "hypothesis_space",
        "partition_geometry",
        "cache",
    )
    model_dir = ROOT_DIR / "src/Bayesian_state/model"

    assert not any((model_dir / path).exists() for path in removed_paths)
    assert not (model_dir / "modules/likelihood.py").exists()
    assert not hasattr(model, "ContinuousPartition")
    assert not hasattr(model, "ContinuousHypothesisSpace")
    for removed_name in (
        "Partition",
        "HypothesisSpace",
        "HypothesisSpec",
        "DiscreteRule",
    ):
        assert not hasattr(hypothesis_space, removed_name)

    hypothesis_dir = ROOT_DIR / "src/Bayesian_state/hypothesis_space"
    for removed_path in (
        "specs.py",
        "continuous.py",
        "discrete.py",
        "partition_base.py",
        "continuous_partition.py",
        "discrete_partition.py",
        "analysis.py",
        "cache",
        "catalogues",
        "partitions",
    ):
        assert not (hypothesis_dir / removed_path).exists()
    assert not (hypothesis_dir / "runtime").exists()
    for package in ("spaces", "geometry", "observation_model", "analysis", "resources"):
        assert (hypothesis_dir / package / "README.md").is_file()


def test_observation_likelihood_is_fixed_runtime_infrastructure() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model import (
        ContinuousPartition,
        ObservationLikelihood,
    )
    from src.Bayesian_state.model.modules import BaseModule

    partition = ContinuousPartition(4, 2)
    evaluator = ObservationLikelihood(
        partition,
        distance_mode="prototype",
        default_beta=7.0,
    )
    observation = (np.asarray([0.2, 0.4, 0.6, 0.8]), 1, 1.0)
    hypotheses = [0, 1, 2]

    actual = evaluator.process(observation, hypotheses)
    expected = partition.calc_likelihood(
        hypotheses,
        ([observation[0]], [1], [1.0]),
        beta=7.0,
        distance_mode="prototype",
        normalized=True,
        feedback_likelihood_mode="category_feedback",
        feedback_lapse=0.0,
    )[0]

    assert not isinstance(evaluator, BaseModule)
    assert actual.shape == (3,)
    assert np.allclose(actual, expected)
    with pytest.raises(ValueError, match="choice and feedback"):
        evaluator.process((observation[0], None, None), hypotheses)


def test_observation_likelihood_can_decouple_evidence_from_action_beta() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model import (
        ContinuousPartition,
        ObservationLikelihood,
    )

    partition = ContinuousPartition(4, 2)
    observation = (np.asarray([0.39, 0.50, 0.50, 0.50]), 1, 1.0)
    hypotheses = [0, 25]
    fixed_evidence_beta = 5.0
    supplied_action_beta = np.asarray([0.2, 25.0])

    fixed = ObservationLikelihood(
        partition,
        distance_mode="boundary",
        default_beta=fixed_evidence_beta,
        beta_source="fixed",
    )
    actual = fixed.process(
        observation,
        hypotheses,
        beta=supplied_action_beta,
    )
    expected = partition.calc_likelihood(
        hypotheses,
        ([observation[0]], [1], [1.0]),
        beta=fixed_evidence_beta,
        distance_mode="boundary",
        normalized=True,
        feedback_likelihood_mode="category_feedback",
        feedback_lapse=0.0,
    )[0]
    assert np.allclose(actual, expected)

    coupled = ObservationLikelihood(
        partition,
        distance_mode="boundary",
        default_beta=fixed_evidence_beta,
        beta_source="action",
    ).process(observation, hypotheses, beta=supplied_action_beta)
    assert not np.allclose(coupled, actual)

    with pytest.raises(ValueError, match="beta_source"):
        ObservationLikelihood(
            partition,
            distance_mode="boundary",
            beta_source="unknown",
        )


@pytest.mark.parametrize("n_cats", [2, 4])
def test_continuous_similarity_uses_versioned_boundary_label_resources(
    n_cats: int,
) -> None:
    from src.Bayesian_state.hypothesis_space.observation_model import ContinuousPartition
    from src.Bayesian_state.hypothesis_space.similarity import (
        SIMILARITY_BASIS,
    )

    partition = ContinuousPartition(4, n_cats)

    assert SIMILARITY_BASIS == "boundary_fixed_labels"
    assert (partition.similarity.RESOURCE_DIR / partition.similarity.filename).is_file()
    assert partition.similarity_matrix.shape == (partition.length, partition.length)


def test_continuous_similarity_runtime_computation_is_seeded_and_validated(
    tmp_path: Path,
) -> None:
    from src.Bayesian_state.hypothesis_space.observation_model import ContinuousPartition
    from src.Bayesian_state.hypothesis_space.similarity import (
        SIMILARITY_COMPUTATION_SEED,
    )

    partition = ContinuousPartition(4, 2)
    similarity = partition.similarity
    assert SIMILARITY_COMPUTATION_SEED == 0
    assert similarity.runtime_filename.endswith("_seed0.npy")
    np.testing.assert_array_equal(
        similarity.compute(n_samples=256),
        similarity.compute(n_samples=256),
    )

    runtime_partition = ContinuousPartition(
        4,
        2,
        similarity_n_samples=256,
        similarity_cache_dir=tmp_path,
    )
    runtime_matrix = runtime_partition.similarity_matrix
    assert runtime_matrix.shape == (runtime_partition.length, runtime_partition.length)
    assert (
        tmp_path / runtime_partition.similarity.runtime_filename
    ).is_file()
    assert not (tmp_path / runtime_partition.similarity.filename).exists()

    valid = np.asarray(partition.similarity_matrix, dtype=float)
    invalid_matrices = []

    invalid_matrices.append(valid[:-1])

    nonfinite = valid.copy()
    nonfinite[0, 1] = nonfinite[1, 0] = np.nan
    invalid_matrices.append(nonfinite)

    outside_probability_range = valid.copy()
    outside_probability_range[0, 1] = outside_probability_range[1, 0] = 1.01
    invalid_matrices.append(outside_probability_range)

    asymmetric = valid.copy()
    asymmetric[0, 1] += 0.01
    invalid_matrices.append(asymmetric)

    wrong_diagonal = valid.copy()
    wrong_diagonal[0, 0] = 0.99
    invalid_matrices.append(wrong_diagonal)

    for index, matrix in enumerate(invalid_matrices):
        path = tmp_path / f"invalid_similarity_{index}.npy"
        np.save(path, matrix)
        assert similarity._load_valid_matrix(path) is None


def test_partition_rejects_unknown_prototype_construction_method() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    with pytest.raises(ValueError, match="Unsupported prototype method"):
        ContinuousPartition(4, 2, prototype_method="manual_points")


def test_partition_is_pickle_safe_for_parallel_execution() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    original = ContinuousPartition(4, 2)
    restored = pickle.loads(pickle.dumps(original))

    assert restored.hypothesis_space.signature == original.hypothesis_space.signature
    assert np.array_equal(
        restored.prototype_geometry.prototypes,
        original.prototype_geometry.prototypes,
    )
    assert np.array_equal(
        restored.prototype_geometry.prototype_mask,
        original.prototype_geometry.prototype_mask,
    )


def test_binary_partition_contains_29_fixed_label_hypotheses() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(n_dims=4, n_cats=2)

    assert partition.length == 29
    assert partition.prototype_geometry.prototypes.shape == (29, 2, 2, 4)
    assert partition.prototype_geometry.prototype_mask.shape == (29, 2, 2)
    assert np.isfinite(partition.prototype_geometry.prototypes).all()
    assert not hasattr(partition, "hypothesis_metadata")

    family_counts = {
        family: sum(split.family == family for split in partition.hypothesis_space)
        for family in set(split.family for split in partition.hypothesis_space)
    }
    assert family_counts == {
        "univariate_threshold": 4,
        "pairwise_order": 6,
        "pairwise_sum_threshold": 6,
        "paired_sum_order": 3,
        "pairwise_similarity_band": 6,
        "univariate_center_band": 4,
    }
    assert [split.family for split in partition.hypothesis_space[:19]] == (
        ["univariate_threshold"] * 4
        + ["pairwise_order"] * 6
        + ["pairwise_sum_threshold"] * 6
        + ["paired_sum_order"] * 3
    )
    assert [split.family for split in partition.hypothesis_space[19:25]] == (
        ["pairwise_similarity_band"] * 6
    )
    assert [split.family for split in partition.hypothesis_space[25:29]] == (
        ["univariate_center_band"] * 4
    )


def test_pairwise_similarity_band_geometry_and_likelihood() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(
        n_dims=4,
        n_cats=2,
        pairwise_similarity_tolerance=0.10,
    )
    hypo = 19  # first appended band: abs(x_0 - x_1) <= 0.10
    stimuli = np.array(
        [
            [0.50, 0.55, 0.50, 0.50],
            [0.80, 0.20, 0.50, 0.50],
            [0.20, 0.80, 0.50, 0.50],
        ],
        dtype=float,
    )

    assignments = partition.boundary_geometry.category_assignments(hypo, stimuli)
    assert assignments.tolist() == [0, 1, 1]
    assert len(partition.hypothesis_space[hypo].categories[1].components) == 2

    data = (stimuli, np.ones(len(stimuli), dtype=int))
    prototype_prob = partition.get_category_probabilities(
        hypo, data, beta=12.0, distance_mode="prototype"
    )
    boundary_prob = partition.get_category_probabilities(
        hypo, data, beta=12.0, distance_mode="boundary"
    )
    assert np.allclose(prototype_prob.sum(axis=0), 1.0)
    assert np.allclose(boundary_prob.sum(axis=0), 1.0)
    assert np.argmax(prototype_prob, axis=0).tolist() == [0, 1, 1]
    assert np.argmax(boundary_prob, axis=0).tolist() == [0, 1, 1]
    assert not np.allclose(prototype_prob, boundary_prob, atol=1e-6)
    assert partition.get_category_prototypes(hypo, 0).shape == (1, 4)
    assert partition.get_category_prototypes(hypo, 1).shape == (2, 4)


def test_univariate_center_band_geometry_and_likelihood() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, 2, center_band_tolerance=0.10)
    hypo = 25  # first center band: abs(x_0 - 0.5) <= 0.10
    stimuli = np.array(
        [
            [0.50, 0.20, 0.50, 0.50],
            [0.20, 0.20, 0.50, 0.50],
            [0.80, 0.20, 0.50, 0.50],
        ],
        dtype=float,
    )

    assignments = partition.boundary_geometry.category_assignments(hypo, stimuli)
    assert assignments.tolist() == [0, 1, 1]
    assert len(partition.hypothesis_space[hypo].categories[1].components) == 2

    data = (stimuli, np.ones(len(stimuli), dtype=int))
    prototype_prob = partition.get_category_probabilities(
        hypo, data, beta=12.0, distance_mode="prototype"
    )
    boundary_prob = partition.get_category_probabilities(
        hypo, data, beta=12.0, distance_mode="boundary"
    )
    assert np.argmax(prototype_prob, axis=0).tolist() == [0, 1, 1]
    assert np.argmax(boundary_prob, axis=0).tolist() == [0, 1, 1]
    assert not np.allclose(prototype_prob, boundary_prob, atol=1e-6)
    assert partition.get_category_prototypes(hypo, 0).shape == (1, 4)
    assert partition.get_category_prototypes(hypo, 1).shape == (2, 4)


def test_prototypes_are_derived_from_category_components() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, 2)

    assert np.allclose(
        partition.get_category_prototypes(0, 0),
        [[0.25, 0.50, 0.50, 0.50]],
    )
    assert np.allclose(
        partition.get_category_prototypes(4, 0),
        [[1.0 / 3.0, 2.0 / 3.0, 0.50, 0.50]],
    )
    assert np.allclose(
        partition.get_category_prototypes(19, 1),
        [[0.30, 0.70, 0.50, 0.50], [0.70, 0.30, 0.50, 0.50]],
    )


def test_paired_sum_centroid_uses_four_dimensional_region_volume() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, 2)

    # x0 <= x1 is a triangular half-square.
    assert np.allclose(
        partition.get_category_prototypes(4, 0),
        [[1.0 / 3.0, 2.0 / 3.0, 0.5, 0.5]],
    )
    # x0 + x1 <= x2 + x3 is half a 4-D cube, not a 2-D triangle.
    assert np.allclose(
        partition.get_category_prototypes(16, 0),
        [[23.0 / 60.0, 23.0 / 60.0, 37.0 / 60.0, 37.0 / 60.0]],
    )


def test_discrete_partition_uses_shared_space_and_rule_geometry() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.discrete_rule_partition import DiscreteRulePartition
    from src.Bayesian_state.hypothesis_space import (
        DiscreteHypothesisSpace,
        DiscreteHypothesisSpec,
    )
    from src.Bayesian_state.hypothesis_space.geometry import DiscreteRuleGeometry

    partition = DiscreteRulePartition(3, 2, include_intercept=True)

    assert isinstance(partition.hypothesis_space, DiscreteHypothesisSpace)
    assert isinstance(partition.rule_geometry, DiscreteRuleGeometry)
    assert partition.rule_geometry.space is partition.hypothesis_space
    assert isinstance(partition.hypothesis_space[0], DiscreteHypothesisSpec)
    assert DiscreteHypothesisSpec((), 1) == partition.hypothesis_space[0]
    assert partition.length == 16
    assert [rule.label for rule in partition.hypothesis_space[:8]] == [
        "Intercept",
        "-Intercept",
        "X1",
        "-X1",
        "X2",
        "-X2",
        "X3",
        "-X3",
    ]
    assert partition.rule_geometry.prediction_table.shape == (16, 8)
    assert not hasattr(partition, "rules")
    assert not hasattr(partition, "_prediction_table")


def test_discrete_partition_refactor_preserves_predictions_and_feedback() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.discrete_rule_partition import DiscreteRulePartition

    partition = DiscreteRulePartition(3, 2, include_intercept=True)
    stimuli = np.asarray(
        [
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    data = (
        stimuli,
        np.asarray([1, 2, 1, 2, 1]),
        np.asarray([1.0, 0.0, 0.5, 1.0, 0.0]),
        np.asarray([1, 2, 1, 2, 1]),
    )
    high = 1.0 / (1.0 + np.exp(-1.7))
    low = 1.0 - high

    probability = partition.get_category_probabilities(3, data, beta=1.7)
    assert np.allclose(
        probability,
        [[high, high, low, low, low], [low, low, high, high, high]],
    )
    # Exp5 historically treats every non-1 feedback value, including 0.5, as
    # binary incorrect feedback rather than Task2's related-family feedback.
    assert np.allclose(partition.calc_likelihood_entry(3, data, 1.7), high)
    assert np.allclose(
        partition.similarity_matrix[:4, :4],
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.0],
            [0.5, 0.5, 0.0, 1.0],
        ],
    )


@pytest.mark.parametrize("n_cats", [2, 4])
def test_automatic_prototypes_lie_inside_their_source_components(n_cats: int) -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, n_cats)
    for hypothesis in partition.hypothesis_space:
        for category_index, category in enumerate(hypothesis.categories):
            prototypes = partition.get_category_prototypes(
                hypothesis.index,
                category_index,
            )
            assert len(prototypes) == len(category.components)
            for prototype, component in zip(prototypes, category.components):
                assert np.all(component.A @ prototype <= component.b + 1e-9)


@pytest.mark.parametrize("n_cats", [2, 4])
@pytest.mark.parametrize("distance_mode", ["prototype", "boundary"])
def test_all_continuous_hypotheses_return_normalized_probabilities(
    n_cats: int,
    distance_mode: str,
) -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, n_cats)
    stimuli = np.random.default_rng(17).random((3, 4))
    data = (stimuli, np.ones(len(stimuli), dtype=int))
    for hypothesis in range(partition.length):
        probability = partition.get_category_probabilities(
            hypothesis,
            data,
            beta=2.0,
            distance_mode=distance_mode,
        )
        assert probability.shape == (n_cats, len(stimuli))
        assert np.isfinite(probability).all()
        assert np.allclose(probability.sum(axis=0), 1.0)


def test_original_binary_families_keep_boundary_prototype_hard_labels() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(4, 2)
    agreement = partition.prototype_boundary_agreement(
        n_samples=2000,
        random_state=91,
    )

    assert np.allclose(agreement[:19], 1.0)
    assert np.all((agreement >= 0.0) & (agreement <= 1.0))


@pytest.mark.parametrize(
    ("parameter", "tolerance"),
    [
        ("pairwise_similarity_tolerance", 0.0),
        ("pairwise_similarity_tolerance", 1.0),
        ("pairwise_similarity_tolerance", np.nan),
        ("center_band_tolerance", 0.0),
        ("center_band_tolerance", 0.5),
        ("center_band_tolerance", np.inf),
    ],
)
def test_binary_band_tolerances_are_validated(
    parameter: str,
    tolerance: float,
) -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    with pytest.raises(ValueError, match=parameter):
        ContinuousPartition(4, 2, **{parameter: tolerance})


def test_four_category_partition_keeps_116_fixed_label_geometric_hypotheses() -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    partition = ContinuousPartition(n_dims=4, n_cats=4)

    assert partition.length == 116
    assert partition.prototype_geometry.prototypes.shape == (116, 1, 4, 4)
    assert np.isfinite(partition.prototype_geometry.prototypes).all()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("include_label_reversals", True),
        ("label_permutations", [(0, 1), (1, 0)]),
    ],
)
def test_partition_rejects_removed_label_permutation_options(
    key: str,
    value: object,
) -> None:
    from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition

    with pytest.raises(TypeError, match=key):
        ContinuousPartition(n_dims=4, n_cats=2, **{key: value})


def test_utils_package_import_has_no_runtime_side_effects() -> None:
    script = """
import logging
from pathlib import Path

def forbidden(*args, **kwargs):
    raise AssertionError("filesystem or logging side effect during utils import")

Path.mkdir = forbidden
Path.glob = forbidden
logging.basicConfig = forbidden
logging.FileHandler = forbidden

import src.Bayesian_state.utils as utils
assert getattr(utils.MODEL_STRUCT, "_values") is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_evaluation_and_metrics_package_imports_are_lazy() -> None:
    script = """
import sys
import src.Bayesian_state.evaluation as evaluation
import src.Bayesian_state.evaluation.particle_filter as particle_filter_evaluation
import src.Bayesian_state.metrics as metrics

assert "src.Bayesian_state.evaluation.evaluator" not in sys.modules
assert "src.Bayesian_state.evaluation.particle_filter.strategy" not in sys.modules
assert "src.Bayesian_state.evaluation.particle_filter.residuals" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
assert not any(
    name.startswith("src.Bayesian_state.metrics.")
    for name in sys.modules
)
assert "ModelEvaluator" in dir(evaluation)
assert "run_particle_filter_strategy_audit" in dir(particle_filter_evaluation)
assert "choice_nll" in dir(metrics)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_module_phase_contract_does_not_depend_on_module_names() -> None:
    from src.Bayesian_state.model import ModelContext
    from src.Bayesian_state.model.engine import BayesianStateEngine, IndexedSet
    from src.Bayesian_state.model.modules import BaseModule, ModulePhase, ModuleRole

    events = []

    class BeforeChoice(BaseModule):
        phase = ModulePhase.PRE_CHOICE
        role = ModuleRole.PERCEPTION

        def process(self, **kwargs):
            events.append(("before", kwargs["marker"]))

    class AfterChoice(BaseModule):
        phase = ModulePhase.POST_CHOICE
        role = ModuleRole.MEMORY

        def process(self, **kwargs):
            events.append(("after", kwargs["marker"]))

    class Likelihood:
        distance_mode = "prototype"

        def process(self, observation, hypotheses, beta=None):
            del observation, hypotheses, beta
            return np.asarray([1.0])

    engine = BayesianStateEngine(
        agenda=["arbitrary_after_name", "arbitrary_before_name"],
        hypotheses_set=IndexedSet([0]),
        partition=object(),
        observation_likelihood=Likelihood(),
        context=ModelContext(),
    )
    engine.register_module("arbitrary_after_name", AfterChoice(engine))
    engine.register_module("arbitrary_before_name", BeforeChoice(engine))

    assert engine.get_module(ModuleRole.PERCEPTION) is engine.modules[
        "arbitrary_before_name"
    ]
    assert engine.get_module(ModuleRole.MEMORY) is engine.modules[
        "arbitrary_after_name"
    ]

    engine.run_phase(ModulePhase.PRE_CHOICE, shared_kwargs={"marker": 1})
    engine.run_phase(ModulePhase.POST_CHOICE, shared_kwargs={"marker": 2})

    assert events == [("before", 1), ("after", 2)]


def test_indexed_set_exposes_a_read_only_inverse_mapping() -> None:
    from src.Bayesian_state.model import IndexedSet

    indexed = IndexedSet(["a", "b"])

    assert indexed.inv["a"] == 0
    with pytest.raises(TypeError):
        indexed.inv["a"] = 1


def test_model_config_requires_agenda_and_modules_to_match() -> None:
    from src.Bayesian_state.model import ModelConfig

    with pytest.raises(ValueError, match="list every configured module"):
        ModelConfig.from_mapping(
            {
                "agenda": [],
                "modules": {
                    "perception": {"class": "package.Perception"},
                },
            }
        )


def test_partition_and_transition_contracts_are_abstract() -> None:
    import inspect

    from src.Bayesian_state.hypothesis_space.observation_model import BasePartition
    from src.Bayesian_state.model import TwoStepHypothesisTransitionMixin

    assert inspect.isabstract(BasePartition)
    assert inspect.isabstract(TwoStepHypothesisTransitionMixin)


def test_trajectory_delegates_selection_rules() -> None:
    from src.Bayesian_state.metrics import selection, trajectory

    assert hasattr(selection, "distribution_selection_metrics")
    for helper in (
        "_upper_bound_violation",
        "_lower_bound_violation",
        "_interval_violation",
        "_ppc_interval_selection",
    ):
        assert not hasattr(trajectory, helper)


def test_grid_and_cd_share_search_base() -> None:
    from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
    from src.Bayesian_state.optimization.search.grid import HyperGridOptimizer
    from src.Bayesian_state.optimization.search.common import HyperSearchBase

    assert issubclass(HyperCDOptimizer, HyperSearchBase)
    assert issubclass(HyperGridOptimizer, HyperSearchBase)
    assert callable(HyperSearchBase.run_subject)
    for name in ("_prepare_stage_config", "_apply_hyperparams", "_build_runner"):
        assert name not in HyperCDOptimizer.__dict__
        assert name not in HyperGridOptimizer.__dict__


def test_combined_workflow_uses_public_in_process_apis() -> None:
    workflow_path = ROOT_DIR / "src/Bayesian_state/run_hyper_then_simulation.py"
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"))

    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "subprocess" not in imported_modules
    assert "_run_subject_pipeline" not in called_attributes
    assert "run_subject" in called_attributes
    assert "run_simulation" in called_names


def test_hyper_evaluation_is_split_by_execution_cost() -> None:
    from src.Bayesian_state.optimization.diagnostics import predictive
    from src.Bayesian_state.optimization.diagnostics import search

    assert callable(search.evaluate_hyper_cd_convergence)
    assert callable(search.evaluate_near_optimal_plateau)
    assert callable(search.evaluate_multiobjective_selection)
    assert not hasattr(search, "diagnose_hyper_accuracy_sampling")
    assert callable(predictive.diagnose_hyper_accuracy_sampling)
    assert callable(predictive.evaluate_volatility_calibration)
