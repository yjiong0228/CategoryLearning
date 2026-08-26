"""Mode-aware similarity operations over hypothesis spaces."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import warnings

import numpy as np

from .geometry import BoundaryGeometry, PrototypeGeometry
from .spaces import ContinuousHypothesisSpace


SIMILARITY_KIND_ASSIGNMENT_AGREEMENT = "assignment_agreement"
SIMILARITY_VERSION = "mode_aware_assignment_agreement_v2"
SIMILARITY_COMPUTATION_SEED = 0


class ContinuousSimilarity:
    """Compute hard-assignment agreement using an explicitly selected encoding."""

    DEFAULT_N_SAMPLES = 100_000
    DEFAULT_RANDOM_STATE = SIMILARITY_COMPUTATION_SEED
    RUNTIME_CACHE_DIR = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "cache"
        / "hypothesis_space"
    )
    RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "similarity"
    _memory_cache: dict[tuple, np.ndarray] = {}

    def __init__(
        self,
        hypothesis_space: ContinuousHypothesisSpace,
        boundary_geometry: BoundaryGeometry,
        prototype_geometry: PrototypeGeometry,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        runtime_cache_dir: str | Path | None = None,
    ) -> None:
        self.space = hypothesis_space
        self.boundary = boundary_geometry
        self.prototype = prototype_geometry
        self.n_samples = int(n_samples)
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples!r}.")
        self.runtime_cache_dir = (
            self.RUNTIME_CACHE_DIR
            if runtime_cache_dir is None
            else Path(runtime_cache_dir)
        )

    @property
    def matrix(self) -> np.ndarray:
        warnings.warn(
            "ContinuousSimilarity.matrix is deprecated; call get_matrix() with "
            "an explicit distance_mode.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_matrix(distance_mode="boundary")

    def _cache_key(
        self,
        *,
        distance_mode: str,
        n_samples: int,
        random_state: int,
        sample_distribution: str,
    ) -> tuple:
        return (
            self.space.signature,
            SIMILARITY_KIND_ASSIGNMENT_AGREEMENT,
            distance_mode,
            self.boundary.method if distance_mode == "boundary" else None,
            self.boundary.tolerance if distance_mode == "boundary" else None,
            (
                self.boundary.projection_iterations
                if distance_mode == "boundary"
                else None
            ),
            int(n_samples),
            int(random_state),
            sample_distribution,
            SIMILARITY_VERSION,
        )

    def _cache_path(self, key: tuple) -> Path:
        digest = sha256(repr(key).encode("utf-8")).hexdigest()[:24]
        return self.runtime_cache_dir / f"similarity_{digest}.npy"

    def get_matrix(
        self,
        *,
        distance_mode: str,
        n_samples: int | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        stimuli: np.ndarray | None = None,
        sample_distribution: str = "uniform",
    ) -> np.ndarray:
        mode = str(distance_mode).strip().lower()
        if mode not in {"prototype", "boundary"}:
            raise ValueError(
                "assignment agreement distance_mode must be prototype or boundary."
            )
        sample_count = self.n_samples if n_samples is None else int(n_samples)
        if sample_count <= 0:
            raise ValueError("n_samples must be positive.")
        distribution = str(sample_distribution).strip().lower()
        if distribution != "uniform":
            raise ValueError("Only uniform similarity sampling is implemented.")
        if stimuli is not None:
            return self.compute(
                distance_mode=mode,
                stimuli=stimuli,
                n_samples=sample_count,
                random_state=random_state,
            )

        key = self._cache_key(
            distance_mode=mode,
            n_samples=sample_count,
            random_state=random_state,
            sample_distribution=distribution,
        )
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached
        path = self._cache_path(key)
        matrix = self._load_valid_matrix(path)
        if matrix is None:
            matrix = self._load_valid_matrix(
                self._compatible_resource_path(mode, sample_count, random_state)
            )
        if matrix is None:
            matrix = self.compute(
                distance_mode=mode,
                n_samples=sample_count,
                random_state=random_state,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        self._memory_cache[key] = matrix
        return matrix

    def _compatible_resource_path(
        self,
        distance_mode: str,
        n_samples: int,
        random_state: int,
    ) -> Path:
        if (
            distance_mode != "boundary"
            or n_samples != self.DEFAULT_N_SAMPLES
            or random_state != self.DEFAULT_RANDOM_STATE
            or self.boundary.method != BoundaryGeometry.METHOD_DYKSTRA
            or not np.isclose(self.boundary.tolerance, 1.0e-9)
            or self.boundary.projection_iterations != 100
            or self.space.parameters.get("label_permutation_policy")
            != "identity_only"
            or self.space.n_dims != 4
            or self.space.parameters.get("pairwise_similarity_tolerance")
            not in (None, 0.10)
            or self.space.parameters.get("center_band_tolerance")
            not in (None, 0.10)
        ):
            return Path("__no_compatible_similarity_resource__")
        if self.space.n_cats == 2:
            return self.RESOURCE_DIR / (
                "similarity_matrix_shared_hypothesis_space_v1_d4_c2_n100000_"
                "pairtol0p1_centertol0p1.npy"
            )
        if self.space.n_cats == 4:
            return self.RESOURCE_DIR / (
                "similarity_matrix_shared_hypothesis_space_v1_d4_c4_n100000.npy"
            )
        return Path("__no_compatible_similarity_resource__")

    def _load_valid_matrix(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        try:
            matrix = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        except (OSError, ValueError):
            return None
        expected = (len(self.space), len(self.space))
        if matrix.shape != expected or not np.all(np.isfinite(matrix)):
            return None
        if np.any(matrix < -1e-12) or np.any(matrix > 1.0 + 1e-12):
            return None
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
            return None
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):
            return None
        return matrix

    def compute(
        self,
        *,
        distance_mode: str,
        n_samples: int | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        stimuli: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = str(distance_mode).strip().lower()
        geometry = self.prototype if mode == "prototype" else self.boundary
        if mode not in {"prototype", "boundary"}:
            raise ValueError("distance_mode must be prototype or boundary.")
        if stimuli is None:
            sample_count = self.n_samples if n_samples is None else int(n_samples)
            if sample_count <= 0:
                raise ValueError("n_samples must be positive.")
            rng = np.random.default_rng(random_state)
            values = rng.random((sample_count, self.space.n_dims))
        else:
            values = np.asarray(stimuli, dtype=float)
            if values.ndim != 2 or values.shape[1] != self.space.n_dims:
                raise ValueError(
                    f"stimuli must have shape [n, {self.space.n_dims}]."
                )
            if values.shape[0] == 0 or not np.all(np.isfinite(values)):
                raise ValueError("stimuli must be non-empty and finite.")
        assignments = np.asarray(
            [geometry.category_assignments(item.index, values) for item in self.space],
            dtype=int,
        )
        return np.mean(
            assignments[:, None, :] == assignments[None, :, :],
            axis=2,
        )


def prototype_center_scores(
    prototype_source,
    *,
    reference_centers: np.ndarray,
    reference_categories: np.ndarray,
    reference_weights: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Score candidates by proximity to weighted reference category centers."""
    centers = np.asarray(reference_centers, dtype=float)
    categories = np.asarray(reference_categories, dtype=int)
    weights = np.asarray(reference_weights, dtype=float)
    candidate_indices = np.asarray(candidates, dtype=int)
    scores = np.zeros(candidate_indices.size, dtype=float)
    for candidate_offset, candidate in enumerate(candidate_indices):
        for center, category, weight in zip(centers, categories, weights):
            candidate_centers = prototype_source.get_category_prototypes(
                int(candidate), int(category)
            )
            distance = np.linalg.norm(candidate_centers - center, axis=1).min()
            scores[candidate_offset] += float(weight) * np.exp(-float(distance))
    return scores


def region_overlap(
    reference_masks: np.ndarray,
    query_mask: np.ndarray,
    *,
    metric: str = "iou",
    box_volume: float = 1.0,
) -> np.ndarray:
    """Score sampled region masks using one named overlap definition."""
    references = np.asarray(reference_masks, dtype=bool)
    query = np.asarray(query_mask, dtype=bool).reshape(-1)
    if references.ndim == 1:
        references = references.reshape(1, -1)
    if references.ndim != 2 or references.shape[1] != query.size:
        raise ValueError("reference_masks and query_mask have incompatible shapes.")
    if query.size == 0:
        raise ValueError("region overlap masks must be non-empty.")
    intersection = np.sum(references & query[None, :], axis=1).astype(float)
    query_count = float(np.sum(query))
    reference_count = np.sum(references, axis=1).astype(float)
    if metric == "iou":
        denominator = reference_count + query_count - intersection
    elif metric == "precision_like":
        denominator = np.full(reference_count.shape, query_count)
    elif metric == "recall_like":
        denominator = reference_count
    elif metric == "intersection":
        return intersection / float(query.size) * float(box_volume)
    else:
        raise ValueError(f"Unsupported overlap metric: {metric}")
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection),
        where=denominator > 0.0,
    )


def prototype_boundary_agreement(
    prototype: PrototypeGeometry,
    boundary: BoundaryGeometry,
    *,
    n_samples: int = 10_000,
    random_state: int = 0,
) -> np.ndarray:
    """Audit two realizations without making either depend on the other."""
    if prototype.space.signature != boundary.space.signature:
        raise ValueError("Prototype and boundary geometry must share one space.")
    rng = np.random.default_rng(random_state)
    stimuli = rng.random((int(n_samples), prototype.space.n_dims))
    return np.asarray(
        [
            np.mean(
                prototype.category_assignments(item.index, stimuli)
                == boundary.category_assignments(item.index, stimuli)
            )
            for item in prototype.space
        ],
        dtype=float,
    )


__all__ = [
    "ContinuousSimilarity",
    "SIMILARITY_COMPUTATION_SEED",
    "SIMILARITY_KIND_ASSIGNMENT_AGREEMENT",
    "SIMILARITY_VERSION",
    "prototype_boundary_agreement",
    "prototype_center_scores",
    "region_overlap",
]
