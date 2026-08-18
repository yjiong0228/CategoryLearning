"""Runtime similarity derivation and cache policy for continuous hypotheses."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .spaces import ContinuousHypothesisSpace
from .geometry import BoundaryGeometry, PrototypeGeometry


SIMILARITY_BASIS = "boundary_fixed_labels"
SIMILARITY_VERSION = "shared_hypothesis_space_v1"
SIMILARITY_COMPUTATION_SEED = 0


class ContinuousSimilarity:
    """Compute and cache agreement of fixed labels under boundary geometry.

    Bundled matrices in ``resources/similarity`` are immutable model resources.
    Matrices for nonstandard parameters are written under ``results/cache`` so
    importing or running the model never modifies source code directories.
    """

    DEFAULT_N_SAMPLES = 100_000
    DEFAULT_RANDOM_STATE = SIMILARITY_COMPUTATION_SEED
    RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "similarity"
    RUNTIME_CACHE_DIR = Path(__file__).resolve().parents[3] / "results" / "cache" / "hypothesis_space"
    _memory_cache: dict[tuple, np.ndarray] = {}

    def __init__(
        self,
        hypothesis_space: ContinuousHypothesisSpace,
        boundary_geometry: BoundaryGeometry,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        runtime_cache_dir: str | Path | None = None,
    ) -> None:
        self.space = hypothesis_space
        self.boundary = boundary_geometry
        self.n_samples = int(n_samples)
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples!r}.")
        self.runtime_cache_dir = (
            self.RUNTIME_CACHE_DIR
            if runtime_cache_dir is None
            else Path(runtime_cache_dir)
        )
        self.filename = self._build_filename()
        self.runtime_filename = self._build_runtime_filename()
        self._matrix: np.ndarray | None = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = self._load_or_compute()
        return self._matrix

    def _build_filename(self) -> str:
        tolerance_suffix = ""
        if self.space.n_cats == 2:
            pair = self.space.parameters["pairwise_similarity_tolerance"]
            center = self.space.parameters["center_band_tolerance"]
            pair_tag = format(float(pair), ".6g").replace(".", "p")
            center_tag = format(float(center), ".6g").replace(".", "p")
            tolerance_suffix = f"_pairtol{pair_tag}_centertol{center_tag}"
        return (
            f"similarity_matrix_{SIMILARITY_VERSION}"
            f"_d{self.space.n_dims}_c{self.space.n_cats}_n{self.n_samples}"
            f"{tolerance_suffix}.npy"
        )

    def _build_runtime_filename(self) -> str:
        """Keep deterministic caches separate from historical unseeded files."""
        source = Path(self.filename)
        return f"{source.stem}_seed{self.DEFAULT_RANDOM_STATE}{source.suffix}"

    def _load_or_compute(self) -> np.ndarray:
        cache_key = (
            self.space.signature,
            self.n_samples,
            SIMILARITY_BASIS,
            SIMILARITY_VERSION,
            self.DEFAULT_RANDOM_STATE,
        )
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached

        for path in (
            self.RESOURCE_DIR / self.filename,
            self.runtime_cache_dir / self.runtime_filename,
        ):
            matrix = self._load_valid_matrix(path)
            if matrix is not None:
                self._memory_cache[cache_key] = matrix
                return matrix

        matrix = self.compute(
            n_samples=self.n_samples,
            random_state=self.DEFAULT_RANDOM_STATE,
        )
        output_path = self.runtime_cache_dir / self.runtime_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, matrix)
        self._memory_cache[cache_key] = matrix
        return matrix

    def _load_valid_matrix(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        try:
            matrix = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        expected = (len(self.space), len(self.space))
        if matrix.shape != expected:
            return None
        if not np.issubdtype(matrix.dtype, np.number):
            return None
        matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(matrix)):
            return None
        tolerance = 1e-12
        if np.any(matrix < -tolerance) or np.any(matrix > 1.0 + tolerance):
            return None
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
            return None
        if not np.allclose(
            np.diag(matrix),
            np.ones(len(self.space), dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            return None
        return matrix

    def compute(
        self,
        *,
        n_samples: int | None = None,
        tol: float = 1e-9,
        random_state: int | None = SIMILARITY_COMPUTATION_SEED,
    ) -> np.ndarray:
        """Return pairwise fixed-label agreement on uniform unit-cube samples."""
        if len(self.space) == 0:
            return np.array([])
        sample_count = self.n_samples if n_samples is None else int(n_samples)
        if sample_count <= 0:
            raise ValueError(f"n_samples must be positive, got {sample_count!r}.")
        rng = np.random.default_rng(random_state)
        stimuli = rng.random((sample_count, self.space.n_dims))
        assignments = np.asarray(
            [
                self.boundary.category_assignments(
                    hypothesis.index,
                    stimuli,
                    tol=tol,
                )
                for hypothesis in self.space
            ],
            dtype=int,
        )
        matrix = np.eye(len(self.space), dtype=float)
        for first in range(len(self.space)):
            for second in range(first + 1, len(self.space)):
                agreement = float(np.mean(assignments[first] == assignments[second]))
                matrix[first, second] = matrix[second, first] = agreement
        return matrix


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
                prototype.category_assignments(hypothesis.index, stimuli)
                == boundary.category_assignments(hypothesis.index, stimuli)
            )
            for hypothesis in prototype.space
        ],
        dtype=float,
    )


__all__ = [
    "ContinuousSimilarity",
    "SIMILARITY_BASIS",
    "SIMILARITY_COMPUTATION_SEED",
    "SIMILARITY_VERSION",
    "prototype_boundary_agreement",
]
