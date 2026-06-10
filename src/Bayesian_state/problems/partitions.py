"""Partition hypotheses and category-likelihood geometry.

The likelihood flow in this file has three layers:

1. Public likelihood API
   ``calc_likelihood`` and ``calc_likelihood_entry`` handle hypothesis loops,
   beta values, normalization, and feedback-code likelihoods.
2. Category-probability geometry
   ``prototype`` and ``boundary`` are parallel distance modes. Both produce a
   ``[n_cats, n_trials]`` probability matrix.
3. Partition construction
   Concrete split definitions, prototype centers, boundary regions, and
   hypothesis similarity live in ``Partition``.
"""
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple
import itertools
from itertools import product

import numpy as np
from pathlib import Path
from .base_problem import softmax, euc_dist


# =============================================================================
# BasePartition: public API, distance modes, and shared helpers
# =============================================================================
class BasePartition(ABC):
    """Shared likelihood API for partition hypothesis spaces."""

    EPS = 1e-12
    DISTANCE_MODE_PROTOTYPE = "prototype"
    DISTANCE_MODE_BOUNDARY = "boundary"
    VALID_DISTANCE_MODES = (DISTANCE_MODE_PROTOTYPE, DISTANCE_MODE_BOUNDARY)
    FEEDBACK_MODE_CATEGORY = "category_feedback"
    FEEDBACK_MODE_BERNOULLI_CHOICE = "bernoulli_choice"
    VALID_FEEDBACK_MODES = (
        FEEDBACK_MODE_CATEGORY,
        FEEDBACK_MODE_BERNOULLI_CHOICE,
    )

    # Class layout:
    # 1. core construction methods
    # 2. public likelihood API
    # 3. distance-mode dispatch
    # 4. category-probability implementations
    # 5. internal helpers

    def __init__(self, n_dims: int, n_cats: int, n_protos: int = 1, **kwargs):
        """Build split definitions, prototype centers, and region cache."""
        self.n_dims = n_dims
        self.n_cats = n_cats
        self.n_protos = n_protos
        self.splits = self.get_all_splits()
        self.prototypes = self.get_prototypes()
        self.regions = list(self.get_regions())

    @property
    def length(self):
        """Number of hypotheses in this partition space."""
        return self.prototypes.shape[0]

    def get_all_splits(self):
        """Return all split definitions for the concrete partition space."""
        raise NotImplementedError

    def get_prototypes(self):
        """Return numeric prototypes with shape [n_hypo, n_proto, n_cat, n_dim]."""
        raise NotImplementedError

    def get_regions(self):
        """Return boundary-region constraints for every split definition."""
        return [self.build_regions(split) for split in self.splits]

    def build_regions(self, split):
        """Build category-region constraints for one split definition."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public likelihood API
    # ------------------------------------------------------------------
    def calc_likelihood(self,
                        hypos: List[int] | Tuple[int],
                        data: list | tuple,
                        beta: list | tuple | float | np.ndarray = 1.,
                        distance_mode: str = DISTANCE_MODE_PROTOTYPE,
                        normalized: bool = True,
                        **kwargs) -> np.ndarray:  # BaseLikelihood:
        """
        Calculate likelihood.

        Parameters
        ----------
        hypos : List[int] | Tuple[int]
            List of hypothesis indices.
        data : list | tuple
            Observation data (stimulus, choices, responses).
        beta : float | list | tuple | np.ndarray
            Softmax inverse temperature. Can be:
            - A scalar (applied to all hypotheses)
            - A list/tuple/array of per-hypothesis beta values
        normalized : bool
            Whether to normalize the result.

        Returns
        -------
        np.ndarray
            Likelihood matrix of shape [n_trials, n_hypos].
        """
        beta_values = self._resolve_beta_vector(beta, len(hypos))
        resolved_mode = self._resolve_distance_mode(distance_mode)
        ret = np.zeros([len(data[2]), len(hypos)], dtype=float)

        for j, h in enumerate(hypos):
            ret[:, j] = self.calc_likelihood_entry(h, data, beta_values[j],
                                                   distance_mode=resolved_mode,
                                                   **kwargs)
        if normalized:
            return ret / np.sum(ret, axis=1, keepdims=True)
        return ret

    def calc_likelihood_entry(self,
                              hypo: int,
                              data: list | tuple,
                              beta: float,
                              distance_mode: str = DISTANCE_MODE_PROTOTYPE,
                              **kwargs) -> np.ndarray:
        """Likelihood for one hypothesis and the observed feedback sequence.

        This method deliberately has two stages:
        1. compute category probabilities via the selected distance mode;
        2. map those category probabilities to feedback likelihood.
        """
        prob = self.get_category_probabilities(
            hypo=hypo,
            data=data,
            beta=beta,
            distance_mode=distance_mode,
            **kwargs,
        )
        return self._feedback_likelihood_from_category_probabilities(
            hypo=hypo,
            prob=prob,
            data=data,
            feedback_likelihood_mode=kwargs.get(
                "feedback_likelihood_mode",
                self.FEEDBACK_MODE_CATEGORY,
            ),
            feedback_lapse=kwargs.get("feedback_lapse", 0.0),
        )

    def calc_trueprob_entry(self,
                            hypo: int,
                            data: list | tuple,
                            beta: float | list | tuple | np.ndarray,
                            distance_mode: str = DISTANCE_MODE_PROTOTYPE,
                            **kwargs) -> np.ndarray:
        """Return the probability assigned to the true category per trial."""

        if isinstance(beta, np.ndarray):
            beta_value = float(beta.flatten()[0])
        elif isinstance(beta, (list, tuple)):
            beta_value = float(beta[0])
        else:
            beta_value = float(beta)

        prob = self.get_category_probabilities(
            hypo=hypo,
            data=data,
            beta=beta_value,
            distance_mode=distance_mode,
            **kwargs,
        )

        category = np.asarray(data[3], dtype=int) - 1
        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)
        return prob[category.flatten(), np.arange(prob.shape[1])]

    # ------------------------------------------------------------------
    # Distance-mode dispatch
    # ------------------------------------------------------------------
    def get_category_probabilities(self,
                                   hypo: int,
                                   data: list | tuple,
                                   beta: float,
                                   distance_mode: str = DISTANCE_MODE_PROTOTYPE,
                                   **kwargs) -> np.ndarray:
        """Return category probabilities for the requested geometry.

        This is the single dispatch point for distance-mode selection. Both
        branches return the same shape: ``[n_cats, n_trials]``.
        """
        mode = self._resolve_distance_mode(distance_mode)
        if mode == self.DISTANCE_MODE_PROTOTYPE:
            return self.calc_category_probabilities_prototype(
                hypo=hypo,
                data=data,
                beta=beta,
                **kwargs,
            )
        if mode == self.DISTANCE_MODE_BOUNDARY:
            return self.calc_category_probabilities_boundary(
                hypo=hypo,
                data=data,
                beta=beta,
                **kwargs,
            )
        raise AssertionError(f"Unhandled distance_mode: {mode}")

    def get_category_assignment(self,
                                hypo: int,
                                stimulus: np.ndarray,
                                distance_mode: str = DISTANCE_MODE_PROTOTYPE,
                                beta: float = 1.0,
                                **kwargs) -> int:
        """Assign a single stimulus to the most probable category."""
        trial_data = ([np.asarray(stimulus, dtype=float)], [1], [1.0])
        prob = self.get_category_probabilities(
            hypo=hypo,
            data=trial_data,
            beta=beta,
            distance_mode=distance_mode,
            **kwargs,
        )
        return int(np.argmax(prob[:, 0]))

    # ------------------------------------------------------------------
    # Category-probability implementation: prototype distance
    # ------------------------------------------------------------------
    def calc_category_probabilities_prototype(self,
                                              hypo: int,
                                              data: list | tuple,
                                              beta: float,
                                              **kwargs) -> np.ndarray:
        """Category probabilities from distance to prototype centers.

        For each category, the stimulus distance is the minimum distance to that
        category's prototype(s). Softmax over ``-beta * distance`` converts the
        distance matrix to probabilities. Returns ``[n_cats, n_trials]``.

        """
        stimulus, _ = data[:2]
        partition = self.prototypes[hypo]
        distances = euc_dist(partition, np.array(stimulus))
        typical_distances = np.min(distances, axis=0)

        prob = softmax(typical_distances, -beta, axis=0)

        return prob

    # ------------------------------------------------------------------
    # Category-probability implementation: boundary distance
    # ------------------------------------------------------------------
    def calc_category_probabilities_boundary(self,
                                             hypo: int,
                                             data: list | tuple,
                                             beta: float,
                                             **kwargs) -> np.ndarray:
        """Category probabilities from distance to boundary regions.

        Subclasses provide regions as dictionaries with A and b arrays, where
        each category region is defined by A @ x <= b. This method only turns
        distances to those regions into category probabilities.
        """
        stimuli, _ = data[:2]
        categories = self.regions[hypo]
        n_cats = len(categories)
        n_trials = len(stimuli)
        dmat = np.zeros((n_trials, n_cats))

        for c, cat in enumerate(categories):
            A, b = cat["A"], cat["b"]
            for t, x in enumerate(stimuli):
                dmat[t, c] = self._distance_to_region(np.array(x), A, b)

        scores = np.exp(-beta * dmat)
        prob = scores / np.sum(scores, axis=1, keepdims=True)
        return prob.T

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _project_to_halfspace(y, a, b):
        """Project a point onto the halfspace a @ x <= b."""
        a = np.asarray(a, dtype=float)
        diff = np.dot(a, y) - b
        if diff <= 0:
            return y
        return y - diff / (np.dot(a, a) + 1e-9) * a

    @staticmethod
    def _project_to_box01(y):
        """Clip a point to the unit cube."""
        return np.clip(y, 0.0, 1.0)

    @classmethod
    def _project_to_polytope(cls, y, A, b, n_iter=100):
        """Project a point onto A @ x <= b with unit-cube bounds."""
        yk = y.copy()
        p_aux = [np.zeros_like(y) for _ in range(len(A))]
        for _ in range(n_iter):
            for i in range(len(A)):
                y_hat = yk + p_aux[i]
                y_new = cls._project_to_halfspace(y_hat, A[i], b[i])
                p_aux[i] = y_hat - y_new
                yk = y_new
            yk = cls._project_to_box01(yk)
        return yk

    @classmethod
    def _distance_to_region(cls, x, A, b):
        """Return distance from a point to a boundary region."""
        vals = A @ x - b
        if np.all(vals <= 0 + 1e-9):
            return 0.0
        proj = cls._project_to_polytope(x, A, b)
        return np.linalg.norm(x - proj)

    @classmethod
    def _resolve_distance_mode(cls, distance_mode: str) -> str:
        """Validate and return a configured distance mode."""
        if distance_mode not in cls.VALID_DISTANCE_MODES:
            raise ValueError(
                f"Unsupported distance_mode '{distance_mode}'. "
                f"Expected one of: {cls.VALID_DISTANCE_MODES}."
            )
        return distance_mode

    @staticmethod
    def _resolve_beta_vector(beta, n_hypos: int) -> list[float]:
        """Return one beta value per hypothesis.

        A scalar beta is broadcast to all hypotheses. If a sequence has the
        wrong length, its first value is used as a conservative fallback; this
        preserves the previous behavior while keeping the entry point compact.
        """
        if isinstance(beta, np.ndarray):
            beta_values = beta.flatten().tolist()
        elif isinstance(beta, (int, float)):
            beta_values = [float(beta)] * n_hypos
        elif isinstance(beta, (list, tuple)):
            beta_values = list(beta)
        else:
            beta_values = [float(beta)] * n_hypos

        if len(beta_values) != n_hypos:
            default_beta = beta_values[0] if beta_values else 1.0
            beta_values = [default_beta] * n_hypos

        return [float(x) for x in beta_values]

    def _feedback_likelihood_from_category_probabilities(
        self,
        hypo: int,
        prob: np.ndarray,
        data: list | tuple,
        feedback_likelihood_mode: str = FEEDBACK_MODE_CATEGORY,
        feedback_lapse: float = 0.0,
    ) -> np.ndarray:
        """Map category probabilities to observed feedback likelihood.

        ``prob`` has shape ``[n_cats, n_trials]``. Feedback is encoded as:
        ``1`` for exact species/category feedback, ``0.5`` for family-level
        feedback, and any other value for wrong feedback.
        """
        choices = np.asarray(data[1], dtype=int).copy() - 1
        responses = np.asarray(data[2])
        n_trials = len(choices)
        mode = self._resolve_feedback_likelihood_mode(feedback_likelihood_mode)
        lapse = self._resolve_feedback_lapse(feedback_lapse)

        p_species = prob[choices, np.arange(n_trials)]
        if mode == self.FEEDBACK_MODE_BERNOULLI_CHOICE:
            chance = 1.0 / float(max(1, prob.shape[0]))
            p_choice = (1.0 - lapse) * p_species + lapse * chance
            responses_float = np.clip(np.asarray(responses, dtype=float), 0.0, 1.0)
            likelihood = np.power(p_choice, responses_float) * np.power(
                1.0 - p_choice,
                1.0 - responses_float,
            )
            return np.clip(likelihood, self.EPS, 1.0 - self.EPS)

        fam_sum = np.zeros(n_trials)
        conn_map = getattr(self, "connectivity_map", {})
        if conn_map:
            mask = np.zeros_like(prob, dtype=bool)
            for t in range(n_trials):
                alt_cats = conn_map[hypo][choices[t]]
                mask[alt_cats, t] = True
            fam_sum = (prob * mask).sum(axis=0)

        p_wrong = 1.0 - p_species
        likelihood = np.where(
            responses == 1,
            p_species,
            np.where(responses == 0.5, fam_sum, p_wrong),
        )
        return np.clip(likelihood, self.EPS, 1.0 - self.EPS)

    @classmethod
    def _resolve_feedback_likelihood_mode(cls, mode: str) -> str:
        mode = str(mode).strip().lower()
        aliases = {
            "category": cls.FEEDBACK_MODE_CATEGORY,
            "categorical": cls.FEEDBACK_MODE_CATEGORY,
            "legacy": cls.FEEDBACK_MODE_CATEGORY,
            "deterministic": cls.FEEDBACK_MODE_CATEGORY,
            "deterministic_feedback": cls.FEEDBACK_MODE_CATEGORY,
            "probabilistic": cls.FEEDBACK_MODE_BERNOULLI_CHOICE,
            "probabilistic_feedback": cls.FEEDBACK_MODE_BERNOULLI_CHOICE,
            "bernoulli": cls.FEEDBACK_MODE_BERNOULLI_CHOICE,
        }
        resolved = aliases.get(mode, mode)
        if resolved not in cls.VALID_FEEDBACK_MODES:
            raise ValueError(
                f"Unsupported feedback_likelihood_mode '{mode}'. "
                f"Expected one of: {cls.VALID_FEEDBACK_MODES}."
            )
        return resolved

    @staticmethod
    def _resolve_feedback_lapse(value: float) -> float:
        lapse = float(value)
        if not np.isfinite(lapse) or lapse < 0.0 or lapse >= 1.0:
            raise ValueError(f"feedback_lapse must be in [0, 1), got {value!r}.")
        return lapse

# =============================================================================
# Partition: concrete split space and boundary geometry
# =============================================================================
@dataclass(frozen=True)
class _SplitSpec:
    """Internal definition of one concrete split hypothesis."""

    type: str
    hyperplanes: list


class Partition(BasePartition):
    """Concrete partition space with prototype and boundary geometry."""
    EPS = 1e-7

    # In-process cache for loaded similarity matrices:
    # {(n_dims, n_cats, n_samples, region_label_version): matrix_array}.
    _loaded_matrices_cache = {}
    REGION_LABEL_VERSION = "prototype_labels_v2"

    # On-disk cache directory for similarity matrices.
    DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

    # Class layout:
    # 1. concrete split and region construction
    # 2. similarity helpers
    # 3. prototype-center construction

    def __init__(self, n_dims: int, n_cats: int, n_protos: int = 1, **kwargs):
        """Build split definitions, prototype centers, and region cache."""
        super().__init__(n_dims, n_cats, n_protos, **kwargs)
        self.connectivity_map = self._compute_connectivity_map()

        # Similarity can be expensive, so only remember how to build/load it.
        # The matrix itself is loaded or computed on first access.
        cache_dir = Path(kwargs.get("cache_dir", self.DEFAULT_CACHE_DIR))
        self._similarity_n_samples = int(kwargs.get("similarity_n_samples", 100000))
        filename = (
            f"similarity_matrix_{self.REGION_LABEL_VERSION}"
            f"_d{n_dims}_c{n_cats}_n{self._similarity_n_samples}.npy"
        )
        self._similarity_matrix_path = cache_dir / filename
        self._similarity_matrix = None

    @property
    def similarity_matrix(self):
        """Load or compute the hypothesis similarity matrix on first access."""
        if self._similarity_matrix is None:
            self._similarity_matrix = self._load_or_compute_similarity(
                self.n_dims,
                self.n_cats,
                self._similarity_matrix_path,
                self._similarity_n_samples,
            )
        return self._similarity_matrix

    # ------------------------------------------------------------------
    # Internal helpers: connectivity and boundary geometry
    # ------------------------------------------------------------------
    def _compute_connectivity_map(self) -> dict[int, dict[int, list[int]]]:
        """Return family-feedback neighbors for each hypothesis/category."""
        conn = {}
        n_cats, n_dims = self.n_cats, self.n_dims
        centers_all = self.prototypes.squeeze(axis=1)

        for h in range(self.length):
            centers = centers_all[h]
            conn[h] = {c: [] for c in range(n_cats)}

            # Family feedback links categories that differ on exactly one axis.
            for a in range(n_cats):
                for b in range(a + 1, n_cats):
                    diff_cnt = np.sum(
                        np.abs(centers[a] - centers[b]) > self.EPS)
                    if diff_cnt == 1:
                        conn[h][a].append(b)
                        conn[h][b].append(a)
        return conn

    # ------------------------------------------------------------------
    # Construction helper: split definitions -> boundary regions
    # ------------------------------------------------------------------
    def build_regions(self, split):
        """Build linear region constraints for each category.

        Each returned category is represented as ``{"A": A, "b": b}``, meaning
        the category region is ``A @ x <= b``.
        """
        split_type = split.type
        hyperplanes = split.hyperplanes
        three_plane_types = {
            "3d_axis_triple", "3d_axis_equality_sum", "4d_equality_axis_pair",
            "4d_sum_axis_pair"
        }

        # Generic split types: category-specific sign combinations. The order
        # must match get_prototypes(), so category c's prototype lies inside
        # category c's boundary region.
        if split_type not in three_plane_types and split_type not in (
                "dimension_max", "dimension_min"):
            sign_orders = {
                # One-plane, two-category splits: low side, then high side.
                "axis": [(1, ), (-1, )],
                "equality": [(1, ), (-1, )],
                "sum": [(1, ), (-1, )],
                "mixed": [(1, ), (-1, )],
                # Two-plane, four-category splits.
                "2d_axis_pair": [(1, 1), (-1, 1), (1, -1), (-1, -1)],
                "2d_equality_sum": [(-1, 1), (1, -1), (-1, -1), (1, 1)],
                "3d_axis_equality": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
                "3d_axis_sum": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
                "4d_equality_pair": [(1, 1), (-1, 1), (1, -1), (-1, -1)],
                "4d_sum_pair": [(1, 1), (-1, 1), (1, -1), (-1, -1)],
            }
            signs_iter = sign_orders.get(split_type)
            if signs_iter is None:
                signs_iter = list(product([1, -1], repeat=len(hyperplanes)))

            categories = []
            for signs in signs_iter:
                A, b = [], []
                for (a, bi), s in zip(hyperplanes, signs):
                    A.append(s * np.array(a))
                    b.append(s * bi)
                categories.append({'A': np.vstack(A), 'b': np.array(b)})
            return categories

        # Three-plane split types have an explicit four-category layout.
        if split_type in three_plane_types:
            (a1, b1), (a2, b2), (a3, b3) = hyperplanes
            a1, a2, a3 = map(np.asarray, (a1, a2, a3))
            cats = [
                {
                    'A': np.vstack([a1, a2]),
                    'b': np.array([b1, b2])
                },
                {
                    'A': np.vstack([a1, -a2]),
                    'b': np.array([b1, -b2])
                },
                {
                    'A': np.vstack([-a1, a3]),
                    'b': np.array([-b1, b3])
                },
                {
                    'A': np.vstack([-a1, -a3]),
                    'b': np.array([-b1, -b3])
                }
            ]
            return cats

        # Dimension-extreme split types select the max/min feature dimension.
        if split_type in ("dimension_max", "dimension_min"):
            n_dims = self.n_dims
            cats = []
            for i in range(n_dims):
                As, bs = [], []
                for j in range(n_dims):
                    if i == j:
                        continue
                    a = np.zeros(n_dims, dtype=float)
                    if split_type == "dimension_max":
                        # x_i >= x_j  <=>  -x_i + x_j <= 0
                        a[i], a[j] = -1., 1.
                        bi = 0.0
                    else:
                        # x_i <= x_j  <=>   x_i - x_j <= 0
                        a[i], a[j] = 1., -1.
                        bi = 0.0
                    As.append(a)
                    bs.append(bi)
                A = np.vstack(As) if As else np.zeros((0, n_dims))
                b = np.array(bs, dtype=float)
                cats.append({'A': A, 'b': b})
            return cats

    # ------------------------------------------------------------------
    # Internal helpers: similarity cache and category assignment
    # ------------------------------------------------------------------

    def _load_or_compute_similarity(self, n_dims, n_cats, file_path, n_samples):
        """Load a similarity matrix from memory/disk or compute it."""
        cache_key = (n_dims, n_cats, int(n_samples), self.REGION_LABEL_VERSION)

        # 1. Check in-process cache.
        if cache_key in Partition._loaded_matrices_cache:
            return Partition._loaded_matrices_cache[cache_key]

        # 2. Check on-disk cache.
        if file_path.exists():
            print(f"Loading similarity matrix from disk: {file_path}")
            try:
                matrix = np.load(file_path)
                # Recompute if the cached matrix was built for another space.
                expected_len = self.length
                if matrix.shape[0] == expected_len and matrix.shape[1] == expected_len:
                    Partition._loaded_matrices_cache[cache_key] = matrix
                    return matrix
                else:
                    print("Cached matrix shape mismatch. Recomputing...")
            except Exception as e:
                print(f"Error loading cache file: {e}. Recomputing...")

        # 3. Compute and persist when no valid cache exists.
        print(f"Computing similarity matrix for d={n_dims}, c={n_cats} (will save to {file_path.name})...")
        matrix = self._compute_hypothesis_similarity_matrix(n_samples)

        # Store both on disk and in memory.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, matrix)

        Partition._loaded_matrices_cache[cache_key] = matrix

        return matrix
    
    def _get_category_assignments_region(
        self,
        hypo: int,
        stimuli: np.ndarray,
        tol: float = 1e-9
    ) -> np.ndarray:
        """Assign stimuli to categories by boundary-region membership.

        A point inside exactly one region gets that category. Boundary ties use
        the smallest violation score; outside points use nearest-region distance.
        Returns an integer array with shape ``[n_samples]``.
        """
        if hypo >= len(self.regions):
            raise IndexError(
                f"Hypothesis index {hypo} out of bounds for regions with length {len(self.regions)}."
            )

        categories = self.regions[hypo]
        n_samples = stimuli.shape[0]
        assignments = np.full(n_samples, -1, dtype=int)

        for i, x in enumerate(stimuli):
            inside_cats = []
            violation_scores = []

            for c, cat in enumerate(categories):
                A = np.asarray(cat['A'], dtype=float)
                b = np.asarray(cat['b'], dtype=float)

                vals = A @ x - b
                pos_violation = np.clip(vals, 0.0, None)
                violation_sum = np.sum(pos_violation)

                violation_scores.append(violation_sum)

                if np.all(vals <= tol):
                    inside_cats.append(c)

            if len(inside_cats) == 1:
                assignments[i] = inside_cats[0]

            elif len(inside_cats) > 1:
                # Boundary points can satisfy multiple regions.
                best_cat = min(inside_cats, key=lambda c: violation_scores[c])
                assignments[i] = best_cat

            else:
                # Outside all regions: use nearest-region distance.
                distances = []
                for cat in categories:
                    A = np.asarray(cat['A'], dtype=float)
                    b = np.asarray(cat['b'], dtype=float)
                    d = self._distance_to_region(x, A, b)
                    distances.append(d)

                assignments[i] = int(np.argmin(distances))

        return assignments

    def _compute_hypothesis_similarity_matrix(
        self,
        n_samples: int = 100000,
        tol: float = 1e-9,
        random_state: int | None = None
    ) -> np.ndarray:
        """Estimate pairwise hypothesis similarity by Monte Carlo sampling.

        Similarity is the fraction of uniformly sampled points in the unit cube
        that receive the same region-based category assignment under two
        hypotheses. Returns a ``[n_hypos, n_hypos]`` matrix.
        """
        n_hypos = self.length
        if n_hypos == 0:
            return np.array([])

        rng = np.random.default_rng(random_state)

        # Sample uniformly from the unit cube.
        X_samples = rng.random((n_samples, self.n_dims))

        # Cache region-based assignments for every hypothesis.
        all_assignments = np.zeros((n_hypos, n_samples), dtype=int)

        for h in range(n_hypos):
            all_assignments[h] = self._get_category_assignments_region(
                hypo=h,
                stimuli=X_samples,
                tol=tol
            )

        # Convert assignment agreement rates to a similarity matrix.
        sim_matrix = np.zeros((n_hypos, n_hypos), dtype=float)

        for i in range(n_hypos):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n_hypos):
                similarity = np.mean(all_assignments[i] == all_assignments[j])
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity

        print("Region-based similarity matrix calculation complete.")
        return sim_matrix

    # ------------------------------------------------------------------
    # Partition construction: split enumeration
    # ------------------------------------------------------------------
    def get_all_splits(self):
        """Enumerate split families and their concrete hyperplanes."""
        splits = []
        n_dims = self.n_dims
        n_cats = self.n_cats

        # Two-category split family.
        if n_cats == 2:
            # Axis-aligned hyperplanes: x_i = 0.5.
            for i in range(n_dims):
                coeff = [0] * n_dims
                coeff[i] = 1
                splits.append(_SplitSpec('axis', [(tuple(coeff), 0.5)]))

            # Equality hyperplanes: x_i = x_j.
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    coeff = [0] * n_dims
                    coeff[i] = 1
                    coeff[j] = -1
                    splits.append(_SplitSpec('equality', [(tuple(coeff), 0)]))

            # Sum hyperplanes: x_i + x_j = 1.
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    coeff = [0] * n_dims
                    coeff[i] = 1
                    coeff[j] = 1
                    splits.append(_SplitSpec('sum', [(tuple(coeff), 1)]))

            # Mixed 4D hyperplanes: x_i + x_j = x_k + x_l.
            if n_dims >= 4:
                dim_pairs = [
                    ((0, 1, 2, 3)),  # x1 + x2 = x3 + x4
                    ((0, 2, 1, 3)),  # x1 + x3 = x2 + x4
                    ((0, 3, 1, 2))  # x1 + x4 = x2 + x3
                ]
                for i, j, k, l in dim_pairs:
                    coeff = [0] * n_dims
                    coeff[i] = coeff[j] = 1
                    coeff[k] = coeff[l] = -1
                    splits.append(_SplitSpec('mixed', [(tuple(coeff), 0)]))

        # Four-category split family.
        elif n_cats == 4:
            # Split families with two hyperplanes.
            # Two-plane splits using two dimensions.
            # Two axis-aligned planes: x_i = 0.5, x_j = 0.5.
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    plane1 = ([0] * n_dims, 0.5)
                    plane2 = ([0] * n_dims, 0.5)
                    plane1[0][i] = 1
                    plane2[0][j] = 1
                    splits.append(_SplitSpec('2d_axis_pair', [plane1, plane2]))

            # Equality plus sum plane: x_i = x_j, x_i + x_j = 1.
            for i, j in itertools.combinations(range(n_dims), 2):
                plane1 = ([0] * n_dims, 0)
                plane2 = ([0] * n_dims, 1)
                plane1[0][i], plane1[0][j] = 1, -1
                plane2[0][i] = plane2[0][j] = 1
                splits.append(_SplitSpec('2d_equality_sum', [plane1, plane2]))

            # Two-plane splits using three dimensions.
            # Axis plus equality plane: x_i = 0.5, x_j = x_k.
            if n_dims >= 3:
                for i in range(n_dims):
                    remaining = [j for j in range(n_dims) if j != i]
                    for j, k in itertools.combinations(remaining, 2):
                        plane1 = ([0] * n_dims, 0.5)
                        plane2 = ([0] * n_dims, 0)
                        plane1[0][i] = 1
                        plane2[0][j], plane2[0][k] = 1, -1
                        splits.append(_SplitSpec('3d_axis_equality', [plane1, plane2]))

            # Axis plus sum plane: x_i = 0.5, x_j + x_k = 1.
            if n_dims >= 3:
                for i in range(n_dims):
                    remaining = [j for j in range(n_dims) if j != i]
                    for j, k in itertools.combinations(remaining, 2):
                        plane1 = ([0] * n_dims, 0.5)
                        plane2 = ([0] * n_dims, 1)
                        plane1[0][i] = 1
                        plane2[0][j] = plane2[0][k] = 1
                        splits.append(_SplitSpec('3d_axis_sum', [plane1, plane2]))

            # Two-plane splits using four dimensions.
            # Two equality planes: x_i = x_j, x_k = x_l.
            if n_dims >= 4:
                dim_pairs = [
                    ((0, 1, 2, 3)),  # x1 = x2, x3 = x4
                    ((0, 2, 1, 3)),  # x1 = x3, x2 = x4
                    ((0, 3, 1, 2))  # x1 = x4, x2 = x3
                ]
                for i, j, k, l in dim_pairs:
                    plane1 = ([0] * n_dims, 0)
                    plane2 = ([0] * n_dims, 0)
                    plane1[0][i], plane1[0][j] = 1, -1
                    plane2[0][k], plane2[0][l] = 1, -1
                    splits.append(_SplitSpec('4d_equality_pair', [plane1, plane2]))

            # Two sum planes: x_i + x_j = 1, x_k + x_l = 1.
            if n_dims >= 4:
                dim_pairs = [
                    ((0, 1, 2, 3)),  # x1 + x2 = 1, x3 + x4 = 1
                    ((0, 2, 1, 3)),  # x1 + x3 = 1, x2 + x4 = 1
                    ((0, 3, 1, 2))  # x1 + x4 = 1, x2 + x3 = 1
                ]
                for i, j, k, l in dim_pairs:
                    plane1 = ([0] * n_dims, 1)
                    plane2 = ([0] * n_dims, 1)
                    plane1[0][i] = plane1[0][j] = 1
                    plane2[0][k] = plane2[0][l] = 1
                    splits.append(_SplitSpec('4d_sum_pair', [plane1, plane2]))

            # Split families with three hyperplanes.
            # Three-plane splits using three dimensions.
            # Three axis-aligned planes: x_i = x_j = x_k = 0.5.
            if n_dims >= 3:
                for i, j, k in itertools.combinations(range(n_dims), 3):
                    # Keep all plane orderings because region labels depend on order.
                    for m, n1, n2 in itertools.permutations([i, j, k]):
                        plane_m = ([0] * n_dims, 0.5)
                        plane_n1 = ([0] * n_dims, 0.5)
                        plane_n2 = ([0] * n_dims, 0.5)
                        plane_m[0][m] = 1
                        plane_n1[0][n1] = 1
                        plane_n2[0][n2] = 1
                        splits.append(
                            _SplitSpec('3d_axis_triple', [plane_m, plane_n1, plane_n2]))

            # Axis, equality, and sum planes.
            if n_dims >= 3:
                for m in range(n_dims):
                    remaining = [i for i in range(n_dims) if i != m]
                    for i, j in itertools.combinations(remaining, 2):
                        plane_m = ([0] * n_dims, 0.5)  # axis
                        plane_eq = ([0] * n_dims, 0)  # equality
                        plane_sum = ([0] * n_dims, 1)  # sum

                        plane_m[0][m] = 1
                        plane_eq[0][i], plane_eq[0][j] = 1, -1
                        plane_sum[0][i] = plane_sum[0][j] = 1

                        splits.append(_SplitSpec('3d_axis_equality_sum',
                                       [plane_m, plane_eq, plane_sum]))
                        splits.append(_SplitSpec('3d_axis_equality_sum',
                                       [plane_m, plane_sum, plane_eq]))

            # Three-plane splits using four dimensions.
            # Equality plus two axis-aligned planes.
            if n_dims >= 4:
                for i, j in itertools.combinations(range(n_dims), 2):
                    remaining = [k for k in range(n_dims) if k not in (i, j)]
                    for k, l in itertools.combinations(remaining, 2):
                        plane_eq = ([0] * n_dims, 0)  # equality
                        plane_axis1 = ([0] * n_dims, 0.5)  # axis
                        plane_axis2 = ([0] * n_dims, 0.5)  # axis

                        plane_eq[0][i], plane_eq[0][j] = 1, -1
                        plane_axis1[0][k] = 1
                        plane_axis2[0][l] = 1

                        # Keep equality first; permute the two axis planes.
                        splits.append(_SplitSpec('4d_equality_axis_pair',
                                       [plane_eq, plane_axis1, plane_axis2]))
                        splits.append(_SplitSpec('4d_equality_axis_pair',
                                       [plane_eq, plane_axis2, plane_axis1]))

            # Sum plus two axis-aligned planes.
            if n_dims >= 4:
                for i, j in itertools.combinations(range(n_dims), 2):
                    remaining = [k for k in range(n_dims) if k not in (i, j)]
                    for k, l in itertools.combinations(remaining, 2):
                        plane_sum = ([0] * n_dims, 1)  # sum
                        plane_axis1 = ([0] * n_dims, 0.5)  # axis
                        plane_axis2 = ([0] * n_dims, 0.5)  # axis

                        plane_sum[0][i], plane_sum[0][j] = 1, 1
                        plane_axis1[0][k] = 1
                        plane_axis2[0][l] = 1

                        # Keep sum first; permute the two axis planes.
                        splits.append(_SplitSpec('4d_sum_axis_pair',
                                       [plane_sum, plane_axis1, plane_axis2]))
                        splits.append(_SplitSpec('4d_sum_axis_pair',
                                       [plane_sum, plane_axis2, plane_axis1]))

            # All pairwise equality planes: x_i = x_j.
            # When n_dims == n_cats == 4, these define dimension-extreme splits.
            if n_dims == n_cats:
                eq_planes = []
                for i, j in itertools.combinations(range(n_dims), 2):
                    plane = ([0] * n_dims, 0)
                    plane[0][i], plane[0][j] = 1, -1  # x_i - x_j = 0
                    eq_planes.append(plane)

                splits.append(_SplitSpec('dimension_max', eq_planes))
                splits.append(_SplitSpec('dimension_min', eq_planes))

        return splits

    # ------------------------------------------------------------------
    # Partition construction: prototypes
    # ------------------------------------------------------------------
    def get_prototypes(self):
        """Return numeric prototypes for every split and category.

        For now ``n_protos`` must be 1, so the returned shape is
        ``[n_hypo, 1, n_cat, n_dim]``.
        """
        if self.n_protos != 1:
            raise NotImplementedError("Only n_protos == 1 is currently supported.")
        n_cats = self.n_cats
        n_dims = self.n_dims
        splits = self.get_all_splits()
        results = []

        for split in splits:
            split_type = split.type
            hyperplanes = split.hyperplanes
            centers = {cat_idx: [] for cat_idx in range(n_cats)}

            # Two-category split family.
            if n_cats == 2:
                # Axis-aligned hyperplane: x_i = 0.5.
                if split_type == 'axis':
                    split_dim = next(
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0)

                    for dim in range(n_dims):
                        if dim == split_dim:
                            centers[0].append(0.25)  # x < 0.5
                            centers[1].append(0.75)  # x > 0.5
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

                # Equality hyperplane: x_i = x_j.
                elif split_type == 'equality':
                    split_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]
                    dim1, dim2 = split_dims[0], split_dims[1]

                    # The two involved dimensions use mirrored centers.
                    for dim in range(n_dims):
                        if dim == dim1:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                        elif dim == dim2:
                            centers[0].append(2 / 3)
                            centers[1].append(1 / 3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

                # Sum hyperplane: x_i + x_j = 1.
                elif split_type == 'sum':
                    split_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]
                    dim1, dim2 = split_dims[0], split_dims[1]

                    # The two involved dimensions move together.
                    for dim in range(n_dims):
                        if dim in [dim1, dim2]:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

                # Mixed 4D hyperplane: x_i + x_j = x_k + x_l.
                elif split_type == 'mixed':
                    pos_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef == 1
                    ]
                    neg_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef == -1
                    ]

                    for dim in range(n_dims):
                        if dim in pos_dims:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                        elif dim in neg_dims:
                            centers[0].append(2 / 3)
                            centers[1].append(1 / 3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

            # Four-category split family.
            elif n_cats == 4:
                # Split families with two hyperplanes.
                # Two axis-aligned planes: x_i = 0.5, x_j = 0.5.
                if split_type == '2d_axis_pair':
                    split_dims = []
                    for hyperplane in hyperplanes:
                        dim_idx = next(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                        split_dims.append(dim_idx)

                    for dim in range(n_dims):
                        if dim == split_dims[0]:
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        elif dim == split_dims[1]:
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Equality plus sum plane: x_i = x_j, x_i + x_j = 1.
                elif split_type == '2d_equality_sum':
                    split_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]

                    for dim in range(n_dims):
                        if dim in split_dims:
                            if dim == split_dims[0]:  # i
                                centers[0].append(1 / 2)
                                centers[1].append(1 / 2)
                                centers[2].append(5 / 6)
                                centers[3].append(1 / 6)
                            else:  # j
                                centers[0].append(1 / 6)
                                centers[1].append(5 / 6)
                                centers[2].append(1 / 2)
                                centers[3].append(1 / 2)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Axis plus equality plane: x_i = 0.5, x_j = x_k.
                # Axis plus sum plane: x_i = 0.5, x_j + x_k = 1.
                elif split_type in ['3d_axis_equality', '3d_axis_sum']:
                    axis_hyperplane = next(plane for plane in hyperplanes
                                           if sum(1 for c in plane[0]
                                                  if c != 0) == 1)
                    other_hyperplane = next(plane for plane in hyperplanes
                                            if plane != axis_hyperplane)

                    axis_dim = next(
                        dim_idx
                        for dim_idx, coef in enumerate(axis_hyperplane[0])
                        if coef != 0)
                    other_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(other_hyperplane[0])
                        if coef != 0
                    ]

                    for dim in range(n_dims):
                        if dim == axis_dim:
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim in other_dims:
                            if split_type == '3d_axis_equality':
                                if dim == other_dims[0]:
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                                else:
                                    centers[0].append(2 / 3)
                                    centers[1].append(1 / 3)
                                    centers[2].append(2 / 3)
                                    centers[3].append(1 / 3)
                            else:  # 3d_axis_sum
                                centers[0].append(1 / 3)
                                centers[1].append(2 / 3)
                                centers[2].append(1 / 3)
                                centers[3].append(2 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Two equality planes: x_i = x_j, x_k = x_l.
                elif split_type == '4d_equality_pair':
                    split_dim_pairs = []
                    for hyperplane in hyperplanes:
                        dim_pair = [
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0
                        ]
                        split_dim_pairs.append(dim_pair)

                    for dim in range(n_dims):
                        if dim in split_dim_pairs[0]:
                            if dim == split_dim_pairs[0][0]:  # i
                                centers[0].append(1 / 3)
                                centers[1].append(2 / 3)
                                centers[2].append(1 / 3)
                                centers[3].append(2 / 3)
                            else:  # j
                                centers[0].append(2 / 3)
                                centers[1].append(1 / 3)
                                centers[2].append(2 / 3)
                                centers[3].append(1 / 3)
                        elif dim in split_dim_pairs[1]:
                            if dim == split_dim_pairs[1][0]:  # k
                                centers[0].append(1 / 3)
                                centers[1].append(1 / 3)
                                centers[2].append(2 / 3)
                                centers[3].append(2 / 3)
                            else:  # l
                                centers[0].append(2 / 3)
                                centers[1].append(2 / 3)
                                centers[2].append(1 / 3)
                                centers[3].append(1 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Two sum planes: x_i + x_j = 1, x_k + x_l = 1.
                elif split_type == '4d_sum_pair':
                    split_dim_pairs = []
                    for hyperplane in hyperplanes:
                        dim_pair = [
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0
                        ]
                        split_dim_pairs.append(dim_pair)

                    for dim in range(n_dims):
                        if dim in split_dim_pairs[0]:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                            centers[2].append(1 / 3)
                            centers[3].append(2 / 3)
                        elif dim in split_dim_pairs[1]:
                            centers[0].append(1 / 3)
                            centers[1].append(1 / 3)
                            centers[2].append(2 / 3)
                            centers[3].append(2 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Split families with three hyperplanes.
                # Three axis-aligned planes: x_i = x_j = x_k = 0.5.
                elif split_type == '3d_axis_triple':
                    split_dims = []
                    for hyperplane in hyperplanes:
                        dim_idx = next(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                        split_dims.append(dim_idx)

                    for dim in range(n_dims):
                        if dim == split_dims[0]:  # first split
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim == split_dims[1]:  # second split, low side
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.5)
                            centers[3].append(0.5)
                        elif dim == split_dims[2]:  # third split, high side
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Axis, equality, and sum planes.
                elif split_type == '3d_axis_equality_sum':
                    axis_dim = next(
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0)
                    other_dims = set()
                    for hyperplane in hyperplanes[1:]:
                        other_dims.update(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                    other_dims = list(other_dims)

                    # The second plane determines which side uses equality.
                    is_second_equality = sum(
                        1 for c in hyperplanes[1][0]
                        if c != 0) == 2 and hyperplanes[1][1] == 0

                    for dim in range(n_dims):
                        if dim == axis_dim:  # x1
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim in other_dims:
                            if is_second_equality:
                                # Low side uses equality; high side uses sum.
                                if dim == other_dims[0]:  # x2
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                                else:  # x3
                                    centers[0].append(2 / 3)
                                    centers[1].append(1 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                            else:
                                # Low side uses sum; high side uses equality.
                                if dim == other_dims[0]:  # x2
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                                else:  # x3
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(2 / 3)
                                    centers[3].append(1 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # Equality plus two axis-aligned planes.
                # Sum plus two axis-aligned planes.
                elif split_type in [
                        '4d_equality_axis_pair', '4d_sum_axis_pair'
                ]:
                    first_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]
                    axis_dims = []
                    for hyperplane in hyperplanes[1:]:
                        dim_idx = next(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                        axis_dims.append(dim_idx)

                    for dim in range(n_dims):
                        if dim in first_dims:
                            if split_type == '4d_equality_axis_pair':
                                if dim == first_dims[0]:  # i
                                    centers[0].append(1 / 3)
                                    centers[1].append(1 / 3)
                                    centers[2].append(2 / 3)
                                    centers[3].append(2 / 3)
                                else:  # j
                                    centers[0].append(2 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(1 / 3)
                            else:  # 4d_sum_axis_pair
                                centers[0].append(1 / 3)
                                centers[1].append(1 / 3)
                                centers[2].append(2 / 3)
                                centers[3].append(2 / 3)
                        elif dim == axis_dims[0]:  # k
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.5)
                            centers[3].append(0.5)
                        elif dim == axis_dims[1]:  # l
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # All pairwise equality planes: x_i = x_j.
                # When n_dims == n_cats == 4, these define dimension-extreme splits.
                # Dimension-max case: category dimension high, others low.
                elif split_type == 'dimension_max':
                    centers_high = {}
                    for cat_idx in range(n_cats):
                        center_coords_high = [0.4] * n_dims
                        center_coords_high[cat_idx] = 0.8
                        centers_high[cat_idx] = tuple(center_coords_high)
                    results.append((split_type, centers_high))
                    continue

                # Dimension-min case: category dimension low, others high.
                elif split_type == 'dimension_min':
                    centers_low = {}
                    for cat_idx in range(n_cats):
                        center_coords_low = [0.8] * n_dims
                        center_coords_low[cat_idx] = 0.4
                        centers_low[cat_idx] = tuple(center_coords_low)
                    results.append((split_type, centers_low))
                    continue

            # Convert mutable lists to tuples for downstream use.
            centers = {k: tuple(v) for k, v in centers.items()}
            results.append((split_type, centers))

        prototype_rows = [
            [[center for _, center in sorted(center_map.items())]]
            for _, center_map in results
        ]
        return np.asarray(prototype_rows, dtype=float)
