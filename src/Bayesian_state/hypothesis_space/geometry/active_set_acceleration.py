"""Two-stage solver: cheap clamp candidates, exact enumeration only as fallback.

A cheap candidate enumerates only the polytope's own halfspaces, projects onto
that affine space and clamps into the unit cube.  Whenever the nearest point of
a region lies on a cube face rather than on the halfspace itself, that rule
overshoots and yields no feasible candidate; the exact active-set enumeration
then supplies the answer.  Measured on the shared task spaces the cheap rule
succeeds for 99-100% of stimuli (fallbacks: 0.4% for 2D/2-cat, 1.2% for
4D/2-cat, 0% for 4D/4-cat).

A category region is a convex polytope ``A x <= b`` intersected with the unit
cube.  Minimality of the cheap rule: clamping into the cube can only decrease
``A z``, so a candidate accepted by the cheap stage already satisfies every
constraint and its distance is exact for its own active set.  The exact stage
enumerates subsets of the *full* constraint set -- the polytope rows followed by
the ``2 * n_dims`` cube faces -- because a cube face can be active at the same
time as a halfspace.  For ``x1 - x2 <= -0.1`` the nearest point to
``(0.041, 0.017)`` is ``(0, 0.1)``, where ``x1 = 0`` and the halfspace are both
active.

Both stages are exact where they accept, so the result never depends on which
stage answered.  Feasibility uses a dedicated, wider tolerance because the
shared task space contains degenerate zero-width polytopes (a boundary plane
through the origin), where "on the plane" and "inside" are indistinguishable at
the geometric tolerance and such a stimulus must resolve to distance zero.

Queries must lie inside the unit cube: the perception module clips every sampled
stimulus into ``[0, 1]``, so a stimulus outside it is a caller error rather than
a value the geometry has to interpret.

The module is self-contained: it imports only NumPy and Numba and holds no
engine, partition or configuration state.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

try:  # pragma: no cover - exercised in environments without Numba.
    from numba import njit
except ImportError:
    njit = None

# A zero-width polytope makes "inside" and "on the plane" indistinguishable at
# the geometric tolerance used for the outside pre-check.  Candidate feasibility
# therefore uses this wider floor so such points resolve to distance zero.
MIN_FEASIBILITY_TOLERANCE = 1.0e-7


def resolve_feasibility_tolerance(tolerance: float) -> float:
    """Return the candidate-feasibility tolerance for a geometry tolerance."""

    return max(float(tolerance), MIN_FEASIBILITY_TOLERANCE)


if njit is None:
    _active_set_impl = None
    _fused_single_stimulus = None
else:

    @njit(cache=True, fastmath=False, inline="always")
    def _cheap_subset_best(
        stimulus,
        all_constraints,
        all_bounds,
        polytope_rows,
        gram_inverse,
        subset_size,
        feasibility_tolerance,
    ):
        """Nearest clamp-feasible candidate over the polytope-row subsets.

        Returns ``inf`` when no subset yields a feasible candidate, which is the
        signal to run the exact enumeration instead.
        """

        n_dims = stimulus.shape[0]
        n_all = all_constraints.shape[0]
        multipliers = np.empty(subset_size, dtype=np.float64)
        for j in range(subset_size):
            accumulated = 0.0
            for m in range(subset_size):
                dot = 0.0
                for dim in range(n_dims):
                    dot += all_constraints[polytope_rows[m], dim] * stimulus[dim]
                accumulated += gram_inverse[j, m] * (
                    dot - all_bounds[polytope_rows[m]]
                )
            multipliers[j] = accumulated

        projected = np.empty(n_dims, dtype=np.float64)
        for dim in range(n_dims):
            value = stimulus[dim]
            for j in range(subset_size):
                value -= multipliers[j] * all_constraints[polytope_rows[j], dim]
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            projected[dim] = value

        for row in range(n_all):
            accumulated = 0.0
            for dim in range(n_dims):
                accumulated += all_constraints[row, dim] * projected[dim]
            if accumulated > all_bounds[row] + feasibility_tolerance:
                return np.inf

        squared = 0.0
        for dim in range(n_dims):
            difference = stimulus[dim] - projected[dim]
            squared += difference * difference
        return squared

    @njit(cache=True, fastmath=False)
    def _cheap_best(
        stimulus,
        all_constraints,
        all_bounds,
        polytope_rows,
        polytope_ginverses,
        polytope_sizes,
        feasibility_tolerance,
    ):
        best_squared = np.inf
        for set_index in range(polytope_sizes.shape[0]):
            squared = _cheap_subset_best(
                stimulus,
                all_constraints,
                all_bounds,
                polytope_rows[set_index],
                polytope_ginverses[set_index],
                polytope_sizes[set_index],
                feasibility_tolerance,
            )
            if squared < best_squared:
                best_squared = squared
        return best_squared

    @njit(cache=True, fastmath=False, inline="always")
    def _exact_subset_best(
        stimulus,
        all_constraints,
        all_bounds,
        subset_rows,
        gram_inverse,
        subset_size,
        feasibility_tolerance,
    ):
        """Distance for one candidate active subset, or ``inf`` if infeasible."""

        n_dims = stimulus.shape[0]
        n_all = all_constraints.shape[0]
        multipliers = np.empty(subset_size, dtype=np.float64)
        for j in range(subset_size):
            accumulated = 0.0
            for m in range(subset_size):
                dot = 0.0
                for dim in range(n_dims):
                    dot += all_constraints[subset_rows[m], dim] * stimulus[dim]
                accumulated += gram_inverse[j, m] * (
                    dot - all_bounds[subset_rows[m]]
                )
            multipliers[j] = accumulated

        projected = np.empty(n_dims, dtype=np.float64)
        for dim in range(n_dims):
            value = stimulus[dim]
            for j in range(subset_size):
                value -= multipliers[j] * all_constraints[subset_rows[j], dim]
            projected[dim] = value

        for row in range(n_all):
            accumulated = 0.0
            for dim in range(n_dims):
                accumulated += all_constraints[row, dim] * projected[dim]
            if accumulated > all_bounds[row] + feasibility_tolerance:
                return np.inf

        squared = 0.0
        for dim in range(n_dims):
            difference = stimulus[dim] - projected[dim]
            squared += difference * difference
        return squared

    @njit(cache=True, fastmath=False)
    def _exact_best(
        stimulus,
        all_constraints,
        all_bounds,
        set_rows,
        set_gram_inverses,
        set_sizes,
        feasibility_tolerance,
    ):
        best_squared = np.inf
        for set_index in range(set_sizes.shape[0]):
            squared = _exact_subset_best(
                stimulus,
                all_constraints,
                all_bounds,
                set_rows[set_index],
                set_gram_inverses[set_index],
                set_sizes[set_index],
                feasibility_tolerance,
            )
            if squared < best_squared:
                best_squared = squared
        return best_squared

    @njit(cache=True, fastmath=False, inline="always")
    def _distance(
        stimulus,
        all_constraints,
        all_bounds,
        polytope_rows,
        polytope_ginverses,
        polytope_sizes,
        set_rows,
        set_gram_inverses,
        set_sizes,
        tolerance,
        feasibility_tolerance,
    ):
        cheap = _cheap_best(
            stimulus, all_constraints, all_bounds,
            polytope_rows, polytope_ginverses, polytope_sizes,
            feasibility_tolerance,
        )
        if np.isfinite(cheap):
            return np.sqrt(cheap)
        exact = _exact_best(
            stimulus, all_constraints, all_bounds,
            set_rows, set_gram_inverses, set_sizes, feasibility_tolerance,
        )
        return np.sqrt(exact)

    @njit(cache=True, fastmath=False)
    def _active_set_impl(
        stimuli,
        outside_rows,
        all_constraints,
        all_bounds,
        polytope_rows,
        polytope_ginverses,
        polytope_sizes,
        set_rows,
        set_gram_inverses,
        set_sizes,
        tolerance,
        feasibility_tolerance,
    ):
        """Distances for the selected rows; unlisted rows are inside, so zero."""

        distances = np.zeros(stimuli.shape[0], dtype=np.float64)
        for outside_index in range(outside_rows.size):
            row = outside_rows[outside_index]
            distances[row] = _distance(
                stimuli[row], all_constraints, all_bounds,
                polytope_rows, polytope_ginverses, polytope_sizes,
                set_rows, set_gram_inverses, set_sizes,
                tolerance, feasibility_tolerance,
            )
        return distances

    @njit(cache=True, fastmath=False)
    def _fused_single_stimulus(
        stimulus,
        all_constraints,
        all_bounds,
        polytope_rows,
        polytope_ginverses,
        polytope_sizes,
        set_rows,
        set_gram_inverses,
        set_sizes,
        tolerance,
        feasibility_tolerance,
    ):
        """Distance for one in-cube stimulus, including the inside test.

        Fusing the inside test into the kernel matters because the model issues
        one stimulus per call and the equivalent NumPy reduction plus broadcast
        costs more than the projection itself at that size.
        """

        n_dims = stimulus.shape[0]
        n_all = all_constraints.shape[0]
        for row in range(n_all):
            accumulated = 0.0
            for dim in range(n_dims):
                accumulated += all_constraints[row, dim] * stimulus[dim]
            if accumulated > all_bounds[row] + tolerance:
                break
        else:
            return 0.0
        return _distance(
            stimulus, all_constraints, all_bounds,
            polytope_rows, polytope_ginverses, polytope_sizes,
            set_rows, set_gram_inverses, set_sizes,
            tolerance, feasibility_tolerance,
        )


def active_set_numba_available() -> bool:
    """Return whether the compiled active-set solver can be used."""

    return _active_set_impl is not None


def _enumerate_subsets(
    all_matrix: np.ndarray,
    rows: tuple[int, ...],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded row indices and inverse Grams for one subset family."""

    row_sets: list[np.ndarray] = []
    gram_inverses: list[np.ndarray] = []
    n_rows = len(rows)
    for size in range(0, n_rows + 1):
        for combination in combinations(range(n_rows), size):
            if size == 0:
                row_sets.append(np.zeros(1, dtype=np.int64))
                gram_inverses.append(np.zeros((1, 1), dtype=np.float64))
                continue
            selected = [rows[index] for index in combination]
            selected_rows = all_matrix[selected]
            if np.linalg.matrix_rank(selected_rows, tol=float(tolerance)) != size:
                continue
            try:
                gram_inverse = np.linalg.inv(selected_rows @ selected_rows.T)
            except np.linalg.LinAlgError:  # pragma: no cover - rank check guards.
                continue
            row_sets.append(np.asarray(selected, dtype=np.int64))
            gram_inverses.append(gram_inverse)

    if not row_sets:  # pragma: no cover - the empty subset is always present.
        raise ValueError("active-set solver requires at least one candidate subset.")

    max_size = max(int(rows_.size) for rows_ in row_sets)
    n_sets = len(row_sets)
    set_rows = np.zeros((n_sets, max_size), dtype=np.int64)
    set_gram_inverses = np.zeros((n_sets, max_size, max_size), dtype=np.float64)
    set_sizes = np.zeros(n_sets, dtype=np.int64)
    for index, (selected, gram_inverse) in enumerate(zip(row_sets, gram_inverses)):
        size = int(selected.size)
        set_sizes[index] = size
        set_rows[index, :size] = selected
        set_gram_inverses[index, :size, :size] = gram_inverse
    return (
        np.ascontiguousarray(set_rows),
        np.ascontiguousarray(set_gram_inverses),
        np.ascontiguousarray(set_sizes),
    )


def build_active_sets(
    all_constraints: np.ndarray,
    all_bounds: np.ndarray,
    n_polytope_rows: int,
    tolerance: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
           tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute the cheap and exact candidate families.

    ``all_constraints`` holds the ``n_polytope_rows`` polytope rows followed by
    the ``2 * n_dims`` unit-cube faces.  The cheap family enumerates subsets of
    the polytope rows only; the exact family enumerates subsets of every row up
    to ``n_dims``, which is what allows cube faces to be active alongside a
    halfspace.
    """

    all_matrix = np.ascontiguousarray(all_constraints, dtype=np.float64)
    n_all, n_dims = all_matrix.shape
    n_polytope_rows = int(n_polytope_rows)
    if not 0 <= n_polytope_rows <= n_all:
        raise ValueError("n_polytope_rows must index into all_constraints.")

    cheap = _enumerate_subsets(
        all_matrix, tuple(range(n_polytope_rows)), tolerance
    )
    exact = _enumerate_subsets(
        all_matrix,
        tuple(range(n_all)),
        tolerance,
    )
    # Trim the exact family to subsets of at most n_dims rows.
    keep = exact[2] <= min(n_dims, n_all)
    exact = (
        np.ascontiguousarray(exact[0][keep]),
        np.ascontiguousarray(exact[1][keep]),
        np.ascontiguousarray(exact[2][keep]),
    )
    return cheap, exact


def active_set_distances_numba(
    stimuli: np.ndarray,
    outside_rows: np.ndarray,
    all_constraints: np.ndarray,
    all_bounds: np.ndarray,
    cheap_sets: tuple[np.ndarray, np.ndarray, np.ndarray],
    exact_sets: tuple[np.ndarray, np.ndarray, np.ndarray],
    tolerance: float,
) -> np.ndarray:
    """Exact distances for the rows known to lie outside the region."""

    if _active_set_impl is None:
        raise RuntimeError(
            "The active-set distance solver was requested, but Numba is "
            "unavailable."
        )
    return _active_set_impl(
        np.ascontiguousarray(stimuli, dtype=np.float64),
        np.ascontiguousarray(outside_rows, dtype=np.int64),
        np.ascontiguousarray(all_constraints, dtype=np.float64),
        np.ascontiguousarray(all_bounds, dtype=np.float64),
        cheap_sets[0], cheap_sets[1], cheap_sets[2],
        exact_sets[0], exact_sets[1], exact_sets[2],
        float(tolerance),
        resolve_feasibility_tolerance(tolerance),
    )


def fused_single_stimulus(
    stimulus: np.ndarray,
    all_constraints: np.ndarray,
    all_bounds: np.ndarray,
    cheap_sets: tuple[np.ndarray, np.ndarray, np.ndarray],
    exact_sets: tuple[np.ndarray, np.ndarray, np.ndarray],
    tolerance: float,
) -> float:
    """Exact distance from one in-cube stimulus, with the inside test included."""

    if _fused_single_stimulus is None:
        raise RuntimeError(
            "The active-set distance solver was requested, but Numba is "
            "unavailable."
        )
    return float(
        _fused_single_stimulus(
            np.ascontiguousarray(stimulus, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(all_constraints, dtype=np.float64),
            np.ascontiguousarray(all_bounds, dtype=np.float64),
            cheap_sets[0], cheap_sets[1], cheap_sets[2],
            exact_sets[0], exact_sets[1], exact_sets[2],
            float(tolerance),
            resolve_feasibility_tolerance(tolerance),
        )
    )


def warmup_active_set_numba() -> bool:
    """Compile or load the cached kernels before process-level parallelism."""

    if _active_set_impl is None:
        return False
    polytope_A = np.asarray([[1.0, 0.0]], dtype=np.float64)
    polytope_b = np.asarray([0.5], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    all_constraints = np.vstack((polytope_A, -identity, identity))
    all_bounds = np.concatenate((polytope_b, np.zeros(2), np.ones(2)))
    cheap, exact = build_active_sets(
        all_constraints, all_bounds, polytope_A.shape[0], 1e-9
    )
    probe = np.asarray([[0.75, 0.25]], dtype=np.float64)
    active_set_distances_numba(
        probe, np.asarray([0], dtype=np.int64),
        all_constraints, all_bounds, cheap, exact, 1e-9,
    )
    fused_single_stimulus(
        probe[0], all_constraints, all_bounds, cheap, exact, 1e-9
    )
    return True


__all__ = [
    "MIN_FEASIBILITY_TOLERANCE",
    "active_set_distances_numba",
    "active_set_numba_available",
    "build_active_sets",
    "fused_single_stimulus",
    "resolve_feasibility_tolerance",
    "warmup_active_set_numba",
]
