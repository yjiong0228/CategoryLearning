"""Task2 block sampling and stopping, translated from the experiment program.

The environment owns categories and feedback. Cognitive modules receive the
stimulus before choice and receive feedback after choice. The MATLAB sources
are exp_program/exp123/{Task2_learning,sample_2}.m. RNG draws use NumPy;
distributional reproduction does not imply MATLAB bitwise reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri


@dataclass(frozen=True)
class TaskBlock:
    stimulus: np.ndarray
    categories: np.ndarray
    ambiguous: np.ndarray


def stimulus_labels(stimulus: np.ndarray, condition: int, ambiguity_width: float = .1) -> tuple[np.ndarray, np.ndarray]:
    """Use experimental feature-role order, before any physical-name permutation."""
    x = np.asarray(stimulus, dtype=float)
    if condition not in (1, 2, 3) or x.ndim != 2 or x.shape[1] != 4:
        raise ValueError('Need condition 1/2/3 and four feature-role columns')
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError('Normalized task features must be finite in [0,1]')
    if not 0 <= ambiguity_width <= 1:
        raise ValueError('Ambiguity width must lie in [0,1]')
    family = (x[:, 0] >= .5).astype(int)
    species = 1 + 2 * family + np.where(family == 0, x[:, 1] >= .5, x[:, 2] >= .5)
    relevant = x[:, :1 if condition == 1 else 3]
    ambiguous = ((relevant > .5 - ambiguity_width / 2) & (relevant < .5 + ambiguity_width / 2)).any(axis=1)
    return (family + 1 if condition == 1 else species).astype(int), ambiguous


def block_performance(categories: np.ndarray, choices: np.ndarray, ambiguous: np.ndarray, n_categories: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-category accuracy and clipped d-prime, excluding ambiguous stimuli.

MATLAB max(NaN,-6) returns -6 (default missing-value omission). np.fmax
preserves this behavior for constant responses and empty category cells.
Accuracy in an empty category remains NaN, so that category cannot pass.
"""
    category, choice, ambiguity = np.asarray(categories), np.asarray(choices), np.asarray(ambiguous, dtype=bool)
    if category.ndim != 1 or category.shape != choice.shape or category.shape != ambiguity.shape:
        raise ValueError('Aligned one-dimensional task arrays are required')
    if n_categories not in (2, 4) or np.any((category < 1) | (category > n_categories)) or np.any((choice < 1) | (choice > n_categories)):
        raise ValueError('Invalid task category or choice')
    form = np.zeros((n_categories, n_categories), dtype=int)
    np.add.at(form, (choice[~ambiguity].astype(int) - 1, category[~ambiguity].astype(int) - 1), 1)
    hit = np.diag(form)
    signal = form.sum(axis=0)
    false_alarm = form.sum(axis=1) - hit
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = hit / signal
        dprime = ndtri(accuracy) - ndtri(false_alarm / (form.sum() - signal))
    return accuracy, np.fmin(np.fmax(dprime, -6.), 6.)


def allocate_species(dprime: np.ndarray, condition: int, block_size: int = 64, beta: float = .2) -> np.ndarray:
    """Translate MATLAB generate(), including its minimum-count remainder rule.

The four returned species allocations can be half integers for condition 1.
sample_2 subsequently applies floor/ceil to the eight cube allocations.
"""
    d = np.asarray(dprime, dtype=float)
    if condition not in (1, 2, 3) or d.shape != (2 if condition == 1 else 4,) or not np.isfinite(d).all():
        raise ValueError('Finite d-prime for every task category is required')
    if block_size <= 0 or block_size % 4 or not np.isfinite(beta) or beta < 0:
        raise ValueError('Block size must be a positive multiple of four; beta nonnegative')
    weight = np.exp(beta * (6 - d))
    percent = weight / weight.sum()
    number = np.floor(percent / percent.sum() * block_size).astype(int)
    indices = np.flatnonzero(number == number.min())
    gap = int(block_size - number.sum())
    if len(indices) == 3:
        number[indices] += gap // 3
        remainder = int(block_size - number.sum()) % 3
        number[np.flatnonzero(number == number.min())[:remainder]] += 1
    elif len(indices) == 2:
        number[indices] += [gap // 2, (gap + 1) // 2]
    elif len(indices) == 1:
        number[indices[0]] += gap
    if number.sum() != block_size:
        raise ValueError('Original allocation rule did not fill this block')
    return np.repeat(number / 2, 2) if condition == 1 else number.astype(float)


def sample_block(species_counts: np.ndarray, condition: int, rng: np.random.Generator, ambiguity_width: float = .1) -> TaskBlock:
    """Sample the eight original feature cubes, then shuffle trial order."""
    n = np.asarray(species_counts, dtype=float)
    if n.shape != (4,) or not np.isfinite(n).all() or np.any(n < 0):
        raise ValueError('Four nonnegative species allocations are required')
    cube_species = np.array([0, 2, 0, 3, 1, 2, 1, 3])
    count = np.array([np.floor(n[0]/2), np.floor(n[2]/2), np.ceil(n[0]/2), np.floor(n[3]/2),
                      np.floor(n[1]/2), np.ceil(n[2]/2), np.ceil(n[1]/2), np.ceil(n[3]/2)], dtype=int)
    if count.sum() != n.sum() or count.sum() == 0:
        raise ValueError('Original cube allocation must produce the requested trial count')
    cube = np.repeat(np.arange(8), count)
    x = rng.random((len(cube), 4))
    # sample_2 feature2 uses the slowest bit, feature3 the middle bit.
    low = np.column_stack([cube % 2, cube // 4, (cube // 2) % 2]) / 2
    x[:, :3] = x[:, :3] / 2 + low
    categories, ambiguous = stimulus_labels(x, condition, ambiguity_width)
    expected = cube_species[cube] + 1
    if condition == 1:
        expected = (expected + 1) // 2
    np.testing.assert_array_equal(categories, expected)
    order = rng.permutation(len(cube))
    return TaskBlock(x[order], categories[order], ambiguous[order])


class AdaptiveTask:
    """Blockwise environment, with a fixed first block at each session start.

Five blocks form a session. Cognitive state persists across sessions. Task RNG
innovations are independent across sessions unless repeat_session_rng=True;
the latter repeats task RNG innovations as a sensitivity check for the original
session seed reset. Neither option reproduces MATLAB's unrelated timing draws.
The explicit horizon is administrative censoring, not an experimental stop.
"""
    def __init__(self, first_block: TaskBlock, *, condition: int, seed: int, max_blocks: int,
                 blocks_per_session: int = 5, threshold: float = .9, beta: float = .2,
                 ambiguity_width: float = .1, repeat_session_rng: bool = False,
                 adaptive: bool = True):
        labels, ambiguous = stimulus_labels(first_block.stimulus, condition, ambiguity_width)
        np.testing.assert_array_equal(labels, first_block.categories)
        np.testing.assert_array_equal(ambiguous, first_block.ambiguous)
        if len(labels) != 64 or max_blocks < 1 or blocks_per_session < 1 or not 0 < threshold <= 1:
            raise ValueError('Need the original 64-trial first block and valid stopping settings')
        self.first_block, self.condition, self.seed = first_block, condition, int(seed)
        self.max_blocks, self.blocks_per_session = int(max_blocks), int(blocks_per_session)
        self.threshold, self.beta, self.ambiguity_width = threshold, beta, ambiguity_width
        self.repeat_session_rng, self.adaptive = repeat_session_rng, adaptive
        self.rng = np.random.default_rng(seed)
        self.current = first_block
        self.history: list[dict] = []
        self.previous_session_dprime: np.ndarray | None = None
        self.finished = False

    def complete_block(self, choices: np.ndarray) -> TaskBlock | None:
        if self.finished:
            raise RuntimeError('Task already stopped or reached its horizon')
        number = len(self.history) + 1
        session = (number - 1) // self.blocks_per_session + 1
        within = (number - 1) % self.blocks_per_session + 1
        accuracy, dprime = block_performance(self.current.categories, choices, self.current.ambiguous, 2 if self.condition == 1 else 4)
        passed = bool(np.all(accuracy >= self.threshold))
        used = (self.previous_session_dprime + dprime) / 2 if session > 1 and within == 2 else dprime
        self.history.append(dict(sequence_block=number, session=session, block=within,
            accuracy=accuracy.copy(), dprime=dprime.copy(), used_dprime=used.copy(), passed=passed))
        if passed or number >= self.max_blocks:
            self.finished = True
            return None
        if within == self.blocks_per_session:
            self.previous_session_dprime = dprime.copy()
            if self.repeat_session_rng:
                self.rng = np.random.default_rng(self.seed)
            self.current = self.first_block
        else:
            allocation = allocate_species(used, self.condition, beta=self.beta) if self.adaptive else np.full(4, 16.)
            self.current = sample_block(allocation, self.condition, self.rng, self.ambiguity_width)
        return self.current


class FixedScheduleTask(AdaptiveTask):
    """Apply the same stopping criterion to an externally supplied schedule."""
    def __init__(self, blocks: list[TaskBlock], *, condition: int, seed: int, threshold: float = .9):
        if not blocks:
            raise ValueError('At least one fixed block is required')
        super().__init__(blocks[0], condition=condition, seed=seed, max_blocks=len(blocks), threshold=threshold)
        for block in blocks:
            labels, ambiguous = stimulus_labels(block.stimulus, condition)
            if len(labels) != 64:
                raise ValueError('Fixed schedule requires complete 64-trial blocks')
            np.testing.assert_array_equal(labels, block.categories)
            np.testing.assert_array_equal(ambiguous, block.ambiguous)
        self.blocks = blocks

    def complete_block(self, choices: np.ndarray) -> TaskBlock | None:
        result = super().complete_block(choices)
        if result is not None:
            self.current = self.blocks[len(self.history)]
        return None if self.finished else self.current
