"""Small Shepard--Hovland--Jenkins six-problem validation schedules."""

from __future__ import annotations

from itertools import product

import numpy as np

from .model import Parameters, SUSTAIN


def six_problems() -> tuple[np.ndarray, np.ndarray]:
    """Nominal binary features and I--VI labels, in the author's public order."""
    features = np.array(list(product((0, 1), repeat=3)))
    labels = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1],
    ])
    return np.eye(2)[features].reshape(8, 6), labels


def run_six_problems(
    parameters: Parameters, *, repetitions: int, blocks: int, seed: int,
    recruitment: str = "prediction_error", criterion_blocks: int | None = None,
) -> dict:
    """Simulate fixed shuffled eight-item blocks, retaining every response path.

    An optional criterion stops learning after consecutive error-free blocks;
    remaining error-curve entries are zero as in the author's demonstration.
    A small run checks behavior; it does not reproduce the paper's group fits.
    """
    if any(isinstance(v, bool) or int(v) != v or v < 1 for v in (repetitions, blocks)):
        raise ValueError("repetitions and blocks must be positive integers")
    if criterion_blocks is not None and (
        isinstance(criterion_blocks, bool) or int(criterion_blocks) != criterion_blocks or criterion_blocks < 1
    ):
        raise ValueError("criterion_blocks must be a positive integer or null")
    repetitions, blocks = int(repetitions), int(blocks)
    x, labels = six_problems()
    paths = []
    mean_errors, mean_clusters = [], []
    streams = np.random.SeedSequence(seed).spawn(6 * repetitions)
    for problem, targets in enumerate(labels):
        problem_curves, cluster_counts = [], []
        for replicate in range(repetitions):
            order_stream, response_stream = streams[problem * repetitions + replicate].spawn(2)
            order_rng, response_rng = np.random.default_rng(order_stream), np.random.default_rng(response_stream)
            model = SUSTAIN((2, 2, 2), 2, parameters, recruitment=recruitment)
            error_curve = np.zeros(blocks)
            consecutive = 0
            rows = []
            for block in range(blocks):
                if criterion_blocks is not None and consecutive >= criterion_blocks:
                    break
                all_correct = True
                for item in order_rng.permutation(8):
                    target = int(targets[item])
                    result = model.step(x[item], target)
                    p = result.prediction.probabilities
                    choice = int(response_rng.choice(2, p=p))
                    error_curve[block] += (1 - p[target]) / 8
                    all_correct &= choice == target
                    rows.append(dict(block=block + 1, item=int(item), target=target,
                                     choice=choice, probabilities=p.tolist(),
                                     n_clusters=len(model.clusters), recruited=result.recruited))
                consecutive = consecutive + 1 if all_correct else 0
            problem_curves.append(error_curve)
            cluster_counts.append(len(model.clusters))
            paths.append(dict(problem=problem + 1, replicate=replicate + 1,
                              trials=rows, final_state=model.state_dict()))
        mean_errors.append(np.mean(problem_curves, axis=0).tolist())
        mean_clusters.append(float(np.mean(cluster_counts)))
    return dict(
        status="small_benchmark_not_full_paper_replication", repetitions=repetitions,
        blocks=blocks, trials_per_block=8, seed=seed, recruitment=recruitment,
        criterion_blocks=criterion_blocks,
        post_criterion_error="zero" if criterion_blocks else "not_applicable",
        mean_error_by_block=mean_errors, mean_final_clusters=mean_clusters, paths=paths,
    )
