"""Task ordering changes wall time without changing numerical jobs."""
from copy import deepcopy

from src.Bayesian_state.workflows.analysis.model_1031_scheduled import ordered_jobs


def test_observed_cost_and_full_length_determine_order_with_unchanged_jobs():
    jobs = [dict(id="long_fast", subject=1, condition=1, kind="fit", config="fast", particles=2,
                 scope="train", train_trials=1000, seed=1),
            dict(id="short_slow", subject=1, condition=1, kind="fit", config="slow", particles=2,
                 scope="train", train_trials=100, seed=2),
            dict(id="full_slow", subject=1, condition=1, kind="fit", config="slow", particles=2,
                 scope="full", train_trials=100, seed=3)]
    records = [dict(job=dict(subject=1, kind="fit", config=c, particles=2), trials=100, cpu_seconds=cost)
               for c, cost in (("fast", .2), ("slow", 20.))]
    before = deepcopy(jobs)
    ordered, estimates = ordered_jobs(jobs, records, {1: 1000})
    assert [job["id"] for job in ordered] == ["full_slow", "short_slow", "long_fast"]
    assert jobs == before and estimates == {"long_fast": 2., "short_slow": 20., "full_slow": 200.}
