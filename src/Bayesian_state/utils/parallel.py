"""CPU scheduling for independent model runs; no cognitive/seed changes."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from joblib import cpu_count, effective_n_jobs, parallel_config
from joblib.parallel import get_active_backend
from threadpoolctl import threadpool_limits


MODEL_0826_PARALLEL_BUDGET = 128


def parallel_job_count(requested: int, task_count: int) -> int:
    """Use the entire requested CPU budget up to the available task count.

    Already-dispatched process tasks execute inner model runs serially: the
    outer pool owns the CPU budget. Negative requests retain joblib semantics.
    """
    if int(task_count) < 1:
        raise ValueError("parallel task_count must be positive")
    if int(requested) == 0:
        raise ValueError("parallel n_jobs must not be zero")
    backend, _ = get_active_backend()
    if (getattr(backend, "nesting_level", 0) or 0) > 0:
        return 1
    return min(effective_n_jobs(int(requested)), cpu_count(), int(task_count))


@contextmanager
def single_threaded_processes() -> Iterator[None]:
    """Limit parent and child numeric libraries, including inherited settings.

    Explicit worker thread limits override e.g. OPENBLAS_NUM_THREADS=64 in the
    caller's environment. Context exit restores the caller's thread settings.
    """
    backend, _ = get_active_backend()
    nested = (getattr(backend, "nesting_level", 0) or 0) > 0
    options = ({"backend": "sequential"} if nested else
               {"backend": "loky", "inner_max_num_threads": 1})
    with threadpool_limits(limits=1), parallel_config(**options):
        yield
