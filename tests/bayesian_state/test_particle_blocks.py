"""Global particle ordering, weights and resampling survive block scheduling."""
from dataclasses import fields, is_dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import numpy as np
import pytest

from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.simulation.config import load_yaml, resolve_engine_config
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.inference.particle_parallel import ProcessAdmission


def assert_exact(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            assert_exact(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_exact(first, second)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    path = Path("configs/meg/simulation_cfg/model0923_task3b.yaml").resolve()
    simulation = load_yaml(path)
    engine = resolve_engine_config(simulation, path.parent)
    engine["oral_readout"] = None
    engine["partition"]["kwargs"]["similarity_cache_dir"] = str(tmp_path_factory.mktemp("block_rules"))
    paths = resolve_dataset_paths(simulation, path.parent)
    loader = SubjectTrialDataLoader(engine, dataset_paths=paths)
    loader.prepare_data(paths["learning_data"])
    arrays = loader._extract_arrays(loader._get_subject_frame(334, 1.0), 24)
    return dict(
        engine_config=engine, subject_id=334, condition=3, stimulus=arrays.stimulus,
        choices=arrays.choices, feedback=arrays.feedback, particle_count=8,
        choice_readout_power=1.0, filter_seed=20261012,
        resample_threshold_fraction=1.0,
        processed_data_dir=paths["processed_dir"], dataset_paths=paths,
    )


@pytest.mark.parametrize("blocks", [2, 4])
def test_all_filter_outputs_and_ancestor_history_exact(case, blocks):
    original = run_state_model_particle_filter(**case)
    parallel = run_state_model_particle_filter(**case, particle_blocks=blocks)
    assert np.any(original.resampled)
    assert_exact(original, parallel)


def test_invalid_block_count_and_audit_fail_before_dispatch(case):
    with pytest.raises(ValueError, match="positive"):
        run_state_model_particle_filter(**case, particle_blocks=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        run_state_model_particle_filter(**case, particle_blocks=16)
    with pytest.raises(ValueError, match="audit disabled"):
        run_state_model_particle_filter(**case, particle_blocks=2, choice_transmission_audit=True)


def test_mixed_block_groups_share_one_budget_and_adapt_to_ready_filters():
    admission = ProcessAdmission(128)
    with admission.plan(24, 4, True) as first:
        assert first == 4
        with admission.plan(56, 2, True) as second:
            assert second == 1
    assert admission.ready_filters == 0
    small = ProcessAdmission(4)
    lock = Lock()
    active = peak = 0

    def run(count):
        nonlocal active, peak
        with small.reserve(count):
            with lock:
                active += count
                peak = max(peak, active)
            time.sleep(0.005)
            with lock:
                active -= count
        return count
    with ThreadPoolExecutor(max_workers=6) as threads:
        assert list(threads.map(run, [1, 4, 2, 1, 4, 1])) == [1, 4, 2, 1, 4, 1]
    assert peak <= 4 and active == 0 and small.available == 4
