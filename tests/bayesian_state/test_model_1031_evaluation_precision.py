"""Optional precision checks cannot displace the registered compute deadline."""
from src.Bayesian_state.workflows.analysis.model_1031_evaluation_precision import admission


def records():
    return [dict(job=dict(kind="fit", scope="full", max_trials=None, subject=1,
                         config="same", particles=256), seconds=100., cpu_seconds=90.)]


def jobs():
    return [dict(subject=1, config="same", particles=1024)] * 4


def test_optional_admission_requires_both_wall_and_cpu_headroom():
    assert admission(records(), jobs(), 1200., 4000.)["admitted"]
    assert not admission(records(), jobs(), 1000., 4000.)["admitted"]
    assert not admission(records(), jobs(), 1200., 3000.)["admitted"]


def test_partial_runs_and_other_parameters_cannot_supply_runtime_bound():
    partial = records()
    partial[0]["job"]["max_trials"] = 24
    assert not admission(partial, jobs(), 1e9, 1e9)["admitted"]
    unrelated = records()
    unrelated[0]["job"]["config"] = "different"
    assert not admission(unrelated, jobs(), 1e9, 1e9)["admitted"]
