from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_utils_package_import_has_no_runtime_side_effects() -> None:
    script = """
import logging
from pathlib import Path

def forbidden(*args, **kwargs):
    raise AssertionError("filesystem or logging side effect during utils import")

Path.mkdir = forbidden
Path.glob = forbidden
logging.basicConfig = forbidden
logging.FileHandler = forbidden

import src.Bayesian_state.utils as utils
assert getattr(utils.MODEL_STRUCT, "_values") is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_trajectory_statistics_delegates_selection_rules() -> None:
    from src.Bayesian_state.metrics import trajectory_selection, trajectory_statistics

    assert hasattr(trajectory_selection, "distribution_selection_metrics")
    for helper in (
        "_upper_bound_violation",
        "_lower_bound_violation",
        "_interval_violation",
        "_ppc_interval_selection",
    ):
        assert not hasattr(trajectory_statistics, helper)


def test_grid_and_cd_share_search_runtime() -> None:
    from src.Bayesian_state.optimization.hyper_cd_optimizer import HyperCDOptimizer
    from src.Bayesian_state.optimization.hyper_grid_optimizer import HyperGridOptimizer
    from src.Bayesian_state.optimization.hyper_search_common import HyperSearchRuntime

    assert issubclass(HyperCDOptimizer, HyperSearchRuntime)
    assert issubclass(HyperGridOptimizer, HyperSearchRuntime)
    assert callable(HyperSearchRuntime.run_subject)
    for name in ("_prepare_stage_config", "_apply_hyperparams", "_build_runner"):
        assert name not in HyperCDOptimizer.__dict__
        assert name not in HyperGridOptimizer.__dict__


def test_combined_workflow_uses_public_in_process_apis() -> None:
    workflow_path = ROOT_DIR / "src/Bayesian_state/run_hyper_then_simulation.py"
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"))

    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "subprocess" not in imported_modules
    assert "_run_subject_pipeline" not in called_attributes
    assert "run_subject" in called_attributes
    assert "run_simulation" in called_names


def test_hyper_evaluation_is_split_by_execution_cost() -> None:
    from src.Bayesian_state.optimization import hyper_predictive_evaluation
    from src.Bayesian_state.optimization import hyper_search_evaluation

    assert callable(hyper_search_evaluation.evaluate_hyper_cd_convergence)
    assert callable(hyper_search_evaluation.evaluate_near_optimal_plateau)
    assert callable(hyper_search_evaluation.evaluate_multiobjective_selection)
    assert not hasattr(hyper_search_evaluation, "diagnose_hyper_accuracy_sampling")
    assert callable(hyper_predictive_evaluation.diagnose_hyper_accuracy_sampling)
    assert callable(hyper_predictive_evaluation.evaluate_volatility_calibration)
