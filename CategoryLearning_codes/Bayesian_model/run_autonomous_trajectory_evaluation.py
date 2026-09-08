"""Compatibility import; implementation lives in src.Bayesian_state.run_autonomous_trajectory_evaluation."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.Bayesian_state.run_autonomous_trajectory_evaluation", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("src.Bayesian_state.run_autonomous_trajectory_evaluation")
