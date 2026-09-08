"""Compatibility import; implementation lives in src.Bayesian_state.model.modules.base_module."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.Bayesian_state.model.modules.base_module", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("src.Bayesian_state.model.modules.base_module")
