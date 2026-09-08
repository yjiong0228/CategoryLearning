"""粒子滤波评估接口；具体绘图模块按需加载。"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ParticleFilterEvaluationMixin": (".summary", "ParticleFilterEvaluationMixin"),
    "run_particle_filter_choice_transmission_audit": (
        ".choice_transmission",
        "run_particle_filter_choice_transmission_audit",
    ),
    "run_particle_filter_residual_diagnostics": (
        ".residuals",
        "run_particle_filter_residual_diagnostics",
    ),
    "run_particle_filter_strategy_audit": (
        ".strategy",
        "run_particle_filter_strategy_audit",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))

__all__ = [
    "ParticleFilterEvaluationMixin",
    "run_particle_filter_choice_transmission_audit",
    "run_particle_filter_residual_diagnostics",
    "run_particle_filter_strategy_audit",
]
