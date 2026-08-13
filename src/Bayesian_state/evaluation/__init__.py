"""仿真结果评价与口述报告对齐的轻量公共入口。"""

from importlib import import_module

__all__ = ["ModelEvaluator"]


def __getattr__(name: str):
    if name != "ModelEvaluator":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.evaluator").ModelEvaluator
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
