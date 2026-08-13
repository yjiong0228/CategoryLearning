"""
Base Module
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Mapping


class ModulePhase(str, Enum):
    """一个认知模块在 trial 生命周期中的执行阶段。"""

    PRE_CHOICE = "pre_choice"
    POST_CHOICE = "post_choice"


class ModuleRole(str, Enum):
    """认知模块在模型中的唯一语义职责。"""

    PERCEPTION = "perception"
    HYPOTHESIS_TRANSITION = "hypothesis_transition"
    MEMORY = "memory"
    BETA = "beta"


class BaseModule(ABC):

    phase: ClassVar[ModulePhase]
    role: ClassVar[ModuleRole]

    def __init__(self, engine, **kwargs):
        """
        Initialize
        """
        self.engine = engine

    def prepare_for_process(self, **kwargs) -> None:
        """在 ``process`` 前从共享 engine 同步模块内部状态。"""

        del kwargs

    @abstractmethod
    def process(self, **kwargs):
        """处理当前 trial；具体认知模块必须实现。"""

    def record_outcome(self, observation: Any) -> None:
        """接收已完成 trial 的 outcome；无历史状态的模块保持空实现。"""

        del observation

    def state_dict(self) -> dict[str, Any]:
        """Return mutable cognitive state for particle resampling.

        Stateless modules may keep the default empty payload. Stateful modules
        override this together with :meth:`load_state_dict`.
        """

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a payload produced by :meth:`state_dict`."""

        if state:
            raise ValueError(
                f"{type(self).__name__} does not implement state restoration."
            )

    def clear_logs(self) -> None:
        """Discard trajectory logs after a particle state is copied."""

    def reseed_future(self, module_seed: int) -> None:
        """Assign a future RNG stream when the module owns stochastic state."""
