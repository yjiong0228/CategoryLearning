"""Bayesian 状态模型的共享状态容器与模块调度器。"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

from .config import ModelContext
from .modules.base_module import BaseModule, ModulePhase, ModuleRole


EPS = 1e-15


class IndexedSet:
    """保持插入顺序、并支持元素到整数位置反查的不可变集合。"""

    def __init__(self, elements: Iterable[Hashable]):
        self.elements = tuple(elements)
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("IndexedSet elements must be unique.")
        self._size = len(self.elements)
        self._index = {element: index for index, element in enumerate(self.elements)}

    @property
    def inv(self) -> Mapping[Hashable, int]:
        """只读的元素到位置映射。"""

        return MappingProxyType(self._index)

    @property
    def length(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key: Hashable) -> int:
        if key not in self._index:
            raise KeyError(f"Invalid key: {key!r}.")
        return self._index[key]

    def __repr__(self) -> str:
        return f"IndexedSet(elements={self.elements!r})"

    def __iter__(self):
        return iter(self.elements)


class BayesianStateEngine:
    """保存一个模型轨迹的状态，并按模块声明的阶段执行 agenda。"""

    upper_numerical_bound = 1e15
    lower_numerical_bound = 1e-15

    def __init__(
        self,
        *,
        agenda: Sequence[str],
        hypotheses_set: IndexedSet,
        partition: Any,
        observation_likelihood: Any,
        context: ModelContext,
        prior: np.ndarray | None = None,
        posterior: np.ndarray | None = None,
        likelihood: np.ndarray | None = None,
        observation: Any = None,
    ) -> None:
        if not isinstance(hypotheses_set, IndexedSet):
            raise TypeError("hypotheses_set must be an IndexedSet.")
        if hypotheses_set.length <= 0:
            raise ValueError("hypotheses_set must be non-empty.")
        if partition is None:
            raise ValueError("BayesianStateEngine requires a partition.")
        if observation_likelihood is None:
            raise ValueError(
                "BayesianStateEngine requires an observation_likelihood evaluator."
            )
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext.")

        self.hypotheses_set = hypotheses_set
        self.set_size = hypotheses_set.length
        self.partition = partition
        self.observation_likelihood = observation_likelihood
        self.context = context
        self.agenda = list(agenda)

        self.hypotheses_mask: np.ndarray | None = None
        self.prior = self._initial_probability(prior, "prior")
        self.posterior = (
            None
            if posterior is None
            else self._validated_probability(posterior, "posterior")
        )
        self.likelihood = (
            np.full(self.set_size, 1.0 / self.set_size, dtype=float)
            if likelihood is None
            else self._validated_vector(likelihood, "likelihood")
        )
        self.state = None
        self.observation = observation
        self.beta: np.ndarray | None = None
        self.modules: dict[str, BaseModule] = {}
        self._modules_by_role: dict[ModuleRole, BaseModule] = {}

        self.log_prior: np.ndarray | None = None
        self.log_likelihood: np.ndarray | None = None
        self.log_posterior: np.ndarray | None = None
        self.last_prior: np.ndarray | None = None

    def _validated_vector(self, values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape != (self.set_size,):
            raise ValueError(
                f"{name} must have shape ({self.set_size},), got {array.shape}."
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative.")
        return array.copy()

    def _validated_probability(self, values: np.ndarray, name: str) -> np.ndarray:
        array = self._validated_vector(values, name)
        total = float(np.sum(array))
        if total <= 0.0:
            raise ValueError(f"{name} must have positive total mass.")
        return array / total

    def _initial_probability(
        self,
        values: np.ndarray | None,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return np.full(self.set_size, 1.0 / self.set_size, dtype=float)
        return self._validated_probability(values, name)

    @staticmethod
    def translate_from_log(log: np.ndarray) -> np.ndarray:
        values = np.asarray(log, dtype=float)
        values -= np.max(values)
        exp = np.exp(values)
        return exp / np.sum(exp)

    @staticmethod
    def translate_to_log(exp: np.ndarray) -> np.ndarray:
        clipped = np.clip(
            exp,
            BayesianStateEngine.lower_numerical_bound,
            BayesianStateEngine.upper_numerical_bound,
        )
        return np.log(clipped)

    def register_module(self, name: str, module: BaseModule) -> None:
        """注册一个已构造模块，并验证其执行阶段契约。"""

        if not name or name.startswith("__"):
            raise ValueError(f"invalid module name: {name!r}.")
        if name in self.modules or hasattr(self, name):
            raise ValueError(f"duplicate engine module name: {name!r}.")
        if not isinstance(module, BaseModule):
            raise TypeError(f"module {name!r} must inherit BaseModule.")
        if module.engine is not self:
            raise ValueError(f"module {name!r} is bound to another engine.")
        if not isinstance(getattr(module, "phase", None), ModulePhase):
            raise TypeError(
                f"module {name!r} must declare phase as a ModulePhase value."
            )
        if not isinstance(getattr(module, "role", None), ModuleRole):
            raise TypeError(
                f"module {name!r} must declare role as a ModuleRole value."
            )
        if module.role in self._modules_by_role:
            raise ValueError(f"duplicate engine module role: {module.role.value!r}.")
        setattr(self, name, module)
        self.modules[name] = module
        self._modules_by_role[module.role] = module

    def get_module(
        self,
        role: ModuleRole,
        *,
        required: bool = False,
    ) -> BaseModule | None:
        """按语义职责取模块，避免运行时代码依赖配置中的实例名。"""

        if not isinstance(role, ModuleRole):
            raise TypeError("role must be a ModuleRole value.")
        module = self._modules_by_role.get(role)
        if module is None and required:
            raise ValueError(f"model requires module role {role.value!r}.")
        return module

    def validate_agenda(self) -> None:
        unknown = [name for name in self.agenda if name not in self.modules]
        if unknown:
            raise ValueError(f"model agenda contains unknown modules: {unknown}.")
        duplicates = [
            name for index, name in enumerate(self.agenda) if name in self.agenda[:index]
        ]
        if duplicates:
            raise ValueError(f"model agenda contains duplicate modules: {duplicates}.")
        unlisted = [name for name in self.modules if name not in self.agenda]
        if unlisted:
            raise ValueError(f"model agenda omits registered modules: {unlisted}.")

    def begin_trial(self, stimulus: np.ndarray) -> None:
        """提交 trial 的物理刺激，并统一执行 ``prior <- posterior``。"""

        if self.prior is None:
            raise ValueError("model prior is not initialized.")
        self.prior = (
            np.asarray(self.posterior, dtype=float).copy()
            if self.posterior is not None
            else np.asarray(self.prior, dtype=float).copy()
        )
        self.observation = (np.asarray(stimulus, dtype=float).copy(), None, None)

    def run_phase(
        self,
        phase: ModulePhase,
        *,
        module_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        shared_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """按 agenda 执行一个模块阶段，不依赖任何模块名称约定。"""

        if not isinstance(phase, ModulePhase):
            raise TypeError("phase must be a ModulePhase value.")
        self.validate_agenda()
        kwargs_by_module = module_kwargs or {}
        common = dict(shared_kwargs or {})
        for module_name in self.agenda:
            module = self.modules[module_name]
            if module.phase is not phase:
                continue
            current_kwargs = dict(kwargs_by_module.get(module_name, {}))
            current_kwargs.update(common)
            module.prepare_for_process(**current_kwargs)
            module.process(**current_kwargs)

    def record_outcome(self, observation: Any) -> None:
        """把已完成 outcome 广播给所有模块的统一生命周期钩子。"""

        for module_name in self.agenda:
            module = self.modules[module_name]
            module.record_outcome(observation)

    @property
    def subject_id(self) -> int | None:
        return self.context.subject_id

    @property
    def processed_data_dir(self):
        return self.context.processed_data_dir

    @property
    def dataset_paths(self):
        return self.context.dataset_paths

    @staticmethod
    def _copy_state_value(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.copy()
        return deepcopy(value)

    def state_dict(self) -> dict[str, Any]:
        """返回粒子重采样所需的 engine 与 module 状态。"""

        core = {
            name: self._copy_state_value(getattr(self, name, None))
            for name in (
                "prior",
                "posterior",
                "likelihood",
                "hypotheses_mask",
                "observation",
                "last_prior",
            )
        }
        modules = {
            name: deepcopy(module.state_dict())
            for name, module in self.modules.items()
        }
        return {"core": core, "modules": modules}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """恢复 :meth:`state_dict` 生成的粒子快照。"""

        for name, value in state.get("core", {}).items():
            setattr(self, name, self._copy_state_value(value))
        for name, payload in state.get("modules", {}).items():
            module = self.modules.get(name)
            if module is None:
                raise ValueError(f"snapshot refers to unavailable module {name!r}.")
            module.load_state_dict(deepcopy(payload))

    def clear_module_logs(self) -> None:
        for module in self.modules.values():
            module.clear_logs()

    @property
    def distance_mode(self) -> str:
        return str(self.observation_likelihood.distance_mode)

    def compute_likelihood(self) -> np.ndarray:
        """针对当前完整 observation 计算并保存 likelihood。"""

        hypothesis_args = tuple(
            self.hypotheses_set[hypothesis] for hypothesis in self.hypotheses_set
        )
        self.likelihood = self.observation_likelihood.process(
            self.observation,
            hypothesis_args,
            beta=self.beta,
        )
        return self.likelihood

    def process(self, **kwargs) -> np.ndarray:
        """执行标准的一步 Bayesian posterior 更新。"""

        del kwargs
        self.log_prior = self.translate_to_log(self.prior)
        self.log_likelihood = self.translate_to_log(self.likelihood)
        self.log_posterior = self.log_prior + self.log_likelihood
        self.posterior = self.translate_from_log(self.log_posterior)
        return self.posterior


__all__ = ["BayesianStateEngine", "EPS", "IndexedSet"]
