"""模型结构配置与单次运行上下文。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..utils.paths import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class ModelConfig:
    """经结构校验且与调用方隔离的模型配置。"""

    _values: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelConfig":
        if not isinstance(config, Mapping):
            raise TypeError("engine_config must be a mapping.")
        values = deepcopy(dict(config))
        agenda = values.get("agenda", [])
        modules = values.get("modules", {})
        if not isinstance(agenda, list) or not all(
            isinstance(name, str) and name for name in agenda
        ):
            raise ValueError("engine_config.agenda must be a list of module names.")
        if len(set(agenda)) != len(agenda):
            raise ValueError("engine_config.agenda contains duplicate module names.")
        if not isinstance(modules, Mapping):
            raise ValueError("engine_config.modules must be a mapping.")
        if not all(isinstance(name, str) and name for name in modules):
            raise ValueError("engine_config.modules keys must be module names.")
        if set(agenda) != set(modules):
            missing = sorted(set(modules) - set(agenda))
            unknown = sorted(set(agenda) - set(modules))
            raise ValueError(
                "engine_config.agenda must list every configured module exactly once; "
                f"unlisted={missing}, unknown={unknown}."
            )
        for name, module_config in modules.items():
            if not isinstance(module_config, Mapping):
                raise ValueError(f"module configuration {name!r} must be a mapping.")
            if "class" not in module_config:
                raise ValueError(f"module configuration {name!r} must include class.")
        return cls(values)

    def to_dict(self) -> dict[str, Any]:
        """返回可安全修改的深拷贝。"""

        return deepcopy(dict(self._values))


@dataclass(frozen=True)
class ModelContext:
    """不属于模型结构、但一次运行必须共享的外部上下文。"""

    condition: int = 1
    subject_id: int | None = None
    processed_data_dir: Path | str | None = None
    dataset_paths: Mapping[str, Path | str] | None = None

    def __post_init__(self) -> None:
        condition = int(self.condition)
        if condition <= 0:
            raise ValueError("condition must be a positive integer.")
        subject_id = None if self.subject_id is None else int(self.subject_id)
        processed_data_dir = Path(
            PROCESSED_DATA_DIR
            if self.processed_data_dir is None
            else self.processed_data_dir
        ).resolve()
        dataset_paths = dict(self.dataset_paths or {})
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "processed_data_dir", processed_data_dir)
        object.__setattr__(self, "dataset_paths", dataset_paths)


__all__ = ["ModelConfig", "ModelContext"]
