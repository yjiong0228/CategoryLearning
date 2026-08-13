"""Shared base implementation for Grid and coordinate-descent searches."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src.Bayesian_state.optimization.artifacts import (
    expand_profile_candidate_hyperparams,
    to_builtin,
    validate_no_nested_hyperparam_paths,
    values_from_json,
    values_product,
)
from src.Bayesian_state.simulation.runner import StateModelSimulationRunner
from src.Bayesian_state.simulation.config import (
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.utils.subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration mappings without mutating inputs."""
    out = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


class HyperSearchBase:
    """Common configuration, candidate application, data, and JSONL helpers."""

    backend_label = "hyperparameter search"
    default_output_dir = "../../results/state-based-hyper/default"

    def __init__(self, config: Mapping[str, Any], config_path: Path) -> None:
        self.config = dict(config)
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent

        self.save_level = str(self.config.get("save_level", "compact")).strip().lower()
        if self.save_level not in {"compact", "full"}:
            raise ValueError("save_level must be 'compact' or 'full'")
        if "hyper_base_seed" not in self.config:
            raise ValueError(f"{self.backend_label} config must include hyper_base_seed.")
        self.hyper_base_seed = int(self.config["hyper_base_seed"])

        base_sim = self.config.get("base_sim_config_path")
        if not base_sim:
            raise ValueError("base_sim_config_path is required")
        base_path = Path(base_sim)
        if not base_path.is_absolute():
            base_path = (self.config_dir / base_path).resolve()
        self.base_sim_config_path = base_path
        self.base_sim_config = load_yaml(self.base_sim_config_path)

        self.output_dir = self._resolve_path(
            self.config.get("output_dir", self.default_output_dir)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, maybe_path: Any) -> Path:
        path = Path(maybe_path)
        if not path.is_absolute():
            path = (self.config_dir / path).resolve()
        return path

    def resolve_subjects(
        self,
        cli_subjects: Sequence[int] | None,
        cli_subject_range: Sequence[int] | None,
    ) -> List[int]:
        if cli_subjects:
            return [int(value) for value in cli_subjects]
        if cli_subject_range:
            start, end = [int(value) for value in cli_subject_range]
            return list(range(start, end + 1))
        if self.config.get("subjects") is not None:
            return [int(value) for value in self.config["subjects"]]
        if self.config.get("subject_range") is not None:
            start, end = [int(value) for value in self.config["subject_range"]]
            return list(range(start, end + 1))
        if self.base_sim_config.get("subjects") is not None:
            return [int(value) for value in self.base_sim_config["subjects"]]
        if self.base_sim_config.get("subject_range") is not None:
            start, end = [int(value) for value in self.base_sim_config["subject_range"]]
            return list(range(start, end + 1))
        raise ValueError(
            "Unable to resolve subjects from CLI/"
            f"{self.backend_label} config/base simulation config"
        )

    def run_subject(
        self,
        subject_id: int,
        stage: str = "all",
        *,
        resume_from_coarse: bool = False,
    ) -> Dict[str, Any]:
        """Run one subject through the backend's configured search pipeline."""
        if stage not in {"coarse", "fine", "all"}:
            raise ValueError("stage must be one of: coarse, fine, all")
        if resume_from_coarse and stage != "fine":
            raise ValueError("resume_from_coarse requires stage='fine'")
        return self._run_subject_pipeline(
            int(subject_id),
            stage,
            self.output_dir,
            resume_from_coarse=resume_from_coarse,
        )

    @staticmethod
    def _linspace_values(spec: Mapping[str, Any]) -> List[float]:
        start = float(spec["start"])
        stop = float(spec["stop"])
        num = int(spec["num"])
        if num < 2:
            return [start]
        return [float(value) for value in np.linspace(start, stop, num=num, endpoint=True)]

    def _hyperparam_values(self, spec: Mapping[str, Any]) -> List[Any]:
        if "values" in spec:
            values = list(spec["values"])
            if not values:
                raise ValueError("hyperparameter values cannot be empty")
            return values
        if "values_from_json" in spec:
            return values_from_json(spec, self.config_dir)
        if "values_product" in spec:
            return values_product(spec)
        if all(key in spec for key in ("start", "stop", "num")):
            return self._linspace_values(spec)
        raise ValueError(
            "Each hyperparameter spec must provide values, values_from_json, "
            "values_product, or (start, stop, num)"
        )

    def _param_specs_for_stage(self, stage_name: str) -> Dict[str, Dict[str, Any]]:
        stage_cfg = (self.config.get("stages") or {}).get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stage config: stages.{stage_name}")
        raw = stage_cfg.get("hyperparam_space", self.config.get("hyperparam_space"))
        if not isinstance(raw, Mapping):
            name = f"stages.{stage_name}.hyperparam_space" if "hyperparam_space" in stage_cfg else "hyperparam_space"
            raise ValueError(f"{name} must be a mapping")
        validate_no_nested_hyperparam_paths(raw)
        return {key: dict(value) for key, value in raw.items()}

    @staticmethod
    def _set_by_path(root: Dict[str, Any], path: str, value: Any) -> None:
        current = root
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = deepcopy(value)

    def _apply_single_hyperparam(
        self,
        key: str,
        value: Any,
        next_sim: Dict[str, Any],
        next_engine: Dict[str, Any],
    ) -> None:
        if key.startswith("engine."):
            self._set_by_path(next_engine, key[len("engine."):], value)
        elif key.startswith("simulation."):
            self._set_by_path(next_sim, key[len("simulation."):], value)
        else:
            raise ValueError(
                f"Hyperparameter key '{key}' must start with 'engine.' or 'simulation.'."
            )

    def _apply_hyperparams(
        self,
        point: Mapping[str, Any],
        sim_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        next_sim = deepcopy(sim_cfg)
        next_engine = deepcopy(engine_cfg)
        for key, value in expand_profile_candidate_hyperparams(point).items():
            self._apply_single_hyperparam(key, value, next_sim, next_engine)
        next_sim["fixed_hyperparams"] = deepcopy(dict(point))
        return next_sim, next_engine

    def _prepare_stage_config(self, stage_name: str) -> Dict[str, Any]:
        stage_cfg = (self.config.get("stages") or {}).get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stages.{stage_name}")
        override = stage_cfg.get("simulation_overrides")
        if override is not None and not isinstance(override, Mapping):
            raise ValueError(f"stages.{stage_name}.simulation_overrides must be a mapping")
        if override is None:
            return deepcopy(self.base_sim_config)
        return deep_update(self.base_sim_config, override)

    def _resolve_sim_components(
        self,
        sim_cfg: Dict[str, Any],
        subject_id: int,
        subjects: Sequence[int],
    ) -> tuple[Any, ...]:
        subject_cfg = resolve_subject_config(sim_cfg, subject_id)
        engine_cfg = resolve_engine_config(
            subject_cfg,
            self.base_sim_config_path.parent,
            subject_id=subject_id,
        )
        prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
        loss_metric = resolve_loss_metric(subject_cfg)
        loss_delta = resolve_loss_delta(subject_cfg, loss_metric)
        window_size = resolve_window_size(subject_cfg, subject_id, subjects)
        n_jobs = int(subject_cfg.get("n_jobs", 1))
        return (
            subject_cfg,
            engine_cfg,
            prediction_mode,
            selection_prediction_mode,
            loss_metric,
            loss_delta,
            window_size,
            n_jobs,
        )

    def _build_runner(
        self,
        sim_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
    ) -> tuple[StateModelSimulationRunner, Mapping[str, Path]]:
        dataset_paths = resolve_dataset_paths(sim_cfg, self.base_sim_config_path.parent)
        runner = StateModelSimulationRunner(
            engine_config=engine_cfg,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            n_jobs=int(sim_cfg.get("n_jobs", 1)),
        )
        runner.prepare_data(dataset_paths["learning_data"])
        return runner, dataset_paths

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(to_builtin(payload), ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot resume fine stage; missing coarse combinations file: {path}"
            )
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL record must be a mapping at {path}:{line_no}")
                records.append(dict(payload))
        return records

    @staticmethod
    def _write_jsonl_records(
        path: Path,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(to_builtin(record), ensure_ascii=False, allow_nan=False) + "\n"
                )


__all__ = ["HyperSearchBase", "deep_update"]
