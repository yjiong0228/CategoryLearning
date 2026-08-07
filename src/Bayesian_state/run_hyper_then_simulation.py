"""Run hyperparameter selection, then materialize and run fixed simulations."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.Bayesian_state.optimization.hyper_cd_optimizer import HyperCDOptimizer
from src.Bayesian_state.optimization.hyper_grid_optimizer import HyperGridOptimizer
from src.Bayesian_state.run_simulation import (
    apply_fixed_hyperparams_to_subject_config,
    infer_fixed_hyperparams_from_engine_config,
    resolve_hyper_base_seed,
    resolve_hyper_candidate_seed as resolve_direct_hyper_candidate_seed,
    resolve_simulation_repeats,
)
from src.Bayesian_state.utils.config_subjects import (
    SUBJECT_OVERRIDE_KEYS,
    resolve_subject_config,
    subject_override_for,
)
from src.Bayesian_state.optimization.optimization_config import (
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.optimization.hyper_utils import (
    build_root_best_payload,
    expand_profile_candidate_hyperparams,
    root_base_sim_config_path,
    root_hyper_base_seed,
    subject_best_hyperparams,
    subject_best_stage,
    subject_hyper_candidate_seed,
    to_builtin as hyper_to_builtin,
)
from src.Bayesian_state.utils.paths import ROOT_DIR


DEFAULT_HYPER_GRID_CONFIG = Path("configs/hyper_grid_cfg/pmh_cond1_hyper_grid_v1.yaml")
DEFAULT_HYPER_CD_CONFIG = Path("configs/hyper_cd_cfg/pmh_cond1_hyper_cd_v1.yaml")
DEFAULT_GENERATED_SIM_CONFIGS = {
    "hyper_grid": Path("configs/simulation_cfg/generated_from_hyper/pmh_cond1_subjectwise_hyper_grid_best.yaml"),
    "hyper_cd": Path("configs/simulation_cfg/generated_from_hyper/pmh_cond1_subjectwise_hyper_cd_best.yaml"),
}
DEFAULT_SIM_OUTPUT_DIRS = {
    "hyper_grid": Path("results/state-based-simulation/pmh/cond1_subjectwise_hyper_grid_best"),
    "hyper_cd": Path("results/state-based-simulation/pmh/cond1_subjectwise_hyper_cd_best"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(payload), f, sort_keys=False, allow_unicode=True)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def resolve_relative_to(base_dir: Path, maybe_path: Any) -> Path:
    path = Path(maybe_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def relative_path_for_yaml(target: Path, yaml_dir: Path) -> str:
    return Path(os.path.relpath(target.resolve(), yaml_dir.resolve())).as_posix()


def rebase_config_relative_path(value: Any, source_yaml_dir: Path, target_yaml_dir: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return path.as_posix()
    absolute = (source_yaml_dir / path).resolve()
    return relative_path_for_yaml(absolute, target_yaml_dir)


def command_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _strip_subject_override_blocks(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in config.items() if k not in SUBJECT_OVERRIDE_KEYS}


def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
    curr = root
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = curr.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested path through non-mapping segment: {path}")
        curr = next_value
    curr[parts[-1]] = deepcopy(value)


def _clear_mapping_replacement_path(root: dict[str, Any], path: str) -> None:
    curr = root
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = curr.get(part)
        if next_value is None:
            next_value = {}
            curr[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot clear nested replacement path through non-mapping segment: {path}")
        curr = next_value
    curr[parts[-1]] = {}


def _engine_mapping_replacement_paths(per_subject_best: Mapping[str, Any]) -> list[str]:
    paths: set[str] = set()
    for subject_payload in per_subject_best.values():
        if not isinstance(subject_payload, Mapping):
            continue
        best_hyperparams = subject_best_hyperparams(subject_payload)
        if not isinstance(best_hyperparams, Mapping):
            continue
        for key, value in expand_profile_candidate_hyperparams(best_hyperparams).items():
            if key.startswith("engine.") and isinstance(value, Mapping):
                paths.add(key[len("engine."):])
    return sorted(paths)


def _clear_engine_mapping_replacements(engine_config: dict[str, Any], paths: Sequence[str]) -> None:
    for path in paths:
        _clear_mapping_replacement_path(engine_config, path)


def _load_raw_base_engine_config(base_sim_cfg: Mapping[str, Any], base_sim_dir: Path) -> dict[str, Any]:
    inline_cfg = base_sim_cfg.get("engine_config")
    path_cfg = base_sim_cfg.get("engine_config_path")
    if inline_cfg is not None and not isinstance(inline_cfg, dict):
        raise ValueError("engine_config must be a mapping when provided")

    base_engine: dict[str, Any] = {}
    if path_cfg:
        engine_path = resolve_relative_to(base_sim_dir, path_cfg)
        base_engine = load_yaml(engine_path)
    if inline_cfg is None and not path_cfg:
        raise ValueError("Base simulation config must provide engine_config or engine_config_path")
    if inline_cfg is not None:
        base_engine = _deep_update(base_engine, inline_cfg)
    return base_engine


def _rebase_generated_sim_paths(
    generated_cfg: dict[str, Any],
    base_sim_dir: Path,
    generated_dir: Path,
) -> None:
    if generated_cfg.get("engine_config_path") is not None:
        generated_cfg["engine_config_path"] = rebase_config_relative_path(
            generated_cfg["engine_config_path"],
            base_sim_dir,
            generated_dir,
        )
    dataset = generated_cfg.get("dataset")
    if isinstance(dataset, dict) and dataset.get("processed_dir") is not None:
        dataset["processed_dir"] = rebase_config_relative_path(
            dataset["processed_dir"],
            base_sim_dir,
            generated_dir,
        )
    if generated_cfg.get("data_path") is not None:
        generated_cfg["data_path"] = rebase_config_relative_path(
            generated_cfg["data_path"],
            base_sim_dir,
            generated_dir,
        )


def _split_hyperparams_for_simulation_override(best_hyperparams: Mapping[str, Any]) -> dict[str, Any]:
    override: dict[str, Any] = {}
    engine_override: dict[str, Any] = {}
    expanded_hyperparams = expand_profile_candidate_hyperparams(best_hyperparams)
    for key, value in expanded_hyperparams.items():
        if key.startswith("engine."):
            _set_by_path(engine_override, key[len("engine."):], value)
        elif key.startswith("simulation."):
            _set_by_path(override, key[len("simulation."):], value)
        else:
            raise ValueError(f"Hyperparameter key must start with 'engine.' or 'simulation.': {key}")
    if engine_override:
        override["engine_config"] = engine_override
    override["fixed_hyperparams"] = deepcopy(expanded_hyperparams)
    return override


def _sorted_unique_subjects(subjects: Sequence[int]) -> list[int]:
    return sorted({int(sid) for sid in subjects})


def _subject_payload_for_sid(per_subject_best: Mapping[str, Any], sid: int) -> Mapping[str, Any] | None:
    for key in (str(int(sid)), int(sid)):
        payload = per_subject_best.get(key)  # type: ignore[arg-type]
        if isinstance(payload, Mapping):
            return payload
    return None


def _filter_hyper_best_payload(
    hyper_best_payload: Mapping[str, Any],
    subjects: Sequence[int] | None,
) -> dict[str, Any]:
    payload = deepcopy(dict(hyper_best_payload))
    per_subject_best = payload.get("per_subject_best")
    if not isinstance(per_subject_best, Mapping) or not per_subject_best:
        raise ValueError("Hyper best payload is missing non-empty per_subject_best.")
    per_subject_outputs = payload.get("per_subject_outputs")
    if not isinstance(per_subject_outputs, Mapping):
        per_subject_outputs = {}
    if subjects is None:
        payload["per_subject_best"] = dict(
            sorted(((str(k), v) for k, v in per_subject_best.items()), key=lambda item: int(item[0]))
        )
        if per_subject_outputs:
            payload["per_subject_outputs"] = dict(
                sorted(((str(k), v) for k, v in per_subject_outputs.items()), key=lambda item: int(item[0]))
            )
        payload["subjects"] = sorted(int(sid) for sid in payload["per_subject_best"].keys())
        return payload

    filtered: dict[str, Any] = {}
    filtered_outputs: dict[str, Any] = {}
    missing: list[int] = []
    for sid in _sorted_unique_subjects(subjects):
        subject_payload = _subject_payload_for_sid(per_subject_best, sid)
        if subject_payload is None:
            missing.append(sid)
            continue
        filtered[str(sid)] = deepcopy(dict(subject_payload))
        subject_outputs = _subject_payload_for_sid(per_subject_outputs, sid)
        if subject_outputs is not None:
            filtered_outputs[str(sid)] = deepcopy(dict(subject_outputs))
    if missing:
        raise FileNotFoundError(f"Hyper best payload is missing subjects: {missing}")
    payload["per_subject_best"] = filtered
    if per_subject_outputs:
        payload["per_subject_outputs"] = filtered_outputs
    payload["subjects"] = _sorted_unique_subjects(subjects)
    return payload


def _base_subject_override_for_generated_config(
    base_sim_cfg: Mapping[str, Any],
    raw_engine_cfg: Mapping[str, Any],
    subject_id: int,
    base_sim_dir: Path,
    generated_dir: Path,
    engine_mapping_replacement_paths: Sequence[str],
) -> dict[str, Any]:
    override = deepcopy(subject_override_for(base_sim_cfg, subject_id))
    _rebase_generated_sim_paths(override, base_sim_dir, generated_dir)

    engine_subject_override = subject_override_for(raw_engine_cfg, subject_id)
    if engine_subject_override:
        existing_engine = override.get("engine_config")
        if existing_engine is not None and not isinstance(existing_engine, Mapping):
            raise ValueError(f"subject {subject_id} engine_config override must be a mapping")
        override["engine_config"] = _deep_update(
            dict(existing_engine or {}),
            engine_subject_override,
        )
    existing_engine = override.get("engine_config")
    if isinstance(existing_engine, dict):
        _clear_engine_mapping_replacements(existing_engine, engine_mapping_replacement_paths)
    return override


def build_subjectwise_simulation_config(
    hyper_best_payload: Mapping[str, Any],
    generated_sim_config_path: Path,
    sim_output_dir: Path,
    keep_logs: bool,
) -> dict[str, Any]:
    per_subject_best = hyper_best_payload.get("per_subject_best")
    if not isinstance(per_subject_best, Mapping) or not per_subject_best:
        raise ValueError("Hyper best payload is missing non-empty per_subject_best.")

    base_sim_path_raw = root_base_sim_config_path(hyper_best_payload)
    if not base_sim_path_raw:
        raise ValueError("Hyper best payload is missing base_sim_config_path.")
    base_sim_path = resolve_project_path(Path(str(base_sim_path_raw)))
    base_sim_cfg = load_yaml(base_sim_path)
    raw_engine_cfg = _load_raw_base_engine_config(base_sim_cfg, base_sim_path.parent)
    base_engine_cfg = _strip_subject_override_blocks(raw_engine_cfg)
    engine_mapping_replacement_paths = _engine_mapping_replacement_paths(per_subject_best)
    _clear_engine_mapping_replacements(base_engine_cfg, engine_mapping_replacement_paths)

    generated_dir = generated_sim_config_path.parent
    sim_output_dir = sim_output_dir.resolve()

    generated_cfg = _strip_subject_override_blocks(base_sim_cfg)
    generated_cfg.pop("engine_config_path", None)
    _rebase_generated_sim_paths(generated_cfg, base_sim_path.parent, generated_dir)
    generated_cfg["engine_config"] = base_engine_cfg
    generated_cfg["subjects"] = sorted(int(sid) for sid in per_subject_best.keys())
    generated_cfg.pop("subject_range", None)
    generated_cfg["output_dir"] = relative_path_for_yaml(sim_output_dir, generated_dir)
    generated_cfg["keep_logs"] = bool(keep_logs)
    hyper_meta = hyper_best_payload.get("hyper")
    hyper_config_name = ""
    if isinstance(hyper_meta, Mapping):
        hyper_config_name = str(hyper_meta.get("config_path", ""))
    objective_paths = []
    for subject_payload in per_subject_best.values():
        if not isinstance(subject_payload, Mapping):
            continue
        objectives = subject_payload.get("objectives")
        if isinstance(objectives, Mapping):
            order = objectives.get("order")
            if isinstance(order, list):
                for item in order:
                    if isinstance(item, Mapping):
                        objective_paths.append(str(item.get("path", "")))
    if "v8" in hyper_config_name or any(path.startswith("statistics.scores.distribution") for path in objective_paths):
        generated_cfg["representative_run_selection"] = "behavior_composite"
        generated_cfg["representative_choice_fraction"] = 0.10
    hyper_base_seed = root_hyper_base_seed(hyper_best_payload)
    if hyper_base_seed is not None:
        generated_cfg["hyper_base_seed"] = int(hyper_base_seed)

    subject_overrides: dict[int, Any] = {}
    for sid_text, subject_payload in sorted(per_subject_best.items(), key=lambda item: int(item[0])):
        if not isinstance(subject_payload, Mapping):
            raise ValueError(f"per_subject_best[{sid_text}] must be a mapping")
        best_hyperparams = subject_best_hyperparams(subject_payload)
        if not isinstance(best_hyperparams, Mapping):
            raise ValueError(f"per_subject_best[{sid_text}] is missing best_hyperparams")
        sid = int(sid_text)
        base_override = _base_subject_override_for_generated_config(
            base_sim_cfg=base_sim_cfg,
            raw_engine_cfg=raw_engine_cfg,
            subject_id=sid,
            base_sim_dir=base_sim_path.parent,
            generated_dir=generated_dir,
            engine_mapping_replacement_paths=engine_mapping_replacement_paths,
        )
        override = _deep_update(base_override, _split_hyperparams_for_simulation_override(best_hyperparams))
        hyper_candidate_seed = subject_hyper_candidate_seed(subject_payload)
        if hyper_candidate_seed is not None:
            override["hyper_candidate_seed"] = int(hyper_candidate_seed)
        subject_overrides[sid] = override

    generated_cfg["subject_overrides"] = subject_overrides
    return generated_cfg


def build_hyper_selector(
    backend: str,
    config_path: Path,
    output_dir: Path | None = None,
) -> HyperGridOptimizer | HyperCDOptimizer:
    cfg = load_yaml(config_path)
    if output_dir is not None:
        cfg = dict(cfg)
        cfg["output_dir"] = str(resolve_project_path(output_dir))
    if backend == "hyper_grid":
        return HyperGridOptimizer(cfg, config_path)
    if backend == "hyper_cd":
        return HyperCDOptimizer(cfg, config_path)
    raise ValueError(f"Unsupported hyper backend: {backend}")


def _subject_best_path(output_dir: Path, subject_id: int) -> Path:
    return output_dir / f"subject_{int(subject_id)}" / "best_hyperparams.json"


def aggregate_per_subject_best(
    output_dir: Path,
    optimizer: HyperGridOptimizer | HyperCDOptimizer,
    config_path: Path,
    backend: str,
    subjects: Sequence[int],
    require_all: bool = False,
) -> dict[str, Any]:
    per_subject_best: dict[str, Any] = {}
    per_subject_outputs: dict[str, Any] = {}

    missing: list[int] = []
    for sid in _sorted_unique_subjects(subjects):
        best_path = _subject_best_path(output_dir, sid)
        if not best_path.is_file():
            missing.append(sid)
            continue
        subject_dir = best_path.parent
        subject_best = json.loads(best_path.read_text(encoding="utf-8"))
        per_subject_best[str(sid)] = subject_best
        outputs = {
            "output_dir": str(subject_dir),
            "all_combinations": str(subject_dir / "all_combinations.jsonl"),
            "stage_summary": str(subject_dir / "stage_summary.json"),
            "best_hyperparams": str(best_path),
        }
        restart_summary = subject_dir / "restart_summary.json"
        coordinate_trace = subject_dir / "coordinate_trace.jsonl"
        if restart_summary.is_file():
            outputs["restart_summary"] = str(restart_summary)
        if coordinate_trace.is_file():
            outputs["coordinate_trace"] = str(coordinate_trace)
        per_subject_outputs[str(sid)] = outputs

    root_payload_kwargs = {
        "backend": backend,
        "config_path": config_path,
        "output_dir": output_dir,
        "base_sim_config_path": optimizer.base_sim_config_path,
        "hyper_base_seed": optimizer.hyper_base_seed,
        "save_level": optimizer.save_level,
        "subjects": subjects,
        "per_subject_best": per_subject_best,
        "per_subject_outputs": per_subject_outputs,
    }
    if hasattr(optimizer, "objective_order_config"):
        root_payload_kwargs["objective_order"] = optimizer.objective_order_config
    else:
        selection_metric = getattr(optimizer, "selection_metric", None)
        if selection_metric is None:
            selection_metric = getattr(optimizer, "tie_break_metric")
        root_payload_kwargs["selection_metric"] = selection_metric
    payload = build_root_best_payload(**root_payload_kwargs)
    if require_all and missing:
        raise FileNotFoundError(f"Missing completed hyper results for subjects: {missing}")
    best_path = output_dir / "best_hyperparams.json"
    save_json(best_path, payload)
    return {
        "output_dir": str(output_dir),
        "per_subject_outputs": dict(sorted(per_subject_outputs.items(), key=lambda item: int(item[0]))),
        "best_hyperparams": str(best_path),
        "best": payload,
    }


def run_command(cmd: Sequence[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(list(cmd), cwd=ROOT_DIR, check=True)


def materialize_simulation_config_from_hyper_best(
    hyper_best_path: Path,
    generated_sim_config_path: Path,
    sim_output_dir: Path,
    keep_logs: bool,
    subjects: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not hyper_best_path.is_file():
        raise FileNotFoundError(f"Hyper best file not found: {hyper_best_path}")
    hyper_best_payload = json.loads(hyper_best_path.read_text(encoding="utf-8"))
    hyper_best_payload = _filter_hyper_best_payload(hyper_best_payload, subjects)
    generated_cfg = build_subjectwise_simulation_config(
        hyper_best_payload=hyper_best_payload,
        generated_sim_config_path=generated_sim_config_path,
        sim_output_dir=sim_output_dir,
        keep_logs=bool(keep_logs),
    )
    save_yaml(generated_sim_config_path, generated_cfg)
    source_payload = dict(hyper_best_payload)
    source_payload.update(
        {
            "hyper_best_path": str(hyper_best_path),
            "generated_sim_config_path": str(generated_sim_config_path),
            "sim_output_dir": str(sim_output_dir),
        }
    )
    save_json(sim_output_dir / "hyper_best_source.json", source_payload)
    print(f"Generated subjectwise simulation config -> {generated_sim_config_path}")
    print(f"Simulation output directory -> {sim_output_dir}")
    return generated_cfg


def run_simulation_for_subject(generated_sim_config_path: Path, subject_id: int) -> None:
    sim_cmd = [
        sys.executable,
        "-m",
        "src.Bayesian_state.run_simulation",
        "--config",
        command_path(generated_sim_config_path),
        "--subjects",
        str(int(subject_id)),
    ]
    run_command(sim_cmd)


def run_simulation_for_all(generated_sim_config_path: Path) -> None:
    sim_cmd = [
        sys.executable,
        "-m",
        "src.Bayesian_state.run_simulation",
        "--config",
        command_path(generated_sim_config_path),
    ]
    run_command(sim_cmd)


def _hyper_stage_satisfies_request(best_stage: Any, requested_stage: str) -> bool:
    stage = str(best_stage or "").strip().lower()
    if requested_stage == "coarse":
        return stage in {"coarse", "fine"}
    if requested_stage in {"fine", "all"}:
        return stage == "fine"
    raise ValueError(f"Unsupported stage: {requested_stage}")


def _subject_hyper_result_satisfies_stage(best_path: Path, requested_stage: str) -> tuple[bool, str]:
    if not best_path.is_file():
        return False, f"missing {best_path}"
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON in {best_path}: {exc}"
    best_stage = subject_best_stage(payload)
    if not _hyper_stage_satisfies_request(best_stage, requested_stage):
        return False, f"best_stage={best_stage!r} does not satisfy requested stage={requested_stage!r}"
    if not isinstance(subject_best_hyperparams(payload), Mapping):
        return False, "missing best_hyperparams"
    if subject_hyper_candidate_seed(payload) is None:
        return False, "missing hyper_candidate_seed"
    return True, "ok"


def _require_subject_hyper_result(best_path: Path, requested_stage: str) -> None:
    ok, reason = _subject_hyper_result_satisfies_stage(best_path, requested_stage)
    if not ok:
        raise FileNotFoundError(f"Existing hyper result is not usable for --skip-hyper: {reason}")


def _expected_simulation_signature(
    generated_cfg: Mapping[str, Any],
    generated_sim_config_path: Path,
    subject_id: int,
    subjects: Sequence[int],
) -> dict[str, Any]:
    sid = int(subject_id)
    subject_cfg = resolve_subject_config(generated_cfg, sid)
    explicit_fixed_hyperparams = dict(subject_cfg.get("fixed_hyperparams") or {})
    subject_cfg = apply_fixed_hyperparams_to_subject_config(subject_cfg, explicit_fixed_hyperparams)
    engine_config = resolve_engine_config(subject_cfg, generated_sim_config_path.parent, subject_id=sid)
    fixed_hyperparams = {
        **infer_fixed_hyperparams_from_engine_config(engine_config),
        **explicit_fixed_hyperparams,
    }
    seed_hyperparams = explicit_fixed_hyperparams or fixed_hyperparams
    hyper_base_seed = resolve_hyper_base_seed(subject_cfg)
    hyper_candidate_seed = resolve_direct_hyper_candidate_seed(
        subject_cfg,
        hyper_base_seed,
        sid,
        seed_hyperparams,
    )
    prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
    loss_metric = resolve_loss_metric(subject_cfg)
    loss_delta = resolve_loss_delta(subject_cfg, loss_metric)
    return {
        "subject_id": sid,
        "fixed_hyperparams": fixed_hyperparams,
        "seed_hyperparams": seed_hyperparams,
        "hyper_base_seed": hyper_base_seed,
        "hyper_candidate_seed": hyper_candidate_seed,
        "simulation_repeats": resolve_simulation_repeats(subject_cfg),
        "window_size": resolve_window_size(subject_cfg, sid, subjects),
        "prediction_mode": prediction_mode,
        "selection_prediction_mode": selection_prediction_mode,
        "loss_metric": loss_metric,
        "loss_delta": loss_delta,
    }


def _values_equal(left: Any, right: Any) -> bool:
    return _to_builtin(left) == _to_builtin(right)


def _simulation_result_satisfies_signature(
    subject_json_path: Path,
    expected: Mapping[str, Any],
) -> tuple[bool, str]:
    if not subject_json_path.is_file():
        return False, f"missing {subject_json_path}"
    try:
        payload = json.loads(subject_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON in {subject_json_path}: {exc}"

    summary = payload.get("simulation")
    if not isinstance(summary, Mapping):
        summary = payload.get("simulation_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        selection = {}
    selection_meta = selection.get("selection_meta")
    if not isinstance(selection_meta, Mapping):
        selection_meta = payload.get("selection_meta") or {}

    checks = {
        "subject_id": payload.get("subject_id"),
        "fixed_hyperparams": payload.get("fixed_hyperparams"),
        "seed_hyperparams": selection_meta.get("seed_hyperparams"),
        "hyper_base_seed": selection.get("hyper_base_seed", payload.get("hyper_base_seed")),
        "hyper_candidate_seed": selection.get("hyper_candidate_seed", payload.get("hyper_candidate_seed")),
        "simulation_repeats": summary.get("simulation_repeats", payload.get("simulation_repeats")),
        "window_size": summary.get("window_size", payload.get("window_size")),
        "prediction_mode": selection.get("prediction_mode", payload.get("prediction_mode")),
        "selection_prediction_mode": selection.get(
            "selection_prediction_mode",
            payload.get("selection_prediction_mode"),
        ),
        "loss_metric": selection.get("loss_metric", payload.get("loss_metric")),
        "loss_delta": selection.get("loss_delta", payload.get("loss_delta")),
    }
    for key, observed in checks.items():
        if not _values_equal(observed, expected.get(key)):
            return False, f"{key} differs"
    return True, "ok"


def run_hyper_resumable(
    backend: str,
    config_path: Path,
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
    stage: str,
    skip_completed: bool,
    hyper_output_dir: Path | None = None,
    resume_from_coarse: bool = False,
) -> dict[str, Any]:
    if resume_from_coarse and stage != "fine":
        raise ValueError("--resume-from-coarse requires --stage fine")
    optimizer = build_hyper_selector(backend, config_path, output_dir=hyper_output_dir)
    resolved_subjects = optimizer.resolve_subjects(subjects, subject_range)

    optimizer.output_dir.mkdir(parents=True, exist_ok=True)
    for sid in resolved_subjects:
        best_path = _subject_best_path(optimizer.output_dir, int(sid))
        if skip_completed:
            complete, reason = _subject_hyper_result_satisfies_stage(best_path, stage)
            if complete:
                print(f"Skipping subject {int(sid)} {backend}; found completed {stage} result: {best_path}")
                continue
            if best_path.is_file():
                print(f"Existing {backend} result for subject {int(sid)} is not complete for {stage}: {reason}")
        print(f"Running {backend} optimization for subject {int(sid)}")
        optimizer._run_subject_pipeline(
            int(sid),
            stage,
            optimizer.output_dir,
            resume_from_coarse=resume_from_coarse,
        )
        aggregate_per_subject_best(
            optimizer.output_dir,
            optimizer,
            config_path,
            backend,
            subjects=resolved_subjects,
            require_all=False,
        )

    result = aggregate_per_subject_best(
        optimizer.output_dir,
        optimizer,
        config_path,
        backend,
        subjects=resolved_subjects,
        require_all=True,
    )
    print(f"{backend} optimization done.")
    print(json.dumps(_to_builtin(result), ensure_ascii=False, indent=2, allow_nan=False))
    return result


def run_per_subject_workflow(
    backend: str,
    hyper_config_path: Path,
    generated_sim_config_path: Path,
    sim_output_dir: Path,
    hyper_output_dir: Path | None,
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
    stage: str,
    skip_hyper: bool,
    skip_completed_hyper: bool,
    skip_simulation: bool,
    skip_completed_simulation: bool,
    keep_logs: bool,
    resume_from_coarse: bool,
) -> None:
    optimizer = build_hyper_selector(backend, hyper_config_path, output_dir=hyper_output_dir)
    resolved_subjects = optimizer.resolve_subjects(subjects, subject_range)
    optimizer.output_dir.mkdir(parents=True, exist_ok=True)

    for sid in resolved_subjects:
        sid = int(sid)
        print(f"\n{'=' * 72}")
        print(f"Subject {sid}: {backend} -> simulation")
        print(f"{'=' * 72}")
        subject_best_path = _subject_best_path(optimizer.output_dir, sid)
        if skip_hyper:
            _require_subject_hyper_result(subject_best_path, stage)
            print(f"Using existing {backend} result for subject {sid}: {subject_best_path}")
        elif skip_completed_hyper:
            complete, reason = _subject_hyper_result_satisfies_stage(subject_best_path, stage)
            if complete:
                print(f"Skipping subject {sid} {backend}; found completed {stage} result: {subject_best_path}")
            else:
                if subject_best_path.is_file():
                    print(f"Existing {backend} result for subject {sid} is not complete for {stage}: {reason}")
                optimizer._run_subject_pipeline(
                    sid,
                    stage,
                    optimizer.output_dir,
                    resume_from_coarse=resume_from_coarse,
                )
        else:
            optimizer._run_subject_pipeline(
                sid,
                stage,
                optimizer.output_dir,
                resume_from_coarse=resume_from_coarse,
            )

        aggregate = aggregate_per_subject_best(
            optimizer.output_dir,
            optimizer,
            hyper_config_path,
            backend,
            subjects=resolved_subjects,
            require_all=False,
        )
        completed_subjects = [
            int(sid_text)
            for sid_text in (aggregate["best"].get("per_subject_best") or {}).keys()
        ]
        generated_cfg = materialize_simulation_config_from_hyper_best(
            hyper_best_path=Path(aggregate["best_hyperparams"]),
            generated_sim_config_path=generated_sim_config_path,
            sim_output_dir=sim_output_dir,
            keep_logs=keep_logs,
            subjects=completed_subjects,
        )

        if skip_simulation:
            continue

        sim_subject_path = sim_output_dir / "subjects" / f"subject_{sid}.json"
        if skip_completed_simulation:
            expected = _expected_simulation_signature(
                generated_cfg,
                generated_sim_config_path,
                sid,
                subjects=[sid],
            )
            complete, reason = _simulation_result_satisfies_signature(sim_subject_path, expected)
            if complete:
                print(f"Skipping subject {sid} simulation; existing result matches current config: {sim_subject_path}")
                continue
            if sim_subject_path.is_file():
                print(f"Existing simulation for subject {sid} is stale: {reason}")
        run_simulation_for_subject(generated_sim_config_path, sid)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hyperparameter selection, generate fixed simulation config, then run simulations.")
    p.add_argument("--backend", choices=("hyper_grid", "hyper_cd"), default="hyper_grid")
    p.add_argument("--hyper-config", type=Path, help="Hyper YAML config for the selected backend")
    p.add_argument("--hyper-grid-config", type=Path, help="Hyper-grid YAML config")
    p.add_argument("--hyper-cd-config", type=Path, help="Hyper-CD YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive subject range")
    p.add_argument("--stage", choices=("coarse", "fine", "all"), default="all")
    p.add_argument("--execution-mode", choices=("batch", "per-subject"), default="batch")
    p.add_argument("--hyper-output-dir", type=Path, help="Override hyper output directory")
    p.add_argument("--generated-sim-config", type=Path)
    p.add_argument("--sim-output-dir", type=Path)
    p.add_argument("--skip-hyper", action="store_true", help="Reuse existing hyper best_hyperparams.json")
    p.add_argument("--skip-completed-hyper", action="store_true", help="Reuse existing subject_<id>/best_hyperparams.json files")
    p.add_argument("--skip-simulation", action="store_true", help="Only generate the subjectwise simulation config")
    p.add_argument("--skip-completed-simulation", action="store_true", help="Reuse existing subjects/subject_<id>.json files")
    p.add_argument("--keep-logs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--resume-from-coarse",
        action="store_true",
        help="With --stage fine, load existing coarse all_combinations.jsonl and rerun only fine.",
    )
    return p.parse_args()


def _resolve_hyper_config_arg(args: argparse.Namespace) -> Path:
    if args.hyper_config is not None:
        return args.hyper_config
    if args.backend == "hyper_cd":
        return args.hyper_cd_config or DEFAULT_HYPER_CD_CONFIG
    return args.hyper_grid_config or DEFAULT_HYPER_GRID_CONFIG


def main() -> None:
    args = parse_args()
    backend = str(args.backend)
    hyper_config_path = resolve_project_path(_resolve_hyper_config_arg(args))
    generated_sim_config_path = resolve_project_path(
        args.generated_sim_config or DEFAULT_GENERATED_SIM_CONFIGS[backend]
    )
    sim_output_dir = resolve_project_path(args.sim_output_dir or DEFAULT_SIM_OUTPUT_DIRS[backend])
    hyper_output_dir = resolve_project_path(args.hyper_output_dir) if args.hyper_output_dir else None
    subject_resolver = build_hyper_selector(backend, hyper_config_path, output_dir=hyper_output_dir)
    resolved_subjects = subject_resolver.resolve_subjects(args.subjects, args.subject_range)

    if args.execution_mode == "per-subject":
        run_per_subject_workflow(
            backend=backend,
            hyper_config_path=hyper_config_path,
            generated_sim_config_path=generated_sim_config_path,
            sim_output_dir=sim_output_dir,
            hyper_output_dir=hyper_output_dir,
            subjects=args.subjects,
            subject_range=args.subject_range,
            stage=args.stage,
            skip_hyper=bool(args.skip_hyper),
            skip_completed_hyper=bool(args.skip_completed_hyper),
            skip_simulation=bool(args.skip_simulation),
            skip_completed_simulation=bool(args.skip_completed_simulation),
            keep_logs=bool(args.keep_logs),
            resume_from_coarse=bool(args.resume_from_coarse),
        )
        return

    if not args.skip_hyper:
        run_hyper_resumable(
            backend=backend,
            config_path=hyper_config_path,
            subjects=args.subjects,
            subject_range=args.subject_range,
            stage=args.stage,
            skip_completed=bool(args.skip_completed_hyper),
            hyper_output_dir=hyper_output_dir,
            resume_from_coarse=bool(args.resume_from_coarse),
        )

    hyper_cfg = load_yaml(hyper_config_path)
    effective_hyper_output_dir = hyper_output_dir or resolve_relative_to(
        hyper_config_path.parent,
        hyper_cfg.get("output_dir"),
    )
    hyper_best_path = effective_hyper_output_dir / "best_hyperparams.json"
    generated_cfg = materialize_simulation_config_from_hyper_best(
        hyper_best_path=hyper_best_path,
        generated_sim_config_path=generated_sim_config_path,
        sim_output_dir=sim_output_dir,
        keep_logs=bool(args.keep_logs),
        subjects=resolved_subjects,
    )

    if args.skip_simulation:
        return
    if args.skip_completed_simulation:
        for sid in resolved_subjects:
            sid = int(sid)
            sim_subject_path = sim_output_dir / "subjects" / f"subject_{sid}.json"
            expected = _expected_simulation_signature(
                generated_cfg,
                generated_sim_config_path,
                sid,
                subjects=[sid],
            )
            complete, reason = _simulation_result_satisfies_signature(sim_subject_path, expected)
            if complete:
                print(f"Skipping subject {sid} simulation; existing result matches current config: {sim_subject_path}")
                continue
            if sim_subject_path.is_file():
                print(f"Existing simulation for subject {sid} is stale: {reason}")
            run_simulation_for_subject(generated_sim_config_path, sid)
        return
    run_simulation_for_all(generated_sim_config_path)


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _to_builtin(obj: Any) -> Any:
    return hyper_to_builtin(obj)


if __name__ == "__main__":
    main()
