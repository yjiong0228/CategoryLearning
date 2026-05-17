"""Run per-subject hyper-optimization, then materialize GRID results."""
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

from src.Bayesian_state.hyper_opt.optimizer import HyperOptimizer
from src.Bayesian_state.utils.config_subjects import SUBJECT_OVERRIDE_KEYS
from src.Bayesian_state.utils.paths import ROOT_DIR


DEFAULT_HYPER_CONFIG = Path("configs/hyper_opt_cfg/pmh_cond1_hyper.yaml")
DEFAULT_GENERATED_GRID_CONFIG = Path("configs/grid_opt_cfg/pmh_cond1_subjectwise_hyper_best.yaml")
DEFAULT_GRID_OUTPUT_DIR = Path("results/state-based-grid-result/pmh/cond1_subjectwise_hyper_best")


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
    path.write_text(json.dumps(_to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def resolve_relative_to(base_dir: Path, maybe_path: Any) -> Path:
    path = Path(maybe_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def relative_path_for_yaml(target: Path, yaml_dir: Path) -> str:
    return Path(os.path.relpath(target.resolve(), yaml_dir.resolve())).as_posix()


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


def _load_base_engine_config(base_grid_cfg: Mapping[str, Any], base_grid_dir: Path) -> dict[str, Any]:
    inline_cfg = base_grid_cfg.get("engine_config")
    path_cfg = base_grid_cfg.get("engine_config_path")

    if inline_cfg is not None and not isinstance(inline_cfg, dict):
        raise ValueError("engine_config must be a mapping when provided")

    base_engine: dict[str, Any] = {}
    if path_cfg:
        engine_path = resolve_relative_to(base_grid_dir, path_cfg)
        base_engine = load_yaml(engine_path)

    if inline_cfg is None and not path_cfg:
        raise ValueError("Base grid config must provide engine_config or engine_config_path")
    if inline_cfg is not None:
        base_engine = _deep_update(base_engine, inline_cfg)

    return _strip_subject_override_blocks(base_engine)


def _split_hyperparams_for_grid_override(best_hyperparams: Mapping[str, Any]) -> dict[str, Any]:
    override: dict[str, Any] = {}
    engine_override: dict[str, Any] = {}

    for key, value in best_hyperparams.items():
        if key.startswith("inner."):
            _set_by_path(override, key[len("inner."):], value)
        elif key.startswith("engine."):
            _set_by_path(engine_override, key[len("engine."):], value)
        else:
            raise ValueError(f"Hyperparameter key must start with 'inner.' or 'engine.': {key}")

    if engine_override:
        override["engine_config"] = engine_override
    return override


def build_subjectwise_grid_config(
    hyper_best_payload: Mapping[str, Any],
    generated_grid_config_path: Path,
    grid_output_dir: Path,
    keep_logs: bool,
) -> dict[str, Any]:
    if hyper_best_payload.get("hyperparam_selection_mode") != "per_subject":
        raise ValueError("This workflow expects hyperparam_selection_mode='per_subject'.")

    per_subject_best = hyper_best_payload.get("per_subject_best")
    if not isinstance(per_subject_best, Mapping) or not per_subject_best:
        raise ValueError("Hyper best payload is missing non-empty per_subject_best.")

    base_grid_path_raw = hyper_best_payload.get("inner_base_config_path")
    if not base_grid_path_raw:
        raise ValueError("Hyper best payload is missing inner_base_config_path.")
    base_grid_path = resolve_project_path(Path(str(base_grid_path_raw)))
    base_grid_cfg = load_yaml(base_grid_path)
    base_engine_cfg = _load_base_engine_config(base_grid_cfg, base_grid_path.parent)

    generated_dir = generated_grid_config_path.parent
    grid_output_dir = grid_output_dir.resolve()

    generated_cfg = _strip_subject_override_blocks(base_grid_cfg)
    generated_cfg.pop("engine_config_path", None)
    generated_cfg["engine_config"] = base_engine_cfg
    generated_cfg["subjects"] = sorted(int(sid) for sid in per_subject_best.keys())
    generated_cfg.pop("subject_range", None)
    generated_cfg["output_dir"] = relative_path_for_yaml(grid_output_dir, generated_dir)
    generated_cfg["keep_logs"] = bool(keep_logs)

    subject_overrides: dict[int, Any] = {}
    for sid_text, subject_payload in sorted(per_subject_best.items(), key=lambda item: int(item[0])):
        if not isinstance(subject_payload, Mapping):
            raise ValueError(f"per_subject_best[{sid_text}] must be a mapping")
        best_hyperparams = subject_payload.get("best_hyperparams")
        if not isinstance(best_hyperparams, Mapping):
            raise ValueError(f"per_subject_best[{sid_text}] is missing best_hyperparams")
        subject_overrides[int(sid_text)] = _split_hyperparams_for_grid_override(best_hyperparams)

    generated_cfg["subject_overrides"] = subject_overrides
    return generated_cfg


def run_hyper(config_path: Path, subjects: Sequence[int] | None, subject_range: Sequence[int] | None, stage: str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    optimizer = HyperOptimizer(cfg, config_path)
    resolved_subjects = optimizer.resolve_subjects(subjects, subject_range)
    result = optimizer.run(subjects=resolved_subjects, stage=stage)
    print("Hyper optimization done.")
    print(json.dumps(_to_builtin(result), ensure_ascii=False, indent=2))
    return result


def _subject_best_path(output_dir: Path, subject_id: int) -> Path:
    return output_dir / f"subject_{int(subject_id)}" / "best_hyperparams.json"


def aggregate_per_subject_best(output_dir: Path, optimizer: HyperOptimizer, config_path: Path) -> dict[str, Any]:
    per_subject_best: dict[str, Any] = {}
    per_subject_outputs: dict[str, Any] = {}

    for best_path in sorted(output_dir.glob("subject_*/best_hyperparams.json")):
        subject_dir = best_path.parent
        try:
            sid = int(subject_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        subject_best = json.loads(best_path.read_text(encoding="utf-8"))
        per_subject_best[str(sid)] = subject_best
        per_subject_outputs[str(sid)] = {
            "output_dir": str(subject_dir),
            "all_combinations": str(subject_dir / "all_combinations.jsonl"),
            "stage_summary": str(subject_dir / "stage_summary.json"),
            "best_hyperparams": str(best_path),
        }

    payload = {
        "selection_metric": optimizer.selection_metric,
        "hyperparam_selection_mode": optimizer.hyperparam_selection_mode,
        "save_level": optimizer.save_level,
        "inner_base_config_path": str(optimizer.inner_base_config_path),
        "hyper_config_path": str(config_path),
        "per_subject_best": dict(sorted(per_subject_best.items(), key=lambda item: int(item[0]))),
    }
    best_path = output_dir / "best_hyperparams.json"
    save_json(best_path, payload)
    return {
        "output_dir": str(output_dir),
        "per_subject_outputs": dict(sorted(per_subject_outputs.items(), key=lambda item: int(item[0]))),
        "best_hyperparams": str(best_path),
        "best": payload,
    }


def run_hyper_resumable(
    config_path: Path,
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
    stage: str,
    skip_completed: bool,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    optimizer = HyperOptimizer(cfg, config_path)
    resolved_subjects = optimizer.resolve_subjects(subjects, subject_range)

    if optimizer.hyperparam_selection_mode != "per_subject":
        if skip_completed:
            raise ValueError("--skip-completed-hyper is only supported for hyperparam_selection_mode='per_subject'.")
        return optimizer.run(subjects=resolved_subjects, stage=stage)

    optimizer.output_dir.mkdir(parents=True, exist_ok=True)
    for sid in resolved_subjects:
        best_path = _subject_best_path(optimizer.output_dir, int(sid))
        if skip_completed and best_path.is_file():
            print(f"Skipping subject {int(sid)}; found {best_path}")
            continue
        print(f"Running hyper optimization for subject {int(sid)}")
        optimizer._run_subject_pipeline(int(sid), stage, optimizer.output_dir)
        aggregate_per_subject_best(optimizer.output_dir, optimizer, config_path)

    result = aggregate_per_subject_best(optimizer.output_dir, optimizer, config_path)
    print("Hyper optimization done.")
    print(json.dumps(_to_builtin(result), ensure_ascii=False, indent=2))
    return result


def run_command(cmd: Sequence[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(list(cmd), cwd=ROOT_DIR, check=True)


def materialize_grid_config_from_hyper_best(
    hyper_best_path: Path,
    generated_grid_config_path: Path,
    grid_output_dir: Path,
    keep_logs: bool,
) -> dict[str, Any]:
    if not hyper_best_path.is_file():
        raise FileNotFoundError(f"Hyper best file not found: {hyper_best_path}")
    hyper_best_payload = json.loads(hyper_best_path.read_text(encoding="utf-8"))
    generated_cfg = build_subjectwise_grid_config(
        hyper_best_payload=hyper_best_payload,
        generated_grid_config_path=generated_grid_config_path,
        grid_output_dir=grid_output_dir,
        keep_logs=bool(keep_logs),
    )
    save_yaml(generated_grid_config_path, generated_cfg)
    save_json(grid_output_dir / "hyper_best_source.json", hyper_best_payload)
    print(f"Generated subjectwise GRID config -> {generated_grid_config_path}")
    print(f"GRID output directory -> {grid_output_dir}")
    return generated_cfg


def run_grid_for_subject(generated_grid_config_path: Path, subject_id: int) -> None:
    grid_cmd = [
        sys.executable,
        "-m",
        "src.Bayesian_state.run_grid_optimization",
        "--config",
        command_path(generated_grid_config_path),
        "--subjects",
        str(int(subject_id)),
    ]
    run_command(grid_cmd)


def run_grid_for_all(generated_grid_config_path: Path) -> None:
    grid_cmd = [
        sys.executable,
        "-m",
        "src.Bayesian_state.run_grid_optimization",
        "--config",
        command_path(generated_grid_config_path),
    ]
    run_command(grid_cmd)


def run_eval(grid_output_dir: Path, generated_grid_config_path: Path) -> None:
    eval_cmd = [
        sys.executable,
        "-m",
        "src.Bayesian_state.eval_grid_results",
        "--input-dir",
        command_path(grid_output_dir),
        "--config",
        command_path(generated_grid_config_path),
    ]
    run_command(eval_cmd)


def run_per_subject_workflow(
    hyper_config_path: Path,
    generated_grid_config_path: Path,
    grid_output_dir: Path,
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
    stage: str,
    skip_hyper: bool,
    skip_completed_hyper: bool,
    skip_grid: bool,
    skip_completed_grid: bool,
    skip_eval: bool,
    keep_logs: bool,
) -> None:
    cfg = load_yaml(hyper_config_path)
    optimizer = HyperOptimizer(cfg, hyper_config_path)
    if optimizer.hyperparam_selection_mode != "per_subject":
        raise ValueError("--execution-mode per-subject requires hyperparam_selection_mode='per_subject'.")

    resolved_subjects = optimizer.resolve_subjects(subjects, subject_range)
    optimizer.output_dir.mkdir(parents=True, exist_ok=True)

    for sid in resolved_subjects:
        sid = int(sid)
        print(f"\n{'=' * 72}")
        print(f"Subject {sid}: hyper -> grid")
        print(f"{'=' * 72}")

        subject_best_path = _subject_best_path(optimizer.output_dir, sid)
        if skip_hyper:
            if not subject_best_path.is_file():
                raise FileNotFoundError(f"--skip-hyper requested but subject best is missing: {subject_best_path}")
            print(f"Using existing hyper result for subject {sid}: {subject_best_path}")
        elif skip_completed_hyper and subject_best_path.is_file():
            print(f"Skipping subject {sid} hyper; found {subject_best_path}")
        else:
            optimizer._run_subject_pipeline(sid, stage, optimizer.output_dir)

        aggregate = aggregate_per_subject_best(optimizer.output_dir, optimizer, hyper_config_path)
        materialize_grid_config_from_hyper_best(
            hyper_best_path=Path(aggregate["best_hyperparams"]),
            generated_grid_config_path=generated_grid_config_path,
            grid_output_dir=grid_output_dir,
            keep_logs=keep_logs,
        )

        if skip_grid:
            continue

        grid_subject_path = grid_output_dir / f"subject_{sid}.json"
        if skip_completed_grid and grid_subject_path.is_file():
            print(f"Skipping subject {sid} GRID; found {grid_subject_path}")
            continue
        run_grid_for_subject(generated_grid_config_path, sid)

    if not skip_grid and not skip_eval:
        run_eval(grid_output_dir, generated_grid_config_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hyper-opt, generate subjectwise GRID config, run GRID, then evaluate.")
    p.add_argument("--hyper-config", type=Path, default=DEFAULT_HYPER_CONFIG)
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive subject range")
    p.add_argument("--stage", choices=("coarse", "fine", "all"), default="all")
    p.add_argument("--execution-mode", choices=("batch", "per-subject"), default="batch")
    p.add_argument("--generated-grid-config", type=Path, default=DEFAULT_GENERATED_GRID_CONFIG)
    p.add_argument("--grid-output-dir", type=Path, default=DEFAULT_GRID_OUTPUT_DIR)
    p.add_argument("--skip-hyper", action="store_true", help="Reuse existing hyper best_hyperparams.json")
    p.add_argument("--skip-completed-hyper", action="store_true", help="In per-subject mode, reuse existing subject_<id>/best_hyperparams.json files")
    p.add_argument("--skip-grid", action="store_true", help="Only generate the subjectwise GRID config")
    p.add_argument("--skip-completed-grid", action="store_true", help="In per-subject mode, reuse existing subject_<id>.json GRID files")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval_grid_results after GRID finishes")
    p.add_argument("--keep-logs", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hyper_config_path = resolve_project_path(args.hyper_config)
    generated_grid_config_path = resolve_project_path(args.generated_grid_config)
    grid_output_dir = resolve_project_path(args.grid_output_dir)

    if args.execution_mode == "per-subject":
        run_per_subject_workflow(
            hyper_config_path=hyper_config_path,
            generated_grid_config_path=generated_grid_config_path,
            grid_output_dir=grid_output_dir,
            subjects=args.subjects,
            subject_range=args.subject_range,
            stage=args.stage,
            skip_hyper=bool(args.skip_hyper),
            skip_completed_hyper=bool(args.skip_completed_hyper),
            skip_grid=bool(args.skip_grid),
            skip_completed_grid=bool(args.skip_completed_grid),
            skip_eval=bool(args.skip_eval),
            keep_logs=bool(args.keep_logs),
        )
        return

    if not args.skip_hyper:
        run_hyper_resumable(
            config_path=hyper_config_path,
            subjects=args.subjects,
            subject_range=args.subject_range,
            stage=args.stage,
            skip_completed=bool(args.skip_completed_hyper),
        )

    hyper_cfg = load_yaml(hyper_config_path)
    hyper_output_dir = resolve_relative_to(hyper_config_path.parent, hyper_cfg.get("output_dir"))
    hyper_best_path = hyper_output_dir / "best_hyperparams.json"
    materialize_grid_config_from_hyper_best(
        hyper_best_path=hyper_best_path,
        generated_grid_config_path=generated_grid_config_path,
        grid_output_dir=grid_output_dir,
        keep_logs=bool(args.keep_logs),
    )

    if args.skip_grid:
        return

    run_grid_for_all(generated_grid_config_path)

    if not args.skip_eval:
        run_eval(grid_output_dir, generated_grid_config_path)


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _to_builtin(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    return obj


if __name__ == "__main__":
    main()
