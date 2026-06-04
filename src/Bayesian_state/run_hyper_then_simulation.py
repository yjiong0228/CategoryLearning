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

from src.Bayesian_state.hyper_cd.optimizer import HyperCDOptimizer
from src.Bayesian_state.hyper_grid.optimizer import HyperGridOptimizer
from src.Bayesian_state.utils.config_subjects import SUBJECT_OVERRIDE_KEYS
from src.Bayesian_state.utils.paths import ROOT_DIR


DEFAULT_HYPER_GRID_CONFIG = Path("configs/hyper_grid_cfg/pmh_cond1_hyper_grid.yaml")
DEFAULT_HYPER_CD_CONFIG = Path("configs/hyper_cd_cfg/pmh_cond1_hyper_cd.yaml")
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
    path.write_text(json.dumps(_to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _load_base_engine_config(base_sim_cfg: Mapping[str, Any], base_sim_dir: Path) -> dict[str, Any]:
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
    return _strip_subject_override_blocks(base_engine)


def _rebase_generated_sim_paths(
    generated_cfg: dict[str, Any],
    base_sim_dir: Path,
    generated_dir: Path,
) -> None:
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
    for key, value in best_hyperparams.items():
        if key.startswith("engine."):
            _set_by_path(engine_override, key[len("engine."):], value)
        elif key.startswith("simulation."):
            _set_by_path(override, key[len("simulation."):], value)
        else:
            raise ValueError(f"Hyperparameter key must start with 'engine.' or 'simulation.': {key}")
    if engine_override:
        override["engine_config"] = engine_override
    override["fixed_hyperparams"] = deepcopy(dict(best_hyperparams))
    return override


def build_subjectwise_simulation_config(
    hyper_best_payload: Mapping[str, Any],
    generated_sim_config_path: Path,
    sim_output_dir: Path,
    keep_logs: bool,
) -> dict[str, Any]:
    if hyper_best_payload.get("hyperparam_selection_mode") != "per_subject":
        raise ValueError("This workflow expects hyperparam_selection_mode='per_subject'.")

    per_subject_best = hyper_best_payload.get("per_subject_best")
    if not isinstance(per_subject_best, Mapping) or not per_subject_best:
        raise ValueError("Hyper best payload is missing non-empty per_subject_best.")

    base_sim_path_raw = hyper_best_payload.get("base_sim_config_path")
    if not base_sim_path_raw:
        raise ValueError("Hyper best payload is missing base_sim_config_path.")
    base_sim_path = resolve_project_path(Path(str(base_sim_path_raw)))
    base_sim_cfg = load_yaml(base_sim_path)
    base_engine_cfg = _load_base_engine_config(base_sim_cfg, base_sim_path.parent)

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
    if "hyper_base_seed" in hyper_best_payload:
        generated_cfg["hyper_base_seed"] = hyper_best_payload["hyper_base_seed"]

    subject_overrides: dict[int, Any] = {}
    for sid_text, subject_payload in sorted(per_subject_best.items(), key=lambda item: int(item[0])):
        if not isinstance(subject_payload, Mapping):
            raise ValueError(f"per_subject_best[{sid_text}] must be a mapping")
        best_hyperparams = subject_payload.get("best_hyperparams")
        if not isinstance(best_hyperparams, Mapping):
            raise ValueError(f"per_subject_best[{sid_text}] is missing best_hyperparams")
        override = _split_hyperparams_for_simulation_override(best_hyperparams)
        if "hyper_candidate_seed" in subject_payload:
            override["hyper_candidate_seed"] = int(subject_payload["hyper_candidate_seed"])
        subject_overrides[int(sid_text)] = override

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
) -> dict[str, Any]:
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

    payload = {
        "selection_metric": optimizer.selection_metric,
        "hyperparam_selection_mode": optimizer.hyperparam_selection_mode,
        "save_level": optimizer.save_level,
        "base_sim_config_path": str(optimizer.base_sim_config_path),
        f"{backend}_config_path": str(config_path),
        "hyper_output_dir": str(output_dir),
        "hyper_backend": backend,
        "hyper_base_seed": optimizer.hyper_base_seed,
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


def run_command(cmd: Sequence[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(list(cmd), cwd=ROOT_DIR, check=True)


def materialize_simulation_config_from_hyper_best(
    hyper_best_path: Path,
    generated_sim_config_path: Path,
    sim_output_dir: Path,
    keep_logs: bool,
) -> dict[str, Any]:
    if not hyper_best_path.is_file():
        raise FileNotFoundError(f"Hyper best file not found: {hyper_best_path}")
    hyper_best_payload = json.loads(hyper_best_path.read_text(encoding="utf-8"))
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

    if optimizer.hyperparam_selection_mode != "per_subject":
        if skip_completed:
            raise ValueError("--skip-completed-hyper is only supported for per_subject mode.")
        return optimizer.run(resolved_subjects, stage=stage, resume_from_coarse=resume_from_coarse)

    optimizer.output_dir.mkdir(parents=True, exist_ok=True)
    for sid in resolved_subjects:
        best_path = _subject_best_path(optimizer.output_dir, int(sid))
        if skip_completed and best_path.is_file():
            print(f"Skipping subject {int(sid)}; found {best_path}")
            continue
        print(f"Running {backend} optimization for subject {int(sid)}")
        optimizer._run_subject_pipeline(
            int(sid),
            stage,
            optimizer.output_dir,
            resume_from_coarse=resume_from_coarse,
        )
        aggregate_per_subject_best(optimizer.output_dir, optimizer, config_path, backend)

    result = aggregate_per_subject_best(optimizer.output_dir, optimizer, config_path, backend)
    print(f"{backend} optimization done.")
    print(json.dumps(_to_builtin(result), ensure_ascii=False, indent=2))
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
    if optimizer.hyperparam_selection_mode != "per_subject":
        raise ValueError("--execution-mode per-subject requires hyperparam_selection_mode='per_subject'.")
    resolved_subjects = optimizer.resolve_subjects(subjects, subject_range)
    optimizer.output_dir.mkdir(parents=True, exist_ok=True)

    for sid in resolved_subjects:
        sid = int(sid)
        print(f"\n{'=' * 72}")
        print(f"Subject {sid}: {backend} -> simulation")
        print(f"{'=' * 72}")
        subject_best_path = _subject_best_path(optimizer.output_dir, sid)
        if skip_hyper:
            if not subject_best_path.is_file():
                raise FileNotFoundError(f"--skip-hyper requested but subject best is missing: {subject_best_path}")
            print(f"Using existing {backend} result for subject {sid}: {subject_best_path}")
        elif skip_completed_hyper and subject_best_path.is_file():
            print(f"Skipping subject {sid} {backend}; found {subject_best_path}")
        else:
            optimizer._run_subject_pipeline(
                sid,
                stage,
                optimizer.output_dir,
                resume_from_coarse=resume_from_coarse,
            )

        aggregate = aggregate_per_subject_best(optimizer.output_dir, optimizer, hyper_config_path, backend)
        materialize_simulation_config_from_hyper_best(
            hyper_best_path=Path(aggregate["best_hyperparams"]),
            generated_sim_config_path=generated_sim_config_path,
            sim_output_dir=sim_output_dir,
            keep_logs=keep_logs,
        )

        if skip_simulation:
            continue

        sim_subject_path = sim_output_dir / "subjects" / f"subject_{sid}.json"
        if skip_completed_simulation and sim_subject_path.is_file():
            print(f"Skipping subject {sid} simulation; found {sim_subject_path}")
            continue
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
    materialize_simulation_config_from_hyper_best(
        hyper_best_path=hyper_best_path,
        generated_sim_config_path=generated_sim_config_path,
        sim_output_dir=sim_output_dir,
        keep_logs=bool(args.keep_logs),
    )

    if args.skip_simulation:
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
