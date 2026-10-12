"""Bounded single-subject timing for restart and particle-block scheduling."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import yaml

from ...optimization.search.coordinate_descent import HyperCDOptimizer
from ...simulation.config import load_yaml, resolve_engine_config
from ...utils.datasets import resolve_dataset_paths
from ...utils.parallel import single_threaded_processes
from ...utils.paths import ROOT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/meg/hyper_cd_cfg/model0923_task3b.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new result directory")
    parser.add_argument("--variant", choices=("serial", "restarts", "blocks"), default="blocks")
    parser.add_argument("--subject", type=int, default=334)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config_path = (ROOT_DIR / args.config).resolve()
    output = (ROOT_DIR / args.output_dir).resolve()
    if args.trials < 9 or args.workers < 1:
        parser.error("trials must be at least9 and workers must be positive")
    output.mkdir(parents=True, exist_ok=False)
    search = load_yaml(config_path)
    base_path = (config_path.parent / search["base_sim_config_path"]).resolve()
    simulation = load_yaml(base_path)
    engine = resolve_engine_config(simulation, base_path.parent)
    engine["partition"]["kwargs"]["similarity_cache_dir"] = str(output / "cache")
    simulation.pop("engine_config_path", None)
    simulation["engine_config"] = engine
    simulation["dataset"] = {key: str(value) for key, value in resolve_dataset_paths(simulation, base_path.parent).items()}
    simulation.update(subjects=[args.subject], max_trials=args.trials, n_jobs=args.workers)
    sim_path = output / "simulation.yaml"
    sim_path.write_text(yaml.safe_dump(simulation, sort_keys=False), encoding="utf-8")
    names = (
        "engine.modules.hypo_transitions_mod.kwargs.search_controller.baseline_probability",
        "engine.modules.beta_mod.kwargs.beta_init", "engine.modules.memory_mod.kwargs.gamma",
    )
    search["hyperparam_space"] = {name: search["hyperparam_space"][name] for name in names}
    search["stages"]["fine"]["hyperparam_space"] = deepcopy(search["hyperparam_space"])
    search.update(subjects=[args.subject], base_sim_config_path=str(sim_path), output_dir=str(output / "fit"))
    search["cd"].update(max_outer_iters=1, parallel_budget=args.workers,
                        parallel_restarts=args.variant != "serial",
                        parallel_final_rescore=args.variant != "serial")
    for name, stage in search["stages"].items():
        stage["simulation_overrides"]["engine_config"]["inference"]["particle_blocks"] = (
            4 if name == "coarse" else 2) if args.variant == "blocks" and args.workers >= 4 else 1
    search["final_rescore"]["simulation_overrides"]["engine_config"]["inference"]["particle_blocks"] = (
        2 if args.variant == "blocks" and args.workers >= 4 else 1)
    path = output / "hyper.yaml"
    path.write_text(yaml.safe_dump(search, sort_keys=False), encoding="utf-8")
    started = perf_counter()
    with single_threaded_processes():
        result = HyperCDOptimizer(search, path).run([args.subject], stage="all")
    report = {"variant": args.variant, "seconds": perf_counter() - started,
              "subject": args.subject, "trials": args.trials, "workers": args.workers,
              "coordinates": list(names), "rounds_per_restart": 1,
              "best_hyperparams": result["best_hyperparams"]}
    (output / "timing.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
