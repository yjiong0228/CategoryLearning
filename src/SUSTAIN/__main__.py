"""Run SUSTAIN scoring, bounded fitting, generation or the six-type benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import yaml

from .benchmark import run_six_problems
from .data import load_csv_history
from .fitting import fit_parameters
from .model import Parameters, SUSTAIN
from .sequence import run_sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(config_path: Path, config: dict) -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    sources = sorted(Path(__file__).parent.glob("*.py"))
    return dict(
        schema="sustain_run_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        config=config, config_sha256=_sha256(config_path),
        code_commit=git("rev-parse", "HEAD"), worktree_status=git("status", "--short"),
        source_sha256={p.name: _sha256(p) for p in sources},
        environment={"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "PyYAML")
        }},
    )


def _write_json(path: Path, value: dict | list) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(config_path: Path, outdir: Path, *, fit: bool = False, simulate: bool = False) -> dict:
    """Execute a declared small or production run in a new output directory."""
    if outdir.exists():
        raise FileExistsError(f"Output already exists: {outdir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "sustain_v1":
        raise ValueError("Expected schema: sustain_v1")
    if fit and simulate:
        raise ValueError("Choose fitting or generation for one invocation")
    parameters = Parameters(**config["parameters"])
    options = dict(config.get("model", {}))
    seed = config["seed"]
    provenance = _provenance(config_path, config)
    if "benchmark" in config:
        if fit or simulate or "data" in config:
            raise ValueError("A benchmark config runs the six-type benchmark only")
        if set(options) - {"recruitment"}:
            raise ValueError("Benchmark model options support recruitment only")
        benchmark = run_six_problems(parameters, seed=seed, **options, **config["benchmark"])
        summary = {k: v for k, v in benchmark.items() if k != "paths"}
        outdir.mkdir(parents=True, exist_ok=False)
        _write_json(outdir / "benchmark.json", benchmark)
    else:
        data = load_csv_history(config["data"])
        options.setdefault("seed", seed)
        n = len(data.targets)
        train_trials = config.get("fit", {}).get("train_trials", n)
        if isinstance(train_trials, bool) or int(train_trials) != train_trials or not 1 <= train_trials <= n:
            raise ValueError("fit.train_trials must select a nonempty available prefix")
        train_trials = int(train_trials)
        optimization = None
        if fit:
            if "fit" not in config:
                raise ValueError("Fitting requires an explicit fit budget and bounds in YAML")
            parameters, optimization = fit_parameters(
                data.stimuli, data.targets, data.choices, data.dimensions, len(data.category_values),
                base=parameters, seed=seed, model_options=options, **config["fit"],
            )
        if options.get("ties", "first") != "first" and not simulate:
            raise ValueError("Scoring requires ties: first; random tie histories are not marginalized")
        model = SUSTAIN(data.dimensions, len(data.category_values), parameters, **options)
        result = run_sequence(
            model, data.stimuli, data.targets, choices=None if simulate else data.choices,
            sample_choices=simulate, seed=seed, trace=config.get("trace", True),
        )
        rows = data.frame[["sustain_source_row", *config["data"]["order_columns"]]].copy()
        rows["target"] = [data.category_values[i] for i in data.targets]
        rows["choice"] = [data.category_values[i] for i in result.choices]
        rows["correct"] = result.choices == data.targets
        for index, value in enumerate(data.category_values):
            rows[f"p_category_{value}"] = result.probabilities[:, index]
        per_trial_nll = -result.log_probabilities[np.arange(n), data.choices]
        if not simulate:
            rows["choice_nll"] = per_trial_nll
            rows["parameter_selection_segment"] = np.where(np.arange(n) < train_trials, "train", "evaluation")
        summary = dict(
            mode="generate" if simulate else "fit" if fit else "score",
            trials=n, category_values=data.category_values, parameters=asdict(parameters),
            model_options=options, dimensions=data.dimensions,
            negative_log_likelihood=result.negative_log_likelihood,
            training_nll=None if simulate else float(per_trial_nll[:train_trials].sum()),
            evaluation_nll=(float(per_trial_nll[train_trials:].sum()) if not simulate and train_trials < n else None),
            evaluation_protocol="sequential_prediction_with_learning_from_past_evaluation_feedback",
            generation_protocol="fixed_recorded_stimulus_and_label_schedule_no_adaptive_stopping",
            final_clusters=len(model.clusters), optimization=optimization,
            status="computational_run_not_parameter_recovery_or_empirical_validation",
        )
        provenance["data"] = dict(path=str(data.source), sha256=data.source_sha256,
                                  selected_source_rows=data.frame.sustain_source_row.tolist())
        outdir.mkdir(parents=True, exist_ok=False)
        rows.to_csv(outdir / "trials.csv", index=False, mode="x")
        _write_json(outdir / "final_state.json", model.state_dict())
        if config.get("trace", True):
            _write_json(outdir / "state_trace.json", result.traces)
    provenance["resolved_parameters"] = asdict(parameters)
    _write_json(outdir / "provenance.json", provenance)
    _write_json(outdir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config; data paths are relative to repository root")
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory; existing directories are refused")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--fit", action="store_true", help="Fit using the explicit YAML budget and training prefix")
    action.add_argument("--simulate", action="store_true", help="Sample a path on the configured fixed schedule")
    args = parser.parse_args()
    summary = run(args.config, args.outdir, fit=args.fit, simulate=args.simulate)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
