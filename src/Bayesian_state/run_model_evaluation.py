"""Run model-evaluation plots for fixed simulation outputs.

Usage:
    python -m src.Bayesian_state.run_model_evaluation \
        --input-dir results/state-based-simulation/pmh/cond1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.Bayesian_state.utils.model_evaluation import ModelEval
from src.Bayesian_state.utils.paths import PROCESSED_DATA_DIR, ROOT_DIR, SIMULATION_RESULTS_DIR


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_DIR = SIMULATION_RESULTS_DIR / "pmh" / "cond1"
DEFAULT_ORAL_DATA = PROCESSED_DATA_DIR / "Task2_processed.csv"


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def resolve_subjects(
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
) -> list[int] | None:
    if subjects:
        return [int(x) for x in subjects]
    if subject_range:
        start, end = [int(x) for x in subject_range]
        return list(range(start, end + 1))
    return None


def subject_json_files(input_dir: Path) -> list[Path]:
    if input_dir.name == "subjects":
        files = sorted(input_dir.glob("subject_*.json"))
    else:
        files = sorted((input_dir / "subjects").glob("subject_*.json"))
    if not files:
        raise FileNotFoundError(f"No subject_*.json files found under {input_dir}")
    return files


def _select_metrics(
    payload: Mapping[str, Any],
    eval_prediction_mode: str | None,
) -> tuple[str | None, Mapping[str, Any]]:
    metrics_by_mode = payload.get("metrics_by_mode")
    if not isinstance(metrics_by_mode, Mapping) or not metrics_by_mode:
        return None, {}

    mode = (
        eval_prediction_mode
        or payload.get("selection_prediction_mode")
        or payload.get("prediction_mode")
        or (payload.get("selection_meta") or {}).get("selection_prediction_mode")
    )
    if mode is None and len(metrics_by_mode) == 1:
        mode = next(iter(metrics_by_mode))
    if mode not in metrics_by_mode:
        available = ", ".join(sorted(str(k) for k in metrics_by_mode))
        raise KeyError(f"metrics_by_mode has no {mode!r}; available modes: {available}")
    metrics = metrics_by_mode[mode]
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics_by_mode[{mode!r}] must be a mapping")
    return str(mode), metrics


def load_simulation_results(
    input_dir: Path,
    subjects: Sequence[int] | None = None,
    eval_prediction_mode: str | None = None,
    window_size: int | None = None,
) -> dict[int, dict[str, Any]]:
    subject_set = {int(x) for x in subjects} if subjects is not None else None
    out: dict[int, dict[str, Any]] = {}
    for path in subject_json_files(input_dir):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        sid = int(payload.get("subject_id", path.stem.replace("subject_", "")))
        if subject_set is not None and sid not in subject_set:
            continue

        mode, metrics = _select_metrics(payload, eval_prediction_mode)
        info = dict(payload)
        info.update(dict(metrics))
        info["subject_id"] = sid
        info["condition"] = int(info.get("condition", -1))
        info["subject_json_path"] = str(path)
        info["eval_prediction_mode"] = mode

        meta = payload.get("selection_meta") or {}
        resolved_window = payload.get("window_size") or meta.get("window_size") or window_size
        if resolved_window is not None:
            info["window_size"] = int(resolved_window)

        if "n_trials" not in info:
            for key in ("true_acc", "pred_acc", "best_step_results", "posterior_log", "prior_log"):
                value = info.get(key)
                if isinstance(value, list) and value:
                    info["n_trials"] = len(value)
                    break

        out[sid] = info

    if not out:
        raise RuntimeError(f"No matching subject results found in {input_dir}")
    return out


def save_manifest(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = output_dir / "evaluation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)
    return path


def run_step(
    records: list[dict[str, Any]],
    name: str,
    func: Callable[[], Any],
    outputs: Sequence[Path] | None = None,
) -> Any:
    try:
        result = func()
        plt.close("all")
        records.append(
            {
                "name": name,
                "status": "ok",
                "outputs": [str(p) for p in outputs or []],
            }
        )
        return result
    except Exception as exc:  # keep one missing plot from aborting the whole report
        plt.close("all")
        LOGGER.warning("Skipping %s: %s", name, exc)
        records.append(
            {
                "name": name,
                "status": "skipped",
                "reason": str(exc),
                "outputs": [str(p) for p in outputs or []],
            }
        )
        return None


def run_basic_plots(
    evaluator: ModelEval,
    results: Mapping[int, Mapping[str, Any]],
    output_dir: Path,
    subjects: Sequence[int] | None,
    window_size: int | None,
    records: list[dict[str, Any]],
    posterior_limit: bool,
) -> None:
    basic_dir = output_dir / "basic"
    basic_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        records,
        "accuracy_comparison",
        lambda: evaluator.plot_accuracy_comparison(
            results,
            subjects=subjects,
            save_path=basic_dir / "accuracy_comparison.png",
            window_size=window_size,
        ),
        [basic_dir / "accuracy_comparison.png"],
    )
    visible_results = evaluator._filter_results(results, subjects)
    if any(int(info.get("condition", -1)) in (2, 3) for info in visible_results.values()):
        run_step(
            records,
            "accuracy_family_comparison",
            lambda: evaluator.plot_accuracy_family_comparison(
                results,
                subjects=subjects,
                save_path=basic_dir / "accuracy_family_comparison.png",
                window_size=window_size,
            ),
            [basic_dir / "accuracy_family_comparison.png"],
        )
    run_step(
        records,
        "posterior_probabilities",
        lambda: evaluator.plot_posterior_probabilities(
            results,
            subjects=subjects,
            save_path=basic_dir / "posterior_probabilities.png",
            limit=posterior_limit,
        ),
        [basic_dir / "posterior_probabilities.png"],
    )
    run_step(
        records,
        "beta_dynamics",
        lambda: evaluator.plot_beta_dynamics(
            results,
            subjects=subjects,
            save_path=basic_dir / "beta_dynamics.png",
        ),
        [basic_dir / "beta_dynamics.png"],
    )
    run_step(
        records,
        "cluster_amount",
        lambda: evaluator.plot_cluster_amount(
            results,
            subjects=subjects,
            save_path=basic_dir / "cluster_amount.png",
            window_size=window_size or 16,
        ),
        [basic_dir / "cluster_amount.png"],
    )
    run_step(
        records,
        "strategy_amount_details",
        lambda: evaluator.plot_strategy_amount_details(
            results,
            subjects=subjects,
            save_path=basic_dir / "strategy_amount_details.png",
            window_size=window_size or 16,
        ),
        [basic_dir / "strategy_amount_details.png"],
    )


def run_trajectory_plots(
    evaluator: ModelEval,
    input_dir: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    trajectory_ranks: Sequence[int] | None,
    posterior_ranks: Sequence[int] | None,
    eval_prediction_mode: str | None,
    posterior_limit: bool,
) -> None:
    accuracy_dir = output_dir / "trajectory_accuracy"
    posterior_dir = output_dir / "trajectory_posterior"
    run_step(
        records,
        "trajectory_accuracy",
        lambda: evaluator.plot_trajectory_analysis(
            input_dir,
            accuracy_dir,
            ranks=trajectory_ranks,
            n_cols=4,
            eval_prediction_mode=eval_prediction_mode,
        ),
        [accuracy_dir],
    )
    run_step(
        records,
        "trajectory_posterior",
        lambda: evaluator.plot_trajectory_posteriors(
            input_dir,
            posterior_dir,
            ranks=posterior_ranks,
            n_cols=4,
            limit=posterior_limit,
        ),
        [posterior_dir],
    )


def run_oral_plots(
    evaluator: ModelEval,
    results: Mapping[int, Mapping[str, Any]],
    oral_data_path: Path,
    output_dir: Path,
    subjects: Sequence[int] | None,
    records: list[dict[str, Any]],
    oral_mode: str,
    window_size: int | None,
    region_n_samples: int,
    region_stimulus_sigma: float | None,
    distribution_model_distribution: str,
    oral_model_distribution: str,
    combine_oral_equivalent: bool,
) -> None:
    oral_mode = str(oral_mode).strip().lower()
    oral_subjects = (
        [int(s) for s in subjects]
        if subjects is not None
        else sorted(int(s) for s in results.keys())
    )
    combine_requested = bool(combine_oral_equivalent)
    if combine_requested:
        eligible_subjects = [
            sid
            for sid in oral_subjects
            if int(results.get(int(sid), {}).get("condition", -1)) in (2, 3)
        ]
        skipped_subjects = sorted(set(oral_subjects) - set(eligible_subjects))
        if not eligible_subjects:
            reason = "combined oral-equivalence alignment is only generated for condition 2/3."
            LOGGER.info("Skipping %s oral combined alignment: %s", oral_mode, reason)
            records.append(
                {
                    "name": f"oral_alignment_{oral_mode}_mode_combined",
                    "status": "skipped",
                    "reason": reason,
                    "subjects": oral_subjects,
                }
            )
            return
        if skipped_subjects:
            LOGGER.info(
                "Excluding condition-1 subject(s) from %s oral combined alignment: %s",
                oral_mode,
                skipped_subjects,
            )
            records.append(
                {
                    "name": f"oral_alignment_{oral_mode}_mode_combined_subject_filter",
                    "status": "ok",
                    "reason": "combined oral-equivalence alignment is only generated for condition 2/3.",
                    "included_subjects": eligible_subjects,
                    "excluded_subjects": skipped_subjects,
                }
            )
        oral_subjects = eligible_subjects

    combine_suffix = "_combined" if combine_requested else ""
    oral_dir = output_dir / f"oral_alignment_{oral_mode}_mode{combine_suffix}"
    oral_dir.mkdir(parents=True, exist_ok=True)
    oral_df = pd.read_csv(oral_data_path)

    oral_mass = run_step(
        records,
        "oral_mass_probabilities",
        lambda: evaluator.compute_oral_mass_probabilities(
            oral_df,
            oral_mode=oral_mode,
            subjects=oral_subjects,
            region_n_samples=region_n_samples,
            region_stimulus_sigma=region_stimulus_sigma,
        ),
        [oral_dir / "oral_mass_probabilities.npz", oral_dir / "oral_mass_probabilities.png"],
    )
    if oral_mass:
        run_step(
            records,
            "save_oral_mass_probabilities",
            lambda: (
                evaluator.save_oral_mass_probabilities(
                    oral_mass,
                    oral_dir / "oral_mass_probabilities.npz",
                ),
                evaluator.plot_oral_mass_probabilities(
                    oral_mass,
                    subjects=oral_subjects,
                    save_path=oral_dir / "oral_mass_probabilities.png",
                ),
            ),
            [oral_dir / "oral_mass_probabilities.npz", oral_dir / "oral_mass_probabilities.png"],
        )

    distribution_results = run_step(
        records,
        "compute_distribution_based_alignment",
        lambda: evaluator.compute_distribution_based_alignment(
            results,
            oral_df,
            oral_mode=oral_mode,
            subjects=oral_subjects,
            region_n_samples=region_n_samples,
            region_stimulus_sigma=region_stimulus_sigma,
            model_distribution=distribution_model_distribution,
            oral_mass_results=oral_mass,
            combine_oral_equivalent=combine_oral_equivalent,
        ),
    )
    if distribution_results is not None:
        run_step(
            records,
            "save_distribution_based_alignment",
            lambda: evaluator.save_distribution_based_alignment_outputs(
                distribution_results,
                oral_dir / "distribution_based_alignment",
                window_size=window_size or 16,
            ),
            [oral_dir / "distribution_based_alignment"],
        )

    oral_based_results = run_step(
        records,
        "compute_oral_based_alignment",
        lambda: evaluator.compute_oral_based_alignment(
            results,
            oral_df,
            oral_mode=oral_mode,
            subjects=oral_subjects,
            region_n_samples=region_n_samples,
            region_stimulus_sigma=region_stimulus_sigma,
            model_distribution=oral_model_distribution,
        ),
    )
    if oral_based_results is not None:
        run_step(
            records,
            "save_oral_based_alignment",
            lambda: evaluator.save_oral_based_alignment_outputs(
                oral_based_results,
                oral_dir / "oral_based_alignment",
                window_size=window_size or 16,
            ),
            [oral_dir / "oral_based_alignment"],
        )

    target_results = run_step(
        records,
        "compute_target_based_alignment",
        lambda: evaluator.compute_target_based_alignment(
            results,
            oral_df,
            oral_mode=oral_mode,
            subjects=oral_subjects,
            region_n_samples=region_n_samples,
            region_stimulus_sigma=region_stimulus_sigma,
            oral_mass_results=oral_mass,
        ),
    )
    if target_results is not None:
        run_step(
            records,
            "save_target_based_alignment",
            lambda: evaluator.save_target_based_alignment_outputs(
                target_results,
                oral_dir / "target_based_alignment",
                window_size=window_size or 16,
            ),
            [oral_dir / "target_based_alignment"],
        )

    hit_results = run_step(
        records,
        "compute_hit_based_alignment",
        lambda: evaluator.compute_hit_based_alignment(
            results,
            oral_df,
            oral_mode=oral_mode,
            subjects=oral_subjects,
            region_n_samples=region_n_samples,
            region_stimulus_sigma=region_stimulus_sigma,
            oral_mass_results=oral_mass,
        ),
    )
    if hit_results is not None:
        run_step(
            records,
            "save_hit_based_alignment",
            lambda: evaluator.save_hit_based_alignment_outputs(
                hit_results,
                oral_dir / "hit_based_alignment",
                window_size=window_size or 16,
            ),
            [oral_dir / "hit_based_alignment"],
        )

    coverage_results = run_step(
        records,
        "compute_coverage_based_alignment",
        lambda: evaluator.compute_coverage_based_alignment(
            results,
            oral_df,
            oral_mode=oral_mode,
            subjects=oral_subjects,
            region_n_samples=region_n_samples,
            region_stimulus_sigma=region_stimulus_sigma,
            oral_mass_results=oral_mass,
        ),
    )
    if coverage_results is not None:
        run_step(
            records,
            "save_coverage_based_alignment",
            lambda: evaluator.save_coverage_based_alignment_outputs(
                coverage_results,
                oral_dir / "coverage_based_alignment",
                window_size=window_size or 16,
            ),
            [oral_dir / "coverage_based_alignment"],
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run plots and model-evaluation reports for simulation outputs")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Simulation result dir with subjects/")
    p.add_argument("--output-dir", type=Path, help="Output dir; defaults to <input-dir>/model_evaluation")
    p.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to evaluate")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive subject range")
    p.add_argument("--eval-prediction-mode", help="Metrics mode to plot, e.g. prior_t")
    p.add_argument("--window-size", type=int, help="Fallback window size for old result JSONs")
    p.add_argument("--skip-basic", action="store_true", help="Skip group-level metric/log plots")
    p.add_argument("--skip-trajectory", action="store_true", help="Skip raw-run trajectory plots")
    p.add_argument("--skip-oral", action="store_true", help="Skip oral/model alignment plots")
    p.add_argument("--oral-data", type=Path, default=DEFAULT_ORAL_DATA, help="Oral/Task2 processed CSV")
    p.add_argument("--oral-mode", choices=("center", "region"), default="center")
    p.add_argument("--region-n-samples", type=int, default=1000)
    p.add_argument(
        "--region-stimulus-sigma",
        type=float,
        default=None,
        help="Deprecated no-op kept for compatibility; region mode now uses unweighted overlap.",
    )
    p.add_argument(
        "--distribution-model-distribution",
        default="prior",
        help="Model state for distribution-based alignment: prior or posterior",
    )
    p.add_argument(
        "--oral-model-distribution",
        default="choice_conditioned_prior",
        help="Model state for oral-based alignment",
    )
    p.add_argument("--combine-oral-equivalent", action="store_true")
    p.add_argument("--trajectory-ranks", nargs="+", type=int, help="Ranks for trajectory accuracy plots")
    p.add_argument("--posterior-ranks", nargs="+", type=int, help="Ranks for posterior trajectory plots")
    p.add_argument("--posterior-limit", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    if input_dir.name == "subjects":
        input_dir = input_dir.parent
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else input_dir / "model_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = resolve_subjects(args.subjects, args.subject_range)
    evaluator = ModelEval()
    records: list[dict[str, Any]] = []

    results = load_simulation_results(
        input_dir,
        subjects=subjects,
        eval_prediction_mode=args.eval_prediction_mode,
        window_size=args.window_size,
    )
    resolved_subjects = sorted(results)
    LOGGER.info("Loaded %d subject result(s): %s", len(resolved_subjects), resolved_subjects)

    if not args.skip_basic:
        run_basic_plots(
            evaluator=evaluator,
            results=results,
            output_dir=output_dir,
            subjects=subjects,
            window_size=args.window_size,
            records=records,
            posterior_limit=bool(args.posterior_limit),
        )

    if not args.skip_trajectory:
        run_trajectory_plots(
            evaluator=evaluator,
            input_dir=input_dir,
            output_dir=output_dir,
            records=records,
            trajectory_ranks=args.trajectory_ranks,
            posterior_ranks=args.posterior_ranks,
            eval_prediction_mode=args.eval_prediction_mode,
            posterior_limit=bool(args.posterior_limit),
        )

    oral_data = resolve_project_path(args.oral_data)
    if not args.skip_oral:
        if oral_data.is_file():
            run_oral_plots(
                evaluator=evaluator,
                results=results,
                oral_data_path=oral_data,
                output_dir=output_dir,
                subjects=subjects,
                records=records,
                oral_mode=str(args.oral_mode),
                window_size=args.window_size,
                region_n_samples=int(args.region_n_samples),
                region_stimulus_sigma=args.region_stimulus_sigma,
                distribution_model_distribution=str(args.distribution_model_distribution),
                oral_model_distribution=str(args.oral_model_distribution),
                combine_oral_equivalent=bool(args.combine_oral_equivalent),
            )
        else:
            records.append(
                {
                    "name": "oral_alignment",
                    "status": "skipped",
                    "reason": f"oral data not found: {oral_data}",
                }
            )

    manifest_path = save_manifest(output_dir, records)
    print(f"Model evaluation done -> {output_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
