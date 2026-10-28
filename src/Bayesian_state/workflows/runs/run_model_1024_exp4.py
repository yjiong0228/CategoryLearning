"""Small fixed-parameter exp4 replay and autonomous closed-loop validation.

Run from the repository root. The output directory must not exist. This entry
point executes one serial worker and keeps the recorded stimulus schedule.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from ...hypothesis_space.similarity import ContinuousSimilarity
from ...inference.dispatcher import run_inference_backend
from ...metrics.losses import choice_nll_loss
from ...metrics.probabilistic_task import probabilistic_task_metrics
from ...model.model_1024 import model_1024_runtime
from ...model.state_model import StateModel
from ...model.modules.readout.choice import read_choice_probabilities_from_model
from ...simulation.exp4_data import file_sha256, load_exp4_subject
from ...simulation.probabilistic import run_autonomous_probabilistic_learning, seeded_cognitive_config
from ...utils.seeding import stable_seed
from ...utils.subjects import deep_update


def load_config(path: Path) -> tuple[dict, dict, Path]:
    """Resolve the carrier plus explicit 1024 coordinates and smoke budget."""
    settings = yaml.safe_load(path.read_text())
    if settings.get("protocol") != "model1024_exp4_closed_loop_v1":
        raise ValueError("Expected model1024_exp4_closed_loop_v1 protocol")
    if settings.get("n_jobs") != 1:
        raise ValueError("The exp4 smoke entry point requires n_jobs=1")
    for key, upper in (("max_trials", 64), ("autonomous_paths", 8)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"Smoke {key} must be an integer in [1, {upper}]")
    base = (path.parent / settings["base_engine_config"]).resolve()
    engine = deep_update(yaml.safe_load(base.read_text()), settings["engine_overrides"])
    inference = engine["inference"]
    if (inference["backend"] != "particle_filter"
            or not 2 <= inference["particle_count"] <= 16
            or inference.get("particle_blocks", 1) != 1):
        raise ValueError("Smoke inference requires 2–16 particles in one block")
    if (engine["provenance"]["condition"] != 1
            or engine["partition"]["kwargs"]["n_cats"] != 2
            or engine["modules"]["perception_mod"]["kwargs"]["noise_mode"] != "uniform"
            or engine["oral_readout"] is not None):
        raise ValueError("exp4 uses binary behavior and measured uniform perception")
    # The PF call below uses this plain choice readout. Reject any YAML change
    # that would silently give replay and generation different choice models.
    if (engine["choice_readout"]["kwargs"] != dict(method="expectation", power=1.,
                                                    strategy_confidence_gain=0.)
            or engine["output_noise"]["kwargs"] != dict(enabled=False)):
        raise ValueError("exp4 smoke requires plain expectation readout and disabled output noise")
    return settings, engine, base


def _check_probabilities(values: np.ndarray, label: str) -> None:
    if (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
            or not np.allclose(values.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError(f"Invalid {label} probabilities")


def _verify_replay(engine: dict, case, generated) -> None:
    """Reconstruct the generated cognitive history from choices and outcomes."""
    model = StateModel(seeded_cognitive_config(engine, generated.trajectory_seed),
                       context=case.model_context())
    trajectory = generated.trajectory
    for t, stimulus in enumerate(trajectory.stimulus):
        pending = model.begin_trial(stimulus)
        probabilities = read_choice_probabilities_from_model(model, pending.perceived_stimulus)
        for actual, expected in ((pending.prior, trajectory.prior[t]),
                                 (pending.beta, trajectory.beta[t]),
                                 (pending.perceived_stimulus, trajectory.perceived_stimulus[t]),
                                 (probabilities, trajectory.cognitive_probabilities[t])):
            np.testing.assert_array_equal(actual, expected)
        posterior, _, _ = model.complete_trial(int(trajectory.choices[t]), trajectory.feedback[t])
        np.testing.assert_array_equal(posterior, trajectory.posterior[t])


def run(config: Path, output: Path) -> dict:
    """Validate and save a small run; all scores use fixed example parameters."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    settings, engine, base_path = load_config(config)
    input_path = (config.parent / settings["input_config"]).resolve()
    case = load_exp4_subject(input_path, settings["subject"])
    n = settings["max_trials"]
    if n > len(case.frame):
        raise ValueError("Requested prefix exceeds the observed stimulus schedule")
    arrays = case.arrays
    stimulus, choices, feedback = arrays.stimulus[:n], arrays.choices[:n], arrays.feedback[:n]
    q = arrays.target_probs[:n]
    root = Path(__file__).resolve().parents[4]
    core = root / "src/Bayesian_state"
    sources = [config.resolve(), input_path, base_path, Path(__file__).resolve()]
    for folder in ("model", "inference", "hypothesis_space", "metrics", "simulation", "utils"):
        sources.extend(sorted((core / folder).rglob("*.py")))
    source_hashes = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): file_sha256(p)
                     for p in sources}
    output.mkdir(parents=True, exist_ok=False)
    inference_seed = stable_seed(dict(seed_role="exp4_observed_filter", seed=settings["seed"],
                                     subject=case.subject_id))
    observed_kwargs = dict(engine_config=engine, subject_id=case.subject_id, condition=1,
                           stimulus=stimulus, choices=choices, feedback=feedback,
                           inference_seed=inference_seed,
                           processed_data_dir=case.dataset_paths["processed_dir"],
                           dataset_paths=case.dataset_paths)
    old_cache = ContinuousSimilarity.RUNTIME_CACHE_DIR
    try:
        with TemporaryDirectory(prefix="exp4_similarity_") as cache, model_1024_runtime():
            ContinuousSimilarity.RUNTIME_CACHE_DIR = Path(cache)
            observed = run_inference_backend(**observed_kwargs)
            repeated = run_inference_backend(**observed_kwargs)
            probabilities = observed.observation_probabilities["prior_t"]
            np.testing.assert_array_equal(probabilities, repeated.observation_probabilities["prior_t"])
            _check_probabilities(probabilities, "observed pre-choice")
            generated = []
            for index in range(settings["autonomous_paths"]):
                trajectory_seed = stable_seed(dict(seed_role="exp4_autonomous", seed=settings["seed"],
                                                   subject=case.subject_id, path=index))
                result = run_autonomous_probabilistic_learning(
                    engine_config=engine, subject_id=case.subject_id, stimulus=stimulus,
                    target_probabilities=q, trajectory_seed=trajectory_seed,
                    processed_data_dir=case.dataset_paths["processed_dir"], dataset_paths=case.dataset_paths)
                for field in ("cognitive_probabilities", "observed_probabilities", "prior", "posterior"):
                    _check_probabilities(getattr(result.trajectory, field), field)
                if not np.isfinite(result.trajectory.beta).all():
                    raise ValueError("Nonfinite generated precision")
                np.testing.assert_array_equal(result.trajectory.feedback,
                                               result.feedback_uniforms < result.selected_probabilities)
                _verify_replay(engine, case, result)
                generated.append(result)
    finally:
        ContinuousSimilarity.RUNTIME_CACHE_DIR = old_cache

    with (output / "observed.npz").open("xb") as stream:
        np.savez_compressed(stream, stimulus=stimulus, choices=choices, feedback=feedback,
                            pre_choice_probabilities=probabilities,
                            pre_choice_ess=observed.pre_choice_ess, post_choice_ess=observed.post_choice_ess,
                            resampled=observed.resampled)
    with (output / "task_metadata.npz").open("xb") as stream:
        np.savez_compressed(stream, target_probabilities=q,
                            trial_ids=case.frame[["iSession", "iBlock", "iTrial"]].to_numpy(dtype=int)[:n],
                            source_rows=case.frame.source_row.to_numpy(dtype=int)[:n],
                            stimulus_ids=case.frame.stiID.to_numpy(dtype=int)[:n])
    summaries = []
    for index, result in enumerate(generated):
        fields = {key: getattr(result.trajectory, key) for key in (
            "stimulus", "perceived_stimulus", "choices", "feedback", "cognitive_probabilities",
            "observed_probabilities", "prior", "posterior", "beta")}
        with (output / f"autonomous_{index:03d}.npz").open("xb") as stream:
            np.savez_compressed(stream, **fields, selected_reward_probabilities=result.selected_probabilities,
                                feedback_uniforms=result.feedback_uniforms)
        summaries.append(dict(path=index, trajectory_seed=result.trajectory_seed,
                              choice_seed=result.trajectory.choice_seed, feedback_seed=result.feedback_seed,
                              metrics=result.metrics))
    metrics = probabilistic_task_metrics(choices=choices, feedback=feedback, target_probabilities=q)
    metrics["choice_nll_per_trial"] = choice_nll_loss(dict(
        pred_category_probs=probabilities, observed_choice_index=choices - 1,
        valid_trial_mask=np.ones(n, dtype=bool)))
    report = dict(protocol=settings["protocol"], status="closed_loop_smoke_passed",
                  scientific_status="fixed_example_interface_validation",
                  model_runtime="model_1024_belief_slot_outcome_error_ADD10_STICKY",
                  settings=settings, resolved_engine=engine, subject_metadata=case.metadata(),
                  executed_trials=n, inference_seed=inference_seed, observed_metrics=metrics,
                  autonomous=summaries, checks=dict(observed_seed_repeat_exact=True,
                  generated_history_replay_exact=True, probabilities_valid=True,
                  feedback_matches_draws=True),
                  conditioning=dict(observed="pre_choice_given_actual_past_choices_and_feedback",
                                    autonomous="pre_choice_given_own_generated_history",
                                    parameters="fixed_example_not_estimated"),
                  protocol_scope="recorded_stimulus_prefix_fixed_length_session_state_carried",
                  unimplemented=["adaptive_stimulus_schedule", "task_stopping_rule",
                                 "instruction_conditioned_uncertainty_prior"],
                  code_and_config_sha256=source_hashes,
                  git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                          capture_output=True, text=True).stdout.strip(),
                  versions={"python": platform.python_version(), **{
                      name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "PyYAML")}},
                  files_sha256={p.name: file_sha256(p) for p in sorted(output.iterdir())})
    with (output / "manifest.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps({key: report[key] for key in ("status", "executed_trials", "checks")}))


if __name__ == "__main__":
    main()
