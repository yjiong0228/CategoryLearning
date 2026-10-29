"""Serial, bounded Exp5 validation of observed filtering and autonomous learning.

The entry point uses fixed example parameters and a recorded stimulus prefix.
It validates the generated-history replay and exercises state recovery on one
known simulated path. Recovery scores describe this tiny interface check only.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import yaml

from ...inference.dispatcher import run_inference_backend
from ...model.config import ModelConfig
from ...model.model_1024 import model_1024_runtime
from ...model.state_model import StateModel
from ...model.modules.base_module import ModuleRole
from ...model.modules.readout.choice import read_choice_probabilities_from_model
from ...simulation.exp5_data import load_exp5_subject, file_sha256, TRIAL_KEYS
from ...simulation.probabilistic import run_autonomous_probabilistic_learning, seeded_cognitive_config
from ...utils.seeding import stable_seed


def load_config(path: Path) -> tuple[dict, dict, Path]:
    """Resolve task settings and enforce a small serial interface-check budget."""
    settings = yaml.safe_load(path.read_text())
    if settings.get("protocol") != "model1024_exp5_closed_loop_v1" or settings.get("n_jobs") != 1:
        raise ValueError("Expected model1024_exp5_closed_loop_v1 with n_jobs=1")
    for key, maximum in (("max_trials", 64), ("autonomous_paths", 8)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"Smoke {key} must be an integer in [1, {maximum}]")
    engine_path = (path.parent / settings["engine_config"]).resolve()
    engine = ModelConfig.from_mapping(yaml.safe_load(engine_path.read_text())).to_dict()
    if engine.get("provenance", {}).get("task") != "exp5_rule1":
        raise ValueError("Smoke engine must be the Exp5 Rule1 baseline")
    inference = engine["inference"]
    particles = inference["particle_count"]
    if (inference["backend"] != "particle_filter" or isinstance(particles, bool)
            or not isinstance(particles, int) or not 2 <= particles <= 16
            or inference.get("particle_blocks", 1) != 1):
        raise ValueError("Smoke inference requires 2–16 particles in one block")
    return settings, engine, engine_path


def _probabilities(values: np.ndarray, label: str, *, normalized: bool = True) -> None:
    if (not np.isfinite(values).all() or np.any((values < 0) | (values > 1 + 1e-12))
            or normalized and not np.allclose(values.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError(f"Invalid {label} probabilities")


def verify_generated_history(engine: dict, case, generated) -> tuple[np.ndarray, np.ndarray]:
    """Replay the saved choices/outcomes through the same cognitive transition."""
    model = StateModel(seeded_cognitive_config(engine, generated.trajectory_seed), context=case.model_context())
    trajectory = generated.trajectory
    active = np.zeros_like(trajectory.prior, dtype=bool)
    executed = []
    for t, stimulus in enumerate(trajectory.stimulus):
        pending = model.begin_trial(stimulus)
        probabilities = read_choice_probabilities_from_model(model, pending.perceived_stimulus)
        transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        active[t, pending.log["active_indices"]] = True
        executed.append(transition.executed_hypothesis)
        expected_active = np.zeros(model.engine.set_size, dtype=bool)
        expected_active[trajectory.step_log[t]["active_indices"]] = True
        np.testing.assert_array_equal(active[t], expected_active)
        assert executed[-1] == trajectory.transition_log[t]["executed_hypothesis"]
        for actual, expected in ((pending.prior, trajectory.prior[t]), (pending.beta, trajectory.beta[t]),
                                 (pending.perceived_stimulus, trajectory.perceived_stimulus[t]),
                                 (probabilities, trajectory.cognitive_probabilities[t])):
            np.testing.assert_array_equal(actual, expected)
        posterior, _, _ = model.complete_trial(int(trajectory.choices[t]), trajectory.feedback[t])
        np.testing.assert_array_equal(posterior, trajectory.posterior[t])
    return active, np.asarray(executed, dtype=int)


def run(config: Path, output: Path) -> dict:
    """Write a new validation bundle, with no optimization or oral analysis."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    settings, engine, engine_path = load_config(config)
    input_path = (config.parent / settings["input_config"]).resolve()
    case = load_exp5_subject(input_path, settings["subject"])
    n = settings["max_trials"]
    if n > len(case.arrays.choices):
        raise ValueError("Requested prefix exceeds the eligible analysis sequence")
    stimulus, choices, feedback = [getattr(case.arrays, key)[:n] for key in ("stimulus", "choices", "feedback")]
    categories = case.arrays.categories[:n]
    q = np.eye(2)[categories - 1]
    root = Path(__file__).resolve().parents[4]
    sources = [config.resolve(), input_path, engine_path, Path(__file__).resolve()]
    for folder in ("model", "hypothesis_space", "inference", "simulation", "utils", "metrics"):
        sources.extend(sorted((root / "src/Bayesian_state" / folder).rglob("*.py")))
    hashes = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): file_sha256(p) for p in sources}
    seed = stable_seed(dict(role="exp5_observed_filter", seed=settings["seed"], subject=case.subject_id))
    kwargs = dict(engine_config=engine, subject_id=case.subject_id, condition=1, stimulus=stimulus,
                  choices=choices, feedback=feedback, inference_seed=seed)
    with model_1024_runtime():
        observed = run_inference_backend(**kwargs)
        repeated = run_inference_backend(**kwargs)
        probabilities = observed.observation_probabilities["prior_t"]
        np.testing.assert_array_equal(probabilities, repeated.observation_probabilities["prior_t"])
        np.testing.assert_array_equal(observed.marginal_active_probability, repeated.marginal_active_probability)
        _probabilities(probabilities, "pre-choice")
        generated = []
        truths = []
        for index in range(settings["autonomous_paths"]):
            trajectory_seed = stable_seed(dict(role="exp5_autonomous", seed=settings["seed"],
                                               subject=case.subject_id, path=index))
            result = run_autonomous_probabilistic_learning(
                engine_config=engine, subject_id=case.subject_id, stimulus=stimulus,
                target_probabilities=q, trajectory_seed=trajectory_seed)
            for key in ("cognitive_probabilities", "observed_probabilities", "prior", "posterior"):
                _probabilities(getattr(result.trajectory, key), key)
            beta = result.trajectory.beta
            true_states = verify_generated_history(engine, case, result)
            active_beta = beta[true_states[0]]
            if not np.isfinite(beta).all() or np.any((active_beta < .1) | (active_beta > 25.)):
                raise ValueError("Generated precision exceeds registered finite bounds")
            np.testing.assert_array_equal(result.trajectory.feedback, result.trajectory.choices == categories)
            truths.append(true_states)
            generated.append(result)
        # Independent inference randomness: the simulator's state/RNG is never supplied to the filter.
        recovery_seed = stable_seed(dict(role="exp5_recovery_filter", seed=settings["seed"]))
        truth = generated[0].trajectory
        recovered = run_inference_backend(**dict(kwargs, choices=truth.choices, feedback=truth.feedback,
                                                 inference_seed=recovery_seed))
        recovered_active = recovered.marginal_active_probability
        _probabilities(recovered_active, "recovered active", normalized=False)
        if recovered_active.shape != truths[0][0].shape:
            raise ValueError("State recovery changed rule/trial coordinates")
    if hashes != {key: file_sha256(root / key) for key in hashes}:
        raise ValueError("Code/configuration changed during the smoke run")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "observed.npz").open("xb") as stream:
        np.savez_compressed(stream, stimulus=stimulus, choices=choices, feedback=feedback,
                            pre_choice_probabilities=probabilities,
                            pre_choice_active_probability=observed.marginal_active_probability,
                            pre_choice_mean_belief=observed.marginal_hypothesis_prior,
                            pre_choice_ess=observed.pre_choice_ess, post_choice_ess=observed.post_choice_ess,
                            resampled=observed.resampled)
    with (output / "task_metadata.npz").open("xb") as stream:
        np.savez_compressed(stream, categories=categories,
                            trial_ids=case.analysis_frame[TRIAL_KEYS].to_numpy(dtype=int)[:n])
    summaries = []
    for index, (result, (active, executed)) in enumerate(zip(generated, truths)):
        fields = {key: getattr(result.trajectory, key) for key in (
            "stimulus", "perceived_stimulus", "choices", "feedback", "cognitive_probabilities",
            "observed_probabilities", "prior", "posterior", "beta")}
        with (output / f"autonomous_{index:03d}.npz").open("xb") as stream:
            np.savez_compressed(stream, **fields, active=active, executed=executed)
        summaries.append(dict(path=index, trajectory_seed=result.trajectory_seed,
                              choice_seed=result.trajectory.choice_seed, feedback_seed=result.feedback_seed,
                              accuracy=float(result.trajectory.feedback.mean()),
                              capacity_min=int(active.sum(axis=1).min()), capacity_max=int(active.sum(axis=1).max())))
    with (output / "recovery_smoke.npz").open("xb") as stream:
        np.savez_compressed(stream, true_pre_choice_active=truths[0][0], pre_choice_active_probability=recovered_active,
                            pre_choice_probabilities=recovered.observation_probabilities["prior_t"])
    report = dict(protocol=settings["protocol"], status="closed_loop_smoke_passed",
                  scientific_status="fixed_example_interface_validation_not_fitted_or_recovery_validated",
                  settings=settings, resolved_engine=engine, subject_metadata=case.metadata(),
                  executed_trials=n, inference_seed=seed,
                  observed=dict(accuracy=float(feedback.mean()), choice_nll_per_trial=float(
                      -np.log(probabilities[np.arange(n), choices - 1]).mean())),
                  autonomous=summaries,
                  recovery_smoke=dict(paths=1, seed=recovery_seed,
                                      pre_choice_active_brier=float(np.mean((recovered_active-truths[0][0])**2)),
                                      acceptance="interface_only_no_recovery_accuracy_claim"),
                  checks=dict(observed_seed_repeat_exact=True, generated_history_replay_exact=True,
                              probabilities_valid=True, feedback_matches_generated_choice=True,
                              recovery_coordinates_valid=True),
                  conditioning=dict(observed="pre_choice_given_actual_past_choices_and_feedback",
                                    autonomous="own_generated_history",
                                    recovery="pre_choice_given_generated_past_choices_and_feedback",
                                    parameters="fixed_example_not_estimated"),
                  unimplemented=["overnight_state_transition", "oral_based_stopping", "historical_sampling_program",
                                 "parameter_estimation", "formal_state_recovery", "oral_validation"],
                  code_and_config_sha256=hashes,
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
    parser.add_argument("--config", type=Path, default=Path("configs/exp5/model1024_smoke.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps({key: report[key] for key in ("status", "executed_trials", "checks")}))


if __name__ == "__main__":
    main()
