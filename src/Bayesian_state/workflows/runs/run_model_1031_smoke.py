"""Small, serial end-to-end check of the Model 1031 experience candidate.

Uses either synthetic binary-feedback stimuli or one subject's observed prefix.
Parameters are fixed examples. This command does not optimize or validate oral
alignment. Existing output directories are rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy
import yaml

from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...model import ModelContext, StateModel
from ...model.modules.readout.choice import resolve_choice_readout_config, resolve_output_noise_config
from ...simulation.data import SubjectTrialDataLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=int, choices=(1, 2), default=1)
    parser.add_argument("--subject-id", type=int, help="omit for synthetic stimuli")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1031)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/exp123/processed/Task2_processed.csv"))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(args) -> dict:
    if not 4 <= args.trials <= 128 or not 2 <= args.particles <= 64:
        raise ValueError("Smoke budget: 4..128 trials, 2..64 particles, one serial worker")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    started = perf_counter()
    config_path = args.config or Path(f"configs/exp123/model_struct/model_1031_cond{args.condition}_experience_v1.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    if cfg["provenance"]["condition"] != args.condition:
        raise ValueError("Configuration condition does not match requested condition")
    cfg["modules"]["hypo_transitions_mod"]["kwargs"]["module_seed"] = args.seed
    data_info = {}
    context = ModelContext(condition=args.condition, subject_id=args.subject_id)
    if args.subject_id is None:
        cfg["modules"]["perception_mod"]["kwargs"] = dict(features=4, mean=[0.] * 4, std=[0.] * 4, module_seed=args.seed)
        stimulus = np.random.default_rng(args.seed).uniform(.05, .95, (args.trials, 4))
        categories = 1 + (stimulus[:, 0] > .5).astype(int)
        if args.condition == 2:
            categories += 2 * (stimulus[:, 1] > .5).astype(int)
        observed_choices = observed_feedback = None
        data_info["source"] = "synthetic_axis_task"
    else:
        # Shared loader preserves original row order and its missing-trial policy.
        loader = SubjectTrialDataLoader(cfg)
        loader.prepare_data(args.data)
        frame = loader._get_subject_frame(args.subject_id, 1.)
        if loader._get_condition_value(frame) != args.condition:
            raise ValueError("Subject condition does not match requested condition")
        arrays = loader._extract_arrays(frame, max_trials=args.trials)
        if arrays.categories is None:
            raise ValueError("Autonomous check requires experimenter category labels")
        stimulus, categories = arrays.stimulus, arrays.categories
        observed_choices, observed_feedback = arrays.choices, arrays.feedback
        if not np.array_equal(observed_feedback, (observed_choices == categories).astype(float)):
            raise ValueError("Recorded binary feedback disagrees with choice/category coordinates")
        cfg["modules"]["perception_mod"].setdefault("kwargs", {})["module_seed"] = args.seed
        data_info = dict(source=str(args.data), sha256=sha256(args.data.read_bytes()).hexdigest(),
                         retained_trial_keys=frame.iloc[:len(stimulus)][[c for c in ("iSession", "iRun", "iBlock", "iTrial") if c in frame]].to_dict("records"))
    generator = StateModel(cfg, context=context)
    generated = generator.generate_step_by_step(
        stimulus, lambda t, choice: float(choice == categories[t]), choice_seed=args.seed + 1)
    replay = StateModel(cfg, context=context)
    max_difference = 0.
    for t, x in enumerate(stimulus):
        replay.begin_trial(x)
        prediction = replay.predict_choice(choices=generated.choices[:t], feedback=generated.feedback[:t],
            choice_readout_config=resolve_choice_readout_config(None, cfg),
            output_noise_config=resolve_output_noise_config(None, cfg), rng=np.random.default_rng(args.seed + 1),
            sticky_state={}, post_error_lapse_state=0.)
        max_difference = max(max_difference, float(np.max(np.abs(prediction.observed_probabilities - generated.observed_probabilities[t]))))
        replay.complete_trial(int(generated.choices[t]), float(generated.feedback[t]))
    if max_difference != 0. or replay.partition_model.nodes != generator.partition_model.nodes:
        raise AssertionError("Autonomous/replayed behavior or representation states disagree")
    if observed_choices is None:
        observed_choices, observed_feedback = generated.choices, generated.feedback
    observed_states = {"predictive": [], "filtered": []}
    def observer(trial, phase, values):
        observed_states[phase].append({key: value.copy() for key, value in values.items()})
    result = run_state_model_particle_filter(
        engine_config=cfg, subject_id=args.subject_id or 0, condition=args.condition,
        stimulus=stimulus, choices=observed_choices, feedback=observed_feedback,
        particle_count=args.particles, choice_readout_power=1., filter_seed=args.seed + 2,
        state_observer=observer, observe_decision_basis=True)
    probabilities = result.marginal_probabilities
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.):
        raise AssertionError("Invalid particle prediction probabilities")
    nll = -np.log(np.maximum(probabilities[np.arange(len(stimulus)), observed_choices - 1], np.finfo(float).tiny)).sum()
    files = [Path(__file__), config_path,
        Path("src/Bayesian_state/model/model_1031.py"), Path("src/Bayesian_state/model/config.py"),
        Path("src/Bayesian_state/model/assembly.py"), Path("src/Bayesian_state/model/state_model.py"),
        Path("src/Bayesian_state/model/engine.py"), Path("src/Bayesian_state/model/modules/beta.py"),
        Path("src/Bayesian_state/model/modules/memory.py"), Path("src/Bayesian_state/model/modules/perception.py"),
        Path("src/Bayesian_state/model/modules/readout/choice.py"),
        Path("src/Bayesian_state/model/modules/hypothesis_transition/experience_workspace.py"),
        Path("src/Bayesian_state/hypothesis_space/observation_model/experience_workspace.py"),
        Path("src/Bayesian_state/inference/backends/particle_filter.py")]
    report = dict(status="implementation_smoke_passed_not_fitted", condition=args.condition,
        subject_id=args.subject_id, trials=len(stimulus), particles=args.particles, workers=1,
        seed=args.seed, data=data_info, autonomous_replay_max_probability_error=max_difference,
        negative_log_likelihood=float(nll), autonomous_correct_fraction=float(generated.feedback.mean()),
        proposal_counts=dict(Counter(event["proposal"] for event in generated.transition_log)),
        representation_groups=result.metadata["representation_coordinates"]["groups"],
        filtered_execution_mean=result.state_probabilities["filtered_representation_execution"].mean(axis=0).tolist(),
        seconds=perf_counter() - started,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, yaml=yaml.__version__),
        source_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p.relative_to(Path.cwd()) if p.is_absolute() else p): sha256(p.read_bytes()).hexdigest() for p in files})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez_compressed(args.output / "behavior.npz", stimulus=stimulus, observed_choices=observed_choices,
        observed_feedback=observed_feedback, predicted_probabilities=probabilities,
        generated_choices=generated.choices, generated_feedback=generated.feedback,
        generated_probabilities=generated.observed_probabilities,
        **{k: v for k, v in result.state_probabilities.items() if "representation" in k})
    for phase, states in observed_states.items():
        np.savez_compressed(args.output / f"{phase}_particle_contents.npz",
                            **{key: np.stack([item[key] for item in states]) for key in states[0]})
    (args.output / "generated_representations.json").write_text(json.dumps(
        dict(pre_choice=[step["representations_pre"] for step in generated.step_log],
             transitions=generated.transition_log), indent=2))
    groups = ", ".join(f"{name}: {value:.1%}" for name, value in zip(report["representation_groups"], report["filtered_execution_mean"]))
    (args.output / "report.md").write_text(
        f"# Model 1031 小规模接入检查\n\n"
        f"条件{args.condition}，被试{args.subject_id or '模拟'}，{len(stimulus)}题、{args.particles}粒子、单进程，耗时{report['seconds']:.1f}秒。\n\n"
        "配置使用开发示例值。本轮检查计算过程；个体参数拟合、数值收敛、状态恢复与口述对齐仍待研究。\n\n"
        f"自主生成后逐题重放，选择概率最大差为{max_difference:g}，最终表示内容完全一致。粒子预测均有限且归一化。\n\n"
        f"实际输入选择的负对数似然为{nll:.4f}；该短记录得分用于排查接口。本次单条自主路径正确率为{generated.feedback.mean():.1%}。\n\n"
        f"观察当前选择、尚未接收当前反馈时，各表示承担执行的平均概率：{groups}。\n\n"
        "`predictive_particle_contents.npz`保存选择前粒子内容，`filtered_particle_contents.npz`保存选择后、反馈前的内容及权重。"
        "`representation_kind`中−1为空位、0为初始未确定、1为模板、2为目录规则；模板保留4维为具体示例、少于4维为局部概括。"
        "`representation_uid`仅在同一祖先路径内标识内容。跨粒子比较需结合特征、类别信念与目录编号。\n\n"
        "`generated_representations.json`记录自主路径每题的表示及来源，`summary.json`记录种子、版本、输入及源码哈希。口述资料未进入本次计算。\n")
    return report


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({key: report[key] for key in ("status", "trials", "particles", "seconds", "autonomous_replay_max_probability_error")}, indent=2))


if __name__ == "__main__":
    main()
