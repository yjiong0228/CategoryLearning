"""Offline comparison and explicitly authorized short Python/native PF screens."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from generate_1031_reference import ROOT, activate_reference, load_engine, provenance


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def parameter_hash(parameters: dict) -> str:
    return sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_file_hash(path: Path, expected: str) -> None:
    if file_hash(path) != expected:
        raise ValueError(f"File changed; results cannot be accepted: {path}")


def guarded_native(command: list, executable: Path, digest: str, timeout: int):
    assert_file_hash(executable, digest)
    try:
        return subprocess.run(command, cwd=ROOT, env=os.environ.copy(), text=True,
                              capture_output=True, timeout=timeout)
    finally:
        assert_file_hash(executable, digest)


def checked_probabilities(values, shape):
    import numpy as np
    p = np.asarray(values, float)
    if p.shape != shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError(f"Invalid probabilities, expected {shape}, got {p.shape}")
    return p


def checked_score_mask(values, trials: int):
    import numpy as np
    mask = np.asarray(values)
    if (mask.shape != (trials,) or mask.dtype not in (np.dtype(bool), np.dtype("uint8"))
            or not np.isin(mask, [0, 1]).all()):
        raise ValueError("score_mask must be a binary bool/uint8 vector of the expected length")
    return mask.astype(bool, copy=False)


def native_probabilities(path: Path, keep, choices, mask, cats: int):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        native_mask = checked_score_mask(saved["score_mask"], len(keep))
        if native_mask[~keep].any() or not np.array_equal(native_mask[keep], mask):
            raise ValueError("Native score mask differs from prespecified first-trial policy")
        if saved["category_probabilities"].shape != (len(keep), cats) or saved["choice_probabilities"].shape != (len(keep),):
            raise ValueError("Native probability arrays do not align with the raw trial prefix")
        p = checked_probabilities(saved["category_probabilities"][keep], (len(choices), cats))
        chosen = checked_probabilities(saved["choice_probabilities"][keep], (len(choices),))
        np.testing.assert_allclose(chosen, p[np.arange(len(choices)), choices - 1], atol=1e-9, rtol=0)
        return p.copy()


def summarize(python, native, choices, mask, thresholds: dict) -> dict:
    """Independent-seed MCSE, including trial correlation within each seed."""
    import numpy as np
    a, b = np.asarray(python), np.asarray(native)
    if a.ndim != 3 or a.shape != b.shape or a.shape[0] < 16:
        raise ValueError("Need >=16 independent seeds per language with identical [seed,trial,category] shape")
    mask = checked_score_mask(mask, a.shape[1])
    if not mask.any():
        raise ValueError("Invalid common score mask")
    for bank in (a, b):
        checked_probabilities(bank, a.shape)
        np.testing.assert_allclose(bank.sum(axis=-1), 1., atol=1e-9, rtol=0)
    chosen = [bank[:, np.arange(len(choices)), choices - 1][:, mask] for bank in (a, b)]
    if any(np.any(p <= 0) for p in chosen):
        raise ValueError("Zero chosen probability: log scores are infinite; no silent floor")
    metrics = {}

    def record(name, x, y, absolute, influences=None):
        means = [x.mean(axis=0), y.mean(axis=0)]
        draws = [x, y] if influences is None else influences
        se = np.sqrt(sum(d.var(axis=0, ddof=1) / len(d) for d in draws))
        error = np.abs(means[0] - means[1])
        limit = absolute + thresholds["mcse_multiplier"] * se
        metrics[name] = dict(python_mean=means[0].tolist(), native_mean=means[1].tolist(),
            absolute_error=error.tolist(), combined_mcse=se.tolist(), limit=limit.tolist(),
            passed=bool(np.all(error <= limit)))

    record("category_probabilities", a[:, mask], b[:, mask], thresholds["probability_abs"])
    record("chosen_probabilities", *chosen, thresholds["probability_abs"])
    per_seed = []
    for p in chosen:
        per_seed.append(dict(mean_probability=p.mean(axis=1).tolist(),
            nll=(-np.log(p).sum(axis=1)).tolist(), mean_nll=(-np.log(p).mean(axis=1)).tolist()))
    record("mean_nll", *[-np.log(p).mean(axis=1) for p in chosen], thresholds["mean_log_abs"])
    # Delta-method influence for log(mean_seed p_t), pooled over scored trials.
    logmeans = [np.log(p.mean(axis=0)).mean() for p in chosen]
    influence = [((p - p.mean(axis=0)) / p.mean(axis=0)).mean(axis=1) for p in chosen]
    record("mean_log_mean_probability", *[np.full(len(a), v) for v in logmeans],
           thresholds["mean_log_abs"], influences=influence)
    return dict(passed=all(m["passed"] for m in metrics.values()), scored_trials=int(mask.sum()),
                per_seed=dict(python=per_seed[0], native=per_seed[1]), metrics=metrics,
                nll_of_seed_mean_probability=dict(python=float(-logmeans[0] * mask.sum()),
                                                 native=float(-logmeans[1] * mask.sum())))


def render_template(value, replacements: dict):
    if isinstance(value, str) and value in replacements:
        return deepcopy(replacements[value])
    if isinstance(value, dict):
        return {k: render_template(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, replacements) for v in value]
    return value


def validate_template(template: dict) -> None:
    text = json.dumps(template)
    for token in ("@SEED@", "@PARTICLES@", "@PARAMETERS@", "@SUBJECT@"):
        if f'"{token}"' not in text:
            raise ValueError(f"Native template must expose whole-value token {token}")
    if not any(f'"{token}"' in text for token in ("@DATA@", "@DATA_ROOT@")):
        raise ValueError("Native template must expose @DATA@ or @DATA_ROOT@")


def validate_input_files(data: Path, reference_root: Path, native_data_root: Path) -> dict:
    """Audit both perception summaries and the behavioral table before any PF run."""
    original = reference_root / "data/exp123/processed"
    native = native_data_root / "exp123/processed"
    hashes = {}
    for name in ("Task2_processed.csv", "Task1b_errorsummary_24.csv", "Task1b_errorsummary_72.csv"):
        reference_hash = sha256((original / name).read_bytes()).hexdigest()
        native_hash = sha256((native / name).read_bytes()).hexdigest()
        if native_hash != reference_hash:
            raise ValueError(f"Native/reference input SHA256 mismatch: {name}")
        hashes[name] = dict(reference=reference_hash, native=native_hash)
    if sha256(data.read_bytes()).hexdigest() != hashes["Task2_processed.csv"]["reference"]:
        raise ValueError("--data differs from the original table used to resolve subject perception")
    return hashes


def prepare_subject(root: Path, data: Path, cfg: dict, subject: int, condition: int, trials: int):
    import numpy as np
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
    processed = root / "data/exp123/processed"
    loader = SubjectTrialDataLoader(cfg)
    loader.prepare_data(data)
    frame = loader._get_subject_frame(subject, 1.)
    if loader._get_condition_value(frame) != condition:
        raise ValueError("Unexpected subject condition")
    arrays = loader._extract_arrays(frame, max_trials=trials)
    if len(arrays.choices) != trials:
        raise ValueError("Insufficient valid trials")
    model = StateModel(cfg, context=ModelContext(condition=condition, subject_id=subject,
                                                processed_data_dir=processed))
    perception = model.engine.get_module(ModuleRole.PERCEPTION)
    raw = loader.learning_data.loc[loader.learning_data["iSub"] == subject]
    # Native retains omitted rows up to the Nth answered trial; Python skips them.
    prefix = raw.iloc[:raw.index.get_loc(frame.index[trials - 1]) + 1]
    keep = prefix["miss"].eq(0).to_numpy() if "miss" in prefix else np.ones(len(prefix), bool)
    if not np.array_equal(prefix.index[keep], frame.index[:trials]):
        raise ValueError("Native raw-prefix/Python retained-trial alignment differs")
    identifiers = [name for name in ("iSession", "iRun", "iBlock", "iTrial") if name in frame]
    metadata = dict(subject=subject, condition=condition, valid_trials=trials,
        native_rows=len(prefix), native_valid_rows=keep.tolist(), columns=identifiers,
        retained=frame.iloc[:trials][identifiers].to_numpy().tolist(),
        omission_policy="original loader skips miss=1; align native retained raw rows before comparison",
        perception=dict(subject=perception.subject_id, mode=perception.noise_mode,
            mean=perception.mean.tolist(), std=perception.std.tolist(),
            uniform_half_range=None if perception.uniform_half_range is None else perception.uniform_half_range.tolist(),
            processed_data_dir=str(processed)))
    return arrays, metadata, keep


def audit_native_perception(metadata: dict, screen_root: Path) -> dict:
    import numpy as np
    subject = metadata["subject"]
    expected = metadata["perception"]
    scale = expected["uniform_half_range"] if expected["mode"] == "uniform" else expected["std"]
    checked, pending = [], []
    for path in sorted((screen_root / f"subject_{subject}").glob(f"native_seed_*/subject_{subject}/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pending.append(str(path))
            continue
        if summary["subject_id"] != subject or summary["condition"] != metadata["condition"]:
            raise AssertionError(f"Native subject/condition mismatch: {path}")
        actual = summary["perception"]
        if actual["mode"] != expected["mode"]:
            raise AssertionError(f"Native/reference perception mode mismatch: {path}")
        np.testing.assert_allclose(actual["mean"], expected["mean"], atol=1e-12, rtol=0,
                                   err_msg=f"Native/reference perception mean: {path}")
        np.testing.assert_allclose(actual["scale"], scale, atol=1e-12, rtol=0,
                                   err_msg=f"Native/reference perception scale: {path}")
        checked.append(summary["seed"])
    return dict(status="matched_available_native_summaries" if checked else "native_summary_pending",
                mode=expected["mode"], checked_seeds=checked, incomplete_json=pending)


def preflight(args) -> None:
    root = activate_reference(args.reference_root)
    data = args.data.resolve(strict=True)
    hashes = validate_input_files(data, root, args.native_data_root.resolve(strict=True))
    template = json.loads(args.native_config_template.read_text(encoding="utf-8"))
    validate_template(template)
    subjects = []
    for condition, subject in enumerate((122, 206, 314), 1):
        _, metadata, _ = prepare_subject(root, data, load_engine(root, condition), subject, condition, args.trials)
        if args.screen_root is not None:
            metadata["native_perception_audit"] = audit_native_perception(metadata, args.screen_root.resolve())
        subjects.append(metadata)
    print(json.dumps(dict(status="input_perception_preflight_only_no_particle_filter",
                          input_sha256=hashes, subjects=subjects), indent=2))


def compare_fixture(args) -> None:
    import numpy as np
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    counts = {}
    for section in ("cases", "proposals"):
        rows = expected.get(section, [])
        observed_rows = actual.get(section, [])
        bank = {case["id"]: case["expected"] for case in observed_rows}
        if len(bank) != len(observed_rows) or set(bank) != {c["id"] for c in rows}:
            raise ValueError(f"{section} IDs must match exactly, without duplicates")
        for case in rows:
            observed = bank[case["id"]]
            if set(observed) != set(case["expected"]):
                raise ValueError(f"Output keys differ: {case['id']}")
            for key, value in case["expected"].items():
                if value is None:
                    if observed[key] is not None:
                        raise AssertionError(f"{case['id']}/{key}: expected null")
                else:
                    x, y = np.asarray(value), np.asarray(observed[key])
                    if x.shape != y.shape or not np.isfinite(y).all():
                        raise AssertionError(f"{case['id']}/{key}: shape or finite-value mismatch")
                    np.testing.assert_allclose(y, x, atol=1e-9, rtol=0, err_msg=f"{case['id']}/{key}")
        counts[section] = len(rows)
    print(json.dumps(dict(passed=True, **counts, absolute_tolerance=1e-9)))


def run_screen(args) -> None:
    if not args.native_ready:
        raise ValueError("Statistical runs require parent confirmation and --native-ready")
    executable = args.native.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("--output must be a NEW directory inside the new repository")
    root = activate_reference(args.reference_root)
    import numpy as np
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.model import ModelContext, StateModel

    parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
    template = json.loads(args.native_config_template.read_text(encoding="utf-8"))
    validate_template(template)
    data = args.data.resolve(strict=True)
    native_data_root = args.native_data_root.resolve(strict=True)
    input_hashes = validate_input_files(data, root, native_data_root)
    # Validate the mapping before creating any output or running either engine.
    for condition in (1, 2, 3):
        load_engine(root, condition, parameters)
    thresholds = dict(probability_abs=args.probability_abs, mean_log_abs=args.mean_log_abs,
                      mcse_multiplier=args.mcse_multiplier)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(model="1031-unified-v2", subjects=[122, 206, 314], trials=args.trials,
        particles=32, workers=1, parameters=parameters, thresholds=thresholds,
        parameters_sha256=parameter_hash(parameters),
        seed_python=list(range(10001, 10001 + args.seeds)),
        seed_native=list(range(90001, 90001 + args.seeds)),
        data_sha256=sha256(data.read_bytes()).hexdigest(), native_sha256=sha256(executable.read_bytes()).hexdigest(),
        template=template, reference=provenance(root),
        input_sha256=input_hashes, native_data_root=str(native_data_root),
        perception_data_sha256={p.name: sha256(p.read_bytes()).hexdigest()
            for p in (root / "data/exp123/processed").glob("Task1b_errorsummary_*.csv")},
        score_policy="first retained trial excluded; all other retained prefix trials scored",
        mcse_policy="independent seed variance, trial-correlated seed means; delta method for log mean probability")
    write_json(output / "protocol.json", protocol)
    results = {}
    for condition, subject in enumerate((122, 206, 314), 1):
        cfg = load_engine(root, condition, parameters)
        arrays, metadata, native_keep = prepare_subject(root, data, cfg, subject, condition, args.trials)
        case_output = output / f"subject_{subject}"
        case_output.mkdir()
        write_json(case_output / "reference_engine.json", cfg)
        write_json(case_output / "trial_selection.json", metadata)
        mask = np.ones(args.trials, bool)
        mask[0] = False
        np.savez_compressed(case_output / "inputs.npz", stimulus=arrays.stimulus,
                            choices=arrays.choices, feedback=arrays.feedback, score_mask=mask)
        banks = [[], []]
        for index in range(args.seeds):
            seed = protocol["seed_python"][index]
            own = deepcopy(cfg)
            own["modules"]["perception_mod"].setdefault("kwargs", {})["module_seed"] = seed
            own["modules"]["hypo_transitions_mod"]["kwargs"]["module_seed"] = seed + 1000000
            # Instantiate the original assembly explicitly; PF constructs independent particles.
            StateModel(own, context=ModelContext(condition=condition, subject_id=subject,
                processed_data_dir=root / "data/exp123/processed"))
            result = run_state_model_particle_filter(engine_config=own, subject_id=subject,
                condition=condition, stimulus=arrays.stimulus, choices=arrays.choices,
                feedback=arrays.feedback, particle_count=32, choice_readout_power=1., filter_seed=seed,
                processed_data_dir=root / "data/exp123/processed")
            banks[0].append(result.marginal_probabilities)
            native_seed = protocol["seed_native"][index]
            run_dir = case_output / f"native_seed_{native_seed}"
            native_cfg = render_template(template, {"@SEED@": native_seed, "@PARTICLES@": 32,
                "@PARAMETERS@": parameters, "@CONDITION@": condition, "@SUBJECT@": subject,
                "@DATA@": str(data), "@DATA_ROOT@": str(native_data_root),
                "@RESOURCE_ROOT@": str(ROOT / "resources")})
            cfg_path = case_output / f"native_seed_{native_seed}.json"
            write_json(cfg_path, native_cfg)
            command = [str(executable), "filter", "--config", str(cfg_path), "--subjects", str(subject),
                       "--max-trials", str(args.trials), "--jobs", "1", "--output", str(run_dir)]
            completed = guarded_native(command, executable, protocol["native_sha256"], args.timeout)
            write_json(case_output / f"native_seed_{native_seed}_process.json",
                       dict(command=command, returncode=completed.returncode,
                            stdout=completed.stdout, stderr=completed.stderr,
                            verified_native_sha256=protocol["native_sha256"]))
            completed.check_returncode()
            banks[1].append(native_probabilities(run_dir / f"subject_{subject}/filter.npz",
                native_keep, arrays.choices, mask, 2 if condition == 1 else 4))
        np.savez_compressed(case_output / "probability_banks.npz", python=banks[0], native=banks[1])
        results[str(subject)] = summarize(*banks, arrays.choices, mask, thresholds)
        write_json(case_output / "comparison.json", results[str(subject)])
    assert_file_hash(executable, protocol["native_sha256"])
    summary = dict(status="statistical_screen", passed=all(v["passed"] for v in results.values()), subjects=results,
                   verified_native_sha256=protocol["native_sha256"])
    write_json(output / "comparison.json", summary)
    print(json.dumps(dict(passed=summary["passed"], output=str(output))))
    if not summary["passed"]:
        raise SystemExit(1)


def recheck_native(args) -> None:
    """Reuse a completed 32x32 Python screen; execute only its 96 native jobs."""
    if not (args.native_ready or args.validate_only):
        raise ValueError("Native recheck requires parent confirmation and --native-ready")
    import numpy as np
    screen, output = args.screen_root.resolve(strict=True), args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists() or output.is_relative_to(screen):
        raise ValueError("Recheck output must be NEW, inside this repository and outside the first screen")
    protocol_path = screen / "protocol.json"
    old = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (old["model"] != "1031-unified-v2" or old["subjects"] != [122, 206, 314]
            or (old["trials"], old["particles"], old["workers"]) != (32, 32, 1)):
        raise ValueError("Recheck is bounded to three subjects, 32 trials, 32 particles and one worker")
    for name in ("seed_python", "seed_native"):
        if len(old[name]) != 32 or len(set(old[name])) != 32 or any(type(s) is not int or s < 0 for s in old[name]):
            raise ValueError("Recheck requires exactly 32 distinct seeds in each original bank")
    if set(old["seed_python"]) & set(old["seed_native"]):
        raise ValueError("Original language seed banks overlap")
    thresholds = old["thresholds"]
    if set(thresholds) != {"probability_abs", "mean_log_abs", "mcse_multiplier"} or any(
            not math.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError("Invalid original thresholds")
    completed = json.loads((screen / "comparison.json").read_text(encoding="utf-8"))
    if completed["status"] != "statistical_screen" or set(completed["subjects"]) != {"122", "206", "314"}:
        raise ValueError("Original screen is incomplete")
    root = activate_reference(args.reference_root)
    data, parameters_path = args.data.resolve(strict=True), args.parameters.resolve(strict=True)
    native_root = Path(old["native_data_root"]).resolve(strict=True)
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    if (parameter_hash(parameters) != parameter_hash(old["parameters"])
            or parameter_hash(parameters) != old.get("parameters_sha256", parameter_hash(old["parameters"]))):
        raise ValueError("Parameters differ from the completed reference screen")
    if provenance(root) != old["reference"] or validate_input_files(data, root, native_root) != old["input_sha256"]:
        raise ValueError("Original Python source/runtime or source/native data changed")
    if file_hash(data) != old["data_sha256"]:
        raise ValueError("Behavioral data changed")
    validate_template(old["template"])
    files = [protocol_path, screen / "comparison.json", parameters_path]
    for subject in old["subjects"]:
        files.extend(screen / f"subject_{subject}" / name for name in ("probability_banks.npz", "inputs.npz",
                     "reference_engine.json", "trial_selection.json", "comparison.json"))
    snapshots = {str(path): file_hash(path) for path in files}
    subjects, bank_hashes = {}, {}
    for condition, subject in enumerate(old["subjects"], 1):
        directory = screen / f"subject_{subject}"
        cfg = load_engine(root, condition, parameters)
        arrays, metadata, keep = prepare_subject(root, data, cfg, subject, condition, 32)
        if (json.loads((directory / "reference_engine.json").read_text()) != cfg
                or json.loads((directory / "trial_selection.json").read_text()) != metadata):
            raise ValueError("Saved reference engine or retained inputs changed")
        mask = np.arange(32) != 0
        with np.load(directory / "inputs.npz", allow_pickle=False) as saved:
            for name, values in dict(stimulus=arrays.stimulus, choices=arrays.choices,
                                     feedback=arrays.feedback, score_mask=mask).items():
                np.testing.assert_array_equal(saved[name], values, err_msg=f"Saved input mismatch: {subject}/{name}")
        with np.load(directory / "probability_banks.npz", allow_pickle=False) as saved:
            python = checked_probabilities(saved["python"], (32, 32, 2 if condition == 1 else 4)).copy()
            previous = summarize(python, saved["native"], arrays.choices, mask, thresholds)
        if previous != completed["subjects"][str(subject)] or previous != json.loads((directory / "comparison.json").read_text()):
            raise ValueError("Saved probability banks disagree with completed screen scores")
        bank_hashes[str(subject)] = sha256(str(python.shape).encode() + python.astype("<f8").tobytes()).hexdigest()
        subjects[subject] = (arrays, metadata, keep, mask, python)
    for path, value in snapshots.items():
        assert_file_hash(Path(path), value)
    parameters_digest = parameter_hash(parameters)
    if args.validate_only:
        print(json.dumps(dict(status="recheck_inputs_validated_no_native_execution", jobs=96,
            reference_bank_sha256=bank_hashes, parameters_sha256=parameters_digest)))
        return
    executable = args.native.resolve(strict=True)
    digest = file_hash(executable)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(deepcopy(old), mode="recheck-native", native_sha256=digest,
        previous_native_sha256=old["native_sha256"], native_executable=str(executable),
        reference_screen=str(screen), reference_files_sha256=snapshots,
        reference_bank_sha256=bank_hashes, parameters_sha256=parameters_digest)
    write_json(output / "protocol.json", protocol)
    results = {}
    for condition, subject in enumerate(old["subjects"], 1):
        arrays, metadata, keep, mask, python = subjects[subject]
        directory = output / f"subject_{subject}"
        directory.mkdir()
        native = []
        for seed in old["seed_native"]:
            run_dir = directory / f"native_seed_{seed}"
            cfg_path = directory / f"native_seed_{seed}.json"
            cfg = render_template(old["template"], {"@SEED@": seed, "@PARTICLES@": 32,
                "@PARAMETERS@": parameters, "@CONDITION@": condition, "@SUBJECT@": subject,
                "@DATA@": str(data), "@DATA_ROOT@": str(native_root), "@RESOURCE_ROOT@": str(ROOT / "resources")})
            write_json(cfg_path, cfg)
            command = [str(executable), "filter", "--config", str(cfg_path), "--subjects", str(subject),
                       "--max-trials", "32", "--jobs", "1", "--output", str(run_dir)]
            process = guarded_native(command, executable, digest, args.timeout)
            write_json(directory / f"native_seed_{seed}_process.json", dict(command=command,
                returncode=process.returncode, stdout=process.stdout, stderr=process.stderr,
                verified_native_sha256=digest))
            process.check_returncode()
            native.append(native_probabilities(run_dir / f"subject_{subject}/filter.npz", keep,
                          arrays.choices, mask, 2 if condition == 1 else 4))
        np.savez_compressed(directory / "probability_banks.npz", python=python, native=native)
        results[str(subject)] = summarize(python, native, arrays.choices, mask, thresholds)
        write_json(directory / "comparison.json", results[str(subject)])
    for path, value in snapshots.items():
        assert_file_hash(Path(path), value)
    if provenance(root) != old["reference"] or validate_input_files(data, root, native_root) != old["input_sha256"]:
        raise ValueError("Reference source or data changed during native recheck")
    assert_file_hash(executable, digest)
    result = dict(status="statistical_screen_native_recheck", passed=all(v["passed"] for v in results.values()),
        subjects=results, verified_native_sha256=digest, reference_bank_sha256=bank_hashes,
        parameters_sha256=parameters_digest, thresholds=thresholds)
    write_json(output / "comparison.json", result)
    print(json.dumps(dict(passed=result["passed"], output=str(output))))
    if not result["passed"]:
        raise SystemExit(1)


def self_test_recheck(template: dict, thresholds: dict) -> None:
    """Exercise all 96 jobs in memory: no files, production imports or processes."""
    from contextlib import ExitStack, nullcontext
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    module = sys.modules[__name__]
    screen, output = ROOT / "mock_screen", ROOT / "mock_output"
    args = SimpleNamespace(screen_root=screen, output=output, native_ready=True, validate_only=False,
        reference_root=ROOT / "mock_reference", data=ROOT / "mock_data.csv", parameters=ROOT / "mock_params.json",
        native=ROOT / "mock.exe", timeout=1)
    reference, hashes, params = {"source_sha256": {"source": "fixed"}}, {"input": "fixed"}, {"capacity": 4}
    old = dict(model="1031-unified-v2", subjects=[122, 206, 314], trials=32, particles=32, workers=1,
        seed_python=list(range(32)), seed_native=list(range(100, 132)), parameters=params,
        thresholds=thresholds, reference=reference, input_sha256=hashes, native_sha256="old executable",
        native_data_root=str(ROOT / "mock_data"), template=template, data_sha256=sha256(b"fixed").hexdigest())
    mask = np.arange(32) != 0
    arrays = SimpleNamespace(stimulus=np.full((32, 4), .2), choices=np.ones(32, int), feedback=np.ones(32))
    text = {str(screen / "protocol.json"): old, str(args.parameters): params}
    loads, banks, scores = {}, {}, {}
    for condition, sid in enumerate(old["subjects"], 1):
        directory = screen / f"subject_{sid}"
        p = np.full((32, 32, 2 if condition == 1 else 4), 1 / (2 if condition == 1 else 4))
        banks[str(directory / "probability_banks.npz")] = dict(python=p, native=p)
        loads[str(directory / "inputs.npz")] = dict(stimulus=arrays.stimulus,
            choices=arrays.choices, feedback=arrays.feedback, score_mask=mask)
        scores[str(sid)] = summarize(p, p, arrays.choices, mask, thresholds)
        for name, value in (("reference_engine.json", {}), ("trial_selection.json", {}),
                             ("comparison.json", scores[str(sid)])):
            text[str(directory / name)] = value
    text[str(screen / "comparison.json")] = dict(status="statistical_screen", subjects=scores)
    written = {}
    with ExitStack() as stack:
        mocks = dict(activate_reference=lambda p: p, provenance=lambda p: reference,
            validate_input_files=lambda *a: hashes, load_engine=lambda *a: {},
            prepare_subject=lambda *a: (arrays, {}, np.ones(32, bool)),
            native_probabilities=lambda *a: np.full((32, a[-1]), 1 / a[-1]),
            write_json=lambda p, v: written.update({str(p): v}))
        for name, value in mocks.items():
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.object(Path, "resolve", lambda p, **kw: p))
        stack.enter_context(patch.object(Path, "exists", return_value=False))
        stack.enter_context(patch.object(Path, "mkdir"))
        stack.enter_context(patch.object(Path, "read_bytes", return_value=b"fixed"))
        stack.enter_context(patch.object(Path, "read_text", lambda p, **kw: json.dumps(text[str(p)])))
        stack.enter_context(patch.object(np, "load", lambda p, **kw: nullcontext((loads | banks)[str(p)])))
        stack.enter_context(patch.object(np, "savez_compressed"))
        stack.enter_context(patch("builtins.print"))
        process = stack.enter_context(patch.object(module, "guarded_native",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")))
        recheck_native(args)
        assert process.call_count == 96
        assert written[str(output / "comparison.json")]["passed"]
        assert written[str(output / "protocol.json")]["thresholds"] == thresholds
        assert len(written[str(output / "protocol.json")]["reference_bank_sha256"]) == 3
        for call in process.call_args_list:
            command = call.args[0]
            assert command[command.index("--jobs") + 1] == "1"
        process.reset_mock()
        for name, value in (("provenance", {"changed": True}), ("validate_input_files", {"changed": True})):
            with patch.object(module, name, return_value=value):
                try:
                    recheck_native(args)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Changed reference identity accepted")
        text[str(args.parameters)] = {"capacity": 5}
        try:
            recheck_native(args)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed cached parameters accepted")
        assert process.call_count == 0


def self_test() -> None:
    import numpy as np
    from unittest.mock import patch
    expected_hash = sha256(b"frozen").hexdigest()
    with patch.object(Path, "read_bytes", return_value=b"frozen"):
        assert_file_hash(Path("binary"), expected_hash)
    with patch.object(Path, "read_bytes", return_value=b"rebuilt"):
        try:
            assert_file_hash(Path("binary"), expected_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed executable accepted")
    with patch.object(Path, "read_bytes", side_effect=[b"frozen", b"rebuilt"]), patch.object(
            subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as process:
        try:
            guarded_native([], Path("binary"), expected_hash, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Mid-run executable rebuild accepted")
        assert process.call_count == 1
    for dtype in (bool, np.uint8):
        np.testing.assert_array_equal(checked_score_mask(np.array([0, 1, 1], dtype=dtype), 3),
                                      [False, True, True])
    for bad in (np.array([0, 2], np.uint8), np.array([0, 1], np.int64),
                np.array([0., 1.]), np.array([[False, True]])):
        try:
            checked_score_mask(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid native mask accepted")
    template = json.loads((ROOT / "tools/native_config_template.json").read_text(encoding="utf-8"))
    validate_template(template)
    rendered = render_template(template, dict({"@SEED@": 90001, "@PARTICLES@": 32,
        "@SUBJECT@": 122, "@PARAMETERS@": {"capacity": 4},
        "@DATA_ROOT@": str(ROOT.parents[1] / "data"), "@RESOURCE_ROOT@": str(ROOT / "resources")}))
    assert rendered["data_root"] == str(ROOT.parents[1] / "data")
    assert rendered["subjects"] == [122] and rendered["evaluation"]["particle_count"] == 32
    assert set(rendered) == {"dataset", "data_root", "resource_root", "subjects", "seed",
                             "parameters", "evaluation"}
    assert "@" not in json.dumps(rendered)
    legacy = deepcopy(template)
    legacy["data_root"] = "@DATA@"
    validate_template(legacy)
    try:
        validate_template(dict(template, data_root="no data token"))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing data token accepted")
    from unittest.mock import patch
    reference_root, native_root = Path("reference"), Path("native")
    data = reference_root / "data/exp123/processed/Task2_processed.csv"
    with patch.object(Path, "read_bytes", return_value=b"identical inputs"):
        assert len(validate_input_files(data, reference_root, native_root)) == 3
    def changed_measurement(path):
        return b"changed" if path == native_root / "exp123/processed/Task1b_errorsummary_72.csv" else b"identical inputs"
    with patch.object(Path, "read_bytes", changed_measurement):
        try:
            validate_input_files(data, reference_root, native_root)
        except ValueError:
            pass
        else:
            raise AssertionError("Perception input hash mismatch accepted")
    rng = np.random.default_rng(33)
    p = rng.dirichlet([2., 2.], (16, 5))
    mask = np.array([False, True, True, True, True])
    thresholds = dict(probability_abs=.05, mean_log_abs=.1, mcse_multiplier=4.)
    self_test_recheck(template, thresholds)
    assert summarize(p, p, np.ones(5, int), mask, thresholds)["passed"]
    a = np.tile([.9, .1], (16, 5, 1))
    assert not summarize(a, a[..., ::-1], np.ones(5, int), mask, thresholds)["passed"]
    try:
        summarize(p[:1], p[:1], np.ones(5, int), mask, thresholds)
    except ValueError:
        pass
    else:
        raise AssertionError("Single-seed comparison accepted")
    assert render_template({"seed": "@SEED@"}, {"@SEED@": 12}) == {"seed": 12}
    # Exercise the fixture comparator's negative paths without writing test files.
    from types import SimpleNamespace
    class MemoryJSON:
        def __init__(self, value):
            self.value = value

        def read_text(self, **kwargs):
            return json.dumps(self.value)

    for mode in ("normal", "uniform"):
        perception = dict(mode=mode, mean=[0.] * 4, std=[.2] * 4, uniform_half_range=[.1] * 4)
        metadata = dict(subject=206, condition=2, perception=perception)
        summary = dict(subject_id=206, condition=2, seed=1, perception=dict(
            mode=mode, mean=[0.] * 4, scale=[.1 if mode == "uniform" else .2] * 4))
        with patch.object(Path, "glob", return_value=[Path("summary.json")]), patch.object(
                Path, "read_text", return_value=json.dumps(summary)):
            assert audit_native_perception(metadata, ROOT)["checked_seeds"] == [1]

    fixture = dict(cases=[dict(id="test", expected=dict(p=[.2, .8], labels=None))])
    combined = dict(fixture, proposals=[dict(id="proposal", expected=dict(weights=[.4, .6]))])
    corrupted = deepcopy(combined)
    corrupted["proposals"][0]["expected"]["weights"] = [.5, .5]
    try:
        compare_fixture(SimpleNamespace(fixture=MemoryJSON(combined), actual=MemoryJSON(corrupted)))
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt proposal output accepted")
    for bad in (dict(p=[.3, .7], labels=None), dict(p=[[.2, .8]], labels=None),
                dict(p=[.2, .8], labels=[]), dict(p=[float("nan"), .8], labels=None)):
        try:
            compare_fixture(SimpleNamespace(fixture=MemoryJSON(fixture),
                actual=MemoryJSON(dict(cases=[dict(id="test", expected=bad)]))))
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupt fixture accepted")
    assert not any(name.startswith("src.Bayesian_state") for name in sys.modules)
    print("self-test passed: mocked 96-job recheck, source/data/parameter rejection, binary rebuild guards, masks, templates, banks; no production imports or native execution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-test")
    audit = sub.add_parser("preflight", help="audit data/perception only; never run native or Python PF")
    audit.add_argument("--reference-root", required=True, type=Path)
    audit.add_argument("--data", required=True, type=Path)
    audit.add_argument("--native-data-root", type=Path, default=ROOT.parents[1] / "data")
    audit.add_argument("--native-config-template", type=Path, default=ROOT / "tools/native_config_template.json")
    audit.add_argument("--trials", type=int, choices=(16, 32), default=16)
    audit.add_argument("--screen-root", type=Path, help="also audit existing native summaries; never launch a run")
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/reference_1031.json")
    fixture.add_argument("--actual", required=True, type=Path)
    recheck = sub.add_parser("recheck-native", help="reuse completed Python banks; rerun only the bounded 96 native jobs")
    recheck.add_argument("--reference-root", required=True, type=Path)
    recheck.add_argument("--screen-root", type=Path, default=ROOT / "results/statistical_screen_1031")
    recheck.add_argument("--native", type=Path, default=ROOT / ".tmp/validation_binary/genmodel_cpp.exe")
    recheck.add_argument("--native-ready", action="store_true")
    recheck.add_argument("--parameters", type=Path, default=ROOT / "tests/fixtures/statistical_parameters.json")
    recheck.add_argument("--data", type=Path, default=ROOT.parents[1] / "data/exp123/processed/Task2_processed.csv")
    recheck.add_argument("--output", type=Path, default=ROOT / "results/statistical_screen_1031_final")
    recheck.add_argument("--timeout", type=int, default=600)
    recheck.add_argument("--validate-only", action="store_true", help="validate cached reference inputs without native execution or output writes")
    run = sub.add_parser("run")
    run.add_argument("--reference-root", required=True, type=Path)
    run.add_argument("--native", required=True, type=Path)
    run.add_argument("--native-ready", action="store_true")
    run.add_argument("--native-config-template", required=True, type=Path)
    run.add_argument("--parameters", required=True, type=Path)
    run.add_argument("--data", required=True, type=Path)
    run.add_argument("--native-data-root", type=Path, default=ROOT.parents[1] / "data")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--trials", type=int, choices=(16, 32), default=16)
    run.add_argument("--seeds", type=int, choices=range(16, 33), default=16)
    run.add_argument("--probability-abs", type=float, default=.05)
    run.add_argument("--mean-log-abs", type=float, default=.1)
    run.add_argument("--mcse-multiplier", type=float, default=4.)
    run.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.mode == "run":
        if any(not math.isfinite(v) or v < 0 for v in
               (args.probability_abs, args.mean_log_abs, args.mcse_multiplier)):
            parser.error("Thresholds must be finite and nonnegative")
        run_screen(args)
    elif args.mode == "recheck-native":
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
        recheck_native(args)
    elif args.mode == "fixture":
        compare_fixture(args)
    elif args.mode == "preflight":
        preflight(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
