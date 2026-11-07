"""Transfer old 1031 parameters into standalone evaluation configs, offline.

Usage: python tools/export_evaluation_configs.py --input BATCH --output NEWDIR
       [--subjects ID ...] [--jobs 12]
Validates all selected inputs before creating NEWDIR. source.json records the
original identities and commands to run separately. C++ appends subject_ID to
each command's output root. NumPy verifies saved scientific input arrays against
the original CSV prefix; forecast/state arrays remain unloaded. No model is
imported, evaluated, or optimized.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = set("capacity similarity_width baseline_probability error_gain failure_decay global_search retention execution_inertia beta_init learning_rate".split())
BUDGET = set("particle_count pf_repeats simulation_repeats band_draws window".split())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value, minimum=1, maximum=2147483647):
    return type(value) is int and minimum <= value <= maximum


def verify_inputs(path: Path, evaluation: dict, schedule: list, condition: int) -> str:
    """Associate declared trials with checksummed input arrays and CSV values."""
    checksum = sha(path)
    require(checksum == evaluation["inputs_sha256"], "evaluation inputs checksum mismatch")
    trials = evaluation["trials"]
    categories = 2 if condition == 1 else 4
    require(evaluation["n_categories"] == categories, "input category count mismatch")
    prefix = schedule[:trials]
    ids = [tuple(int(r[k]) for k in ("iSession", "iBlock", "iTrial")) for r in prefix]
    require(ids == sorted(set(ids)), "CSV trial IDs must be unique and ordered")
    valid = np.array([r.get("miss", "0") == "0" for r in prefix], dtype="uint8")
    score = valid.copy()
    score[np.flatnonzero(score)[0]] = 0
    # C++ stores category-coded responses as zero-based; missed choices are -1.
    expected = dict(trial_ids=np.array(ids, dtype="int64"),
                    session_ids=np.array([key[0] for key in ids], dtype="int64"),
                    valid=valid, score_mask=score,
                    stimuli=np.array([[float(r[f"feature{k}"]) for k in range(1, 5)] for r in prefix], dtype="float64"),
                    observed_choices=np.array([int(r["choice"]) - 1 if v else -1 for r, v in zip(prefix, valid)], dtype="int64"),
                    observed_feedback=np.array([float(r["feedback"].strip() or "nan") for r in prefix], dtype="float64"),
                    true_categories=np.array([int(r["category"]) - 1 for r in prefix], dtype="int64"),
                    category_to_response=np.tile(np.arange(categories, dtype="int64"), (trials, 1)))
    require(np.isfinite(expected["stimuli"]).all(), "nonfinite CSV stimuli")
    with np.load(path, allow_pickle=False) as inputs:
        for name, reference in expected.items():
            require(name in inputs.files, f"missing scientific input array: {name}")
            actual = inputs[name]  # NPZ access is lazy: only these input members are loaded.
            metadata = evaluation["arrays"][name]
            require(actual.shape == reference.shape and metadata["shape"] == list(reference.shape), f"input shape/trial mismatch: {name}")
            require(actual.dtype == reference.dtype and metadata["dtype"] == str(actual.dtype), f"input dtype mismatch: {name}")
            require(np.array_equal(actual, reference, equal_nan=True), f"input differs from CSV prefix: {name}")
    return checksum


def export_evaluation_configs(batch: Path, output: Path, subjects=None, jobs: int = 12, *, repo: Path = ROOT) -> dict:
    """Validate metadata and source bytes, then publish new JSON configurations."""
    batch, output, repo = Path(batch).resolve(), Path(output).resolve(), Path(repo).resolve()
    require(batch.is_dir(), "input batch directory missing")
    require(not output.exists(), "output already exists")
    require(not output.is_relative_to(batch), "output must be outside the original batch")
    require(integer(jobs, maximum=12), "jobs must be an integer in 1..12")
    selected = None if subjects is None else set(subjects)
    require(selected is None or (selected and len(selected) == len(subjects) and all(integer(s) for s in selected)), "invalid/duplicate subjects")
    directories = []
    for directory in sorted(batch.glob("cond*/subject_*")):
        match = re.fullmatch(r"cond([123])/subject_([1-9][0-9]*)", directory.relative_to(batch).as_posix())
        require(match is not None and directory.is_dir(), f"invalid subject path: {directory}")
        condition, sid = map(int, match.groups())
        if selected is None or sid in selected:
            directories.append((directory, condition, sid))
    ids = [sid for _, _, sid in directories]
    require(ids and len(ids) == len(set(ids)) and (selected is None or selected == set(ids)), "missing or duplicated selected subjects")
    repository = repo.parents[1]
    data_root = repository / "data"
    source_path = repo / "resources/data_sources.json"
    source_hashes = {}
    for item in read(source_path)["files"]:
        path = (repository / item["path"]).resolve()
        require(path.is_relative_to(data_root), "source manifest path outside data")
        actual = sha(path)
        require(actual in {item["sha256"], item.get("git_sha256")}, f"source hash mismatch: {path}")
        source_hashes[path.relative_to(repository).as_posix()] = actual
    csv_path = data_root / "exp123/processed/Task2_processed.csv"
    require(csv_path.relative_to(repository).as_posix() in source_hashes, "behavior CSV missing from source manifest")
    rows = {sid: [] for sid in ids}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = int(row["iSub"])
            if sid in rows:
                require(row.get("miss", "0") in ("0", "1"), "invalid CSV miss mask")
                rows[sid].append(row)
    space = repo / "configs/parameter_space.yaml"
    require(space.is_file(), "parameter_space missing")
    entries, configs = [], {}
    for directory, condition, sid in directories:
        fitpath = directory / f"fit/subject_{sid}/best_parameters.json"
        evalpath = directory / f"evaluate/subject_{sid}/evaluation_manifest.json"
        fit, evaluation = read(fitpath), read(evalpath)
        require(evaluation["scientific_model_contract"] == "1031-unified-v2", "scientific model contract mismatch")
        require(evaluation["protocol"]["dataset"] == "exp123", "dataset mismatch")
        code = 1 if condition == 1 else 2
        resource = hashlib.sha256("".join(sha(repo / f"resources/catalogue_c{code}.{ext}") for ext in ("json", "npz")).encode()).hexdigest()
        for document in (fit, evaluation):
            require(document["subject_id"] == sid and document["condition"] == condition, "subject/condition mismatch")
            require(document["resource_fingerprint"] == resource, "resource fingerprint mismatch")
            provenance = document["provenance"]
            require(provenance["model_version"] == "1031-unified-v2:cpp-v1" and provenance["source_data_scope"] == "exp123"
                    and provenance["source_manifest"] in {"data/source_1031.json", "src/cpp/resources/data_sources.json"}, "foreign fit provenance")
            require(provenance["catalogue_fingerprint"] == resource, "provenance resource mismatch")
        require(fit["provenance"] == evaluation["provenance"], "fit/evaluation provenance mismatch")
        require(fit["model_version"] == fit["provenance"]["scientific_fingerprint"]
                and re.fullmatch(r"[0-9a-f]{64}", fit["model_version"]), "original model identity mismatch")
        require(fit["data_fingerprint"] == evaluation["data_fingerprint"]
                and re.fullmatch(r"[0-9a-f]{64}", fit["data_fingerprint"]), "data fingerprint mismatch")
        params = fit["best_parameters"]
        require(params == evaluation["parameters"] and set(params) == PARAMETERS, "parameter mismatch or incomplete ten-parameter dictionary")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in params.values()), "nonfinite/nonnumeric parameter")
        require(integer(params["capacity"]) and params["similarity_width"] > 0 and 0 < params["baseline_probability"] < 1
                and params["error_gain"] >= 0 and params["execution_inertia"] >= 0 and .1 <= params["beta_init"] <= 25
                and all(0 <= params[k] <= 1 for k in ("failure_decay", "global_search", "retention", "learning_rate")), "parameter outside domain")
        budget = evaluation["config"].copy()
        require(BUDGET <= budget.keys() and budget.keys() <= BUDGET | {"seed", "jobs", "plots", "keep_states", "export_particles"}, "unknown/incomplete evaluation budget")
        seed = budget.pop("seed")
        require(integer(seed, 0, 9223372036854775807), "invalid root seed")
        require(integer(budget.pop("jobs", 1), maximum=64), "invalid saved jobs")
        require(all(integer(budget[k], maximum=536870911) for k in BUDGET)
                and all(type(budget[k]) is bool for k in ("plots", "keep_states", "export_particles") if k in budget), "invalid evaluation budget")
        trials, schedule = evaluation["trials"], rows[sid]
        require(integer(trials) and trials <= len(schedule) and all(int(r["condition"]) == condition for r in schedule), "trial count/CSV condition mismatch")
        valid_count = sum(r.get("miss", "0") == "0" for r in schedule[:trials])
        require(valid_count > 0 and (trials == len(schedule) or schedule[trials - 1].get("miss", "0") == "0"), "trial prefix cannot be represented by max_trials")
        inputpath = evalpath.with_name("evaluation_inputs.npz")
        inputsha = verify_inputs(inputpath, evaluation, schedule, condition)
        cfg = dict(dataset="exp123", subjects=[sid], parameters=params, seed=seed, jobs=jobs, evaluation=budget)
        for key, path in (("data_root", data_root), ("resource_root", repo / "resources"), ("parameter_space", space)):
            try:
                cfg[key] = Path(os.path.relpath(path, output)).as_posix()
            except ValueError:  # Windows paths on different drives cannot be relative.
                cfg[key] = path.as_posix()
        if trials < len(schedule):
            cfg["max_trials"] = valid_count
        name = f"subject_{sid}.json"
        configs[name] = cfg
        entries.append(dict(subject_id=sid, condition=condition, config=name, fit=str(fitpath), evaluation=str(evalpath),
                            fit_sha256=sha(fitpath), evaluation_sha256=sha(evalpath), original_provenance=fit["provenance"],
                            evaluation_inputs=str(inputpath), inputs_sha256=inputsha,
                            parameter_origin="best_parameters", original_model_version=fit["model_version"],
                            data_fingerprint=fit["data_fingerprint"], resource_fingerprint=resource, trials=trials,
                            command=f'genmodel_cpp evaluate --config "{(output / name).as_posix()}" --output NEWRESULTROOT/subject_{sid}'))
    manifest = dict(schema_version="gm1031-parameter-transfer-v1", scientific_model_contract="1031-unified-v2",
                    source_batch=str(batch), source_manifest_sha256=sha(source_path), source_hashes=source_hashes,
                    note="Explicit parameter transfer; original implementation identity retained here. Commands are not executed. C++ appends subject_ID beneath each output root.", exports=entries)
    payloads = {name: json.dumps(value, indent=2, allow_nan=False) + "\n" for name, value in {**configs, "source.json": manifest}.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(payload)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    try:
        manifest = export_evaluation_configs(args.input, args.output, args.subjects, args.jobs)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"export failed: {error}\n")
    print(f"Exported {len(manifest['exports'])} configs to {args.output}; no evaluation executed.")


if __name__ == "__main__":
    main()
