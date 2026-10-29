"""Prepare all Exp5 behavior in a new directory, including deferred day-two rows."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import yaml

from ...simulation.exp5_data import load_exp5_subject, file_sha256, TRIAL_KEYS


def prepare(config: Path, output: Path) -> dict:
    """Validate the registered cohort and write behavior, observations and provenance."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    settings = yaml.safe_load(config.read_text())
    cases = [load_exp5_subject(config, subject) for subject in settings["subjects"]]
    if not cases:
        raise ValueError("Input cohort is empty")
    output.mkdir(parents=True, exist_ok=False)
    pd.concat([case.frame for case in cases], ignore_index=True).to_csv(output / "behavior.csv", index=False)
    for case in cases:
        folder = output / str(case.subject_id)
        folder.mkdir()
        arrays, frame = case.arrays, case.analysis_frame
        with (folder / "observations.npz").open("xb") as stream:
            np.savez_compressed(stream, stimulus=arrays.stimulus, choices=arrays.choices, feedback=arrays.feedback)
        with (folder / "task_metadata.npz").open("xb") as stream:
            np.savez_compressed(stream, categories=arrays.categories, presskeys=arrays.presskeys,
                                trial_ids=frame[TRIAL_KEYS].to_numpy(dtype=int),
                                blocks=frame.iBlock.to_numpy(dtype=int),
                                source_rows=frame.source_row.to_numpy(dtype=int))
        with (folder / "metadata.json").open("x") as stream:
            json.dump(case.metadata(), stream, ensure_ascii=False, indent=2, allow_nan=False)
    from ...simulation import data, exp5_data
    manifest = dict(protocol=settings["protocol"], status="inputs_validated", model_execution="not_run",
                    recorded_trials=sum(len(case.frame) for case in cases),
                    analysis_trials=sum(len(case.arrays.choices) for case in cases),
                    subjects=settings["subjects"], input_config=settings,
                    versions={"python": platform.python_version(), **{
                        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "PyYAML")}},
                    code_sha256={module.__name__: file_sha256(Path(module.__file__)) for module in (data, exp5_data)},
                    entry_sha256=file_sha256(Path(__file__)),
                    files_sha256={str(path.relative_to(output)): file_sha256(path)
                                  for path in sorted(output.rglob("*")) if path.is_file()})
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp5/model1024_inputs.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.config, args.output)
    print(json.dumps({key: report[key] for key in ("status", "subjects", "recorded_trials", "analysis_trials")}))


if __name__ == "__main__":
    main()
