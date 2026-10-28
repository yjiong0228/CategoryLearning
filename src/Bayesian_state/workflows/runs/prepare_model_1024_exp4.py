"""Prepare validated exp4 input bundles for the shared Model 1024 workflow.

Run from the repository root with --config and a new --output directory.
This entry point reads observed data only and does not execute or fit a model.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from ...simulation.exp4_data import load_exp4_subject, file_sha256


def prepare(config: Path, output: Path, subjects: list[int] | None = None) -> dict:
    """Validate all selected subjects, then publish into a fresh directory."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    settings = yaml.safe_load(config.read_text())
    selected = settings["subjects"] if subjects is None else subjects
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select a nonempty list of distinct subjects")
    cases = [load_exp4_subject(config, subject) for subject in selected]
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for case in cases:
        folder = output / str(case.subject_id)
        folder.mkdir()
        arrays = case.arrays
        with (folder / "observations.npz").open("xb") as stream:
            np.savez_compressed(stream, stimulus=arrays.stimulus, choices=arrays.choices,
                                feedback=arrays.feedback, presskeys=arrays.presskeys)
        with (folder / "task_metadata.npz").open("xb") as stream:
            np.savez_compressed(stream, target_probs=arrays.target_probs,
                                recorded_target_probs=case.frame[["probCat1", "probCat2"]].to_numpy(),
                                trial_ids=case.frame[["iSession", "iBlock", "iTrial"]].to_numpy(dtype=int),
                                source_rows=case.frame.source_row.to_numpy(dtype=int),
                                stimulus_ids=case.frame.stiID.to_numpy(dtype=int),
                                ambiguous=case.frame.ambiguous.to_numpy(dtype=int),
                                stimulus_ambiguous=case.frame.stimulus_ambiguous.to_numpy(dtype=int))
        with (folder / "metadata.json").open("x") as stream:
            json.dump(case.metadata(), stream, ensure_ascii=False, indent=2, allow_nan=False)
        for path in sorted(folder.iterdir()):
            files[str(path.relative_to(output))] = file_sha256(path)
    from ...simulation import data, exp4_data
    from ...model.modules import perception
    manifest = dict(protocol=settings["protocol"], status="inputs_validated",
                    model_execution="not_run", subjects=selected,
                    input_config=settings,
                    total_trials=sum(len(case.frame) for case in cases),
                    versions={"python": platform.python_version(), **{
                        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "PyYAML")}},
                    code_sha256={module.__name__: file_sha256(Path(module.__file__))
                                 for module in (data, exp4_data, perception)},
                    entry_sha256=file_sha256(Path(__file__)), files_sha256=files)
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp4/model1024_inputs.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+")
    args = parser.parse_args()
    result = prepare(args.config, args.output, args.subjects)
    print(json.dumps({key: result[key] for key in ("status", "subjects", "total_trials")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
