"""The C++ subtree shares data but has an independent build and Python runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]


def executable(name):
    return ROOT / (f"build/Release/{name}.exe" if os.name == "nt" else f"build/{name}")


def test_no_copied_research_data():
    assert (ROOT / "CMakeLists.txt").is_file()
    assert not (ROOT / "data").exists()


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_all_condition_configs_share_main_data(condition):
    paths = list((ROOT / "configs").rglob(f"cond{condition}*.yaml"))
    paths += list((ROOT / "configs").glob(f"cond{condition}*.json"))
    assert paths
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (path.parent / cfg["data_root"]).resolve() == REPOSITORY / "data"
        assert (path.parent / cfg["resource_root"]).resolve() == ROOT / "resources"
        assert (path.parent / cfg["parameter_space"]).is_file()


def test_shared_source_hashes():
    source = json.loads((ROOT / "resources/data_sources.json").read_text(encoding="utf-8"))
    assert len(source["files"]) == 3
    for record in source["files"]:
        path = REPOSITORY / record["path"]
        assert path.is_relative_to(REPOSITORY / "data")
        allowed = {(record["bytes"], record["sha256"])}
        if "git_sha256" in record:
            allowed.add((record["git_bytes"], record["git_sha256"]))
        assert (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()) in allowed


def test_byte_identified_files_disable_checkout_conversion():
    for pattern in ("resources/catalogue_*.json -text", "tests/fixtures/* -text", "src/*.cpp -text"):
        assert pattern in (ROOT / ".gitattributes").read_text(encoding="utf-8")


@pytest.mark.parametrize("working,spelling", [(REPOSITORY, "src/cpp"), (REPOSITORY, "src/cpp/"), (ROOT, ".")])
def test_benchmark_root_spellings(working, spelling):
    native = executable("gm_benchmark")
    if not native.is_file():
        pytest.skip("Build the benchmark first")
    result = subprocess.run([str(native), spelling, "122", "2", "2", "1", "1"],
                            cwd=working, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["trials"] == 2


def test_predictions_from_main_and_subtree(tmp_path):
    native = executable("genmodel_cpp")
    if not native.is_file():
        pytest.skip("Build the model first")
    outputs = []
    for i, working in enumerate((REPOSITORY, ROOT)):
        output = tmp_path / str(i)
        config = os.path.relpath(ROOT / "configs/cond1.yaml", working)
        result = subprocess.run([str(native), "filter", "--config", config, "--subjects", "122",
                                 "--max-trials", "4", "--jobs", "1", "--output", str(output)],
                                cwd=working, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(output / "subject_122/filter.npz")
    with np.load(outputs[0]) as a, np.load(outputs[1]) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
