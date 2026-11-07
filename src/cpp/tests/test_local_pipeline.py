"""Bounded runner checks: dry plans and mocked external commands, no native fits."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run_cond123_local.ps1"
SHELL = shutil.which("powershell") or shutil.which("pwsh")


def run_script(*args, cwd=None, env=None, prefix="", script=None):
    if not SHELL:
        raise unittest.SkipTest("PowerShell is required")
    assert SCRIPT.is_file(), "local runner is not implemented"
    # -Command preserves typed arrays in both Windows PowerShell and PowerShell 7.
    quoted = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = prefix + " & " + quoted(script or SCRIPT) + " " + " ".join(args)
    return subprocess.run([SHELL, "-NoProfile", "-NonInteractive", "-Command", command],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=60)


def check_full_configs_match_csv_and_preserve_fit_budget(condition):
    path = ROOT / f"configs/cond{condition}_full_local.yaml"
    assert path.is_file(), "local full config is not implemented"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    with (ROOT.parents[1] / "data/exp123/processed/Task2_processed.csv").open(encoding="utf-8-sig", newline="") as stream:
        ids = {int(row["iSub"]) for row in csv.DictReader(stream)
               if int(row["condition"]) == condition}
    assert len(ids) == 32
    assert cfg["subjects"] == sorted(ids)
    assert "max_trials" not in cfg
    prior = yaml.safe_load((ROOT / "configs/cond1_full.yaml").read_text(encoding="utf-8"))
    assert cfg["fit"] == prior["fit"]
    assert cfg["parameter_space"] == prior["parameter_space"]
    assert cfg["seed"] == prior["seed"]
    assert (path.parent / cfg["parameter_space"]).is_file()
    assert (path.parent / cfg["data_root"]).resolve() == ROOT.parents[1] / "data"
    assert cfg["jobs"] == 12
    assert cfg["evaluation"]["pf_repeats"] == 12
    assert cfg["evaluation"]["simulation_repeats"] == 128


def check_default_plan_is_full_sequential_and_creates_nothing(tmp_path):
    output = tmp_path / "new output"
    result = run_script("-PlanOnly", f"-OutputRoot '{output}'", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert not output.exists()
    steps = plan["steps"]
    assert len(steps) == 96 * 3
    assert [(s["condition"], s["subject"], s["stage"]) for s in steps] == [
        (c, sid, stage) for c in (1, 2, 3) for sid in range(c * 100 + 1, c * 100 + 33)
        for stage in ("fit", "evaluate", "plot")]
    for step in steps:
        args = step["arguments"]
        if step["stage"] != "plot":
            assert args[args.index("--jobs") + 1] == "12"
            assert args[args.index("--subjects") + 1] == str(step["subject"])
            assert "--max-trials" not in args and "--smoke" not in args
        assert Path(step["program"]).is_absolute()


def check_smoke_plan_filters_subjects_and_uses_matching_trial_limits(tmp_path):
    result = run_script("-PlanOnly -Jobs 2 -Conditions 1,3 -Subjects 122,314 -MaxTrials 4 -Smoke",
                        f"-OutputRoot '{tmp_path / 'smoke'}'", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    steps = json.loads(result.stdout)["steps"]
    assert [s["subject"] for s in steps] == [122] * 3 + [314] * 3
    for step in steps:
        if step["stage"] != "plot":
            args = step["arguments"]
            assert args[args.index("--max-trials") + 1] == "4"
            assert args[args.index("--jobs") + 1] == "2"
            assert "--smoke" in args


def check_invalid_selection_fails_before_outputs(tmp_path, options):
    output = tmp_path / "invalid"
    result = run_script(options, "-PlanOnly", f"-OutputRoot '{output}'")
    assert result.returncode != 0
    assert not output.exists()


def mock_commands(tmp_path):
    mock = tmp_path / "mock external.ps1"
    mock.write_text("""$record = @{args=@($args); cwd=$pwd.Path; omp=$env:OMP_NUM_THREADS; blas=$env:OPENBLAS_NUM_THREADS; mkl=$env:MKL_NUM_THREADS} | ConvertTo-Json -Compress
Add-Content -LiteralPath $env:LOCAL_PIPELINE_CALLS -Value $record
Write-Output 'mock progress'
$global:LASTEXITCODE = 0
$stage = $args[0]
if ($args -contains '-c') { $stage = 'preflight' }
if ($stage -eq $env:LOCAL_PIPELINE_FAIL) { $global:LASTEXITCODE = 7 }
""", encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    env = dict(os.environ, LOCAL_PIPELINE_CALLS=str(calls))
    options = f"-Executable '{mock}' -PythonExe '{mock}' -Jobs 2 -Conditions 1,2 -Subjects 122,206 -Smoke -MaxTrials 4"
    return options, calls, env


def check_execution_order_logs_and_fail_fast(tmp_path, mock_commands, failure, expected_calls):
    options, calls, env = mock_commands
    env["LOCAL_PIPELINE_FAIL"] = failure
    output = tmp_path / "fresh output"
    result = run_script(options, f"-OutputRoot '{output}'", cwd=tmp_path, env=env)
    assert (result.returncode == 0) == (not failure), result.stdout + result.stderr
    records = [json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()]
    assert len(records) == expected_calls
    assert [r["args"][0] for r in records] == ["-B", "fit", "evaluate", "-B", "fit", "evaluate", "-B"][:expected_calls]
    probe = records[0]["args"]
    assert probe[:2] == ["-B", "-c"]
    assert all(f"import {name}" in probe[2] for name in ("numpy", "scipy", "matplotlib"))
    assert all(r["omp"] == r["blas"] == r["mkl"] == "1" for r in records)
    assert all(Path(r["cwd"]) == ROOT for r in records)
    assert len(list((output / "logs").glob("*.log"))) == expected_calls - 1
    assert (output / "plan.json").is_file()
    assert (output / "progress.log").is_file()
    if failure:
        assert "FAILED" in (output / "progress.log").read_text(encoding="utf-8-sig")
    else:
        assert "COMPLETE" in (output / "progress.log").read_text(encoding="utf-8-sig")
    again = run_script(options, f"-OutputRoot '{output}'", env=env)
    assert again.returncode != 0
    assert len(calls.read_text(encoding="utf-8-sig").splitlines()) == expected_calls


def check_existing_empty_directory_is_refused(tmp_path):
    result = run_script("-PlanOnly", f"-OutputRoot '{tmp_path}'")
    assert result.returncode != 0


def test_windows_runner_rejects_shared_data_outputs(tmp_path, directory_alias):
    repository = tmp_path / "repository"
    cpp = repository / "src/cpp"
    script = cpp / "tools/run_cond123_local.ps1"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    configs = cpp / "configs"
    configs.mkdir()
    for condition in (1, 2, 3):
        name = f"cond{condition}_full_local.yaml"
        shutil.copyfile(ROOT / "configs" / name, configs / name)
    data = repository / "data"
    data.mkdir()
    alias = repository / "data-alias"
    directory_alias(alias, data)
    for base in (data, alias):
        output = base / "new-output"
        result = run_script("-PlanOnly -Conditions 1 -Subjects 122", f"-OutputRoot '{output}'", script=script)
        assert result.returncode != 0, result.stdout + result.stderr
        assert "data" in result.stderr.lower()
        assert not (data / "new-output").exists()


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local pipeline tests ")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_configs(self):
        for condition in (1, 2, 3):
            with self.subTest(condition=condition):
                check_full_configs_match_csv_and_preserve_fit_budget(condition)

    def test_default_plan(self):
        check_default_plan_is_full_sequential_and_creates_nothing(self.path)

    def test_recording_config_set_plan_and_export(self):
        output = self.path / "records"
        result = run_script("-PlanOnly -Jobs 8 -ConfigDirectory configs/local_records8 -ExportPaper",
                            f"-OutputRoot '{output}'")
        assert result.returncode == 0, result.stdout + result.stderr
        plan = json.loads(result.stdout)
        assert not output.exists()
        assert len(plan["steps"]) == 96 * 4
        assert plan["jobs"] == 8
        assert [s["stage"] for s in plan["steps"][:4]] == ["fit", "evaluate", "plot", "export-paper"]
        for condition in (1, 2, 3):
            path = ROOT / f"configs/local_records8/cond{condition}_full_local.yaml"
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert cfg["jobs"] == 8 and "max_trials" not in cfg
            assert cfg["subjects"] == list(range(100*condition+1, 100*condition+33))
            assert cfg["evaluation"]["export_particles"] is True
            assert cfg["evaluation"]["particle_count"] == 512
            assert cfg["evaluation"]["pf_repeats"] == 16
            assert cfg["evaluation"]["simulation_repeats"] == 512
            assert (path.parent / cfg["data_root"]).resolve() == ROOT.parents[1] / "data"
            assert (path.parent / cfg["resource_root"]).resolve() == ROOT / "resources"
            assert (path.parent / cfg["parameter_space"]).is_file()
        for step in plan["steps"]:
            if step["stage"] in ("fit", "evaluate"):
                args = step["arguments"]
                assert args[args.index("--jobs")+1] == "8"
                assert "local_records8" in args[args.index("--config")+1]
            elif step["stage"] == "export-paper":
                assert "export_paper_tables.py" in step["arguments"][1]
                assert step["arguments"][-2:] == ["--window", "16"]

    def test_export_paper_executes_after_plot(self):
        options, calls, env = mock_commands(self.path)
        output = self.path / "export_mock"
        result = run_script(options, "-ExportPaper", f"-OutputRoot '{output}'", env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        records = [json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()]
        assert len(records) == 9  # dependency probe plus four stages per subject
        assert "plot_results.py" in records[3]["args"][1]
        assert "export_paper_tables.py" in records[4]["args"][1]
        assert all(r["omp"] == r["blas"] == r["mkl"] == "1" for r in records)

    def test_smoke_plan(self):
        check_smoke_plan_filters_subjects_and_uses_matching_trial_limits(self.path)

    def test_invalid_selections(self):
        for options in ("-Conditions 1,1", "-Conditions 4", "-Jobs 13",
                        "-Conditions 1 -Subjects 201", "-Subjects 122,122", "-MaxTrials 0"):
            with self.subTest(options=options):
                check_invalid_selection_fails_before_outputs(self.path, options)

    def test_external_commands(self):
        for failure, count in (("", 7), ("fit", 2), ("evaluate", 3), ("-B", 4)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=self.path) as directory:
                path = Path(directory)
                check_execution_order_logs_and_fail_fast(path, mock_commands(path), failure, count)

    def test_missing_python_dependencies_fail_before_outputs_or_fit(self):
        options, calls, env = mock_commands(self.path)
        env["LOCAL_PIPELINE_FAIL"] = "preflight"
        output = self.path / "must not exist"
        result = run_script(options, f"-OutputRoot '{output}'", env=env)
        assert result.returncode != 0, result.stdout + result.stderr
        assert "Python dependency preflight failed" in result.stdout + result.stderr
        records = [json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()]
        assert len(records) == 1
        assert records[0]["args"][:2] == ["-B", "-c"]
        assert not output.exists()

    def test_real_python_import_failure_creates_no_outputs(self):
        _, calls, env = mock_commands(self.path)
        mock = self.path / "mock external.ps1"
        # Force a reproducible failed import without relying on installed packages.
        (self.path / "numpy.py").write_text("raise ImportError('forced missing NumPy')\n", encoding="utf-8")
        env["PYTHONPATH"] = str(self.path)
        options = f"-Executable '{mock}' -PythonExe '{sys.executable}' -Jobs 2 -Conditions 1 -Subjects 122 -Smoke -MaxTrials 4"
        output = self.path / "no real outputs"
        result = run_script(options, f"-OutputRoot '{output}'", env=env)
        assert result.returncode != 0, result.stdout + result.stderr
        assert "Python dependency preflight failed" in result.stdout + result.stderr
        assert "forced missing NumPy" in result.stdout + result.stderr
        assert not calls.exists()
        assert not output.exists()

    def test_existing_output(self):
        check_existing_empty_directory_is_refused(self.path)

    def test_plan_needs_no_executables_and_relative_output_is_repo_relative(self):
        relative = f"results/{self.path.name}"
        options = f"-Executable '{self.path / 'missing native.exe'}' -PythonExe '{self.path / 'missing python.exe'}'"
        result = run_script(options, "-PlanOnly -Jobs 2 -Conditions 1 -Subjects 122",
                            f"-OutputRoot '{relative}'", cwd=self.path)
        assert result.returncode == 0, result.stdout + result.stderr
        plan = json.loads(result.stdout)
        output = ROOT / relative
        assert Path(plan["output_root"]) == output
        assert not output.exists()
        result = run_script(options, "-Jobs 2 -Conditions 1 -Subjects 122",
                            f"-OutputRoot '{relative}'", cwd=self.path)
        assert result.returncode != 0
        assert not output.exists()


if __name__ == "__main__":
    unittest.main()
