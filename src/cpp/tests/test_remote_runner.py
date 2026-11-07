"""Mock-only remote orchestration tests; never launch the native model."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIT = dict(initial_starts=64, rounds=3, proposals_per_round=32, top_k=4,
           search_particles=48, search_seeds=4, rescore_top=4, rescore_diverse=2,
           rescore_particles=192, rescore_seeds=4, audit_top=2,
           audit_particles=512, audit_seeds=8)
EVAL = dict(particle_count=512, pf_repeats=32, simulation_repeats=512,
            band_draws=512, window=16, plots=True, export_particles=True)


def test_remote_runner_exists():
    script = ROOT / "tools/run_cond123_remote.py"
    assert script.is_file(), "remote runner is not implemented"
    spec = importlib.util.spec_from_file_location("remote_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.fixture
def runner():
    test_remote_runner_exists()
    spec = importlib.util.spec_from_file_location("remote_runner", ROOT / "tools/run_cond123_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("condition", [1, 2, 3])
def test_full_configs(condition):
    path = ROOT / f"configs/cond{condition}_full_remote.json"
    assert path.is_file(), "remote config is not implemented"
    cfg = json.loads(path.read_text())
    assert cfg["subjects"] == list(range(100*condition+1, 100*condition+33))
    assert cfg["fit"] == FIT and cfg["evaluation"] == EVAL
    assert cfg["jobs"] == 64 and "max_trials" not in cfg
    assert cfg["seed"] == 1031202611 and cfg["simulation_repeats"] == 512
    for key, value in dict(data_root="../../../data", resource_root="../resources",
                           parameter_space="parameter_space.yaml").items():
        assert cfg[key] == value and (path.parent / value).exists()


def test_plan_is_sequential_and_read_only(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "resource_gate", Mock(side_effect=AssertionError("gate called")))
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(side_effect=AssertionError("job called")))
    out = tmp_path / "fresh"
    assert runner.main(["--output", str(out), "--plan-only"]) == 0
    assert not out.exists()
    plan = runner.make_plan(out)
    assert len(plan["steps"]) == 384
    assert [(s["condition"], s["subject"], s["stage"]) for s in plan["steps"]] == [
        (c, s, stage) for c in (1, 2, 3) for s in range(c*100+1, c*100+33)
        for stage in ("fit", "evaluate", "plot", "export-paper")]
    assert all(c["budgets"]["fit"] == FIT for c in plan["configs"])
    for step in plan["steps"]:
        cmd = step["command"]
        if step["stage"] in ("fit", "evaluate"):
            assert cmd[cmd.index("--jobs")+1] == "64"
            assert "--smoke" not in cmd and "--max-trials" not in cmd
        else:
            source = Path(cmd[cmd.index("--input")+1])
            target = Path(cmd[cmd.index("--output")+1])
            assert source not in target.parents and target not in source.parents
            assert source.name == f"subject_{step['subject']}"
            assert target.name == ("png" if step["stage"] == "plot" else "paper")


@pytest.mark.parametrize("subjects", [[999], [101, 101], []])
def test_bad_subjects(runner, tmp_path, subjects):
    with pytest.raises(ValueError):
        runner.make_plan(tmp_path / "new", subjects=subjects)


def test_subset_keeps_budgets(runner, tmp_path):
    plan = runner.make_plan(tmp_path / "new", subjects=[314, 122])
    assert [s["subject"] for s in plan["steps"]] == [122]*4 + [314]*4
    assert all(c["budgets"]["evaluation"] == EVAL for c in plan["configs"])


@pytest.mark.parametrize("hours", [0, -1, 4.01, float("nan"), float("inf")])
def test_deadline_bounds(runner, hours):
    with pytest.raises(ValueError):
        runner.deadline_after(hours, now=10)


def test_total_budget(runner):
    assert runner.deadline_after(4, now=10) == 14410
    assert runner.remaining(100, now=98) == 2
    with pytest.raises(TimeoutError):
        runner.remaining(100, now=100)


def test_paths(runner, tmp_path):
    with pytest.raises(ValueError):
        runner.make_plan(tmp_path)
    with pytest.raises(ValueError):
        runner.make_plan(ROOT / "data" / "new_run")
    with pytest.raises(ValueError):
        runner.make_plan(ROOT.parents[1] / "data" / "new_run")
    link = Mock(spec=Path)
    link.is_symlink.return_value = True
    with pytest.raises(ValueError):
        runner.fresh_output(link)


def test_cpu_equivalents(runner):
    before = "cpu0 10 0 10 100 0 0 0 0 8 0\ncpu1 0 0 0 100 0 0 0 0\n"
    after = "cpu0 60 0 10 150 0 0 0 0 30 0\ncpu1 0 0 0 200 0 0 0 0\n"
    a, b = runner.parse_cpu_stat(before), runner.parse_cpu_stat(after)
    assert runner.free_cpu_equivalents(a, b, {0}) == .5
    assert runner.free_cpu_equivalents(a, b, {0, 1}) == 1.5
    b[0] = (a[0][0]-1, a[0][1])
    with pytest.raises(ValueError):
        runner.free_cpu_equivalents(a, b, {0})


@pytest.mark.parametrize("free,disk,quota", [(63.999, 150, 128), (64, 99.99, 128), (128, 150, 63)])
def test_insufficient_resources(runner, free, disk, quota):
    with pytest.raises(RuntimeError):
        runner.check_resources(free, int(disk*2**30), quota)


def test_gate_boundary(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "sample_free_cpus", Mock(return_value=64.0))
    monkeypatch.setattr(runner, "cpu_quota", lambda: 128.0)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: Mock(free=100*2**30))
    assert runner.resource_gate(tmp_path, runner.deadline_after(1))["free_cpu_equivalents"] == 64
    assert runner.sample_free_cpus.call_args.kwargs["seconds"] == 3.0
    assert runner.MIN_DISK == 100*2**30


@pytest.mark.parametrize("args", [[], ["--output", "fresh", "--jobs", "65"],
                                  ["--output", "fresh", "--force"]])
def test_no_bypass_or_missing_output(runner, args):
    with pytest.raises(SystemExit):
        runner.main(args)


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_ancestor_quota(runner, monkeypatch, version):
    if version == 2:
        files = {"/proc/self/cgroup": "0::/batch/child\n",
                 "/proc/self/mountinfo": "1 0 0:1 / /cgroup rw - cgroup2 cgroup rw\n",
                 "/cgroup/cpu.max": "max 100000", "/cgroup/batch/cpu.max": "3200000 100000",
                 "/cgroup/batch/child/cpu.max": "6400000 100000"}
    else:
        files = {"/proc/self/cgroup": "1:cpu,cpuacct:/child\n",
                 "/proc/self/mountinfo": "1 0 0:1 / /cgroup rw - cgroup cgroup rw,cpu,cpuacct\n",
                 "/cgroup/cpu.cfs_quota_us": "3200000", "/cgroup/cpu.cfs_period_us": "100000",
                 "/cgroup/child/cpu.cfs_quota_us": "6400000", "/cgroup/child/cpu.cfs_period_us": "100000"}
    monkeypatch.setattr(runner.Path, "read_text", lambda p, **kw: files[p.as_posix()])
    monkeypatch.setattr(runner.Path, "exists", lambda p: p.as_posix() in files)
    assert runner.cpu_quota() == 32


def test_sampling_deadline_and_affinity_change(runner, monkeypatch):
    monkeypatch.setattr(runner.os, "sched_getaffinity", Mock(side_effect=[{0}, {1}]), raising=False)
    monkeypatch.setattr(runner, "read_cpu_stat", lambda: {0: (100, 100)})
    sleep = Mock()
    monkeypatch.setattr(runner.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="affinity changed"):
        runner.sample_free_cpus(deadline=runner.deadline_after(1))
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: {0})
    sleep.reset_mock()
    with pytest.raises(TimeoutError):
        runner.sample_free_cpus(deadline=runner.time.monotonic()+1)
    assert not sleep.called


def test_three_second_sampling(runner, monkeypatch):
    stat = Mock(side_effect=[{0: (100, 50)}, {0: (200, 150)}])
    monkeypatch.setattr(runner, "read_cpu_stat", stat)
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    sleep = Mock()
    monkeypatch.setattr(runner.time, "sleep", sleep)
    assert runner.sample_free_cpus(deadline=runner.deadline_after(1)) == 1
    sleep.assert_called_once_with(3.0)
    assert stat.call_count == 2


def test_timeout_stops_foreground(runner, tmp_path, monkeypatch):
    proc = Mock()
    proc.wait.side_effect = [subprocess.TimeoutExpired(["native"], 2), 0]
    popen = Mock(return_value=proc)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(TimeoutError):
        runner.run_stage({"command": ["native"]}, tmp_path / "stage.log", runner.deadline_after(1))
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert proc.wait.call_count == 2
    assert popen.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert popen.call_args.kwargs.get("shell", False) is False


def test_expired_deadline_no_process(runner, tmp_path, monkeypatch):
    popen = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(TimeoutError):
        runner.run_stage({"command": ["native"]}, tmp_path / "none.log", 0)
    assert not popen.called and not (tmp_path / "none.log").exists()


def test_gate_before_any_output(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "resource_gate", Mock(side_effect=RuntimeError("busy")))
    stage = Mock()
    monkeypatch.setattr(runner, "run_stage", stage)
    out = tmp_path / "new"
    with pytest.raises(RuntimeError, match="busy"):
        runner.execute(runner.make_plan(out, subjects=[101]), runner.deadline_after(1))
    assert not out.exists() and not stage.called


def test_between_subject_gate(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "resource_gate", Mock(side_effect=[{}, RuntimeError("busy")]))
    stage = Mock()
    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner, "validate_packet", Mock())
    with pytest.raises(RuntimeError, match="busy"):
        runner.execute(runner.make_plan(tmp_path / "new", subjects=[101, 102]), runner.deadline_after(1))
    assert stage.call_count == 4
    rows = [json.loads(s) for s in (tmp_path / "new/progress.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "failed"


def test_stage_failure_fail_fast(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "resource_gate", lambda *a: {})
    stage = Mock(side_effect=RuntimeError("fit failed"))
    monkeypatch.setattr(runner, "run_stage", stage)
    with pytest.raises(RuntimeError, match="fit failed"):
        runner.execute(runner.make_plan(tmp_path / "new", subjects=[101]), runner.deadline_after(1))
    assert stage.call_count == 1


@pytest.mark.parametrize("damage", [None, "hash", "phase", "path", "subject", "missing", "budget"])
def test_packet_validation(runner, tmp_path, damage):
    m = dict(schema_version="gm1031-evaluation-inputs-v1", subject_id=101,
             condition=1, trials=10, parameters={"capacity": 2}, config=EVAL,
             pf_seeds=list(range(32)), simulation_seeds=list(range(512)),
             arrays={"valid": {"shape": [10], "dtype": "uint8"}},
             protocol={"feature_order": ["tail", "head", "neck", "leg"]},
             catalogue_metadata=[{"index": 0}], provenance={"model_version": "test"})
    for file in ("evaluation_inputs.npz", "conditional_particle_contents.npz"):
        (tmp_path / file).write_bytes(b"synthetic hash-only fixture")
    digest = hashlib.sha256(b"synthetic hash-only fixture").hexdigest()
    m["inputs_sha256"] = digest
    p = dict(status="available", file="conditional_particle_contents.npz", sha256=digest,
             seed=0, pf_repeat_index=0, particle_count=512, capacity=2, pairing_axis=1,
             phase="current_choice_filtered_before_feedback_before_resampling",
             arrays={"particle_weights": {"shape": [10, 512], "dtype": "float64"}})
    m["conditional_particle_contents"] = p
    if damage == "hash":
        p["sha256"] = "0"*64
    elif damage == "phase":
        p["phase"] = "after_feedback"
    elif damage == "path":
        p["file"] = "../escape.npz"
    elif damage == "subject":
        m["subject_id"] = 102
    elif damage == "missing":
        del m["arrays"]
    elif damage == "budget":
        p["particle_count"] = 8
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(m))
    if damage:
        with pytest.raises(ValueError):
            runner.validate_packet(tmp_path, 101, 1, EVAL)
    else:
        runner.validate_packet(tmp_path, 101, 1, EVAL)
