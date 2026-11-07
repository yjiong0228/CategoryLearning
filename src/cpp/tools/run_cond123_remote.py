"""Foreground Linux orchestration for fixed-budget native Model 1031 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = 64
MIN_DISK = 100 * 2**30
PHASE = "current_choice_filtered_before_feedback_before_resampling"


def deadline_after(hours: float, *, now: float | None = None) -> float:
    if not math.isfinite(hours) or not 0 < hours <= 4:
        raise ValueError("--max-hours must be finite, positive and <= 4")
    return (time.monotonic() if now is None else now) + hours * 3600


def remaining(deadline: float, *, now: float | None = None) -> float:
    seconds = deadline - (time.monotonic() if now is None else now)
    if seconds <= 0:
        raise TimeoutError("total run deadline reached")
    return seconds


def fresh_output(path: Path) -> Path:
    if path.is_symlink() or path.exists():
        raise ValueError(f"output must be fresh: {path}")
    path = path.absolute()
    resolved = path.resolve()
    if resolved == ROOT or resolved in ROOT.parents:
        raise ValueError("output cannot contain the repository")
    for name in ("data", "resources", "configs", "src", "include", "tools", "tests", "evaluation", "docs", "build", "third_party"):
        protected = ROOT / name
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"protected output path: {path}")
    for name in ("data", "configs", "tests", ".git"):
        protected = ROOT.parents[1] / name
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"protected shared repository path: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("output ancestors must not be symlinks; use the physical path")
    return resolved


def make_plan(output: Path, subjects: list[int] | None = None) -> dict:
    output = fresh_output(output)
    configs = []
    steps = []
    available = {c*100+s for c in (1, 2, 3) for s in range(1, 33)}
    if subjects is not None and (not subjects or len(set(subjects)) != len(subjects)
                                 or not set(subjects) <= available):
        raise ValueError("--subjects must be unique IDs from the 96 configured subjects")
    for condition in (1, 2, 3):
        path = ROOT / f"configs/cond{condition}_full_remote.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg["jobs"] != JOBS or "max_trials" in cfg:
            raise ValueError("remote configs require jobs=64 and all trials")
        ids = cfg["subjects"]
        if ids != list(range(condition*100+1, condition*100+33)):
            raise ValueError("remote config subjects changed")
        configs.append(dict(path=str(path), sha256=hash_file(path), budgets=cfg))
        for sid in ids:
            if subjects is not None and sid not in subjects:
                continue
            base = output / f"cond{condition}" / f"subject_{sid}"
            fit = base / "fit"
            evaluate = base / "evaluate"
            bundle = evaluate / f"subject_{sid}"
            common = ["--config", str(path), "--subjects", str(sid), "--jobs", str(JOBS)]
            commands = [
                ("fit", [str(ROOT / "build/genmodel_cpp"), "fit", *common, "--output", str(fit)]),
                ("evaluate", [str(ROOT / "build/genmodel_cpp"), "evaluate", *common,
                              "--fit-dir", str(fit), "--output", str(evaluate)]),
                ("plot", [sys.executable, "-B", str(ROOT / "evaluation/plot_results.py"),
                          "--input", str(bundle), "--output", str(base / "png")]),
                ("export-paper", [sys.executable, "-B", str(ROOT / "evaluation/export_paper_tables.py"),
                                  "--input", str(bundle), "--output", str(base / "paper"),
                                  "--window", str(cfg["evaluation"]["window"])])]
            for stage, command in commands:
                steps.append(dict(condition=condition, subject=sid, stage=stage, command=command,
                                  bundle=str(bundle), base=str(base), evaluation=cfg["evaluation"]))
    return dict(output=str(output), jobs=JOBS, configs=configs, steps=steps,
                resource_gate=dict(free_cpu_equivalents=64, sample_seconds=3, disk_gib=100))


def parse_cpu_stat(text: str) -> dict[int, tuple[int, int]]:
    result = {}
    for line in text.splitlines():
        words = line.split()
        if words and words[0].startswith("cpu") and words[0][3:].isdigit():
            counters = [int(v) for v in words[1:9]]
            if len(counters) < 4 or min(counters) < 0:
                raise ValueError("invalid /proc/stat counters")
            # Guest times are already included in user/nice. Only idle is free;
            # iowait/steal are conservatively treated as unavailable capacity.
            result[int(words[0][3:])] = (sum(counters), counters[3])
    if not result:
        raise ValueError("no per-CPU counters in /proc/stat")
    return result


def read_cpu_stat() -> dict[int, tuple[int, int]]:
    return parse_cpu_stat(Path("/proc/stat").read_text())


def free_cpu_equivalents(before: dict, after: dict, affinity: set[int]) -> float:
    free = 0.0
    if not affinity:
        raise ValueError("empty CPU affinity")
    for cpu in affinity:
        if cpu not in before or cpu not in after:
            raise ValueError("CPU went offline or counters missing")
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        if total <= 0 or not 0 <= idle <= total:
            raise ValueError("invalid CPU counter deltas")
        free += idle / total
    return free


def sample_free_cpus(*, deadline: float, seconds: float = 3.0) -> float:
    affinity = os.sched_getaffinity(0)
    if remaining(deadline) <= seconds:
        raise TimeoutError("deadline too close for CPU resource sample")
    before = read_cpu_stat()
    time.sleep(seconds)
    after = read_cpu_stat()
    remaining(deadline)
    if affinity != os.sched_getaffinity(0):
        raise RuntimeError("CPU affinity changed during sample")
    return free_cpu_equivalents(before, after, affinity)


def cpu_quota() -> float:
    """Read effective visible ancestor quotas on cgroup v1/v2; fail closed."""
    groups = {}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        for controller in controllers.split(","):
            groups[controller] = path
    limits = []
    found = False
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        sep = fields.index("-")
        kind = fields[sep+1]
        if kind == "cgroup2" and "" in groups:
            group, version = groups[""], 2
        elif kind == "cgroup" and "cpu" in fields[sep+3].split(",") and "cpu" in groups:
            group, version = groups["cpu"], 1
        else:
            continue
        found = True
        mount = Path(fields[4])
        root = Path(fields[3])
        relative = Path(group).relative_to(root)
        if ".." in relative.parts:
            raise RuntimeError("unresolvable cgroup namespace")
        current = mount / relative
        while True:
            if version == 2:
                file = current / "cpu.max"
                if file.exists():
                    quota, period = file.read_text().split()
                    if quota != "max":
                        limits.append(int(quota) / int(period))
            else:
                quota = int((current / "cpu.cfs_quota_us").read_text())
                period = int((current / "cpu.cfs_period_us").read_text())
                if quota >= 0:
                    limits.append(quota / period)
            if current == mount:
                break
            current = current.parent
    if not found:
        raise RuntimeError("cannot determine CPU cgroup quota")
    return min(limits, default=math.inf)


def check_resources(free: float, disk: int, quota: float) -> None:
    if not math.isfinite(free) or free < JOBS or math.isnan(quota) or quota < JOBS or disk < MIN_DISK:
        raise RuntimeError(f"resources insufficient: free CPUs={free:.4f}, quota={quota}, "
                           f"disk GiB={disk/2**30:.2f}; require >=64 CPUs and >=100 GiB")


def resource_gate(output: Path, deadline: float) -> dict:
    free = sample_free_cpus(deadline=deadline, seconds=3.0)
    quota = cpu_quota()
    parent = output
    while not parent.exists():
        parent = parent.parent
    disk = shutil.disk_usage(parent).free
    check_resources(free, disk, quota)
    remaining(deadline)
    return dict(free_cpu_equivalents=free, quota_cpus=None if math.isinf(quota) else quota,
                disk_free_bytes=disk, sample_seconds=3)


def hash_file(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if deadline is not None:
                remaining(deadline)
            block = stream.read(1024*1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_packet(bundle: Path, subject: int, condition: int, evaluation: dict,
                    deadline: float | None = None) -> None:
    """Check new native manifest identity/budgets and stream hashes, without loading arrays."""
    m = json.loads((bundle / "evaluation_manifest.json").read_text(encoding="utf-8"))
    required = {"schema_version", "subject_id", "condition", "trials", "parameters", "config",
                "pf_seeds", "simulation_seeds", "arrays", "protocol", "catalogue_metadata",
                "provenance", "inputs_sha256", "conditional_particle_contents"}
    if not required <= m.keys():
        raise ValueError("evaluation manifest missing required fields")
    if (m["schema_version"] != "gm1031-evaluation-inputs-v1" or m["subject_id"] != subject
            or m["condition"] != condition or type(m["trials"]) is not int or m["trials"] <= 0):
        raise ValueError("evaluation identity/schema/trials mismatch")
    if not all(m["config"].get(k) == v for k, v in evaluation.items()):
        raise ValueError("evaluation budget mismatch")
    if (len(m["pf_seeds"]) != evaluation["pf_repeats"]
            or len(m["simulation_seeds"]) != evaluation["simulation_repeats"]
            or not m["arrays"] or not m["protocol"].get("feature_order")
            or not m["catalogue_metadata"] or not m["provenance"]):
        raise ValueError("evaluation metadata incomplete")
    p = m["conditional_particle_contents"]
    checks = dict(status="available", file="conditional_particle_contents.npz", phase=PHASE,
                  pf_repeat_index=0, seed=m["pf_seeds"][0], particle_count=evaluation["particle_count"],
                  capacity=m["parameters"]["capacity"], pairing_axis=3 if condition == 3 else 1)
    if not all(p.get(k) == v for k, v in checks.items()) or not p.get("arrays"):
        raise ValueError("conditional particle metadata mismatch")
    for name, digest in [("evaluation_inputs.npz", m["inputs_sha256"]), (p["file"], p.get("sha256"))]:
        file = bundle / name
        if file.is_symlink() or hash_file(file, deadline) != digest:
            raise ValueError(f"packet hash mismatch: {name}")


def run_stage(step: dict, log: Path, deadline: float) -> None:
    remaining(deadline)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env.update(OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    with log.open("x", encoding="utf-8") as stream:
        proc = subprocess.Popen(step["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, shell=False)
        try:
            code = proc.wait(timeout=remaining(deadline))
        except BaseException as error:
            # Native work uses threads in this foreground process. Reap it before returning.
            try:
                proc.terminate()
            finally:
                proc.kill()
                proc.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError("total run deadline reached; foreground process stopped") from error
            raise
        if code:
            raise RuntimeError(f"stage exited {code}: {step['command']}; see {log}")
    remaining(deadline)


def execute(plan: dict, deadline: float) -> None:
    output = fresh_output(Path(plan["output"]))
    resources = resource_gate(output, deadline)
    remaining(deadline)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    def progress(event: str, **fields) -> None:
        row = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   remaining_seconds=max(0, deadline-time.monotonic()), **fields)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    try:
        progress("started", resources=resources)
        previous = None
        for step in plan["steps"]:
            remaining(deadline)
            sid = step["subject"]
            if sid != previous:
                if previous is not None:
                    resources = resource_gate(output, deadline)
                    progress("resource_check", subject=sid, resources=resources)
                previous = sid
                Path(step["base"]).mkdir(parents=True, exist_ok=False)
            progress("stage_started", subject=sid, stage=step["stage"])
            log = Path(step["base"]) / (step["stage"] + ".log")
            run_stage(step, log, deadline)
            if step["stage"] == "evaluate":
                validate_packet(Path(step["bundle"]), sid, step["condition"], step["evaluation"], deadline)
            progress("stage_completed", subject=sid, stage=step["stage"])
        remaining(deadline)
        progress("complete")
    except BaseException as error:
        progress("failed", error=str(error), error_type=type(error).__name__)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        deadline = deadline_after(args.max_hours)
        plan = make_plan(args.output, args.subjects)
        plan["max_hours"] = args.max_hours
        if args.plan_only:
            print(json.dumps(plan, indent=2))
            return 0
        if sys.platform != "linux":
            raise RuntimeError("execution requires Linux /proc and CPU affinity support")
        binary = ROOT / "build/genmodel_cpp"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"native executable is unavailable: {binary}")
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        old_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            execute(plan, deadline)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
    except (ValueError, OSError, RuntimeError, TimeoutError, KeyboardInterrupt) as error:
        print(f"Run stopped: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
