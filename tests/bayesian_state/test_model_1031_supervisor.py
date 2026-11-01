"""The watchdog stops only the process group it created, including orphan workers."""
import os
from pathlib import Path
import subprocess
import sys
import time

from src.Bayesian_state.workflows.runs.supervise_model_1031_finalization import stop_group, tree_cpu


def test_cpu_counter_includes_children_launched_from_other_threads(tmp_path):
    def process(pid, ticks, children_by_thread):
        root = tmp_path / str(pid)
        root.mkdir()
        fields = ["0"] * 20
        for index, value in zip((11, 12, 13, 14), ticks):
            fields[index] = str(value)
        (root / "stat").write_text(f"{pid} (python) " + " ".join(fields))
        for thread, children in children_by_thread.items():
            path = root / "task" / str(thread)
            path.mkdir(parents=True)
            (path / "children").write_text(" ".join(map(str, children)))

    process(100, (10, 20, 30, 40), {100: [], 101: [200], 102: [300]})
    process(200, (100, 200, 300, 400), {200: [400]})
    process(300, (1, 2, 3, 4), {300: []})
    process(400, (11, 12, 13, 14), {400: []})
    assert tree_cpu(100, tmp_path) == 1160 / os.sysconf("SC_CLK_TCK")


def test_stop_group_terminates_a_running_owned_phase():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        assert os.getpgid(process.pid) == process.pid
        assert tree_cpu(process.pid) >= 0.
        stop_group(process)
        assert process.poll() is not None
        stop_group(process)
    finally:
        stop_group(process)


def test_stop_group_includes_worker_after_the_phase_parent_exits():
    code = ("import subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(child.pid,flush=True)")
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        worker = int(process.stdout.readline())
        process.wait(timeout=5.)
        stop_group(process)
        deadline = time.monotonic() + 2.
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{worker}/stat").read_text().split(") ", 1)[1].split()[0]
            except FileNotFoundError:
                break
            if status == "Z":
                break
            time.sleep(.02)
        else:
            raise AssertionError("Owned worker survived process-group termination")
    finally:
        stop_group(process)
        process.stdout.close()
