"""The watchdog stops only the process group it created, including orphan workers."""
import os
from pathlib import Path
import subprocess
import sys
import time

from src.Bayesian_state.workflows.runs.supervise_model_1031_finalization import stop_group, tree_cpu


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
