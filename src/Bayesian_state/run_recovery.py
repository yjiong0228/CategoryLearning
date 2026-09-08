"""Public recovery CLI; phase orchestration lives in workflows.recovery.run."""
import sys
from src.Bayesian_state.workflows.recovery import run as _workflow

if __name__ == "__main__":
    _workflow.main()
else:
    sys.modules[__name__] = _workflow
