"""Compatibility entrypoint for the shared Model 0826 recovery workflow."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.Bayesian_state import run_recovery as _shared

if __name__ == "__main__":
    _shared.main()
else:
    sys.modules[__name__] = _shared
