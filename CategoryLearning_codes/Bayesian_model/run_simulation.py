"""Run journal simulations using the shared model and journal data scope."""
from pathlib import Path
from typing import Sequence
from src.Bayesian_state import run_simulation as _shared
from .workflow import validate_simulation_config


def __getattr__(name):
    return getattr(_shared, name)


def run_simulation(config_path: str | Path, *, subjects: Sequence[int] | None = None,
                   subject_range: Sequence[int] | None = None) -> list[Path]:
    path = validate_simulation_config(config_path, subjects=subjects, subject_range=subject_range)
    return _shared.run_simulation(path, subjects=subjects, subject_range=subject_range)


def main() -> None:
    _shared.configure_logging()
    args = _shared.parse_args()
    run_simulation(args.config, subjects=args.subjects, subject_range=args.subject_range)


if __name__ == "__main__":
    main()
