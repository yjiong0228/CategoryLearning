"""Journal recovery entrypoint; orchestration and numerical code are shared."""
from pathlib import Path
from typing import Sequence
from src.Bayesian_state import run_recovery as _shared
from src.Bayesian_state.workflows.recovery.design import load_recovery_design
from .workflow import PACKAGE_DIR, validate_simulation_config

DEFAULT_CONFIG = PACKAGE_DIR / "configs/recovery_v1.yaml"


def __getattr__(name):
    return getattr(_shared, name)


def build_parser():
    return _shared.build_parser(default_config=DEFAULT_CONFIG)


def run(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    design = load_recovery_design(args.config.resolve())
    validate_simulation_config(design.base_simulation_config, subjects=list(design.subject_trial_counts))
    _shared.run(argv, default_config=DEFAULT_CONFIG)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
