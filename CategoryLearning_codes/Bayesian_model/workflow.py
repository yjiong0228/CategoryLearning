"""Journal data policy; experiment semantics and numeric code belong to the shared core."""
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.Bayesian_state.simulation.config import load_yaml, resolve_subjects
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.subjects import resolve_subject_config

PACKAGE_DIR = Path(__file__).resolve().parent


def validate_dataset(config: Mapping[str, Any], yaml_dir: Path) -> None:
    """Reject journal inputs outside data/exp123/, including symlinks and absolute overrides."""
    allowed = (ROOT_DIR / "data" / "exp123").resolve()
    for name, path in resolve_dataset_paths(config, yaml_dir).items():
        if not path.resolve().is_relative_to(allowed):
            raise ValueError(f"Journal {name} must be inside data/exp123/: {path}")


def validate_simulation_config(config_path: str | Path, *,
                               subjects: Sequence[int] | None = None,
                               subject_range: Sequence[int] | None = None) -> Path:
    """Validate the resolved configuration for every selected subject before running."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()
    config = load_yaml(path)
    validate_dataset(config, path.parent)
    for subject in resolve_subjects(subjects, subject_range, config):
        validate_dataset(resolve_subject_config(config, subject), path.parent)
    return path
