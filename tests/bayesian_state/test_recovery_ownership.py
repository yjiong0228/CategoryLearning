"""Recovery ownership, compatibility, and provenance after workflow pruning."""
from pathlib import Path
import ast
import importlib

from src.Bayesian_state import run_recovery
from src.Bayesian_state.evaluation import model_recovery as compatibility
from src.Bayesian_state.workflows.recovery.design import load_recovery_design

ROOT = Path(__file__).resolve().parents[2]


def test_recovery_operations_have_explicit_owners():
    owners = {
        'fit_recovery_dataset': 'optimization.recovery',
        'score_frozen_candidate': 'evaluation.recovery',
        'summarize_parameter_recovery': 'evaluation.recovery',
        'load_recovery_design': 'workflows.recovery.design',
        'generate_synthetic_dataset': 'workflows.recovery.generation',
        'synthetic_dataset_frame': 'simulation.recovery',
    }
    for name, owner in owners.items():
        module = importlib.import_module('src.Bayesian_state.' + owner)
        assert getattr(compatibility, name) is getattr(module, name)
        assert getattr(compatibility, name).__module__ == module.__name__
    from src.Bayesian_state.workflows.recovery import run
    from src.Bayesian_state.workflows.runs import run_model_0826_recovery
    assert run_recovery is run_model_0826_recovery is run


def test_no_production_imports_reference_models():
    assert not (ROOT / 'src/Bayesian_state/reference_models').exists()
    for path in (ROOT / 'src/Bayesian_state').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'reference_models' not in (node.module or ''), path
            elif isinstance(node, ast.Import):
                assert all('reference_models' not in alias.name for alias in node.names), path


def test_recovery_fingerprint_tracks_each_split_implementation(monkeypatch):
    design = load_recovery_design(ROOT / 'configs/exp123/specific_models/model_0826_recovery_v1.yaml')
    before = run_recovery._design_fingerprint(design)
    original_hash = run_recovery._file_sha256
    changed = {'path': None}

    def controlled_hash(path):
        return '0' * 64 if Path(path) == changed['path'] else original_hash(path)

    monkeypatch.setattr(run_recovery, '_file_sha256', controlled_hash)
    for relative in [
        'simulation/recovery.py', 'optimization/recovery.py',
        'optimization/recovery_parameters.py', 'evaluation/recovery.py',
        'utils/recovery_artifacts.py', 'workflows/recovery/design.py',
        'workflows/recovery/generation.py', 'workflows/recovery/run.py',
    ]:
        changed['path'] = ROOT / 'src/Bayesian_state' / relative
        assert run_recovery._design_fingerprint(design) != before, relative
