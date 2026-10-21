"""Check closed coordinates and matched, training-only candidate construction."""
from pathlib import Path

import yaml

from src.Bayesian_state.workflows.analysis import model_1021_simplification as study


def test_parameter_counts_and_shared_supports():
    settings = yaml.safe_load(Path('configs/shared/model1021_simplification.yaml').read_text())
    for condition, count in [(1, 8), (2, 8), (3, 9)]:
        reference = study.model_space(settings, 'REFERENCE', condition)
        compact = study.model_space(settings, 'COMPACT', condition)
        assert sum(len(values) > 1 for values in reference.values()) == 13
        assert sum(len(values) > 1 for values in compact.values()) == count
        fixed = study.fixed_coordinates('COMPACT', condition)
        for key, values in compact.items():
            assert values == ([fixed[key]] if key in fixed else reference[key])


def test_training_banks_have_equal_counts_and_closed_coordinates(monkeypatch):
    settings = yaml.safe_load(Path('configs/shared/model1021_simplification.yaml').read_text())
    old_read, old_digest = study.io.read, study.io.digest
    touched = []

    def guard(path, function):
        path = Path(path)
        assert 'sealed_test' not in str(path) and 'oral' not in str(path)
        touched.append(path)
        return function(path)

    monkeypatch.setattr(study.io, 'read', lambda p: guard(p, old_read))
    monkeypatch.setattr(study.io, 'digest', lambda p: guard(p, old_digest))
    banks, required, spaces, inputs = study.starting_banks(settings, False)
    assert len(banks) == 6 and inputs and touched
    for key, rows in banks.items():
        assert len(rows) == len({r['id'] for r in rows}) == 40
        assert required[key] == rows[0]
        for row in rows:
            assert all(row['hyperparams'][k] in values for k, values in spaces[key].items())
