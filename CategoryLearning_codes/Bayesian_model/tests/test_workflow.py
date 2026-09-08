"""Project policy must not alter the shared dataset resolver or scientific settings."""
from pathlib import Path
import pytest
import yaml
from CategoryLearning_codes.Bayesian_model.workflow import (
    PACKAGE_DIR, validate_dataset, validate_simulation_config,
)
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.datasets import resolve_dataset_paths


@pytest.mark.parametrize('folder', ['data/exp123', 'data/exp4', 'data/exp5', 'data/meg'])
def test_shared_dataset_resolution_is_project_independent(folder):
    config = {'dataset': {'processed_dir': f'{folder}/processed', 'learning_data': 'example.csv'}}
    paths = resolve_dataset_paths(config, ROOT_DIR)
    assert paths['learning_data'] == ROOT_DIR / folder / 'processed/example.csv'
    if folder == 'data/exp123':
        validate_dataset(config, ROOT_DIR)
    else:
        with pytest.raises(ValueError, match='inside data/exp123/'):
            validate_dataset(config, ROOT_DIR)


@pytest.mark.parametrize('key', ['learning_data', 'feature_order_data', 'perception_summary', 'perception_summary_72'])
def test_journal_rejects_file_override_outside_data(key):
    config = {'dataset': {key: str(ROOT_DIR / 'data/exp4/processed/example.csv')}}
    with pytest.raises(ValueError, match='inside data/exp123/'):
        validate_dataset(config, ROOT_DIR)


def test_journal_validates_selected_subject_overrides(tmp_path):
    config = yaml.safe_load((PACKAGE_DIR / 'configs/smoke_simulation.yaml').read_text())
    config['dataset']['processed_dir'] = str(ROOT_DIR / 'data/exp123/processed')
    config['subject_overrides'] = {'101': {'dataset': {'processed_dir': str(ROOT_DIR / 'data/meg/processed')}}}
    path = tmp_path / 'simulation.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='inside data/exp123/'):
        validate_simulation_config(path)
    assert validate_simulation_config(path, subjects=[102]) == path


def test_journal_configs_and_defaults():
    for name in ['smoke_simulation.yaml', 'recovery_simulation.yaml']:
        validate_simulation_config(PACKAGE_DIR / 'configs' / name)
    from CategoryLearning_codes.Bayesian_model import run_hyper_evaluation as paper
    from src.Bayesian_state import run_hyper_evaluation as shared
    kw = dict(hyper_config_path=None, stage='fine', candidates_json=None, candidate_key=None)
    assert paper.infer_candidate_source(**kw) == (None, 'cond1')
    assert shared.infer_candidate_source(**kw) == (shared.DEFAULT_CANDIDATES_JSON, 'cond1')
    assert paper.parse_args([]).input_dir == paper.DEFAULT_INPUT_DIR
    assert shared.parse_args([]).input_dir == shared.DEFAULT_INPUT_DIR
    from CategoryLearning_codes.Bayesian_model.optimization.parameter_space import load_model_parameter_space
    with pytest.raises(ValueError, match='model_0826'):
        load_model_parameter_space(PACKAGE_DIR / 'configs/parameter_space_0826.yaml', expected_model_id='model_0818')
