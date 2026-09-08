"""Dataset namespaces and configuration lookup after experiment consolidation."""
from pathlib import Path
import pytest
import yaml
from src.Bayesian_state.utils.paths import ROOT_DIR, PROCESSED_DATA_DIR
from src.Bayesian_state.utils.config import MODEL_STRUCT
from src.Bayesian_state.utils.datasets import resolve_dataset_paths


def test_model_lookup_includes_shared_and_exp123_definitions():
    assert 'p_model' in MODEL_STRUCT
    assert 'pmh_model_cond1_0826' in MODEL_STRUCT
    assert PROCESSED_DATA_DIR == ROOT_DIR / 'data/exp123/processed'
    assert (PROCESSED_DATA_DIR / 'Task2_processed.csv').is_file()


@pytest.mark.parametrize('path', [
    'configs/exp123/simulation_cfg/pmh_cond1_simulation.yaml',
    'configs/exp4/simulation_cfg/pmh_prob_proto_simulation.yaml',
    'configs/exp5/simulation_cfg/pmh_rule1_simulation.yaml',
])
def test_experiment_configuration_resolves_inputs_and_output_root(path):
    source = ROOT_DIR / path
    config = yaml.safe_load(source.read_text())
    dataset = resolve_dataset_paths(config, source.parent)
    assert dataset['learning_data'].is_file()
    assert (source.parent / config['engine_config_path']).resolve().is_file()
    assert (source.parent / config['output_dir']).resolve().is_relative_to(ROOT_DIR / 'results')
