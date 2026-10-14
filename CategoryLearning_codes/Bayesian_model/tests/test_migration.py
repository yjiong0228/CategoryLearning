"""Shared-core regressions against saved pre-consolidation numerical outputs."""
from pathlib import Path
import ast
import importlib
import hashlib
import json
import numpy as np
import pytest
import yaml
from CategoryLearning_codes.Bayesian_model.tests.reference_cases import run_cases, ROOT, PACKAGE


def test_frozen_numeric_reference():
    reference = PACKAGE / 'tests/fixtures/pre_shared_core.npz'
    metadata = json.loads(reference.with_suffix('.json').read_text())
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == metadata['sha256'][str(reference.relative_to(ROOT))]
    with np.load(reference, allow_pickle=False) as expected:
        actual = run_cases('src.Bayesian_state')
        assert set(actual) == set(expected.files)
        for key in expected.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize('module', [
    'model.engine', 'model.state_model', 'inference.backends.particle_filter',
    'simulation.runner', 'evaluation.oral.scoring', 'hypothesis_space.similarity',
])
def test_compatibility_modules_are_the_same_object(module):
    old = importlib.import_module('CategoryLearning_codes.Bayesian_model.' + module)
    shared = importlib.import_module('src.Bayesian_state.' + module)
    assert old is shared


def journal_imports(source):
    # Artifact paths and downstream plotting CLI names are valid metadata.
    # The shared implementation must not import the journal wrapper package.
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or '')
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ''
            if name in ('__import__', 'import_module') and isinstance(node.args[0], ast.Constant):
                names.append(str(node.args[0].value))
    return [name for name in names if name.split('.')[0] == 'CategoryLearning_codes']


@pytest.mark.parametrize('source', [
    'import CategoryLearning_codes.Bayesian_model',
    'from CategoryLearning_codes import Bayesian_model',
    "importlib.import_module('CategoryLearning_codes.Bayesian_model')",
    "__import__('CategoryLearning_codes.Bayesian_model')",
])
def test_dependency_check_detects_static_and_literal_dynamic_imports(source):
    assert journal_imports(source)


def test_dependency_check_accepts_artifact_paths():
    assert not journal_imports("path = 'CategoryLearning_codes/figures/outputs'")


def test_shared_core_has_no_journal_dependency():
    for file in (ROOT / 'src/Bayesian_state').rglob('*.py'):
        assert not journal_imports(file.read_text()), str(file)


def test_recovery_entrypoints_share_implementation():
    from src.Bayesian_state.workflows.runs import run_model_0826_recovery as legacy
    from src.Bayesian_state import run_recovery as shared
    from CategoryLearning_codes.Bayesian_model import run_recovery as journal
    assert legacy is shared
    assert journal.build_parser().parse_args([]).config == PACKAGE / 'configs/recovery_v1.yaml'
    assert shared.build_parser().parse_args([]).config == ROOT / 'configs/exp123/specific_models/model_0826_recovery_v1.yaml'


def test_frozen_config_and_resource_match_manuscript():
    import hashlib
    old=(ROOT/'configs/exp123/model_struct/pmh_model_cond1_0826.yaml').read_text()
    config=yaml.safe_load((PACKAGE/'configs/model_0826.yaml').read_text())
    assert config==yaml.safe_load(old)
    assert hashlib.sha256((ROOT/config['provenance']['manuscript_path']).read_bytes()).hexdigest()==config['provenance']['manuscript_sha256']
    similarity=config['provenance']['hypothesis_similarity']
    resource=ROOT/'src/Bayesian_state/hypothesis_space/resources/similarity'/similarity['resource_filename']
    assert hashlib.sha256(resource.read_bytes()).hexdigest()==similarity['resource_sha256']
    assert np.load(resource).shape==(29,29)
    from CategoryLearning_codes.Bayesian_model.evaluation.model_recovery import load_recovery_design
    for version in [1,2]:
        design=load_recovery_design(PACKAGE/f'configs/recovery_v{version}.yaml')
        assert design.model_engine_config.is_relative_to(PACKAGE)
        assert design.parameter_space_path.is_relative_to(PACKAGE)
        assert design.base_simulation_config.is_relative_to(PACKAGE)
