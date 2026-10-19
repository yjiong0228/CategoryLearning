"""Check chronological information, response alphabets and fixed-state probes."""
from copy import deepcopy
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.Bayesian_state.workflows.analysis.model_0923_rnn_comparator import (
    ROOT, RNN, ENTRY, digest, replay, reusable_fits, rnn_module, validate_arrays,
)


def arrays():
    return dict(stimulus=np.array([[.2,.3,.4,.5],[.6,.2,.7,.4],[.3,.9,.1,.8]]),
                choices=np.array([1,2,1]),feedback=np.array([0.,1.,0.]))


@pytest.mark.parametrize('responses',[2,4])
def test_replay_preserves_original_choice_and_rho_and_actual_probe(responses):
    torch.set_num_threads(1);torch.manual_seed(13)
    module=rnn_module(ROOT/RNN,responses);model=module.TinyRNN(16)
    data=arrays();validate_arrays(data,responses)
    before={k:v.clone() for k,v in model.state_dict().items()}
    captured=replay(module,model,data,reflections=True)
    expected=module.get_predictions(model,data['stimulus'],data['choices'],data['feedback'])
    np.testing.assert_array_equal(captured['probabilities'].argmax(1),expected)
    np.testing.assert_array_equal(captured['rho'],module.compute_rho_traj(model,data['stimulus'],data['choices'],data['feedback']))
    assert captured['probabilities'].shape==(3,responses)
    np.testing.assert_allclose(captured['reflection_probabilities'][:,0],captured['probabilities'],atol=2e-7,rtol=0)
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])


def test_current_choice_feedback_and_future_inputs_do_not_change_current_prediction():
    torch.manual_seed(17);module=rnn_module(ROOT/RNN,4);model=module.TinyRNN(16)
    data=arrays();changed=deepcopy(data)
    changed['choices'][1:]=[3,4];changed['feedback'][1:]=[.5,.5];changed['stimulus'][2]=.1
    a,b=replay(module,model,data),replay(module,model,changed)
    np.testing.assert_array_equal(a['probabilities'][:2],b['probabilities'][:2])
    np.testing.assert_array_equal(a['rho'][:2],b['rho'][:2])


def test_actual_input_changes_leave_pre_stimulus_state_fixed():
    torch.manual_seed(17);module=rnn_module(ROOT/RNN,4);model=module.TinyRNN(16)
    data=arrays();changed=deepcopy(data);changed['stimulus'][1]=1.-data['stimulus'][1]
    a,b=replay(module,model,data),replay(module,model,changed)
    np.testing.assert_array_equal(a['rho'][:2],b['rho'][:2])
    assert not np.array_equal(a['probabilities'][1],b['probabilities'][1])


@pytest.mark.parametrize('changed_calculation', [False, True])
def test_reuse_accepts_json_subject_keys_but_rejects_changed_scoring(tmp_path, changed_calculation):
    settings = {'expected_boundaries': {107: [1, 128, 64]}}
    text = (ROOT/ENTRY).read_text()
    text = text.replace('atol=p.shape[1]*np.finfo(p.dtype).eps', 'atol=2e-7')
    if changed_calculation:
        text = text.replace('np.clip(p, 1e-12, 1.)', 'np.clip(p, 1e-6, 1.)')
    old = tmp_path/'original.py'; old.write_text(text)
    protocol = dict(settings=settings, smoke=False, jobs=[],
                    inputs_sha256={str((ROOT/ENTRY).resolve()): digest(old)})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    if changed_calculation:
        with pytest.raises(ValueError, match='scoring calculation changed'):
            reusable_fits(tmp_path, old, settings, {})
    else:
        assert reusable_fits(tmp_path, old, settings, {}) == {}
        (tmp_path/'nominees.json').write_text('{}')
        with pytest.raises(ValueError, match='remain frozen before test'):
            reusable_fits(tmp_path, old, settings, {})
