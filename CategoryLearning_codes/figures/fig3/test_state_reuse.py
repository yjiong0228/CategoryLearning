"""Cache reuse must preserve subject, nominee, seed and numerical budget."""
import json

import numpy as np
import pytest

from CategoryLearning_codes.figures.fig3.nine_subject_states import reusable_states
from src.Bayesian_state.utils.seeding import stable_seed


def cache(tmp_path):
    config={'subjects':[104,215],'source_runs':['old','new'],'particle_count':128,
            'selected_repeats':1,'alternative_repeats':1,'seed_base':2026092003}
    old=dict(config,subjects=[104],source_runs=['old'])
    (tmp_path/'manifest.json').write_text(json.dumps({'config':old,'smoke':False,
        'cohort':[{'subject':104,'source':'old','selected':'nominee','alternative':'near'}]}))
    (tmp_path/'completion.json').write_text('{}')
    rows=[{'source':'old','fit':{'subject':104,'condition':1,'selected':'nominee'},
           'alternative':{'id':'near'}}]
    (tmp_path/'S104').mkdir()
    for variant,point in [('selected','nominee'),('alternative','near')]:
        np.savez(tmp_path/'S104'/f'{variant}_00.npz',subject=104,condition=1,
                 point_id=point,particles=128,
                 seed=stable_seed({'role':'nine_subject_figures','base':config['seed_base'],
                                   'subject':104,'repeat':0}))
    return rows,config


def test_compatible_subset_can_be_reused_without_writing(tmp_path):
    rows,config=cache(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    files=reusable_states(tmp_path,rows,config)
    assert len(files)==2 and all(receipt['cached'] for _,receipt in files)
    assert all(path.read_bytes()==value for path,value in before.items())


@pytest.mark.parametrize('field,value',[('particle_count',256),('seed_base',42)])
def test_changed_replay_settings_cannot_reuse_cache(tmp_path,field,value):
    rows,config=cache(tmp_path);config[field]=value
    with pytest.raises(ValueError,match='settings differ'):
        reusable_states(tmp_path,rows,config)


def test_changed_nominee_cannot_reuse_cache(tmp_path):
    rows,config=cache(tmp_path);rows[0]['fit']['selected']='different'
    with pytest.raises(ValueError,match='nominee/source changed'):
        reusable_states(tmp_path,rows,config)


def test_corrupted_state_identity_is_rejected(tmp_path):
    rows,config=cache(tmp_path)
    path=tmp_path/'S104'/'selected_00.npz'
    with np.load(path) as saved:
        values=dict(saved)
    values['seed']=np.array(1)
    np.savez(path,**values)
    with pytest.raises(ValueError,match='identity mismatch'):
        reusable_states(tmp_path,rows,config)
