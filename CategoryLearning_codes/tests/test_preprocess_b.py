"""Check integrated process_use encoding with counterbalanced part assignments."""
import pandas as pd
import pytest
from src.preprocess_b import Preprocessor_B


@pytest.mark.parametrize('names,text,expected',[
    (['tail','head','leg','neck'],'头长尾巴短',[1,1,0,0]),
    (['head','neck','leg','tail'],'头长脖子短',[1,1,0,0]),
    (['tail','leg','neck','head'],'头长脖子短',[0,0,1,1]),
    (['tail','leg','head','neck'],'尾比脚长',[1,0,0,0]),
    (['tail','head','leg','neck'],'其他都一样',[0,0,0,0]),
    (['tail','head','leg','neck'],None,[0,0,0,0]),
    (['blue','pink','green','yellow'],'绿色比蓝长',[1,0,1,0]),
])
def test_process_codes_mentions_in_feature_order(names,text,expected):
    stimulus=pd.DataFrame({'stiID':[1],**{f'feature{i}_name':[n] for i,n in enumerate(names,1)},
                           **{f'feature{i}':[.2] for i in range(1,5)}})
    behavior=pd.DataFrame({'stiID':[1], 'condition':[3], 'iSession':[1], 'iBlock':[1],
                          'iTrial':[1], 'category':[2], 'choice':[2], 'feedback':[1],
                          'ambiguous':[0], 'choRT':[1.2]})
    recording=pd.DataFrame({'iSession':[1], 'iTrial':[1], 'text':[text]})
    feature_map=({'neck':0,'head':1,'leg':2,'tail':3} if 'tail' in names
                 else {'green':0,'yellow':1,'pink':2,'blue':3})
    args=(1,feature_map,stimulus,behavior)
    out=Preprocessor_B().process(*args,recording)
    cols=[f'feature{i}_use' for i in range(1,5)]
    assert out[cols].iloc[0].tolist()==expected
    assert out.category.tolist()==[2]
    assert Preprocessor_B().process(*args)[cols].eq(0).all().all()
