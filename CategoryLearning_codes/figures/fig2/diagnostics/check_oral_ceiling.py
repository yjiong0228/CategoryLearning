"""Reproduce oral target mass using unchanged production encoder."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from CategoryLearning_codes.Bayesian_model.evaluation.oral.scoring import OralAlignmentScoringMixin as Encoder
from CategoryLearning_codes.Bayesian_model.hypothesis_space import ContinuousPartition

ROOT=Path(__file__).resolve().parents[4]

def main():
    out=ROOT/'CategoryLearning_codes/figures/outputs/fig2/oral_alignment_audit_v1'
    out.mkdir(exist_ok=False,parents=True)
    e=Encoder(); p=ContinuousPartition(n_dims=4,n_cats=2)
    rows=[]
    for sigma in [.05,.075,.1,.15,.2]:
        latest={}
        for cat in [1,2]:
            center=e._category_prototypes(p,0,cat-1)[0]
            q,d=e._center_oral_distribution(center,cat,p,center_sigma=sigma,return_diagnostics=True)
            latest[cat]={'distribution':q,'diagnostics':d}
            state,diag=e._category_state_distribution(latest,p,'center',center_sigma=sigma)
            rows.append({'sigma':sigma,'categories_seen':cat,'center':center.tolist(),'instantaneous_h0':q[0],'state_h0':state[0]})
            if sigma==.1 and cat==2:
                pd.DataFrame({'hypothesis':range(p.length),'mass':state}).to_csv(out/'ideal_report_distribution.csv',index=False)
    df=pd.read_csv(ROOT/'data/processed/Task2_processed.csv').query('iSub == 101')
    oral=e.compute_oral_mass_probabilities(df,subjects=[101],partitions_by_subject={101:p})[101]
    np.savez_compressed(out/'s101_oral.npz',oral_mass=oral['oral_mass'],instantaneous=oral['instantaneous_oral_mass'])
    print('ideal',rows)
    print('S101',np.nanmax(oral['oral_mass'][:,0]),np.nanmax(oral['instantaneous_oral_mass'][:,0]))
    pd.DataFrame(rows).to_csv(out/'ideal_sigma_sensitivity.csv',index=False)
    pd.DataFrame({'trial':range(1,len(df)+1),'oral_target_mass':oral['oral_mass'][:,0], 'instantaneous_target_mass':oral['instantaneous_oral_mass'][:,0], 'valid_report':oral['valid_oral_report']}).to_csv(out/'s101_oral_target.csv',index=False)
    (out/'encoder_metadata.json').write_text(json.dumps({k:v for k,v in oral.items() if isinstance(v,(str,int,float))},indent=2))

if __name__=='__main__': main()
