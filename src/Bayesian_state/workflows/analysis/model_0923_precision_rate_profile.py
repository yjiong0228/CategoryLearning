"""Profile the additive learning rate while fixing all other fitted coordinates.

This diagnostic separates rate changes from changes in the common H parameters.
It leaves the nominated cohort fits immutable and uses only training choices.
Rates are expressed in the effective additive coordinate; the existing ADD10
adapter stores rate / 10. New output directories retain all seeds and sources.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import model_0923_precision_refit_recovery as recovery
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed
from ...evaluation.model_0923_h_validation import nll, jackknife_scores, jackknife_se


def run(fit: Path, config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():raise FileExistsError('Use a new rate-profile directory')
    base=recovery.base;v=base.validation();audit=yaml.safe_load(config.read_text())
    frozen=v.read(fit/'nominees.json')
    if frozen['smoke'] or frozen['status']!='frozen_before_test':raise ValueError('Need frozen scientific fits')
    for name,sha in frozen['source_sha256'].items():
        if v.digest(base.REPO/name)!=sha:raise ValueError('Fitted implementation changed')
    subject,model=audit['subject'],audit['model']
    nominee=next(r for r in frozen['nominees'] if r['subject']==subject and r['model']==model)
    arm=frozen['settings']['arms'][model]
    if arm['precision']!='fast' or arm['rate_scale']!=10. or arm['upper']!=25.:
        raise ValueError('This profile uses the declared additive 0–10 adapter')
    reference=nominee['candidate']['hyperparams'];reference_rate=reference[base.RATE]*arm['rate_scale']
    rates=audit['effective_rates']
    if len(set(rates))!=len(rates) or reference_rate not in rates or any(not 0<=r<=10 for r in rates):
        raise ValueError('Need distinct supported rates, including the frozen reference')
    if smoke:rates=[rates[0],1.,reference_rate]
    points=[candidate(dict(reference,**{base.RATE:rate/arm['rate_scale']}),'fixed_other_parameters_rate_profile') for rate in rates]
    if len({r['id'] for r in points})!=len(points):raise ValueError('Duplicate profile points')
    for point in points:
        assert all(value==reference[key] for key,value in point['hyperparams'].items() if key!=base.RATE)
    reference_id=next(r['id'] for r in points if r['hyperparams']==reference)
    assert reference_id==nominee['candidate']['id']
    settings=deepcopy(frozen['settings'])
    settings.update(protocol=audit['protocol'],seed=audit['seed'],parallel_budget=audit['parallel_budget'])
    settings['budgets']['profile']=dict(particles=audit['particles'],seeds=audit['seeds'])
    recovery.configure()
    inherited=base.source_hashes
    def hashes(worker):
        result=inherited(worker);result[str(Path(__file__).resolve().relative_to(base.REPO))]=worker.digest(Path(__file__))
        return result
    base.source_hashes=hashes
    output.mkdir(parents=True);sources=hashes(v);base.archive(output,config,sources)
    v.write(output/'protocol.json',dict(fit=str(fit.resolve()),nomination_sha256=v.digest(fit/'nominees.json'),
        audit=audit,smoke=smoke,settings=settings,source_sha256=sources,reference_id=reference_id,
        reference_parameters=reference,effective_rates=rates,
        data_use='training-only diagnostic; other parameters fixed; cohort nominations remain immutable'))
    rows=base.stage(settings,Path(frozen['data']),output,{(subject,model):points},'profile',smoke)[subject,model]
    arrays={};seed_bank=None;inputs={}
    for row in rows:
        file=Path(row['score_file']);record=v.read(file)
        assert record['trials']==(8 if smoke else v.read(Path(frozen['data'])/f'subject_{subject}'/'prepared.json')['training_trials'])
        assert v.digest(file.with_suffix('.npz'))==record['predictions_sha256']
        inputs[str(file.with_suffix('.npz'))]=record['predictions_sha256']
        with np.load(file.with_suffix('.npz')) as z:
            arrays[row['id']]=z['observed_probability'];seeds=z['seeds']
        if seed_bank is None:seed_bank=seeds
        np.testing.assert_array_equal(seeds,seed_bank)
    previous=frozen['settings'];old_seeds=set()
    for stage,budget in previous['budgets'].items():
        old_seeds.update(stable_seed(dict(protocol=previous['protocol'],base=previous['seed'],stage=stage,
            subject=subject,repeat=i)) for i in range(budget['seeds']))
    assert not set(seed_bank)&old_seeds
    anchor=arrays[reference_id];records=[]
    for row in rows:
        values=arrays[row['id']];jk=jackknife_scores(values)
        records.append(dict(subject=subject,model=model,candidate=row['id'],
            effective_rate=row['hyperparams'][base.RATE]*arm['rate_scale'],reference=row['id']==reference_id,
            choice_nll=nll(values),mc_se=jackknife_se(jk),delta_from_reference=nll(values)-nll(anchor),
            delta_mc_se=jackknife_se(jk-jackknife_scores(anchor)),particles=2 if smoke else audit['particles'],seeds=len(seed_bank)))
    pd.DataFrame(records).sort_values('effective_rate').to_csv(output/'rate_profile.csv',index=False)
    assert v.digest(fit/'nominees.json')==v.read(output/'protocol.json')['nomination_sha256']
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,points=len(points),
        nomination_sha256=v.digest(fit/'nominees.json'),protocol_sha256=v.digest(output/'protocol.json'),
        inputs_sha256=inputs,summary_sha256=v.digest(output/'rate_profile.csv')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.fit,args.config,args.output,args.smoke)
