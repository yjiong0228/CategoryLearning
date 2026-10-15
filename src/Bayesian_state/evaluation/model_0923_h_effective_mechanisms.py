"""Report whether a fitted R1 window can affect behavior.

At exit_rate=0, R1 and B have identical behavioral transitions at the same
parameters. The R1 evidence buffer still changes internally. This diagnostic
preserves the fitted points and identifies exact matches among their effective
intervention settings. It does not select another parameter point or establish
general model identifiability. --fit supplies frozen nominees; --output must
be a new directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd

from .model_0923_h_validation import digest, load


def summarize(fit: Path, output: Path) -> None:
    """Keep configured model names separate from their effective mechanisms."""
    if output.exists():
        raise FileExistsError('Use a new effective-mechanism diagnostic destination')
    frozen = load(fit/'nominees.json')
    source = 'src/Bayesian_state/model/modules/hypothesis_transition/mechanistic_rule_search.py'
    repo = Path(__file__).resolve().parents[3]
    if digest(repo/source) != frozen['source_sha256'][source]:
        raise ValueError('Fitted H implementation differs from the audited zero-exit equivalence')
    rows = []
    for nominee in frozen['nominees']:
        mechanism = deepcopy(frozen['settings']['models'][nominee['model']])
        point = nominee['candidate']['hyperparams']
        exit_rate = point.get('engine.modules.hypo_transitions_mod.kwargs.exit_rate')
        r1_closed = mechanism['kind'] == 'R1' and exit_rate == 0.
        effective = {'kind':'B'} if r1_closed else mechanism
        signature = json.dumps(dict(intervention=effective,parameters=point),sort_keys=True)
        rows.append(dict(subject=nominee['subject'],model=nominee['model'],candidate=nominee['candidate']['id'],
            exit_rate=exit_rate,r1_window=mechanism.get('window') if mechanism['kind']=='R1' else None,
            r1_window_can_affect_behavior=(not r1_closed) if mechanism['kind']=='R1' else None,
            effective_intervention=json.dumps(effective,sort_keys=True),
            behavioral_point_sha256=hashlib.sha256(signature.encode()).hexdigest(),
            interpretation=('退出率为0；该参数点的R1行为等价于相同参数的B，窗口只更新内部记录'
                            if r1_closed else '保留原机制设定；其参数可区分性需单独检验')))
    frame = pd.DataFrame(rows)
    matches = frame.groupby(['subject','behavioral_point_sha256']).model.agg(list).to_dict()
    frame['other_fitted_models_at_same_behavioral_point'] = [
        ';'.join(model for model in matches[(row.subject,row.behavioral_point_sha256)] if model!=row.model)
        for row in frame.itertuples(index=False)]
    output.mkdir(parents=True)
    frame.to_csv(output/'effective_mechanisms.csv',index=False)
    with (output/'validation.json').open('x') as handle:
        json.dump(dict(status='complete',fit=str(fit.resolve()),groups=len(frame),
            smoke=frozen['smoke'],data_use_status=frozen.get('data_use_status',frozen['settings'].get('data_use_status','development')),
            nomination_sha256=digest(fit/'nominees.json'),source_sha256=digest(Path(__file__)),
            h_source_sha256=digest(repo/source),
            equivalence='R1 with exit_rate=0 and B at identical parameters; behavioral transitions and RNG consumption coincide',
            scope='exact parameter-boundary diagnostic; fitted parameters and predictions unchanged'),handle,indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();summarize(args.fit,args.output)


if __name__ == '__main__':main()
