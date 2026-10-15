"""Screen every one-step parameter neighbor, then independently confirm one.

This bounded training-only continuation completes coordinate coverage after
the preceding search's ordered ray budget. It reports screened local evidence;
one confirmed challenger does not establish a global or exact local optimum.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from .model_0923_h_validation import REPO, read, write, digest, model_space
from .model_0923_h_training_refinement import score_stage, paired_comparison
from ...optimization.search.adaptive_proposals import candidate, neighbor_values


def coordinate_bank(point: dict, space: dict) -> list[dict]:
    """Retain the anchor and every valid adjacent value of every coordinate."""
    if set(point) != set(space) or any(value not in space[key] for key,value in point.items()):
        raise ValueError('Anchor must belong to the complete declared support')
    anchor = candidate(point, 'coordinate_check_anchor')
    bank = {anchor['id']: anchor}
    for key, values in space.items():
        for value in neighbor_values(key, point[key], values):
            proposal = deepcopy(point); proposal[key] = deepcopy(value)
            row = candidate(proposal, 'complete_coordinate_neighbor', anchor=anchor['id'], parameter=key)
            bank[row['id']] = row
    return list(bank.values())


def reference_fit(fit: Path, frozen: dict) -> str:
    """Keep the original prediction reference through development continuations."""
    if frozen['status'] == 'frozen_for_development_evaluation':
        return frozen['parent_fit']
    return str(fit.resolve())


def run(fit: Path, config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new coordinate-check destination')
    frozen = read(fit/'nominees.json'); settings = yaml.safe_load(config.read_text())
    if frozen['smoke'] or frozen['status'] not in ('frozen_before_test','frozen_for_development_evaluation'):
        raise ValueError('Need frozen scientific starting points')
    for name, sha in frozen['source_sha256'].items():
        if digest(REPO/name) != sha:
            raise ValueError('Frozen implementation changed')
    selected = frozen['nominees'][:1] if smoke else frozen['nominees']
    banks, originals = {}, {}
    for row in selected:
        key = row['subject'],row['model']; originals[key] = row
        space,_ = model_space(frozen['settings'],row['model'])
        bank = coordinate_bank(row['candidate']['hyperparams'],space)
        banks[key] = bank[:3] if smoke else bank
    output.mkdir(parents=True)
    write(output/'protocol.json',dict(fit=str(fit.resolve()),parent_nomination_sha256=digest(fit/'nominees.json'),
        settings=settings,smoke=smoke,source_sha256=digest(Path(__file__)),
        data_boundary='training.npz only',data_use_status='development',
        scope='complete one-step coordinate neighborhood; independent confirmation of screening minimum',
        banks=[dict(subject=k[0],model=k[1],candidates=rows) for k,rows in banks.items()]))
    screened = score_stage(frozen,settings,banks,'screen',output,smoke)
    banks, challengers = {}, {}
    for key, rows in screened.items():
        anchor = originals[key]['candidate']; challengers[key] = rows[0]['id']
        banks[key] = [anchor] + ([rows[0]] if rows[0]['id'] != anchor['id'] else [])
    write(output/'confirmation_selection.json',dict(groups=[dict(subject=k[0],model=k[1],candidates=rows) for k,rows in banks.items()]))
    confirmed = score_stage(frozen,settings,banks,'confirm',output,smoke)
    nominees, decisions = [], []
    for key, rows in confirmed.items():
        indexed={r['id']:r for r in rows};anchor=indexed[originals[key]['candidate']['id']];challenger=indexed[challengers[key]]
        decision=paired_comparison(anchor,challenger,settings['mcse_multiplier'])
        decisions.append(dict(subject=key[0],model=key[1],neighbors_screened=len(screened[key])-1,**decision))
        nominee=challenger if decision['accepted'] else anchor
        nominees.append(dict(originals[key],candidate=nominee,finalists=rows,coordinate_neighbors=screened[key],
            previous_candidate_id=anchor['id'],scientific_status='bounded_coordinate_screen_with_independent_numerical_confirmation'))
    pd.DataFrame(decisions).to_csv(output/'decisions.csv',index=False)
    write(output/'nominees.json',dict(status='frozen_for_development_evaluation',settings=frozen['settings'],
        source_sha256=frozen['source_sha256'],smoke=smoke,data=frozen['data'],nominees=nominees,
        parent_fit=reference_fit(fit,frozen),coordinate_parent_fit=str(fit.resolve()),
        coordinate_settings=settings,data_use_status='development'))
    write(output/'completion.json',dict(status='complete',smoke=smoke,groups=len(nominees),
        accepted=sum(r['accepted'] for r in decisions),test_arrays_opened=False))
    print('Complete coordinate-neighbor screen and independent confirmation finished',flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.fit,args.config,args.output,args.smoke)


if __name__=='__main__':main()
