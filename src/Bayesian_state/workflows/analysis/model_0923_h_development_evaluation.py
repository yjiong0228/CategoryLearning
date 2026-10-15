"""Evaluate confirmed H parameter updates on already inspected participants.

Reuses hash-verified predictions when a parameter point is unchanged; scores
changed points through the existing archived-runtime worker. The same filter
and autonomous seed banks preserve paired comparisons. The chronological test
suffix remains excluded from training, and its prior use for mechanism design
is recorded explicitly as development evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from math import ceil
from pathlib import Path

from .model_0923_h_validation import REPO, read, write, digest, source_hashes, score_jobs, execute, combine_evaluation_chunks
from ...evaluation.model_0923_h_validation import summarize
from ...utils.seeding import stable_seed


def reuse_record(nominee: dict, reference_row: dict, reference: Path, output: Path) -> dict:
    """Reuse numerical arrays only for an identical cognitive parameter point."""
    if nominee['candidate']['hyperparams'] != reference_row['candidate']['hyperparams']:
        raise ValueError('Changed parameters cannot reuse old predictions')
    if nominee['subject'] != reference_row['subject'] or nominee['model'] != reference_row['model']:
        raise ValueError('Subject/model reference mismatch')
    source = reference / 'evaluate' / f'subject_{nominee["subject"]}' / nominee['model'] / (reference_row['candidate']['id'] + '.npz')
    if digest(source) != reference_row['predictions_sha256']:
        raise ValueError('Saved reference predictions changed')
    target = output / 'evaluate' / f'subject_{nominee["subject"]}' / nominee['model'] / (nominee['candidate']['id'] + '.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())
    row = dict(reference_row, candidate=nominee['candidate'], reused=True,
               reused_predictions=str(source.resolve()), reused_record=reference_row)
    write(target.with_suffix('.json'), row)
    return row


def evaluate(frozen: dict, fit: Path, reference: Path, output: Path, generate: bool) -> None:
    output.mkdir()
    old_protocol = read(reference / 'evaluation_protocol.json')
    reference_rows = read(reference / 'evaluation.json')
    if reference_rows['status'] != 'complete':
        raise ValueError('Reference evaluation incomplete')
    original = {(r['subject'], r['model']): r for r in reference_rows['rows']}
    reused, banks = [], {}
    for nominee in frozen['nominees']:
        key = nominee['subject'], nominee['model']; old = original[key]
        if old['candidate']['hyperparams'] == nominee['candidate']['hyperparams']:
            reused.append(reuse_record(nominee, old, reference, output))
        else:
            banks[key] = [nominee['candidate']]
    write(output / 'evaluation_protocol.json', dict(nomination_sha256=digest(fit / 'nominees.json'),
        fit=str(fit.resolve()), source_sha256=source_hashes(), reference=str(reference.resolve()),
        reference_nomination_sha256=old_protocol['nomination_sha256'],
        test_opening='previously_inspected_development_suffix; parameters_frozen_before_this_evaluation',
        data_use_status='development', reused_groups=len(reused), computed_groups=len(banks)))
    jobs = score_jobs(Path(frozen['data']), output, frozen['settings'], banks, 'evaluate', False)
    groups, chunks = {}, []
    for job in jobs:
        if generate:
            job['phase'] = 'autonomous'
            subject = read(Path(job['folder']) / 'prepared.json')['subject']
            job['seeds'] = [stable_seed(dict(protocol=frozen['settings']['protocol'], base=frozen['settings']['seed'],
                role='autonomous', subject=subject, path=i)) for i in range(frozen['settings']['autonomous_paths'])]
        job['source_sha256'][str(Path(__file__).resolve().relative_to(REPO))] = digest(Path(__file__))
        groups[job['name']] = []
        default_width = frozen['settings']['autonomous_chunk' if generate else 'evaluate_seed_chunk']
        desired_chunks = ceil(frozen['settings']['parallel_budget'] / len(jobs))
        width = min(default_width, max(1, len(job['seeds']) // desired_chunks))
        for first in range(0, len(job['seeds']), width):
            chunk = deepcopy(job); target = Path(job['target'])
            chunk.update(seeds=job['seeds'][first:first+width], name=job['name']+f'_C{first:03d}',
                target=str(target.parent/'chunks'/(target.name+f'_C{first:03d}')))
            chunks.append(chunk); groups[job['name']].append(chunk)
    if chunks:
        execute(chunks, output, frozen['settings']['parallel_budget'])
    rows = reused + [combine_evaluation_chunks(job, groups[job['name']], generate) for job in jobs]
    rows.sort(key=lambda r: (r['subject'], r['model']))
    write(output / 'evaluation.json', dict(status='complete', rows=rows, data_use_status='development',
        role='autonomous_from_start_prefix_fitted' if generate else 'sequential_heldout_prechoice_predictions'))


def run(fit: Path, conditional_reference: Path, autonomous_reference: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new development-evaluation destination')
    frozen = read(fit / 'nominees.json')
    if frozen['status'] != 'frozen_for_development_evaluation' or frozen['smoke']:
        raise ValueError('Need confirmed scientific development nominees')
    parent = Path(frozen['parent_fit']); parent_frozen = read(parent / 'nominees.json')
    if frozen['data'] != parent_frozen['data'] or frozen['settings'] != parent_frozen['settings']:
        raise ValueError('Data or seed/budget protocol changed from original comparison')
    for name, sha in frozen['source_sha256'].items():
        if digest(REPO / name) != sha:
            raise ValueError('Frozen scientific implementation changed')
    for reference in (conditional_reference, autonomous_reference):
        if read(reference/'evaluation_protocol.json')['nomination_sha256'] != digest(parent/'nominees.json'):
            raise ValueError('Reference differs from original fitted comparison')
    output.mkdir(parents=True)
    write(output/'development_context.json', dict(status='parameters_frozen', fit=str(fit.resolve()),
        source_sha256=digest(Path(__file__)), role='development reevaluation after training-only numerical refinement',
        original_nomination_sha256=digest(parent/'nominees.json'), updated_nomination_sha256=digest(fit/'nominees.json')))
    evaluate(frozen, fit, conditional_reference, output/'conditional', False)
    evaluate(frozen, fit, autonomous_reference, output/'autonomous', True)
    summarize(fit, output/'conditional', output/'autonomous', output/'summary')
    write(output/'completion.json', dict(status='complete', data_use_status='development', groups=len(frozen['nominees']),
        reused_conditional_groups=read(output/'conditional/evaluation_protocol.json')['reused_groups'],
        reused_autonomous_groups=read(output/'autonomous/evaluation_protocol.json')['reused_groups']))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','conditional-reference','autonomous-reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); run(args.fit,args.conditional_reference,args.autonomous_reference,args.output)


if __name__=='__main__':main()
