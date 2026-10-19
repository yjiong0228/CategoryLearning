"""Cross belief persistence and precision learning at two training-only anchors.

Run --config YAML --output NEW_DIRECTORY [--smoke]. Anchors are the best completed
SEPARATE screen/refinement SEARCH points at the declared gamma values. Both
ONLINE and SEPARATE receive the same bounded parameter bank and new common
filter seeds. Only original training choices are scored; original nominations,
later evaluations and oral data stay outside this diagnostic. This is a local
mechanism/parameter interaction study, not a replacement full refit.

The existing numerical workers execute all cognitive calculations. Smoke runs
the two original anchors under both mechanisms, one seed, four particles and
the first eight training trials, using one worker.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PROBE = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_probe.py'
GAMMA = 'engine.modules.memory_mod.kwargs.gamma'
RATE = 'engine.modules.beta_mod.kwargs.learning_rate'
TRANSFER = 'engine.modules.hypo_transitions_mod.kwargs.transfer_scale'
MODES = {'ONLINE': 'online_retention_evidence', 'SEPARATE': 'separate_gamma_belief_evidence'}


def helpers():
    spec = importlib.util.spec_from_file_location('belief_precision_worker', ROOT / PROBE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    job = json.loads(file.read_text())
    if job['phase'] != 'training' or job.get('diagnostic') != 'belief_precision_training_grid':
        raise ValueError('This diagnostic scores original training choices only')
    mode = job.get('probe_mechanism')
    if (mode not in MODES or job.get('belief_intervention') != MODES[mode]
            or job.get('online_belief') is not (mode == 'ONLINE')):
        raise ValueError('Diagnostic worker mechanism differs from its declared cell')
    helpers().worker(file)


def build_bank(anchors: list[dict], gammas: list[float], rates: list[float], space: dict) -> list[dict]:
    """Change only the two declared coordinates; retain both exact anchors."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, point_id
    if (not anchors or len(gammas) != len(set(gammas)) or len(rates) != len(set(rates))
            or not gammas or not rates or not set(gammas) <= set(space[GAMMA])
            or not set(rates) <= set(space[RATE])):
        raise ValueError('Require unique nonempty grid values within the original supports')
    bank = {}
    for anchor in anchors:
        original = anchor['hyperparams']
        if (set(original) != set(space) or any(original[k] not in values for k, values in space.items())
                or point_id(original) != anchor['id']):
            raise ValueError('Anchor identity or declared parameter support changed')
        points = [deepcopy(original)] + [dict(deepcopy(original), **{GAMMA: gamma, RATE: rate})
                                        for gamma in gammas for rate in rates]
        for point in points:
            own = candidate(point, 'training_anchor_belief_precision_grid', anchor_id=anchor['id'])
            item = bank.setdefault(own['id'], dict(own, anchor_ids=[]))
            if anchor['id'] not in item['anchor_ids']: item['anchor_ids'].append(anchor['id'])
    return list(bank.values())


def check_result(job: dict, result: dict, arrays: dict, belief: dict, precision: dict) -> None:
    """Verify actual training-only numerical execution and both state updates."""
    from ...evaluation import model_0923_separate_belief_refit as summary
    if (job['phase'] != 'training' or result.get('role') != 'training'
            or job.get('diagnostic') != 'belief_precision_training_grid'
            or result.get('subject') != 208 or result.get('condition') != 2
            or result['trials'] != (8 if job['smoke'] else job['training_trials'])):
        raise ValueError('Diagnostic crossed its original training boundary')
    summary.check_training_arrays(job, result, arrays)
    if job['probe_mechanism'] == 'SEPARATE':
        summary.check_execution(job, result, belief, precision)
        return
    copies = len(job['seeds']) * job['particles']
    if (job['probe_mechanism'] != 'ONLINE' or job.get('online_belief') is not True
            or job.get('belief_intervention') != MODES['ONLINE']
            or job.get('fixed_coordinates') != {TRANSFER: .01}
            or job['candidate']['hyperparams'].get(TRANSFER) != .01
            or job.get('retain_precision') is not True
            or job.get('structural_intervention') != 'retained_rule_precision'
            or belief.get('enabled') is not True or belief.get('belief_intervention') != MODES['ONLINE']
            or belief.get('fixed_coordinates') != job['fixed_coordinates']
            or belief.get('exact_transition_timing') is not True
            or belief.get('evidence_matches_retained_state') is not True
            or belief.get('reconstructions') != copies * (result['trials'] - 1)
            or precision.get('retained_precision') is not True
            or precision.get('structural_intervention') != 'retained_rule_precision'
            or precision.get('transitions') != copies * result['trials']):
        raise ValueError('ONLINE diagnostic execution or precision storage differs')


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict, object]:
    from ...evaluation.model_0923_h_validation import load, digest
    from ...utils.seeding import stable_seed
    settings = yaml.safe_load(config.read_text())
    if (settings['subject'] != 208 or settings['arm'] != 'SEARCH'
            or not 1 <= settings['parallel_budget'] <= 24
            or settings['anchor_gammas'] != [0., 1.] or settings['particles'] != 128
            or settings['filter_seeds'] != 2):
        raise ValueError('Declare the bounded training diagnostic and its shared worker allocation')
    source = ROOT / settings['coverage']; proof = load(source / 'completion.json')
    if proof['status'] != 'complete': raise ValueError('Need the completed training coverage audit')
    inputs = dict(proof['inputs_sha256'])
    for name, sha in proof['artifacts'].items(): inputs[str((source / name).resolve())] = sha
    for path, sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Completed training coverage input changed')
    def bind(path):
        path = Path(path).resolve(); inputs[str(path)] = digest(path); return path
    bind(config); bind(source / 'completion.json')
    fit = ROOT / settings['fit']; protocol = load(bind(fit / 'protocol.json'))
    spaces = load(bind(fit / 'parameter_spaces.json')); space = spaces[settings['arm']]
    if protocol['smoke'] or protocol['settings']['belief_intervention'] != MODES['SEPARATE']:
        raise ValueError('Require the scientific SEPARATE fitting policy')
    candidates = pd.read_csv(source / 'unique_candidates.csv'); anchors = []
    for gamma in settings['anchor_gammas']:
        rows = candidates[(candidates.mechanism == 'SEPARATE') & (candidates.arm == settings['arm'])
                          & (candidates.gamma == gamma)].sort_values(['training_nll', 'id'])
        if rows.empty: raise ValueError('Missing completed training anchor stratum')
        chosen = rows.iloc[0]; stage = load(bind(fit / (chosen.stage + '_completion.json')))
        group = next(g for g in stage['groups'] if (g['subject'], g['model']) == (settings['subject'], settings['arm']))
        original = next(r for r in group['rows'] if r['id'] == chosen.id)
        if (original['hyperparams'] != json.loads(chosen.hyperparams)
                or Path(original['score_file']).resolve() != Path(chosen.score_file).resolve()):
            raise ValueError('Training anchor belongs to a different fit or candidate')
        np.testing.assert_allclose(original['mean_nll'], chosen.training_nll, rtol=0, atol=1e-12)
        anchors.append(deepcopy(original))
    bank = build_bank(anchors, settings['gammas'], settings['learning_rates'], space)
    if smoke: bank = [point for point in bank if point['id'] in {a['id'] for a in anchors}]
    source_jobs = sorted((fit / 'jobs').glob(f'screen_S{settings["subject"]}_{settings["arm"]}_*.json'))
    if not source_jobs: raise ValueError('Missing actual fitting worker template')
    template = load(bind(source_jobs[0])); folder = Path(template['folder']); meta = load(bind(folder / 'prepared.json'))
    if (template['phase'] != 'training' or template['smoke'] or template['model'] != settings['arm']
            or meta['subject'] != settings['subject'] or meta['training_trials'] != 384
            or meta['condition'] != 2
            or Path(template['folder']).resolve() != (ROOT / protocol['settings']['data'] / 'subject_208').resolve()):
        raise ValueError('Template differs from the declared original training task')
    for name, sha in template['source_sha256'].items():
        if digest(bind(ROOT / name)) != sha: raise ValueError('Original fitting implementation changed')
    hashes = dict(template['source_sha256'])
    for name in (ENTRY, PROBE): hashes[name] = digest(bind(ROOT / name))
    seeds = [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'], subject=settings['subject'], repeat=i))
             for i in range(1 if smoke else settings['filter_seeds'])]
    jobs = []
    for point in bank:
        for mode, intervention in MODES.items():
            name = mode + '_' + point['id']
            job = dict(deepcopy(template), candidate=deepcopy(point), name=name, seeds=seeds,
                particles=4 if smoke else settings['particles'], smoke=smoke,
                target=str((output / 'chunks' / name).resolve()), source_sha256=hashes,
                probe_mechanism=mode, belief_intervention=intervention, online_belief=mode == 'ONLINE',
                diagnostic='belief_precision_training_grid', training_trials=meta['training_trials'])
            jobs.append(job)
    recalled = helpers().helpers().helpers().helpers().helpers()
    original_settings = deepcopy(protocol['settings']); original_settings['parallel_budget'] = settings['parallel_budget']
    return dict(settings=original_settings, diagnostic_settings=settings, smoke=smoke, jobs=jobs,
        anchors=anchors, bank=bank, input_sha256=inputs, source_sha256=hashes,
        workers=1 if smoke else min(settings['parallel_budget'], len(jobs)), numerical_threads=1,
        training_trials=meta['training_trials'], scope='Training-only gamma/precision interaction at two fixed anchors',
        selection='Best completed SEARCH training score within gamma0 and gamma1; all other coordinates fixed per anchor',
        original_nominations_changed=False, oral_used=False, later_choices_scored=False), recalled


def summarize(output: Path, protocol: dict) -> None:
    from ...evaluation.model_0923_h_validation import load, digest, jackknife_scores, jackknife_se
    from .model_0923_h_validation import write
    inputs = dict(protocol['input_sha256']); scores, blocks, banks = [], [], {}
    job_inputs = {}
    for job in protocol['jobs']:
        target = Path(job['target']); paths = [target.with_suffix(s) for s in ('.json', '.npz', '.precision.json', '.experiment.json')]
        paths.append(target.with_suffix('.belief.json' if job['probe_mechanism'] == 'SEPARATE' else '.online.json'))
        paths.append(output / 'jobs' / (job['name'] + '.json'))
        job_inputs.update({str(p.resolve()): digest(p) for p in paths})
        if load(paths[-1]) != job: raise ValueError('Executed worker differs from the declared grid')
        result, precision, belief = load(paths[0]), load(paths[2]), load(paths[4])
        if digest(paths[1]) != result['predictions_sha256']: raise ValueError('Training prediction array changed')
        with np.load(paths[1]) as z: arrays = {k: z[k].copy() for k in z.files}
        check_result(job, result, arrays, belief, precision)
        training = Path(job['folder']) / 'training.npz'; inputs[str(training.resolve())] = digest(training)
        with np.load(training) as z: correct = z['feedback'][:result['trials']] == 1
        probability = arrays['observed_probability']; loss = -np.log(np.maximum(probability.mean(0), np.finfo(float).tiny))
        point = job['candidate']; values = point['hyperparams']
        tags = dict(mechanism=job['probe_mechanism'], candidate=point['id'], anchor_ids=json.dumps(point['anchor_ids']),
                    gamma=values[GAMMA], learning_rate=values[RATE], trials=result['trials'])
        scores.append(dict(tags, training_nll=loss.mean(), nll_mc_se=jackknife_se(jackknife_scores(probability)) if len(probability)>1 else None,
                           correct_loss=(loss * correct).mean(), error_loss=(loss * ~correct).mean()))
        width = 8 if protocol['smoke'] else 64
        for first in range(0, len(loss), width):
            stop = min(len(loss), first + width); index = slice(first, stop)
            blocks.append(dict(tags, first=first + 1, last=stop, training_nll=loss[index].mean(),
                               correct_loss=(loss[index] * correct[index]).mean(), error_loss=(loss[index] * ~correct[index]).mean()))
        banks[job['probe_mechanism'], point['id']] = arrays
    contrasts = []
    for point in protocol['bank']:
        old, new = (banks[m, point['id']] for m in MODES)
        np.testing.assert_array_equal(old['seeds'], new['seeds'])
        a, b = old['observed_probability'], new['observed_probability']
        difference = np.log(np.maximum(a.mean(0), np.finfo(float).tiny)) - np.log(np.maximum(b.mean(0), np.finfo(float).tiny))
        contrasts.append(dict(candidate=point['id'], gamma=point['hyperparams'][GAMMA], learning_rate=point['hyperparams'][RATE],
            delta_training_nll=difference.mean(), paired_mc_se=jackknife_se(jackknife_scores(b)-jackknife_scores(a)) if len(a)>1 else None))
    inputs.update(job_inputs)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Diagnostic input changed during execution')
    tables = dict(scores=pd.DataFrame(scores), training_block64=pd.DataFrame(blocks), contrasts=pd.DataFrame(contrasts))
    for name, table in tables.items(): table.to_csv(output / (name + '.csv'), index=False)
    write(output / 'completion.json', dict(status='complete', smoke=protocol['smoke'], jobs=len(protocol['jobs']),
        parameters=len(protocol['bank']), inputs_sha256=inputs, original_nominations_changed=False,
        oral_used=False, later_choices_scored=False, protocol_sha256=digest(output / 'protocol.json'),
        artifacts={name + '.csv': digest(output / (name + '.csv')) for name in tables},
        interpretation='Local training-only interaction; common points and seeds; both anchor contexts retained'))


def run(config: Path, output: Path, smoke: bool = False) -> None:
    from .model_0923_h_validation import write
    if output.exists(): raise FileExistsError('Use a new belief/precision diagnostic directory')
    protocol, recalled = prepare(config, output, smoke)
    output.mkdir(parents=True); write(output / 'protocol.json', protocol)
    shutil.copy2(config, output / 'config.yaml'); shutil.copy2(__file__, output / 'analysis_source.py')
    recalled.ENTRY = ENTRY
    recalled.execute(protocol['jobs'], output, protocol['workers'])
    summarize(output, protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'): parser.add_argument('--' + name, type=Path)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Use --config and --output, or internal --worker')
