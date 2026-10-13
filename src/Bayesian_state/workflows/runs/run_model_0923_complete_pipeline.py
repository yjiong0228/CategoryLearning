"""Continue completed Plus fits once, then automatically deliver subject reports.

Preparation preserves the original model snapshot and overlays the current
analysis layer. Each subject receives bounded support probes, an independent
fixed-bank audit, and evaluation v2, including unresolved results. Source fits
stay read-only. Stage caches and completed report hashes support safe resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import yaml

from src.Bayesian_state.evaluation.model_0923_protocol import load_protocol
from src.Bayesian_state.optimization.adaptive_runtime import digest, freeze_json
from src.Bayesian_state.optimization.model_0923_continuation import load_policy
from src.Bayesian_state.utils.paths import ROOT_DIR


DEFAULT_POLICY = ROOT_DIR / 'configs/shared/model0923_continuation.yaml'


def now() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def prepare(source_queue: Path, output: Path, policy_path: Path, *, smoke: bool = False) -> dict:
    source_queue, output, policy_path = source_queue.resolve(), output.resolve(), policy_path.resolve()
    original = json.loads((source_queue / 'queue_plan.json').read_text())
    archived_source = Path(original['source'])
    for name, expected in original['snapshot_sha256'].items():
        if digest(archived_source / name) != expected:
            raise ValueError(f'Original queued snapshot changed: {name}')
    policy = load_policy(policy_path, smoke)
    queued = {j['subject']: j for j in original['jobs']}
    for job in policy['jobs']:
        if job['subject'] not in queued or not (Path(queued[job['subject']]['output']) / 'fit_results.json').is_file():
            raise ValueError('Every continuation subject needs a completed source fit')
    project = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT_DIR, text=True).strip())
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    shutil.copytree(archived_source, source, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for path in source.rglob('*'):
        if path.is_dir():
            path.chmod(0o755)
    source.chmod(0o755)
    # Preserve every old scientific source file. Updated report code belongs
    # to the analysis paths permitted by the saved-fit integrity adapter.
    for folder in ('src/Bayesian_state/evaluation', 'src/Bayesian_state/workflows/analysis',
                   'CategoryLearning_codes/figures'):
        for path in (ROOT_DIR / folder).rglob('*.py'):
            if 'outputs' in path.parts or '__pycache__' in path.parts:
                continue
            target = source / path.relative_to(ROOT_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for name in ('src/oral_coding.py', 'src/Bayesian_state/optimization/model_0923_continuation.py',
                 'src/Bayesian_state/workflows/runs/run_model_0923_complete_pipeline.py'):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT_DIR / name, target)
    canonical_figures = project / 'CategoryLearning_codes/figures/outputs'
    canonical_figures.mkdir(parents=True, exist_ok=True)
    (source / 'CategoryLearning_codes/figures/outputs').symlink_to(canonical_figures, target_is_directory=True)
    runtime = source / 'continuation_runtime'
    runtime.mkdir()
    # Runtime absolute paths are confined to generated artifacts.
    frozen_policy = yaml.safe_load(policy_path.read_text())
    for job in frozen_policy['jobs']:
        job['evaluation_config'] = next(j['evaluation_config'] for j in policy['jobs'] if j['subject'] == job['subject'])
    (runtime / 'policy.yaml').write_text(yaml.safe_dump(frozen_policy, sort_keys=False))
    jobs, inputs = [], {str(policy_path): digest(policy_path), str(source_queue / 'queue_plan.json'): digest(source_queue / 'queue_plan.json')}
    for job in policy['jobs']:
        sid = job['subject']
        template = Path(job['evaluation_config'])
        settings = load_protocol(template, smoke=smoke)
        original_result = json.loads((Path(queued[sid]['output']) / 'fit_results.json').read_text())['subjects'][str(sid)]
        needs_continuation = bool({'boundary_review_required', 'independent_audit_unresolved'} & set(original_result['issues']))
        followup = output / f'subject_{sid}' / 'continuation'
        report = output / f'subject_{sid}' / 'evaluation'
        figures = canonical_figures / 'fig3' / output.name / f'subject_{sid}'
        if report.exists() or figures.exists():
            raise FileExistsError('Choose new report and PNG directories')
        settings.update(fit_run=str(followup) if needs_continuation else queued[sid]['output'],
                        output_dir=str(report), figure_output_dir=str(figures), subjects=[sid])
        evaluation = runtime / f'evaluate_{sid}.yaml'
        evaluation.write_text(yaml.safe_dump(settings, sort_keys=False))
        jobs.append(dict(subject=sid, source_fit=queued[sid]['output'], continuation=str(followup), needs_continuation=needs_continuation,
                         evaluation=str(report), figures=str(figures), evaluation_config=str(evaluation)))
        inputs[str(template)] = digest(template)
        criterion = settings['task_criterion']
        if criterion['mode'] == 'exp123':
            inputs[criterion['source_program']] = digest(Path(criterion['source_program']))
        oral = settings.get('oral_alignment', {})
        if oral.get('mode') == 'predict':
            inputs[oral['trials_path']] = digest(Path(oral['trials_path']))
    files = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file() and not p.is_symlink()}
    plan = dict(protocol=policy['protocol'], source=str(source), original_queue=str(source_queue),
        policy=str(runtime / 'policy.yaml'), jobs=jobs, smoke=smoke, evaluation_attempts=policy['evaluation_attempts'], prepared_at=now(),
        source_model_commit=original.get('source_commit'),
        analysis_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT_DIR, text=True).strip(),
        snapshot_sha256=files, input_sha256=inputs,
        scope='One bounded continuation per subject, then evaluation with explicit acceptance and missing-data statuses')
    freeze_json(output / 'pipeline_plan.json', plan)
    freeze_json(output / 'figure_contract.json', dict(backend='python', reuse='existing evaluation v2 unchanged',
        conclusion='Assess individual behavioral agreement, internal states and autonomous learning with recorded fit acceptance',
        archetype='quantitative grid', export='PNG only',
        evidence=['behavior_alignment', 'oral_alignment', 'representations', 'behavior', 'diagnostics'],
        uncertainty='Fixed-parameter particle marginals and autonomous simulation variation; numerical seeds are integration noise',
        exclusions='First valid trial initializes the model and is excluded from choice NLL; missing oral reports retain unavailable status'))
    for path in source.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    write_index(output, plan, dict(status='prepared', jobs=[]))
    return plan


def write_index(output: Path, plan: dict, state: dict) -> None:
    rows = []
    statuses = {j['subject']: j for j in state['jobs']}
    for job in plan['jobs']:
        sid = job['subject']
        status = statuses.get(sid, {})
        report = Path(status.get('evaluation_output', job['evaluation'])) / 'report.html'
        link = f'<a href="{html.escape(os.path.relpath(report, output))}">逐人报告</a>' if report.is_file() else '报告待生成'
        figure_link = html.escape(os.path.relpath(Path(status.get('figure_directory', job['figures'])), output))
        rows.append(f'<tr><td>{sid}</td><td>{html.escape(status.get("stage", "pending"))}</td>'
                    f'<td>{html.escape(status.get("fit_status", "pending"))}</td><td>{link}</td>'
                    f'<td><a href="{figure_link}">PNG目录</a></td></tr>')
    text = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Model 0923 完整流程</title>'
            '<style>body{max-width:1000px;margin:32px auto;padding:20px;font:17px/1.7 sans-serif}'
            'td,th{padding:8px 16px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
            '<h1>Model 0923：拟合后的自动续接</h1>'
            '<p>每名被试依次完成一次有上限的边界补查、独立评分和逐人评估。'
            '原拟合结果保留；参数验收状态随新报告交付。未解决状态继续生成报告。</p>'
            f'<p>流程状态：{html.escape(state["status"])}。<a href="pipeline_status.json">详细进度</a></p>'
            '<table><tr><th>被试</th><th>阶段</th><th>参数验收</th><th>报告</th><th>图</th></tr>'
            + ''.join(rows) + '</table><p>口述缺失、任务标准未配置、恢复与留出预测等项目按实际状态记录。</p></html>')
    temporary = output / '.index.tmp'
    temporary.write_text(text)
    temporary.replace(output / 'index.html')


def run_prepared(output: Path) -> dict:
    output = output.resolve()
    plan = json.loads((output / 'pipeline_plan.json').read_text())
    source = Path(plan['source'])
    for name, expected in plan['snapshot_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError(f'Frozen pipeline source changed: {name}')
    for name, expected in plan['input_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Frozen pipeline input changed: {name}')
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR=str(output / 'mpl_cache'),
        NUMBA_CACHE_DIR=str(output / 'numba_cache'), MPLBACKEND='Agg')
    state = dict(status='running', started_at=now(), jobs=[])

    def publish() -> None:
        state['updated_at'] = now()
        temporary = output / '.pipeline_status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output / 'pipeline_status.json')
        write_index(output, plan, state)

    def child(command: list[str], log: Path, row: dict) -> None:
        with log.open('a') as handle:
            process = subprocess.Popen(command, cwd=source, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            publish()
            code = process.wait()
        row['returncode'] = code
        if code:
            row['stage'] += '_failed'
            raise RuntimeError(f'Subject {row["subject"]} stage failed; inspect {log}')

    with (output / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for job in plan['jobs']:
                sid = job['subject']
                row = dict(subject=sid, stage='continuation', started_at=now())
                state['jobs'].append(row)
                publish()
                followup = Path(job['continuation'])
                if job['needs_continuation']:
                    command = [sys.executable, '-m', 'src.Bayesian_state.optimization.model_0923_continuation',
                               '--source-fit', job['source_fit'], '--policy', plan['policy'], '--output-dir', str(followup)]
                    if (followup / 'manifest.json').exists():
                        command.append('--resume')
                    if plan['smoke']:
                        command.append('--smoke')
                    child(command, output / f'subject_{sid}_continuation.log', row)
                else:
                    followup = Path(job['source_fit'])
                result = json.loads((followup / 'fit_results.json').read_text())['subjects'][str(sid)]
                row.update(stage='evaluation', fit_status=result['status'], fit_issues=result['issues'],
                           selected=result['selected'], selected_mean_nll=result['selected_mean_nll'])
                publish()
                report, figures, evaluation_config = evaluation_location(job, output, plan['evaluation_attempts'])
                row.update(evaluation_output=str(report), figure_directory=str(figures), evaluation_config=str(evaluation_config))
                publish()
                if report.exists():
                    manifest_path = report / 'manifest.json'
                    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                else:
                    command = [sys.executable, '-m', 'CategoryLearning_codes.figures.fig3.model0923_evaluation',
                               '--config', str(evaluation_config)]
                    if plan['smoke']:
                        command += ['--smoke', '--output-dir', str(report), '--figure-output-dir', str(figures)]
                    child(command, output / f'subject_{sid}_evaluation.log', row)
                    manifest = json.loads((report / 'manifest.json').read_text())
                if manifest.get('status') != 'completed':
                    raise RuntimeError('Evaluation returned without a completed report bundle')
                verify_report_artifacts(report, figures, manifest['artifacts_sha256'])
                for name, expected in manifest['input_sha256'].items():
                    if digest(Path(name)) != expected:
                        raise ValueError('Completed evaluation input changed')
                row.update(stage='completed', ended_at=now())
                publish()
            state.update(status='completed', ended_at=now())
            publish()
        except Exception as error:
            state.update(status='failed', error=str(error), ended_at=now())
            publish()
            raise
    return state


def evaluation_location(job: dict, output: Path, attempts: int) -> tuple[Path, Path, Path]:
    """Retain partial reports and choose a bounded new retry directory on resume."""
    for attempt in range(1, attempts + 1):
        report = Path(job['evaluation'])
        figures = Path(job['figures'])
        config = Path(job['evaluation_config'])
        if attempt > 1:
            report = report.with_name(f'{report.name}_retry_{attempt}')
            figures = figures.with_name(f'{figures.name}_retry_{attempt}')
            settings = yaml.safe_load(Path(job['evaluation_config']).read_text())
            settings.update(output_dir=str(report), figure_output_dir=str(figures))
            config = output / 'runtime' / f'evaluate_{job["subject"]}_retry_{attempt}.yaml'
            config.parent.mkdir(parents=True, exist_ok=True)
            if not config.exists():
                with config.open('x') as stream:
                    yaml.safe_dump(settings, stream, sort_keys=False)
            elif yaml.safe_load(config.read_text()) != settings:
                raise ValueError('Frozen evaluation retry configuration differs')
        manifest = report / 'manifest.json'
        if manifest.is_file() and json.loads(manifest.read_text()).get('status') == 'completed':
            return report, figures, config
        if not report.exists() and not figures.exists():
            return report, figures, config
    raise RuntimeError('Evaluation retry budget exhausted; partial outputs retained')


def verify_report_artifacts(report: Path, figures: Path, hashes: dict[str, str]) -> None:
    """Resolve portable report/figure receipts against their separate roots."""
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'report', 'figures'}:
            raise ValueError('Invalid report artifact receipt path')
        root = report if relative.parts[0] == 'report' else figures
        if digest(root.joinpath(*relative.parts[1:])) != expected:
            raise ValueError('Completed evaluation artifact changed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-queue', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.run_prepared:
        parser.error('Choose prepare-only or run-prepared')
    if not args.run_prepared:
        if args.source_queue is None:
            parser.error('Preparation requires --source-queue')
        plan = prepare(args.source_queue, args.output_dir, args.policy, smoke=args.smoke)
        if args.prepare_only:
            print(json.dumps({k: plan[k] for k in ('protocol', 'jobs', 'smoke')}, ensure_ascii=False, indent=2))
            return
    print(json.dumps(run_prepared(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
